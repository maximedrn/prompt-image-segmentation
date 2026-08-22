"""The generated API document, and the pages that render it.

No model and no Redis: building the transport is enough to produce the
document, which is the point of generating it from the routes rather
than maintaining it by hand.
"""

# pytest collects the test functions by name; nothing in the module
# calls them, which is what ``allow-global-unused-variables`` flags.
# pylint: disable=unused-variable

from http import HTTPStatus
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from app.interfaces.http.application import create_app
from app.interfaces.http.constants import (
    HttpRoute,
    HttpVerb,
    OpenApiKey,
    OpenApiTag,
)
from app.interfaces.http.schemas import JsonValue
from app.settings import AuthMode, Settings

type Document = dict[str, JsonValue]

#: Sphinx field syntax. The docstrings use it, the published document
#: must not: FastAPI falls back to the docstring for any route that
#: does not pass ``description``.
SPHINX_MARKUP: Final[tuple[str, ...]] = (
    ":param ",
    ":type ",
    ":returns:",
    ":rtype:",
    ":raises ",
)
DOCUMENTATION: Final[tuple[str, ...]] = (
    HttpRoute.docs,
    HttpRoute.redoc,
    HttpRoute.openapi,
)
#: Every route the document is expected to carry. The socket is absent
#: on purpose: OpenAPI cannot describe one.
DOCUMENTED_ROUTES: Final[tuple[tuple[str, str], ...]] = (
    (HttpVerb.get, HttpRoute.health),
    (HttpVerb.get, HttpRoute.ready),
    (HttpVerb.get, HttpRoute.segmenters),
    (HttpVerb.post, HttpRoute.jobs),
    (HttpVerb.get, HttpRoute.job),
    (HttpVerb.delete, HttpRoute.job),
)


def _mapping(value: JsonValue) -> Document:
    # ``JsonValue`` is a union all the way down; narrowing as it is
    # walked is what keeps this file free of casts and of ignores.
    assert isinstance(value, dict), f"expected an object, got {type(value)}"
    return value


def _text(value: JsonValue) -> str:
    assert isinstance(value, str), f"expected a string, got {type(value)}"
    return value


def _transport() -> FastAPI:
    return create_app(Settings(AUTH_MODE=AuthMode.NONE, _env_file=None))


@pytest.fixture(name="document")
def document_fixture() -> Document:
    """Build the transport and return the document it generates.

    :returns: The OpenAPI document.
    :rtype: dict[str, JsonValue]
    """
    return _transport().openapi()


def _operations(document: Document) -> dict[tuple[str, str], Document]:
    """Flatten the document's paths into one operation per entry.

    :param document: The OpenAPI document.
    :type document: dict[str, JsonValue]
    :returns: Operations keyed by ``(verb, path)``.
    :rtype: dict[tuple[str, str], dict[str, JsonValue]]
    """
    return {
        (verb, path): _mapping(operation)
        for path, operations in _mapping(document[OpenApiKey.paths]).items()
        for verb, operation in _mapping(operations).items()
    }


def _statuses(document: Document, verb: str, path: str) -> set[str]:
    """Return the statuses one operation declares.

    :param document: The OpenAPI document.
    :type document: dict[str, JsonValue]
    :param verb: HTTP method, lowercased.
    :type verb: str
    :param path: Templated route path.
    :type path: str
    :returns: The declared status codes, as the document spells them.
    :rtype: set[str]
    """
    operation: Document = _operations(document)[(verb, path)]
    return set(_mapping(operation[OpenApiKey.responses]))


def test_every_route_is_documented(document: Document) -> None:
    """The document is the endpoint listing, so it has to be complete.

    :param document: The OpenAPI document.
    :type document: dict[str, JsonValue]
    """
    assert set(_operations(document)) == set(DOCUMENTED_ROUTES)


def test_no_route_publishes_its_sphinx_docstring(document: Document) -> None:
    """A reader of ``/docs`` is not a reader of the codebase.

    Every route passes an explicit ``description``. Dropping one makes
    FastAPI fall back to the docstring, which would publish the whole
    Sphinx field list to the page.

    :param document: The OpenAPI document.
    :type document: dict[str, JsonValue]
    """
    verb: str
    path: str
    operation: Document
    for (verb, path), operation in _operations(document).items():
        description: str = _text(operation[OpenApiKey.description])
        assert description, f"{verb} {path} has no description"
        for markup in SPHINX_MARKUP:
            assert (
                markup not in description
            ), f"{verb} {path} publishes {markup!r} to /docs"


def test_the_job_routes_declare_only_what_they_can_answer(
    document: Document,
) -> None:
    """Polling cannot answer 413, and acceptance cannot answer 409.

    One shared response list is easy and wrong: it tells a caller to
    handle statuses the route has no way of producing.

    :param document: The OpenAPI document.
    :type document: dict[str, JsonValue]
    """
    too_large: str = str(int(HTTPStatus.CONTENT_TOO_LARGE))
    conflict: str = str(int(HTTPStatus.CONFLICT))

    acceptance: set[str] = _statuses(document, HttpVerb.post, HttpRoute.jobs)
    assert too_large in acceptance
    assert conflict not in acceptance

    polling: set[str] = _statuses(document, HttpVerb.get, HttpRoute.job)
    assert too_large not in polling
    assert conflict not in polling

    assert conflict in _statuses(document, HttpVerb.delete, HttpRoute.job)


def test_the_socket_is_described_where_openapi_cannot_reach(
    document: Document,
) -> None:
    """OpenAPI has no socket, so the prose has to carry it.

    :param document: The OpenAPI document.
    :type document: dict[str, JsonValue]
    """
    information: Document = _mapping(document[OpenApiKey.info])
    assert HttpRoute.job_events in _text(information[OpenApiKey.description])


def test_basic_auth_is_offered_so_the_page_can_authorise(
    document: Document,
) -> None:
    """Without a declared scheme there is no Authorize button.

    :param document: The OpenAPI document.
    :type document: dict[str, JsonValue]
    """
    components: Document = _mapping(document[OpenApiKey.components])
    assert _mapping(components[OpenApiKey.security_schemes])

    operation: Document = _operations(document)[
        (HttpVerb.post, HttpRoute.jobs)
    ]
    assert operation[OpenApiKey.security]


def test_both_tags_carry_a_description(document: Document) -> None:
    """A bare heading tells a reader nothing about the group.

    :param document: The OpenAPI document.
    :type document: dict[str, JsonValue]
    """
    tags: JsonValue = document[OpenApiKey.tags]
    assert isinstance(tags, list)
    described: dict[str, str] = {
        _text(_mapping(tag)[OpenApiKey.name]): _text(
            _mapping(tag)[OpenApiKey.description]
        )
        for tag in tags
    }
    assert set(described) == {OpenApiTag.meta, OpenApiTag.segmentation}
    assert all(described.values())


@pytest.mark.parametrize("path", DOCUMENTATION)
def test_the_documentation_pages_are_served(path: str) -> None:
    """The point of the whole thing is that a browser can open it.

    :param path: Documentation path under test.
    :type path: str
    """
    response: Response = TestClient(_transport()).get(path)
    assert response.status_code == HTTPStatus.OK
