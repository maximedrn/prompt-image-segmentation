"""Trust-boundary behavior: auth, upload caps, rate limit, error shape.

None of these reach the models: every check fires in a dependency or in
the upload reader, so the suite stays fast. The transport is built
without running the lifespan, which is what keeps the weights out of it.
"""

# pytest collects the test functions by name; nothing in the module
# calls them, which is what ``allow-global-unused-variables`` flags.
# pylint: disable=unused-variable

from http import HTTPStatus
from io import BytesIO
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from PIL import Image as PilImage

from app.interfaces.http.application import create_app
from app.interfaces.http.constants import (
    HEADER,
    ROUTE,
    ErrorCode,
    HealthState,
)
from app.settings import AuthMode, Settings
from tests.conftest import PASSWORD, USERNAME

AUTH: Final[tuple[str, str]] = (USERNAME, PASSWORD)
#: Small enough that a test payload crosses it cheaply.
UPLOAD_LIMIT: Final[int] = 2048
BUDGET: Final[int] = 5


def _settings(**overrides: object) -> Settings:
    """Build settings for a test transport.

    :param overrides: Field overrides.
    :type overrides: object
    :returns: Validated settings.
    :rtype: app.settings.Settings
    """
    values: dict[str, object] = {
        "AUTH_MODE": AuthMode.BASIC,
        "SEGMENTATION_USERNAME": USERNAME,
        "SEGMENTATION_PASSWORD": PASSWORD,
        "ENABLE_UI": False,
        "MAX_UPLOAD_BYTES": UPLOAD_LIMIT,
        "RATE_LIMIT_REQUESTS": BUDGET,
        "RATE_LIMIT_WINDOW_SECONDS": 60.0,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    """Return a transport with no lifespan run, so no model loads.

    :returns: A test client over a freshly built transport.
    :rtype: fastapi.testclient.TestClient
    """
    transport: FastAPI = create_app(_settings())
    return TestClient(transport)


def _png(width: int = 8, height: int = 8) -> bytes:
    """Encode a small valid PNG.

    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :returns: PNG bytes.
    :rtype: bytes
    """
    buffer: BytesIO = BytesIO()
    PilImage.new("RGB", (width, height), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def _post(
    client: TestClient,
    payload: bytes,
    auth: tuple[str, str] | None = AUTH,
) -> Response:
    """POST a payload to the segmentation route.

    :param client: Test client.
    :type client: fastapi.testclient.TestClient
    :param payload: Raw file bytes.
    :type payload: bytes
    :param auth: Basic credentials, or ``None`` to omit them.
    :type auth: tuple[str, str] | None
    :returns: The HTTP response.
    :rtype: httpx.Response
    """
    return client.post(
        ROUTE.segment,
        files={"image": ("input.png", payload, "image/png")},
        data={"prompt": "dog"},
        auth=auth,
    )


def test_liveness_needs_no_auth(client: TestClient) -> None:
    """Liveness must answer for the container runtime."""
    response: Response = client.get(ROUTE.health)
    assert response.status_code == HTTPStatus.OK
    assert response.json()["status"] == HealthState.OK


def test_readiness_is_503_before_the_lifespan_runs(
    client: TestClient,
) -> None:
    """Readiness keeps traffic away while nothing is wired yet."""
    response: Response = client.get(ROUTE.ready)
    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["error"] == ErrorCode.NOT_READY


def test_segment_requires_credentials(client: TestClient) -> None:
    """The API is closed without valid Basic credentials."""
    assert (
        _post(client, _png(), auth=None).status_code == HTTPStatus.UNAUTHORIZED
    )
    unauthorized: Response = _post(client, _png(), auth=("wrong", "wrong"))
    assert unauthorized.status_code == HTTPStatus.UNAUTHORIZED
    assert unauthorized.json()["error"] == ErrorCode.UNAUTHORIZED
    assert HEADER.authenticate in unauthorized.headers


def test_oversized_upload_is_refused(client: TestClient) -> None:
    """An upload past the cap is a 413, never fully buffered."""
    response: Response = _post(
        client, b"\x89PNG\r\n\x1a\n" + b"\x00" * (UPLOAD_LIMIT * 2)
    )
    assert response.status_code == HTTPStatus.CONTENT_TOO_LARGE
    assert response.json()["error"] == ErrorCode.UPLOAD_TOO_LARGE


def test_empty_upload_is_refused(client: TestClient) -> None:
    """An empty body is a client error, not a traceback."""
    response: Response = _post(client, b"")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_CONTENT
    assert response.json()["error"] == ErrorCode.INVALID_IMAGE


def test_undecodable_upload_is_refused(client: TestClient) -> None:
    """Bytes Pillow cannot open yield 422 rather than a 500."""
    response: Response = _post(client, b"definitely not an image")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_CONTENT
    assert response.json()["error"] == ErrorCode.INVALID_IMAGE


def test_rate_limit_returns_429_with_retry_after(
    client: TestClient,
) -> None:
    """Past the budget the caller is refused and told when to retry."""
    for _ in range(BUDGET):
        _post(client, b"not an image")
    response: Response = _post(client, b"not an image")
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    assert response.json()["error"] == ErrorCode.RATE_LIMITED
    assert int(response.headers["retry-after"]) >= 1


def test_basic_auth_without_credentials_refuses_to_start() -> None:
    """An empty secret must fail loudly instead of serving openly."""
    with pytest.raises(ValueError, match="AUTH_MODE=basic requires"):
        _settings(SEGMENTATION_USERNAME=None, SEGMENTATION_PASSWORD=None)


def test_auth_mode_none_is_explicit() -> None:
    """Serving without auth stays possible, but only on purpose."""
    settings: Settings = _settings(
        AUTH_MODE=AuthMode.NONE,
        SEGMENTATION_USERNAME=None,
        SEGMENTATION_PASSWORD=None,
    )
    assert settings.auth_enabled is False
