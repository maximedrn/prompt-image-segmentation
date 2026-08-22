"""Trust-boundary behavior: auth, upload caps, rate limit, error shape.

None of these reach the models: every check fires in a dependency or in
the upload reader, so the suite stays fast. The transport is built
without running the lifespan, which is what keeps the weights out of it.
"""

# pytest collects the test functions by name; nothing in the module
# calls them, which is what ``allow-global-unused-variables`` flags.
# pylint: disable=unused-variable

from base64 import b64encode
from http import HTTPStatus
from io import BytesIO
from typing import Final, final

import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient
from httpx import Response
from PIL import Image as PilImage

from app.application.jobs import JobResult
from app.domain import Job, JobState, queued, start, succeed
from app.interfaces.http.application import create_app
from app.interfaces.http.auth import socket_authorised
from app.interfaces.http.constants import (
    BasicScheme,
    ErrorCode,
    FormField,
    HeaderName,
    HealthState,
    HttpRoute,
)
from app.interfaces.http.schemas import ErrorSchema, HealthSchema, JobSchema
from app.settings import AuthMode, EnvVar, Settings
from tests.conftest import (
    BUDGET,
    PASSWORD,
    UPLOAD_LIMIT,
    USERNAME,
    WINDOW_SECONDS,
)

type SettingValue = str | int | float | bool | AuthMode | None

#: pydantic-settings reads this, so it is spelled the way it spells it.
NO_ENV_FILE: Final[str] = "_env_file"

AUTH: Final[tuple[str, str]] = (USERNAME, PASSWORD)
JOB_ID: Final[str] = "job-under-test"
POLICY_VIOLATION: Final[int] = 1008
NORMAL_CLOSE: Final[int] = 1000
TRY_AGAIN_LATER: Final[int] = 1013
#: Small enough that a PNG crossing it stays quick to encode.
PIXEL_CEILING: Final[int] = 250_000
PROMPT: Final[str] = "dog"
UPLOAD_NAME: Final[str] = "input.png"
UPLOAD_CONTENT_TYPE: Final[str] = "image/png"


@final
class FakeStore:
    """A store the socket can read, with no Redis behind it."""

    def __init__(self) -> None:
        """Start with nothing stored."""
        self.jobs: dict[str, tuple[Job, JobResult | None]] = {}

    async def read(
        self, identifier: str
    ) -> tuple[Job, JobResult | None] | None:
        """Return what the test put there.

        :param identifier: Job identity.
        :type identifier: str
        :returns: The job and its result, or ``None``.
        :rtype: tuple[app.domain.Job, app.application.jobs.JobResult
            | None] | None
        """
        return self.jobs.get(identifier)


def _basic() -> dict[str, str]:
    """Return the header a socket carries its credentials in.

    :returns: An ``Authorization`` header.
    :rtype: dict[str, str]
    """
    credentials: str = f"{USERNAME}{BasicScheme.separator}{PASSWORD}"
    encoded: str = b64encode(credentials.encode()).decode()
    return {BasicScheme.header: f"{BasicScheme.prefix}{encoded}"}


def _settings(**overrides: SettingValue) -> Settings:
    """Build settings for a test transport.

    :param overrides: Field overrides, keyed by environment name.
    :type overrides: SettingValue
    :returns: Validated settings.
    :rtype: app.settings.Settings
    """
    values: dict[str, SettingValue] = {
        EnvVar.AUTH_MODE: AuthMode.BASIC,
        EnvVar.SEGMENTATION_USERNAME: USERNAME,
        EnvVar.SEGMENTATION_PASSWORD: PASSWORD,
        EnvVar.ENABLE_UI: False,
        EnvVar.MAX_UPLOAD_BYTES: UPLOAD_LIMIT,
        EnvVar.RATE_LIMIT_REQUESTS: BUDGET,
        EnvVar.RATE_LIMIT_WINDOW_SECONDS: float(WINDOW_SECONDS),
        # Pydantic settings' own switch, not one of ours: it stops the
        # repository's .env from reaching a test.
        NO_ENV_FILE: None,
    }
    values.update(overrides)
    return Settings(**values)


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
    **options: str,
) -> Response:
    """POST a payload to the job route.

    :param client: Test client.
    :type client: fastapi.testclient.TestClient
    :param payload: Raw file bytes.
    :type payload: bytes
    :param auth: Basic credentials, or ``None`` to omit them.
    :type auth: tuple[str, str] | None
    :param options: Extra form fields, sent verbatim.
    :type options: str
    :returns: The HTTP response.
    :rtype: httpx.Response
    """
    return client.post(
        HttpRoute.jobs,
        files={FormField.image: (UPLOAD_NAME, payload, UPLOAD_CONTENT_TYPE)},
        data={FormField.prompt: PROMPT, **options},
        auth=auth,
    )


def _failure(response: Response) -> ErrorSchema:
    """Validate a failure body into the schema it claims to be.

    Reading ``response.json()["error"]`` would assert on a key rather
    than on the contract; this fails if the shape drifts at all.

    :param response: What the transport answered.
    :type response: httpx.Response
    :returns: The parsed failure body.
    :rtype: app.interfaces.http.schemas.ErrorSchema
    """
    return ErrorSchema.model_validate(response.json())


def test_liveness_needs_no_auth(client: TestClient) -> None:
    """Liveness must answer for the container runtime."""
    response: Response = client.get(HttpRoute.health)
    assert response.status_code == HTTPStatus.OK
    assert (
        HealthSchema.model_validate(response.json()).status == HealthState.OK
    )


def test_readiness_is_503_before_the_lifespan_runs(
    client: TestClient,
) -> None:
    """Readiness keeps traffic away while nothing is wired yet."""
    response: Response = client.get(HttpRoute.ready)
    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert _failure(response).error == ErrorCode.NOT_READY


def test_jobs_require_credentials(client: TestClient) -> None:
    """The API is closed without valid Basic credentials."""
    assert (
        _post(client, _png(), auth=None).status_code == HTTPStatus.UNAUTHORIZED
    )
    unauthorized: Response = _post(client, _png(), auth=("wrong", "wrong"))
    assert unauthorized.status_code == HTTPStatus.UNAUTHORIZED
    assert _failure(unauthorized).error == ErrorCode.UNAUTHORIZED
    assert HeaderName.authenticate in unauthorized.headers


def test_oversized_upload_is_refused(client: TestClient) -> None:
    """An upload past the cap is a 413, never fully buffered."""
    response: Response = _post(
        client, b"\x89PNG\r\n\x1a\n" + b"\x00" * (UPLOAD_LIMIT * 2)
    )
    assert response.status_code == HTTPStatus.CONTENT_TOO_LARGE
    assert _failure(response).error == ErrorCode.UPLOAD_TOO_LARGE


def test_empty_upload_is_refused(client: TestClient) -> None:
    """An empty body is a client error, not a traceback."""
    response: Response = _post(client, b"")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_CONTENT
    assert _failure(response).error == ErrorCode.INVALID_IMAGE


def test_undecodable_upload_is_refused(client: TestClient) -> None:
    """Bytes Pillow cannot open yield 422 rather than a 500."""
    response: Response = _post(client, b"definitely not an image")
    assert response.status_code == HTTPStatus.UNPROCESSABLE_CONTENT
    assert _failure(response).error == ErrorCode.INVALID_IMAGE


def test_rate_limit_returns_429_with_retry_after(
    client: TestClient,
) -> None:
    """Past the budget the caller is refused and told when to retry."""
    for _ in range(BUDGET):
        _post(client, b"not an image")
    response: Response = _post(client, b"not an image")
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    assert _failure(response).error == ErrorCode.RATE_LIMITED
    assert int(response.headers[HeaderName.retry_after]) >= 1


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


def test_out_of_range_options_are_refused_at_the_boundary(
    client: TestClient,
) -> None:
    """The option form validates before anything reaches a model.

    The transport carries no wired application, so a request that got
    past validation would answer 503. A 422 is therefore proof that the
    bounds were checked, and checked first.
    """
    refused: Response = _post(client, _png(), minimum_confidence="5.0")
    assert refused.status_code == HTTPStatus.UNPROCESSABLE_CONTENT

    accepted: Response = _post(
        client, _png(), minimum_confidence="0.5", split_masks="true"
    )
    assert accepted.status_code == HTTPStatus.SERVICE_UNAVAILABLE


def test_the_event_socket_refuses_a_caller_without_credentials(
    client: TestClient,
) -> None:
    """A socket cannot answer 401, so it closes on a policy violation.

    Refused before the accept, which is why the connect itself is what
    raises here.
    """
    with (
        pytest.raises(WebSocketDisconnect) as raised,
        client.websocket_connect(f"/jobs/{JOB_ID}/events"),
    ):
        pass
    assert raised.value.code == POLICY_VIOLATION


def test_the_event_socket_closes_on_a_job_it_cannot_find(
    client: TestClient,
) -> None:
    """An expired identifier ends the subscription rather than hanging.

    The credentials pass, so the socket is accepted and only then
    closed: the disconnect surfaces on the first read.
    """
    store: FakeStore = FakeStore()
    client.app.state.store = (
        store  # pyright: ignore[reportAttributeAccessIssue]
    )
    with (
        client.websocket_connect(
            f"/jobs/{JOB_ID}/events", headers=_basic()
        ) as socket,
        pytest.raises(WebSocketDisconnect) as raised,
    ):
        socket.receive_json()
    assert raised.value.code == POLICY_VIOLATION


def test_the_event_socket_streams_transitions_then_closes(
    client: TestClient,
) -> None:
    """Each state is pushed once, and the terminal one ends it."""
    store: FakeStore = FakeStore()
    job: Job = queued(JOB_ID, 1000.0)
    store.jobs[JOB_ID] = (job, None)
    client.app.state.store = (
        store  # pyright: ignore[reportAttributeAccessIssue]
    )

    with client.websocket_connect(
        f"/jobs/{JOB_ID}/events", headers=_basic()
    ) as socket:
        assert (
            JobSchema.model_validate(socket.receive_json()).state
            == JobState.QUEUED
        )

        running: Job = start(job, 1001.0)
        store.jobs[JOB_ID] = (running, None)
        assert (
            JobSchema.model_validate(socket.receive_json()).state
            == JobState.RUNNING
        )

        store.jobs[JOB_ID] = (
            succeed(running, 1002.0),
            JobResult(error=None, body=None),
        )
        assert (
            JobSchema.model_validate(socket.receive_json()).state
            == JobState.SUCCEEDED
        )
        with pytest.raises(WebSocketDisconnect) as raised:
            socket.receive_json()
    assert raised.value.code == NORMAL_CLOSE


def test_an_image_past_the_pixel_ceiling_is_refused() -> None:
    """The configured ceiling is the ceiling, not twice it.

    Pillow raises only past double its limit and merely warns in
    between, so a bomb sized at 1.3x would decode unless that warning is
    promoted to an error. Its own client, because the shared one caps
    uploads far below what a large image weighs.
    """
    transport: FastAPI = create_app(
        _settings(MAX_IMAGE_PIXELS=PIXEL_CEILING, MAX_UPLOAD_BYTES=1_000_000)
    )
    side: int = int(PIXEL_CEILING**0.5 * 1.3)
    response: Response = _post(TestClient(transport), _png(side, side))
    assert response.status_code == HTTPStatus.UNPROCESSABLE_CONTENT
    assert _failure(response).error == ErrorCode.INVALID_IMAGE


def test_the_event_socket_is_budgeted_like_any_other_request(
    client: TestClient,
) -> None:
    """A subscription costs a poll every quarter second, so it counts.

    Without this a caller holds open as many sockets as it likes, each
    one reading the store four times a second, none of them charged
    against the budget the HTTP routes share.
    """
    store: FakeStore = FakeStore()
    client.app.state.store = (
        store  # pyright: ignore[reportAttributeAccessIssue]
    )
    for _ in range(BUDGET):
        _post(client, b"not an image")

    with (
        pytest.raises(WebSocketDisconnect) as raised,
        client.websocket_connect(f"/jobs/{JOB_ID}/events", headers=_basic()),
    ):
        pass
    assert raised.value.code == TRY_AGAIN_LATER


def test_non_ascii_socket_credentials_are_refused_not_raised() -> None:
    """A header the comparison cannot digest is a refusal, not a crash.

    ``compare_digest`` rejects a non-ASCII ``str`` outright, and the
    socket decodes its own header rather than letting FastAPI do it, so
    this path is the one that had to compare bytes.
    """
    settings: Settings = _settings()
    encoded: str = b64encode(
        f"ünicode{BasicScheme.separator}password".encode()
    ).decode()
    headers: dict[str, str] = {
        BasicScheme.header: f"{BasicScheme.prefix}{encoded}"
    }

    assert socket_authorised(headers, settings) is False
