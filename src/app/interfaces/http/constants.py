"""HTTP transport vocabulary.

Routes, headers, error codes and caller-facing messages, grouped so no
handler ever spells a protocol literal inline.
"""

from dataclasses import dataclass
from enum import StrEnum, unique
from http import HTTPStatus
from typing import ClassVar, Literal, final


@unique
class ErrorCode(StrEnum):
    """Machine-readable code every failure body carries.

    Clients branch on this and never parse ``message``.
    """

    UNAUTHORIZED = "unauthorized"
    INVALID_PROMPT = "invalid_prompt"
    NO_DETECTION = "no_detection"
    UNKNOWN_BACKEND = "unknown_backend"
    UNKNOWN_JOB = "unknown_job"
    QUEUE_FULL = "queue_full"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    ALREADY_STARTED = "already_started"
    INVALID_CALLBACK = "invalid_callback"
    INVALID_IMAGE = "invalid_image"
    UPLOAD_TOO_LARGE = "upload_too_large"
    RATE_LIMITED = "rate_limited"
    OUT_OF_MEMORY = "out_of_memory"
    UNAVAILABLE_FEATURE = "unavailable_feature"
    STORE_UNAVAILABLE = "store_unavailable"
    NOT_READY = "not_ready"
    INTERNAL = "internal"


@unique
class HealthState(StrEnum):
    """Values the two probes report."""

    OK = "ok"
    READY = "ready"


@final
@dataclass(frozen=True, slots=True)
class HttpRoute:
    """Every public path, in one place."""

    health: ClassVar[str] = "/healthz"
    ready: ClassVar[str] = "/readyz"
    segmenters: ClassVar[str] = "/segmenters"
    jobs: ClassVar[str] = "/jobs"
    job: ClassVar[str] = "/jobs/{identifier}"
    job_events: ClassVar[str] = "/jobs/{identifier}/events"
    #: FastAPI's own, at their default locations.
    docs: ClassVar[str] = "/docs"
    redoc: ClassVar[str] = "/redoc"
    openapi: ClassVar[str] = "/openapi.json"


@final
@dataclass(frozen=True, slots=True)
class HttpVerb:
    """Methods, lowercased the way the generated document spells them."""

    get: ClassVar[str] = "get"
    post: ClassVar[str] = "post"
    delete: ClassVar[str] = "delete"


@final
@dataclass(frozen=True, slots=True)
class SocketRules:
    """How the event socket behaves, and how it says goodbye.

    Close codes are the RFC 6455 ones: 1000 for a clean end, 1008 for a
    policy violation, which covers both a refused credential and a job
    that does not exist.
    """

    normal_close: ClassVar[int] = 1000
    policy_violation: ClassVar[int] = 1008
    #: RFC 6455 has no "too many requests", and 1013 is the registered
    #: way to say the same thing to a socket.
    try_again_later: ClassVar[int] = 1013
    #: Deliberate: the socket polls the store rather than subscribing to
    #: it. One small read every quarter second per connection costs
    #: nothing next to an inference; Redis pub/sub is the upgrade if a
    #: crowd ever watches the same queue.
    poll_seconds: ClassVar[float] = 0.25


@final
@dataclass(frozen=True, slots=True)
class Documented:
    """Statuses each job route declares in its OpenAPI responses.

    One tuple per route rather than one shared list: a caller reading
    the document should not be told that polling can answer ``413``, or
    that acceptance can answer ``409``. Neither can.
    """

    #: Shared by all three: the credentials, the budget, and the catch
    #: of last resort apply wherever the router does.
    common: ClassVar[tuple[HTTPStatus, ...]] = (
        HTTPStatus.UNAUTHORIZED,
        HTTPStatus.TOO_MANY_REQUESTS,
        HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    #: ``POST /jobs``: everything the upload itself can be wrong about,
    #: plus the 503 a caller meets while the models are still loading.
    acceptance: ClassVar[tuple[HTTPStatus, ...]] = common + (
        HTTPStatus.BAD_REQUEST,
        HTTPStatus.CONTENT_TOO_LARGE,
        HTTPStatus.UNPROCESSABLE_CONTENT,
        HTTPStatus.SERVICE_UNAVAILABLE,
    )
    #: ``GET /jobs/{identifier}``: it exists or it does not.
    polling: ClassVar[tuple[HTTPStatus, ...]] = common + (
        HTTPStatus.NOT_FOUND,
    )
    #: ``DELETE /jobs/{identifier}``: and it may be too late.
    withdrawal: ClassVar[tuple[HTTPStatus, ...]] = common + (
        HTTPStatus.NOT_FOUND,
        HTTPStatus.CONFLICT,
    )


@final
@dataclass(frozen=True, slots=True)
class RetryAfter:
    """How long a refused caller is told to wait, in seconds."""

    #: The queue drains at the pace of one accelerator, so a short
    #: retry is honest: a full queue is not full for long.
    queue_full: ClassVar[int] = 5
    #: An outage is not measured in seconds the way a full queue is, so
    #: this is a hint to back off rather than a promise.
    store_unavailable: ClassVar[int] = 10


@final
@dataclass(frozen=True, slots=True)
class HeaderName:
    """Response headers this service sets."""

    retry_after: ClassVar[str] = "Retry-After"
    authenticate: ClassVar[str] = "WWW-Authenticate"
    #: Read from the request, not set on the response. Named here with
    #: the others because it is part of the same published contract.
    idempotency_key: ClassVar[str] = "Idempotency-Key"


@final
@dataclass(frozen=True, slots=True)
class BasicScheme:
    """How Basic credentials arrive, and how they are taken apart."""

    header: ClassVar[str] = "authorization"
    prefix: ClassVar[str] = "Basic "
    separator: ClassVar[str] = ":"


@final
@dataclass(frozen=True, slots=True)
class Decoding:
    """How far Pillow is allowed to go with a malformed upload."""

    #: Pillow refuses a truncated file by default. Accepting one is
    #: deliberate - a browser upload cut short is the caller's accident,
    #: not an attack - and only holds because the pixel ceiling caps
    #: what a malformed file can allocate.
    load_truncated: ClassVar[bool] = True
    #: Turns Pillow's decompression-bomb warning into the exception the
    #: decoder already treats as the client's problem. Typed as the
    #: literal ``warnings.filterwarnings`` accepts, not as ``str``.
    bomb_action: ClassVar[Literal["error"]] = "error"


@final
@dataclass(frozen=True, slots=True)
class RateLimitValues:
    """What the limiter answers with when nothing is refused."""

    #: A client whose address the transport could not determine. They
    #: share one budget, which is the safe direction to err in.
    unknown_client: ClassVar[str] = "unknown"
    no_wait: ClassVar[float] = 0.0
    empty_window: ClassVar[tuple[int, float]] = (0, 0.0)


@final
@dataclass(frozen=True, slots=True)
class AuthScheme:
    """Authentication scheme presented in a challenge."""

    basic: ClassVar[str] = "Basic"


@final
@dataclass(frozen=True, slots=True)
class FormField:
    """Multipart field names of ``POST /jobs``."""

    image: ClassVar[str] = "image"
    prompt: ClassVar[str] = "prompt"
    person_mode: ClassVar[str] = "person_mode"
    segmenter: ClassVar[str] = "segmenter"


@final
@dataclass(frozen=True, slots=True)
class Message:
    """Caller-facing text. Deliberately free of internal detail."""

    authentication_required: ClassVar[str] = "Authentication required."
    invalid_credentials: ClassVar[str] = "Invalid credentials."
    rate_limited: ClassVar[str] = "Rate limit exceeded."
    empty_upload: ClassVar[str] = "Empty upload."
    models_loading: ClassVar[str] = "Models are still loading."
    internal: ClassVar[str] = "Internal server error."
    feature_unavailable: ClassVar[str] = (
        "This feature is not installed on the server."
    )
    ui_unavailable: ClassVar[str] = (
        "ENABLE_UI is on, but gradio is not installed. Install the "
        "'ui' extra, or rebuild the image with ENABLE_UI=true."
    )
    out_of_memory: ClassVar[str] = (
        "Not enough device memory for this image. Retry with a smaller one."
    )


@final
@dataclass(frozen=True, slots=True)
class StateKey:
    """Names of what the factory hangs off ``app.state``.

    ``getattr`` needs the name as text, which is exactly the kind of
    string that goes stale silently when the attribute is renamed.
    """

    application: ClassVar[str] = "application"


@final
@dataclass(frozen=True, slots=True)
class OpenApiKey:
    """Keys FastAPI and Pydantic expect in the mappings handed to them."""

    model: ClassVar[str] = "model"
    name: ClassVar[str] = "name"
    description: ClassVar[str] = "description"
    examples: ClassVar[str] = "examples"
    #: Read back out of the generated document rather than put into it.
    paths: ClassVar[str] = "paths"
    info: ClassVar[str] = "info"
    tags: ClassVar[str] = "tags"
    components: ClassVar[str] = "components"
    responses: ClassVar[str] = "responses"
    security: ClassVar[str] = "security"
    security_schemes: ClassVar[str] = "securitySchemes"


@final
@dataclass(frozen=True, slots=True)
class JobField:
    """Field names of the job body, where they have to be spelled.

    A schema names its own fields in code; these exist for the places
    that cannot, such as the example the document advertises.
    """

    identifier: ClassVar[str] = "identifier"
    state: ClassVar[str] = "state"
    created_at: ClassVar[str] = "created_at"
    updated_at: ClassVar[str] = "updated_at"
    queue_position: ClassVar[str] = "queue_position"


@final
@dataclass(frozen=True, slots=True)
class Serialisation:
    """How a Pydantic model is dumped for the wire.

    ``json`` rather than ``python``: the transport sends JSON, so enums
    and floats have to be resolved before they reach a response.
    """

    json_mode: ClassVar[Literal["json"]] = "json"


@final
@dataclass(frozen=True, slots=True)
class FailureText:
    """Templates for the caller-facing half of a typed failure.

    Formatted at the boundary rather than spelled inline, so the wording
    a caller reads lives beside every other string this layer emits.
    """

    unknown_segmenter: ClassVar[str] = (
        "Unknown segmenter {requested!r}. Available: {available}."
    )
    invalid_image: ClassVar[str] = "Invalid image: {detail}"
    upload_too_large: ClassVar[str] = "Upload exceeds {limit} bytes."
    no_detection: ClassVar[str] = "No detection for prompt {prompt!r}."


@final
@dataclass(frozen=True, slots=True)
class LogMessage:
    """Format strings the transport logs with.

    ``%s`` placeholders rather than f-strings: logging interpolates
    lazily, so a suppressed record costs nothing to not format.
    """

    device_exhausted: ClassVar[str] = "Device exhausted: %s"
    face_unavailable: ClassVar[str] = "Face analysis unavailable: %s"
    model_unavailable: ClassVar[str] = "Model %s unavailable: %s"
    store_unavailable: ClassVar[str] = "Job store unavailable: %s"
    unhandled: ClassVar[str] = "Unhandled defect: %s"
    startup_failed: ClassVar[str] = "Startup failed, staying unready"


@final
@dataclass(frozen=True, slots=True)
class JobMessage:
    """Caller-facing text the queue and its deliveries produce."""

    unknown: ClassVar[str] = "No such job, or it has expired."
    queue_full: ClassVar[str] = "The queue is full. Retry shortly."
    idempotency_conflict: ClassVar[str] = (
        "This Idempotency-Key was already used for a different request. "
        "Reuse the key only to retry the identical submission."
    )
    store_unavailable: ClassVar[str] = (
        "The job store is unreachable. This is an outage, not your "
        "request: retry shortly."
    )
    callback_disabled: ClassVar[str] = (
        "Webhooks are not configured on this server: set "
        "WEBHOOK_SIGNING_SECRET, or poll instead."
    )
    callback_refused: ClassVar[str] = (
        "callback_url must be an https address that resolves to a public host."
    )
    already_started: ClassVar[str] = (
        "This job has already started; only a queued job can be cancelled."
    )


@final
@dataclass(frozen=True, slots=True)
class OpenApiTag:
    """Groups the routes are documented under."""

    meta: ClassVar[str] = "meta"
    segmentation: ClassVar[str] = "segmentation"

    #: Handed to FastAPI so each group carries a sentence in the docs
    #: rather than a bare heading.
    descriptions: ClassVar[tuple[dict[str, str], ...]] = (
        {
            OpenApiKey.name: meta,
            OpenApiKey.description: (
                "Probes and the backend listing. Open: a probe has to "
                "answer before any secret is configured, and none of "
                "these exposes image data."
            ),
        },
        {
            OpenApiKey.name: segmentation,
            OpenApiKey.description: (
                "Accept work, report on it, withdraw it. HTTP Basic, "
                "and rate limited per client address."
            ),
        },
    )


@final
@dataclass(frozen=True, slots=True)
class RouteDoc:
    """One route's summary and prose, as the documents render them.

    Kept out of the docstrings on purpose: those are written for the
    codebase, in the Sphinx field syntax the rest of the project uses,
    and FastAPI would otherwise publish ``:param request:`` to every
    reader of ``/docs``.
    """

    summary: str
    description: str


@final
@dataclass(frozen=True, slots=True)
class RouteDocs:
    """What ``/docs`` says about each route."""

    liveness: ClassVar[RouteDoc] = RouteDoc(
        summary="Liveness probe",
        description=(
            "Answers as soon as the process is up, before the models "
            "finish loading. A container runtime uses this to tell a "
            "cold start from a wedged process."
        ),
    )
    readiness: ClassVar[RouteDoc] = RouteDoc(
        summary="Readiness probe",
        description=(
            "`503` until every model is resident. Segmentation is "
            "refused until this answers `200`."
        ),
    )
    segmenters: ClassVar[RouteDoc] = RouteDoc(
        summary="List the wired backends",
        description=(
            "The names `POST /jobs` accepts in its `segmenter` field. "
            "Omitting that field uses the configured default."
        ),
    )
    create_job: ClassVar[RouteDoc] = RouteDoc(
        summary="Queue a segmentation",
        description=(
            "Answers `202` with an identifier to poll. Segmentation "
            "runs on a single accelerator, so the work is queued "
            "rather than held on an open connection.\n\n"
            "What can be judged while the caller still holds the "
            "connection is judged here: an oversized upload, an image "
            "Pillow cannot open, an empty prompt, an unknown backend, "
            "a callback URL pointing somewhere internal, and a queue "
            "already at its ceiling.\n\n"
            "`queue_position` in the response is how many jobs are "
            "ahead of this one."
        ),
    )
    read_job: ClassVar[RouteDoc] = RouteDoc(
        summary="Poll a job",
        description=(
            "The same body acceptance returned, with `state` moving to "
            "`running` and then to one of `succeeded`, `failed` or "
            "`cancelled`. A succeeded job carries `result`.\n\n"
            "Jobs expire: an identifier older than the retention "
            "window answers `404`."
        ),
    )
    cancel_job: ClassVar[RouteDoc] = RouteDoc(
        summary="Withdraw a queued job",
        description=(
            "Withdraws a job that has not started. A running one "
            "answers `409`: the accelerator is already busy with it, "
            "and there is no way out of `running` but to finish."
        ),
    )


@final
@dataclass(frozen=True, slots=True)
class ApiMetadata:
    """What the generated OpenAPI document announces."""

    title: ClassVar[str] = "Prompt image segmentation"
    #: Markdown, rendered at the top of ``/docs`` and ``/redoc``. It
    #: carries the two things the generated document cannot: how to
    #: authenticate in the browser, and the event socket, which OpenAPI
    #: has no way to describe.
    description: ClassVar[str] = (
        "Prompt-driven segmentation via GroundingDINO + SAM 2.1.\n\n"
        "Give it an image and any text prompt, get back a binary mask, "
        "the cropped region, and a score telling you whether to trust "
        "them.\n\n"
        "### Trying it here\n\n"
        "The `/jobs` routes use HTTP Basic. Press **Authorize** and "
        "enter `SEGMENTATION_USERNAME` / `SEGMENTATION_PASSWORD`, then "
        "**Try it out** on any route below. Running with "
        "`AUTH_MODE=none` skips this entirely.\n\n"
        "Work is queued: `POST /jobs` answers `202` and the result "
        "arrives later. Collect it by polling `GET /jobs/{identifier}`, "
        "by subscribing to the socket below, or by passing a "
        "`callback_url`.\n\n"
        "### `GET /jobs/{identifier}/events` (WebSocket)\n\n"
        "Not listed below because OpenAPI cannot describe a socket. It "
        "pushes the same `JobSchema` body as polling, once per "
        "transition, and closes itself on a terminal state. Credentials "
        "ride on the upgrade request, which browsers cannot set - so it "
        "suits server-to-server callers, and polling remains for "
        "everything else.\n\n"
        "```bash\n"
        "websocat -H \"Authorization: Basic $(printf 'user:pass' | "
        'base64)" \\\n'
        "    ws://localhost:7860/jobs/<identifier>/events\n"
        "```\n\n"
        "### Webhooks\n\n"
        "Pass `callback_url` and the outcome is delivered there once, "
        'signed with `X-Signature` over `"<timestamp>." + body`. '
        "Deliveries go only to public `https` addresses and are never "
        "redirected. Unset `WEBHOOK_SIGNING_SECRET` refuses a "
        "`callback_url` rather than sending it unsigned."
    )


__all__: list[str] = [
    "ApiMetadata",
    "AuthScheme",
    "BasicScheme",
    "Decoding",
    "Documented",
    "ErrorCode",
    "FailureText",
    "FormField",
    "HeaderName",
    "HealthState",
    "HttpRoute",
    "HttpVerb",
    "JobField",
    "JobMessage",
    "LogMessage",
    "Message",
    "OpenApiKey",
    "OpenApiTag",
    "RateLimitValues",
    "RetryAfter",
    "RouteDoc",
    "RouteDocs",
    "Serialisation",
    "StateKey",
    "SocketRules",
]
