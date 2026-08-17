"""Route handlers.

Each one parses, validates, calls a wired use case, and translates the
typed outcome. No orchestration lives here (``SKILL.md`` section 31).
"""

from http import HTTPStatus
from typing import Annotated, Final

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.bootstrap import Application, SegmentOutcome
from app.domain import (
    DeviceExhausted,
    ImageDecodeFailed,
    ModelUnavailable,
    NoDetection,
    PersonPayload,
    Prompt,
    SegmentedImage,
    SourceImage,
    UnknownBackend,
    UploadTooLarge,
)
from app.infrastructure.imaging import decode_image, encode_png
from app.interfaces.http.auth import require_credentials
from app.interfaces.http.constants import (
    MESSAGE,
    ROUTE,
    TAG,
    ErrorCode,
    HealthState,
)
from app.interfaces.http.errors import (
    failure_response,
    map_device_exhausted,
    map_no_detection,
)
from app.interfaces.http.rate_limit import enforce_rate_limit
from app.interfaces.http.schemas import (
    ErrorSchema,
    HealthSchema,
    SegmentSchema,
    SegmentersSchema,
)
from app.settings import Settings

meta_router: APIRouter = APIRouter(tags=[TAG.meta])
segmentation_router: APIRouter = APIRouter(
    tags=[TAG.segmentation],
    dependencies=[
        Depends(require_credentials),
        Depends(enforce_rate_limit),
    ],
)

#: Statuses ``POST /segment`` documents, all sharing the error body.
_FAILURE_STATUSES: Final[tuple[HTTPStatus, ...]] = (
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.UNAUTHORIZED,
    HTTPStatus.CONTENT_TOO_LARGE,
    HTTPStatus.UNPROCESSABLE_CONTENT,
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.SERVICE_UNAVAILABLE,
)
_FAILURE_SCHEMA: Final[dict[int | str, dict[str, type[ErrorSchema]]]] = {
    status: {"model": ErrorSchema} for status in _FAILURE_STATUSES
}


def _settings(request: Request) -> Settings:
    """Return the configuration, available before the models are.

    :param request: Incoming request.
    :type request: fastapi.Request
    :returns: The validated configuration.
    :rtype: app.settings.Settings
    """
    settings: Settings = request.app.state.settings
    return settings


def _application(request: Request) -> Application:
    """Return the wired application for this request.

    :param request: Incoming request.
    :type request: fastapi.Request
    :returns: The application assembled at startup.
    :rtype: app.bootstrap.Application
    :raises app.domain.errors.ModelUnavailable: While startup has not
        finished, so a request that needs a model is refused with the
        same 503 the readiness probe reports.
    """
    # Annotated rather than a walrus: the state attribute is untyped, and
    # the annotation is what keeps this from handing back Any.
    # pylint: disable=consider-using-assignment-expr
    wired: Application | None = request.app.state.application
    if wired is None:
        raise ModelUnavailable(
            model=_settings(request).default_segmenter,
            detail=MESSAGE.models_loading,
        )
    return wired


def _read_upload(upload: UploadFile, settings: Settings) -> SourceImage:
    """Read and decode an upload, under the configured ceilings.

    Reads one byte past the limit rather than the whole body, so an
    oversized upload is rejected without ever being held in memory.

    :param upload: Multipart file from the request.
    :type upload: fastapi.UploadFile
    :param settings: Wired configuration.
    :type settings: app.settings.Settings
    :returns: The decoded image.
    :rtype: app.domain.models.SourceImage
    :raises app.domain.errors.UploadTooLarge: Past the byte ceiling.
    :raises app.domain.errors.ImageDecodeFailed: On an empty or
        undecodable payload.
    """
    limit: int = settings.max_upload_bytes
    payload: bytes = upload.file.read(limit + 1)
    if len(payload) > limit:
        raise UploadTooLarge(limit_bytes=limit)
    if not payload:
        raise ImageDecodeFailed(detail=MESSAGE.empty_upload)
    return decode_image(payload)


@meta_router.get(ROUTE.health, response_model=HealthSchema)
def liveness() -> HealthSchema:
    """Report that the process is up, before the models are.

    Answers immediately so a container runtime can tell a cold start
    from a wedged process.

    :returns: The liveness payload.
    :rtype: app.interfaces.http.schemas.HealthSchema
    """
    return HealthSchema(status=HealthState.OK)


@meta_router.get(
    ROUTE.ready,
    response_model=HealthSchema,
    responses={HTTPStatus.SERVICE_UNAVAILABLE: {"model": ErrorSchema}},
)
def readiness(request: Request) -> HealthSchema | object:
    """Report that every model is resident and no download is pending.

    :param request: Incoming request.
    :type request: fastapi.Request
    :returns: The readiness payload, or a 503 while still loading.
    :rtype: app.interfaces.http.schemas.HealthSchema | object
    """
    if (wired := getattr(request.app.state, "application", None)) is None:
        return failure_response(
            HTTPStatus.SERVICE_UNAVAILABLE,
            ErrorCode.NOT_READY,
            MESSAGE.models_loading,
        )
    del wired
    return HealthSchema(status=HealthState.READY)


@meta_router.get(ROUTE.segmenters, response_model=SegmentersSchema)
def list_segmenters(request: Request) -> SegmentersSchema:
    """Enumerate the wired segmentation backends.

    :param request: Incoming request.
    :type request: fastapi.Request
    :returns: The backend listing.
    :rtype: app.interfaces.http.schemas.SegmentersSchema
    """
    return SegmentersSchema(available=sorted(_application(request).backends))


@segmentation_router.post(
    ROUTE.segment,
    response_model=SegmentSchema,
    responses=_FAILURE_SCHEMA,
)
def segment_endpoint(
    request: Request,
    image: Annotated[UploadFile, File(description="Image to segment.")],
    prompt: Annotated[
        str, Form(description="Text prompt (comma or dot separated).")
    ],
    person_mode: Annotated[
        bool, Form(description="Enable face analysis.")
    ] = False,
    segmenter: Annotated[str | None, Form(description="Backend name.")] = None,
) -> SegmentSchema | object:
    """Segment an image and translate the typed outcome.

    :param request: Incoming request.
    :type request: fastapi.Request
    :param image: Uploaded image.
    :type image: fastapi.UploadFile
    :param prompt: Raw prompt text.
    :type prompt: str
    :param person_mode: Whether to attach a face summary.
    :type person_mode: bool
    :param segmenter: Optional backend override.
    :type segmenter: str | None
    :returns: The response body, or a failure response.
    :rtype: app.interfaces.http.schemas.SegmentSchema | object
    :raises app.domain.errors.UnknownBackend: If ``segmenter`` names a
        backend that is not wired.
    """
    source: SourceImage = _read_upload(image, _settings(request))
    application: Application = _application(request)
    parsed: Prompt = Prompt.parse(prompt)

    backend: str = segmenter or application.settings.default_segmenter
    if backend not in application.backends:
        raise UnknownBackend(
            requested=backend,
            available=tuple(sorted(application.backends)),
        )

    person: PersonPayload | None = (
        application.analyse_faces(source) if person_mode else None
    )
    outcome: SegmentOutcome = application.segment(
        backend, source, parsed, person
    )
    match outcome:
        case NoDetection():
            return map_no_detection(outcome)
        case DeviceExhausted():
            return map_device_exhausted(outcome)
        case SegmentedImage():
            return SegmentSchema.of(
                result=outcome,
                prompt=parsed.text,
                segmenter=backend,
                mask=encode_png(outcome.cropped_mask),
                image=encode_png(outcome.cropped_image),
            )


__all__: list[str] = [
    "liveness",
    "list_segmenters",
    "meta_router",
    "readiness",
    "segment_endpoint",
    "segmentation_router",
]
