"""What every route reads off the request, and the upload guard.

Plain functions rather than FastAPI dependencies: they are called from
the handlers, and one of them is called from a WebSocket, where a
``Request`` does not exist.
"""

from fastapi import Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.application.capabilities import JobStore
from app.application.jobs import JobResult
from app.bootstrap import Application
from app.domain import ImageDecodeFailed, ModelUnavailable, UploadTooLarge
from app.infrastructure.imaging.imaging import decode_image
from app.interfaces.http.constants import Message
from app.interfaces.http.schemas import SegmentSchema
from app.settings import Settings


def settings_of(request: Request) -> Settings:
    """Return the configuration, available before the models are.

    :param request: Incoming request.
    :type request: fastapi.Request
    :returns: The validated configuration.
    :rtype: app.settings.Settings
    """
    settings: Settings = request.app.state.settings
    return settings


def application_of(request: Request) -> Application:
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
            model=settings_of(request).default_segmenter,
            detail=Message.models_loading,
        )
    return wired


def body_of(result: JobResult | None) -> SegmentSchema | None:
    """Return the answer a job produced, if it produced one.

    No validation left to do here: ``JobResult.body`` is the model, and
    the store validated it on the way out of Redis.

    :param result: What the store had, if anything.
    :type result: app.application.jobs.JobResult | None
    :returns: The body, or ``None``.
    :rtype: app.application.jobs.results.SegmentSchema | None
    """
    return None if result is None else result.body


def store_of(request: Request) -> JobStore:
    """Return the job store this transport was built with.

    :param request: Incoming request.
    :type request: fastapi.Request
    :returns: The store queued work lives in.
    :rtype: app.application.capabilities.JobStore
    """
    store: JobStore = request.app.state.store
    return store


def read_bytes(upload: UploadFile, settings: Settings) -> bytes:
    """Read an upload under the configured ceilings.

    Reads one byte past the limit rather than the whole body, so an
    oversized upload is rejected without ever being held in memory.

    :param upload: Multipart file from the request.
    :type upload: fastapi.UploadFile
    :param settings: Wired configuration.
    :type settings: app.settings.Settings
    :returns: The raw bytes, size-checked but not yet decoded.
    :rtype: bytes
    :raises app.domain.errors.UploadTooLarge: Past the byte ceiling.
    :raises app.domain.errors.ImageDecodeFailed: On an empty payload.
    """
    limit: int = settings.max_upload_bytes
    payload: bytes = upload.file.read(limit + 1)
    if len(payload) > limit:
        raise UploadTooLarge(limit_bytes=limit)
    if not payload:
        raise ImageDecodeFailed(detail=Message.empty_upload)
    return payload


def _read_and_decode(upload: UploadFile, settings: Settings) -> bytes:
    payload: bytes = read_bytes(upload, settings)
    # Decoded and thrown away: the worker decodes the stored bytes
    # again, but a caller deserves to hear about a bad image now, while
    # it is still holding the connection.
    decode_image(payload)
    return payload


async def accepted_bytes(upload: UploadFile, settings: Settings) -> bytes:
    """Read and validate an upload without blocking the event loop.

    Both halves are blocking - starlette spools a large upload to disk,
    and decoding a forty-megapixel image is pure CPU - so they share one
    hop into the threadpool rather than stalling every other request in
    flight.

    :param upload: Multipart file from the request.
    :type upload: fastapi.UploadFile
    :param settings: Wired configuration.
    :type settings: app.settings.Settings
    :returns: The raw bytes, size-checked and known to decode.
    :rtype: bytes
    :raises app.domain.errors.UploadTooLarge: Past the byte ceiling.
    :raises app.domain.errors.ImageDecodeFailed: On an empty payload, or
        one Pillow cannot open.
    """
    return await run_in_threadpool(_read_and_decode, upload, settings)


__all__: list[str] = [
    "accepted_bytes",
    "application_of",
    "body_of",
    "read_bytes",
    "settings_of",
    "store_of",
]
