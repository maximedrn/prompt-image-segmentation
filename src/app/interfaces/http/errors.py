"""Typed outcome to HTTP response.

Every recoverable failure has exactly one status and one code here, so
the mapping is auditable in one screen (``SKILL.md`` section 31). The
only broad catch in the application sits at the very bottom of this
module: a process boundary, where it reports rather than pretends the
operation succeeded (section 13).
"""

from http import HTTPStatus
from logging import Logger, getLogger

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.domain import (
    DeviceExhausted,
    FaceAnalysisUnavailable,
    ImageDecodeFailed,
    InvalidPrompt,
    ModelUnavailable,
    NoDetection,
    RateLimited,
    UnknownBackend,
    UploadTooLarge,
)
from app.interfaces.http.constants import HEADER, MESSAGE, ErrorCode
from app.interfaces.http.schemas import ErrorSchema

_LOGGER: Logger = getLogger(__name__)


def failure_response(
    status: HTTPStatus,
    code: ErrorCode,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the uniform failure body.

    :param status: HTTP status to answer with.
    :type status: http.HTTPStatus
    :param code: Machine-readable failure code.
    :type code: app.interfaces.http.constants.ErrorCode
    :param message: Caller-facing explanation, free of internal detail.
    :type message: str
    :param headers: Extra response headers, if the failure carries any.
    :type headers: dict[str, str] | None
    :returns: The JSON response.
    :rtype: fastapi.responses.JSONResponse
    """
    body: ErrorSchema = ErrorSchema(error=code, message=message)
    return JSONResponse(
        status_code=status,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


def map_no_detection(failure: NoDetection) -> JSONResponse:
    """Nothing matched the prompt: the request was fine, the image was not.

    :param failure: The typed failure.
    :type failure: app.domain.errors.NoDetection
    :returns: A 422 response.
    :rtype: fastapi.responses.JSONResponse
    """
    return failure_response(
        HTTPStatus.UNPROCESSABLE_CONTENT,
        ErrorCode.NO_DETECTION,
        f"No detection for prompt {failure.prompt!r}.",
    )


def map_device_exhausted(failure: DeviceExhausted) -> JSONResponse:
    """The accelerator ran out of memory: retryable with a smaller image.

    :param failure: The typed failure.
    :type failure: app.domain.errors.DeviceExhausted
    :returns: A 503 response.
    :rtype: fastapi.responses.JSONResponse
    """
    _LOGGER.warning("Device exhausted: %s", failure.detail)
    return failure_response(
        HTTPStatus.SERVICE_UNAVAILABLE,
        ErrorCode.OUT_OF_MEMORY,
        MESSAGE.out_of_memory,
    )


def _register_client_failures(application: FastAPI) -> None:
    """Attach the failures the caller can fix themselves.

    :param application: The application to configure.
    :type application: fastapi.FastAPI
    """

    @application.exception_handler(InvalidPrompt)
    async def _invalid_prompt(  # pyright: ignore[reportUnusedFunction]
        _request: Request, failure: InvalidPrompt
    ) -> JSONResponse:
        return failure_response(
            HTTPStatus.UNPROCESSABLE_CONTENT,
            ErrorCode.INVALID_PROMPT,
            failure.reason,
        )

    @application.exception_handler(NoDetection)
    async def _no_detection(  # pyright: ignore[reportUnusedFunction]
        _request: Request, failure: NoDetection
    ) -> JSONResponse:
        return map_no_detection(failure)

    @application.exception_handler(UnknownBackend)
    async def _unknown_backend(  # pyright: ignore[reportUnusedFunction]
        _request: Request, failure: UnknownBackend
    ) -> JSONResponse:
        return failure_response(
            HTTPStatus.BAD_REQUEST,
            ErrorCode.UNKNOWN_BACKEND,
            f"Unknown segmenter {failure.requested!r}. "
            f"Available: {list(failure.available)}.",
        )

    @application.exception_handler(ImageDecodeFailed)
    async def _invalid_image(  # pyright: ignore[reportUnusedFunction]
        _request: Request, failure: ImageDecodeFailed
    ) -> JSONResponse:
        return failure_response(
            HTTPStatus.UNPROCESSABLE_CONTENT,
            ErrorCode.INVALID_IMAGE,
            f"Invalid image: {failure.detail}",
        )

    @application.exception_handler(UploadTooLarge)
    async def _upload_too_large(  # pyright: ignore[reportUnusedFunction]
        _request: Request, failure: UploadTooLarge
    ) -> JSONResponse:
        return failure_response(
            HTTPStatus.CONTENT_TOO_LARGE,
            ErrorCode.UPLOAD_TOO_LARGE,
            f"Upload exceeds {failure.limit_bytes} bytes.",
        )

    @application.exception_handler(RateLimited)
    async def _rate_limited(  # pyright: ignore[reportUnusedFunction]
        _request: Request, failure: RateLimited
    ) -> JSONResponse:
        return failure_response(
            HTTPStatus.TOO_MANY_REQUESTS,
            ErrorCode.RATE_LIMITED,
            MESSAGE.rate_limited,
            headers={HEADER.retry_after: str(failure.retry_after_seconds)},
        )


def _register_server_failures(application: FastAPI) -> None:
    """Attach the failures that are the deployment's problem.

    :param application: The application to configure.
    :type application: fastapi.FastAPI
    """

    @application.exception_handler(DeviceExhausted)
    async def _device_exhausted(  # pyright: ignore[reportUnusedFunction]
        _request: Request, failure: DeviceExhausted
    ) -> JSONResponse:
        return map_device_exhausted(failure)

    @application.exception_handler(FaceAnalysisUnavailable)
    async def _face_unavailable(  # pyright: ignore[reportUnusedFunction]
        _request: Request, failure: FaceAnalysisUnavailable
    ) -> JSONResponse:
        _LOGGER.error("Face analysis unavailable: %s", failure.detail)
        return failure_response(
            HTTPStatus.NOT_IMPLEMENTED,
            ErrorCode.UNAVAILABLE_FEATURE,
            MESSAGE.feature_unavailable,
        )

    @application.exception_handler(ModelUnavailable)
    async def _model_unavailable(  # pyright: ignore[reportUnusedFunction]
        _request: Request, failure: ModelUnavailable
    ) -> JSONResponse:
        _LOGGER.error(
            "Model %s unavailable: %s", failure.model, failure.detail
        )
        return failure_response(
            HTTPStatus.SERVICE_UNAVAILABLE,
            ErrorCode.NOT_READY,
            MESSAGE.models_loading,
        )

    @application.exception_handler(Exception)
    async def _unhandled(  # pyright: ignore[reportUnusedFunction]
        _request: Request, failure: Exception
    ) -> JSONResponse:
        # The transport boundary, and the only broad catch in the code
        # base. Logged with its traceback, opaque to the caller: the
        # message could carry a path, a prompt or a checkpoint name.
        _LOGGER.exception("Unhandled defect: %s", failure)
        return failure_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            ErrorCode.INTERNAL,
            MESSAGE.internal,
        )


def register_error_handlers(application: FastAPI) -> None:
    """Attach every failure to its response.

    :param application: The application to configure.
    :type application: fastapi.FastAPI
    """
    _register_client_failures(application)
    _register_server_failures(application)


__all__: list[str] = [
    "failure_response",
    "map_device_exhausted",
    "map_no_detection",
    "register_error_handlers",
]
