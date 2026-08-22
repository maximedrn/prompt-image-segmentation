"""Typed outcome to HTTP response.

Each failure gets a plain function that turns it into a body. Starlette
wants a handler taking ``Exception``, which no precise function can
satisfy, so :func:`register_transport_failure` bridges the two: it
narrows once, at the one place that knows which type it registered for.
That is what lets every responder below keep its real parameter type,
with no cast and no suppression anywhere.
"""

from collections.abc import Callable
from http import HTTPStatus
from logging import Logger, getLogger

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from app.domain import (
    DeviceExhausted,
    FaceAnalysisUnavailable,
    ImageDecodeFailed,
    InvalidPrompt,
    JobStoreUnavailable,
    ModelUnavailable,
    NoDetection,
    RateLimited,
    UnknownBackend,
    UploadTooLarge,
)
from app.interfaces.http.constants import (
    ErrorCode,
    FailureText,
    HeaderName,
    JobMessage,
    LogMessage,
    Message,
    RetryAfter,
    Serialisation,
)
from app.interfaces.http.schemas import ErrorSchema

_logger: Logger = getLogger(__name__)

type Responder[FailedT: Exception] = Callable[[FailedT], JSONResponse]
"""Turns one kind of failure into the body a caller reads."""


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
        content=body.model_dump(mode=Serialisation.json_mode),
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
        FailureText.no_detection.format(prompt=failure.prompt),
    )


def map_device_exhausted(failure: DeviceExhausted) -> JSONResponse:
    """The accelerator ran out of memory: retryable with a smaller image.

    :param failure: The typed failure.
    :type failure: app.domain.errors.DeviceExhausted
    :returns: A 503 response.
    :rtype: fastapi.responses.JSONResponse
    """
    _logger.warning(LogMessage.device_exhausted, failure.detail)
    return failure_response(
        HTTPStatus.SERVICE_UNAVAILABLE,
        ErrorCode.OUT_OF_MEMORY,
        Message.out_of_memory,
    )


def _invalid_prompt(failure: InvalidPrompt) -> JSONResponse:
    return failure_response(
        HTTPStatus.UNPROCESSABLE_CONTENT,
        ErrorCode.INVALID_PROMPT,
        failure.reason,
    )


def _unknown_backend(failure: UnknownBackend) -> JSONResponse:
    return failure_response(
        HTTPStatus.BAD_REQUEST,
        ErrorCode.UNKNOWN_BACKEND,
        FailureText.unknown_segmenter.format(
            requested=failure.requested, available=list(failure.available)
        ),
    )


def _invalid_image(failure: ImageDecodeFailed) -> JSONResponse:
    return failure_response(
        HTTPStatus.UNPROCESSABLE_CONTENT,
        ErrorCode.INVALID_IMAGE,
        FailureText.invalid_image.format(detail=failure.detail),
    )


def _upload_too_large(failure: UploadTooLarge) -> JSONResponse:
    return failure_response(
        HTTPStatus.CONTENT_TOO_LARGE,
        ErrorCode.UPLOAD_TOO_LARGE,
        FailureText.upload_too_large.format(limit=failure.limit_bytes),
    )


def _rate_limited(failure: RateLimited) -> JSONResponse:
    return failure_response(
        HTTPStatus.TOO_MANY_REQUESTS,
        ErrorCode.RATE_LIMITED,
        Message.rate_limited,
        headers={HeaderName.retry_after: str(failure.retry_after_seconds)},
    )


def _face_unavailable(failure: FaceAnalysisUnavailable) -> JSONResponse:
    _logger.error(LogMessage.face_unavailable, failure.detail)
    return failure_response(
        HTTPStatus.NOT_IMPLEMENTED,
        ErrorCode.UNAVAILABLE_FEATURE,
        Message.feature_unavailable,
    )


def _model_unavailable(failure: ModelUnavailable) -> JSONResponse:
    _logger.error(LogMessage.model_unavailable, failure.model, failure.detail)
    return failure_response(
        HTTPStatus.SERVICE_UNAVAILABLE,
        ErrorCode.NOT_READY,
        Message.models_loading,
    )


def _store_unavailable(failure: JobStoreUnavailable) -> JSONResponse:
    _logger.error(LogMessage.store_unavailable, failure.detail)
    return failure_response(
        HTTPStatus.SERVICE_UNAVAILABLE,
        ErrorCode.STORE_UNAVAILABLE,
        JobMessage.store_unavailable,
        headers={HeaderName.retry_after: str(RetryAfter.store_unavailable)},
    )


def _unhandled(failure: Exception) -> JSONResponse:
    # The transport boundary, and the only broad catch in the code base.
    # Logged with its traceback, opaque to the caller: the message could
    # carry a path, a prompt or a checkpoint name.
    _logger.exception(LogMessage.unhandled, failure)
    return failure_response(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        ErrorCode.INTERNAL,
        Message.internal,
    )


def register_transport_failure[FailureT: Exception](
    application: FastAPI,
    failure_type: type[FailureT],
    respond: Responder[FailureT],
) -> None:
    """Attach one responder to the failure it answers for.

    :param application: The application to configure.
    :type application: fastapi.FastAPI
    :param failure_type: What this responder was written for.
    :type failure_type: type[FailureT]
    :param respond: Turns that failure into a body.
    :type respond: Responder[FailureT]
    """

    async def handle(request: Request, failure: Exception) -> Response:
        """Narrow to the registered type, then answer.

        :param request: Unused: a failure body depends on the failure.
        :type request: fastapi.Request
        :param failure: Whatever starlette caught.
        :type failure: Exception
        :returns: The failure body.
        :rtype: starlette.responses.Response
        """
        del request
        if not isinstance(failure, failure_type):
            # Starlette dispatches on the type registered here, so this
            # is unreachable. Re-raising rather than guessing keeps it
            # that way instead of inventing a response for it.
            raise failure
        return respond(failure)

    application.add_exception_handler(failure_type, handle)


def _register_client_failures(application: FastAPI) -> None:
    """Attach the failures the caller can fix themselves.

    :param application: The application to configure.
    :type application: fastapi.FastAPI
    """
    register_transport_failure(application, InvalidPrompt, _invalid_prompt)
    register_transport_failure(application, NoDetection, map_no_detection)
    register_transport_failure(application, UnknownBackend, _unknown_backend)
    register_transport_failure(application, ImageDecodeFailed, _invalid_image)
    register_transport_failure(application, UploadTooLarge, _upload_too_large)
    register_transport_failure(application, RateLimited, _rate_limited)


def _register_server_failures(application: FastAPI) -> None:
    """Attach the failures that are the deployment's problem.

    :param application: The application to configure.
    :type application: fastapi.FastAPI
    """
    register_transport_failure(
        application, DeviceExhausted, map_device_exhausted
    )
    register_transport_failure(
        application, FaceAnalysisUnavailable, _face_unavailable
    )
    register_transport_failure(
        application, ModelUnavailable, _model_unavailable
    )
    register_transport_failure(
        application, JobStoreUnavailable, _store_unavailable
    )
    register_transport_failure(application, Exception, _unhandled)


def register_error_handlers(application: FastAPI) -> None:
    """Attach every failure to its response.

    :param application: The application to configure.
    :type application: fastapi.FastAPI
    """
    _register_client_failures(application)
    _register_server_failures(application)


__all__: list[str] = [
    "Responder",
    "failure_response",
    "map_device_exhausted",
    "map_no_detection",
    "register_error_handlers",
    "register_transport_failure",
]
