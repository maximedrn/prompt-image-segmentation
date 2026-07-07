"""Central error -> HTTP response mapping."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core import (
    BackendUnavailableError,
    InvalidPromptError,
    NoDetectionError,
    SegmenterError,
)


def _payload(error: str, message: str) -> dict[str, str]:
    """Shape the JSON body every error handler returns.

    :param error: Short machine-readable error code.
    :type error: str
    :param message: Human-readable explanation.
    :type message: str
    :returns: ``{"error": ..., "message": ...}``.
    :rtype: dict[str, str]
    """
    return {"error": error, "message": message}


def register_error_handlers(app: FastAPI) -> None:
    """Attach the four domain-error -> HTTP response mappings.

    :param app: The FastAPI application to configure.
    :type app: fastapi.FastAPI
    :returns: ``None``.
    :rtype: None
    """

    @app.exception_handler(NoDetectionError)
    async def _no_detection(  # pyright: ignore[reportUnusedFunction]
        _request: Request, exception: NoDetectionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_payload("no_detection", str(exception)),
        )

    @app.exception_handler(InvalidPromptError)
    async def _invalid_prompt(  # pyright: ignore[reportUnusedFunction]
        _request: Request, exception: InvalidPromptError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_payload("invalid_prompt", exception.reason),
        )

    @app.exception_handler(BackendUnavailableError)
    async def _unknown_backend(  # pyright: ignore[reportUnusedFunction]
        _request: Request, exception: BackendUnavailableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_payload("unknown_backend", str(exception)),
        )

    @app.exception_handler(SegmenterError)
    async def _generic(  # pyright: ignore[reportUnusedFunction]
        _request: Request, exception: SegmenterError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_payload("internal", str(exception)),
        )


__all__: list[str] = ["register_error_handlers"]
