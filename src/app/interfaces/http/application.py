"""FastAPI application factory.

Assembles the transport around the wired application. Model construction
happens in the lifespan, off the event loop, so the liveness probe keeps
answering while several hundred megabytes of weights are fetched.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from http import HTTPStatus
from logging import Logger, getLogger
from typing import Final

from PIL import Image as PilImage, ImageFile
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.bootstrap import Application, build
from app.interfaces.http.auth import Unauthorized
from app.interfaces.http.constants import (
    API,
    HEADER,
    SCHEME,
    ErrorCode,
)
from app.interfaces.http.errors import (
    failure_response,
    register_error_handlers,
)
from app.interfaces.http.rate_limit import FixedWindowLimiter
from app.interfaces.http.routes import meta_router, segmentation_router
from app.settings import Settings

_LOGGER: Logger = getLogger(__name__)

# A truncated JPEG should still yield a usable image, but that leniency
# only holds because the pixel ceiling below caps what a malformed file
# can allocate.
_LOAD_TRUNCATED: Final[bool] = True


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncGenerator[None]:
    """Build the wired application off the event loop.

    A failure is logged rather than fatal: the process stays up and the
    readiness probe keeps answering 503, which is a recoverable state
    instead of a restart loop.

    :param application: The FastAPI application being started.
    :type application: fastapi.FastAPI
    :returns: An async iterator yielding once.
    :rtype: collections.abc.AsyncGenerator[None]
    """
    settings: Settings = application.state.settings
    try:
        wired: Application = await run_in_threadpool(build, settings)
    # Model loading touches network, disk and driver: none of those may
    # take the process down at boot.
    except Exception as failure:  # pylint: disable=broad-except
        _LOGGER.exception("Startup failed, staying unready: %s", failure)
    else:
        application.state.application = wired
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Compose the transport.

    :param settings: Configuration to run with. Loaded from the
        environment when omitted.
    :type settings: app.settings.Settings | None
    :returns: A ready-to-serve application.
    :rtype: fastapi.FastAPI
    """
    resolved: Settings = settings if settings is not None else Settings()
    ImageFile.LOAD_TRUNCATED_IMAGES = _LOAD_TRUNCATED
    PilImage.MAX_IMAGE_PIXELS = resolved.max_image_pixels

    application: FastAPI = FastAPI(
        title=API.title,
        version=__version__,
        description=API.description,
        lifespan=_lifespan,
    )
    application.state.settings = resolved
    application.state.application = None
    application.state.limiter = FixedWindowLimiter(
        resolved.rate_limit_policy()
    )

    register_error_handlers(application)

    @application.exception_handler(Unauthorized)
    async def _unauthorized(  # pyright: ignore[reportUnusedFunction]
        _request: Request, failure: Unauthorized
    ) -> JSONResponse:
        return failure_response(
            HTTPStatus.UNAUTHORIZED,
            ErrorCode.UNAUTHORIZED,
            failure.message,
            headers={HEADER.authenticate: SCHEME.basic},
        )

    application.include_router(meta_router)
    application.include_router(segmentation_router)

    if resolved.enable_ui:
        # Imported here so an API-only deployment never pays for gradio:
        # a heavy import and an extra slice of attack surface.
        # pylint: disable=import-outside-toplevel
        from app.interfaces.ui import mount_ui

        mount_ui(application, resolved)

    return application


__all__: list[str] = ["create_app"]
