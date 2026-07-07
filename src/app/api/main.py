"""FastAPI application factory + Gradio mount + lifespan warmup."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from PIL import ImageFile
from fastapi import FastAPI
from gradio import Blocks, mount_gradio_app

from app import __version__
from app.api.errors import register_error_handlers
from app.api.routes import meta_router, segmentation_router
from app.config import get_settings
from app.config.settings import Settings
from app.managers import ModelManager
from app.ui import build_gradio_blocks

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _gradio_auth(
    settings_username: str | None,
    settings_password: str | None,
) -> tuple[str, str] | None:
    """Build the ``auth`` tuple Gradio expects, or ``None`` if unset.

    :param settings_username: Username from settings (may be ``None``).
    :type settings_username: str | None
    :param settings_password: Password from settings (may be ``None``).
    :type settings_password: str | None
    :returns: ``(user, pass)`` when both are set, else ``None``.
    :rtype: tuple[str, str] | None
    """
    if settings_username and settings_password:
        return settings_username, settings_password
    return None


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan: warm up every model on startup.

    :param _app: The FastAPI application (unused; kept for the API
        contract).
    :type _app: fastapi.FastAPI
    :returns: An async iterator yielding once.
    :rtype: collections.abc.AsyncIterator[None]
    """
    ModelManager().warmup()
    yield


def create_app() -> FastAPI:
    """Compose the FastAPI app + routers + Gradio mount + handlers.

    :returns: A ready-to-serve FastAPI instance.
    :rtype: fastapi.FastAPI
    """
    settings: Settings = get_settings()
    fastapi_app: FastAPI = FastAPI(
        title="Prompt image segmentation",
        version=__version__,
        description=(
            "Prompt-driven segmentation via GroundingDINO + SAM. "
            "Layered architecture: config / core / domain / "
            "infrastructure / models / managers / segmenters / "
            "services / api / ui."
        ),
        lifespan=_lifespan,
    )
    register_error_handlers(fastapi_app)
    fastapi_app.include_router(meta_router)
    fastapi_app.include_router(segmentation_router)
    blocks: Blocks = build_gradio_blocks()
    mount_gradio_app(
        fastapi_app,
        blocks,
        path=settings.ui_mount_path,
        auth=_gradio_auth(
            settings.segmentation_username,
            settings.segmentation_password,
        ),
    )
    return fastapi_app


app: FastAPI = create_app()


__all__: list[str] = ["app", "create_app"]
