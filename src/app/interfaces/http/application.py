"""FastAPI application factory.

Assembles the transport around the wired application. Model construction
happens in the lifespan, off the event loop, so the liveness probe keeps
answering while several hundred megabytes of weights are fetched.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from http import HTTPStatus
from logging import Logger, getLogger
from warnings import filterwarnings

from anyio import create_task_group
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image as PilImage, ImageFile
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.application.capabilities import JobStore
from app.application.jobs import JobPayload, JobResult
from app.application.policies import WebhookPolicy
from app.application.use_cases.process_job import Segmenter, consume
from app.bootstrap import Application, SegmentOutcome, build
from app.domain import (
    DeviceExhausted,
    NoDetection,
    PersonPayload,
    Prompt,
    SegmentedImage,
    SourceImage,
)
from app.infrastructure.imaging.imaging import decode_image, encode_png
from app.infrastructure.jobs import build_store
from app.infrastructure.webhooks.notifier import (
    SignedWebhookNotifier,
    client as webhook_client,
)
from app.interfaces.http.auth import Unauthorized
from app.interfaces.http.constants import (
    ApiMetadata,
    AuthScheme,
    Decoding,
    ErrorCode,
    HeaderName,
    LogMessage,
    Message,
    OpenApiTag,
)
from app.interfaces.http.errors import (
    failure_response,
    register_error_handlers,
    register_transport_failure,
)
from app.interfaces.http.rate_limit import FixedWindowLimiter
from app.interfaces.http.routes import (
    events_router,
    meta_router,
    segmentation_router,
)
from app.interfaces.http.schemas import RegionSchema, SegmentSchema
from app.settings import Settings

_logger: Logger = getLogger(__name__)


def _unauthorized(failure: Unauthorized) -> JSONResponse:
    return failure_response(
        HTTPStatus.UNAUTHORIZED,
        ErrorCode.UNAUTHORIZED,
        failure.message,
        headers={HeaderName.authenticate: AuthScheme.basic},
    )


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
    store: JobStore = application.state.store
    # Before the models: the store is what a probe and a poll need, and
    # neither should wait on several hundred megabytes of weights.
    await store.open()
    try:
        wired: Application = await run_in_threadpool(build, settings)
    # Model loading touches network, disk and driver: none of those may
    # take the process down at boot.
    except Exception:  # pylint: disable=broad-except
        _logger.exception(LogMessage.startup_failed)
        yield
        await store.close()
        return

    application.state.application = wired
    # One consumer per process, which is what the device lock already
    # implies: a second would only queue behind the first.
    async with create_task_group() as workers:
        workers.start_soon(
            consume,
            store,
            _segmenter(wired),
            lambda: application.state.consuming,
            application.state.notifier,
        )
        yield
        # The claim call wakes up on its own timeout, so clearing the
        # flag is enough to end the loop without cancelling mid-job.
        application.state.consuming = False
    await store.close()


def _segmenter(wired: Application) -> Segmenter:
    """Bind the wired application into what a worker can call.

    Translation lives here rather than in the worker: mapping a typed
    outcome onto a wire body is the transport's job, and the worker has
    no business knowing what a caller will read.

    :param wired: The assembled application.
    :type wired: app.bootstrap.Application
    :returns: A callable turning one payload into one result.
    :rtype: app.application.use_cases.process_job.Segmenter
    """

    def run(payload: JobPayload) -> JobResult:
        """Run one payload to a result.

        :param payload: What the caller submitted.
        :type payload: app.application.jobs.JobPayload
        :returns: The body, or the failure code that stopped it.
        :rtype: app.application.jobs.JobResult
        """
        source: SourceImage = decode_image(payload.image)
        prompt: Prompt = Prompt.parse(payload.prompt)
        person: PersonPayload | None = (
            wired.analyse_faces(source) if payload.person_mode else None
        )
        outcome: SegmentOutcome = wired.segment(
            payload.backend, source, prompt, person, payload.options
        )
        match outcome:
            case NoDetection():
                return JobResult(error=ErrorCode.NO_DETECTION)
            case DeviceExhausted():
                return JobResult(error=ErrorCode.OUT_OF_MEMORY)
            case SegmentedImage():
                return JobResult(
                    body=SegmentSchema.of(
                        result=outcome,
                        prompt=prompt.text,
                        segmenter=payload.backend,
                        regions=RegionSchema.of_all(
                            outcome.regions, encode_png
                        ),
                    )
                )

    return run


def create_app(settings: Settings | None = None) -> FastAPI:
    """Compose the transport.

    :param settings: Configuration to run with. Loaded from the
        environment when omitted.
    :type settings: app.settings.Settings | None
    :returns: A ready-to-serve application.
    :rtype: fastapi.FastAPI
    :raises RuntimeError: If the UI is enabled but its optional extra is
        not installed.
    """
    resolved: Settings = settings if settings is not None else Settings()
    ImageFile.LOAD_TRUNCATED_IMAGES = Decoding.load_truncated
    PilImage.MAX_IMAGE_PIXELS = resolved.max_image_pixels
    # Pillow only raises past *twice* the ceiling; between one and two
    # times it emits a warning and decodes anyway. Promoting that
    # warning is what makes the configured number the actual limit
    # rather than half of it.
    filterwarnings(
        Decoding.bomb_action, category=PilImage.DecompressionBombWarning
    )

    application: FastAPI = FastAPI(
        title=ApiMetadata.title,
        version=__version__,
        description=ApiMetadata.description,
        openapi_tags=list(OpenApiTag.descriptions),
        lifespan=_lifespan,
    )
    application.state.settings = resolved
    application.state.application = None
    application.state.store = build_store(resolved.job_policy())
    webhooks: WebhookPolicy = resolved.webhook_policy()
    application.state.notifier = (
        SignedWebhookNotifier(webhook_client(), webhooks)
        if webhooks.enabled
        else None
    )
    application.state.consuming = True
    application.state.limiter = FixedWindowLimiter(
        resolved.rate_limit_policy()
    )

    register_error_handlers(application)
    register_transport_failure(application, Unauthorized, _unauthorized)

    application.include_router(meta_router)
    application.include_router(segmentation_router)
    application.include_router(events_router)

    if resolved.enable_ui:
        # Imported here so an API-only deployment never pays for gradio:
        # a heavy import and an extra slice of attack surface. It is an
        # optional extra, hence absent from the published images, so the
        # failure has to name the flag that asked for it.
        # pylint: disable=import-outside-toplevel
        try:
            from app.interfaces.ui import mount_ui
        except ImportError as error:
            raise RuntimeError(Message.ui_unavailable) from error

        mount_ui(application, resolved)

    return application


__all__: list[str] = ["create_app"]
