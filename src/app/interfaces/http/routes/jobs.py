"""The job API: accept work, report on it, withdraw it.

Segmentation runs on a single accelerator, so a request is queued rather
than held on an open connection. What a caller can be told immediately -
an unreadable image, an unknown backend, a full queue - is answered here,
while it still holds that connection.
"""

from http import HTTPStatus
from time import time
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import JSONResponse

from app.application.capabilities import JobStore
from app.application.jobs import JobPayload
from app.application.policies import WebhookPolicy
from app.bootstrap import Application
from app.domain import Job, JobState, Prompt, UnknownBackend, cancel, queued
from app.infrastructure.webhooks.notifier import acceptable
from app.interfaces.http.auth import require_credentials
from app.interfaces.http.constants import (
    Documented,
    ErrorCode,
    HeaderName,
    HttpRoute,
    JobMessage,
    OpenApiKey,
    OpenApiTag,
    RetryAfter,
    RouteDocs,
)
from app.interfaces.http.errors import failure_response
from app.interfaces.http.rate_limit import enforce_rate_limit
from app.interfaces.http.routes.dependencies import (
    accepted_bytes,
    application_of,
    body_of,
    settings_of,
    store_of,
)
from app.interfaces.http.schemas import (
    ErrorSchema,
    JobSchema,
    SegmentForm,
    SegmentSchema,
)
from app.settings import Settings

segmentation_router: APIRouter = APIRouter(
    tags=[OpenApiTag.segmentation],
    dependencies=[
        Depends(require_credentials),
        Depends(enforce_rate_limit),
    ],
)

#: Derived, not a literal: the statuses live in the vocabulary and
#: this is the shape FastAPI wants them in.
type _Responses = dict[int | str, dict[str, type[ErrorSchema]]]


def _failures(statuses: tuple[HTTPStatus, ...]) -> _Responses:
    """Declare one route's failure bodies for the OpenAPI document.

    :param statuses: Statuses this route can actually answer with.
    :type statuses: tuple[http.HTTPStatus, ...]
    :returns: The ``responses`` mapping FastAPI expects.
    :rtype: dict[int | str, dict[str, type[ErrorSchema]]]
    """
    return {status: {OpenApiKey.model: ErrorSchema} for status in statuses}


@segmentation_router.post(
    HttpRoute.jobs,
    response_model=JobSchema,
    status_code=HTTPStatus.ACCEPTED,
    responses=_failures(Documented.acceptance),
    summary=RouteDocs.create_job.summary,
    description=RouteDocs.create_job.description,
)
async def create_job(
    request: Request,
    form: Annotated[SegmentForm, Form()],
) -> JobSchema | JSONResponse:
    """Accept a segmentation and hand back something to poll.

    :param request: Incoming request.
    :type request: fastapi.Request
    :param form: The whole multipart body, already validated.
    :type form: app.interfaces.http.schemas.SegmentForm
    :returns: The accepted job, or a failure response.
    :rtype: app.interfaces.http.schemas.JobSchema
        | fastapi.responses.JSONResponse
    :raises app.domain.errors.UnknownBackend: If ``segmenter`` names a
        backend that is not wired.
    """
    # Ceilings first, and before readiness: an oversized or unreadable
    # upload is the caller's problem whether or not the models are up,
    # and answering that costs no round trip to anything.
    settings: Settings = settings_of(request)
    payload: bytes = await accepted_bytes(form.image, settings)
    prompt: Prompt = Prompt.parse(form.prompt)

    application: Application = application_of(request)
    store: JobStore = store_of(request)

    backend: str = form.segmenter or application.settings.default_segmenter
    if backend not in application.backends:
        raise UnknownBackend(
            requested=backend,
            available=tuple(sorted(application.backends)),
        )

    if form.callback_url is not None:
        webhooks: WebhookPolicy = settings.webhook_policy()
        if not webhooks.enabled:
            return failure_response(
                HTTPStatus.UNPROCESSABLE_CONTENT,
                ErrorCode.INVALID_CALLBACK,
                JobMessage.callback_disabled,
            )
        if not await acceptable(form.callback_url, webhooks):
            return failure_response(
                HTTPStatus.UNPROCESSABLE_CONTENT,
                ErrorCode.INVALID_CALLBACK,
                JobMessage.callback_refused,
            )

    # One round trip, and the only one that decides: ``enqueue`` checks
    # the depth and stores the job in the same transaction, so two
    # callers racing at the ceiling cannot both be let through. Reading
    # the depth here first would add a round trip to every accepted job
    # to save one decode on the rare refused one.
    job: Job = queued(str(uuid4()), time())
    position: int | None = await store.enqueue(
        job,
        JobPayload(
            image=payload,
            prompt=prompt.text,
            backend=backend,
            person_mode=form.person_mode,
            options=form.to_options(),
            callback_url=form.callback_url,
        ),
    )
    if position is None:
        return failure_response(
            HTTPStatus.TOO_MANY_REQUESTS,
            ErrorCode.QUEUE_FULL,
            JobMessage.queue_full,
            headers={HeaderName.retry_after: str(RetryAfter.queue_full)},
        )
    return JobSchema.of(job, queue_position=position)


@segmentation_router.get(
    HttpRoute.job,
    response_model=JobSchema,
    responses=_failures(Documented.polling),
    summary=RouteDocs.read_job.summary,
    description=RouteDocs.read_job.description,
)
async def read_job(
    request: Request, identifier: str
) -> JobSchema | JSONResponse:
    """Report where a job is, and its answer once it has one.

    :param request: Incoming request.
    :type request: fastapi.Request
    :param identifier: What acceptance handed back.
    :type identifier: str
    :returns: The job, or a failure response.
    :rtype: app.interfaces.http.schemas.JobSchema
        | fastapi.responses.JSONResponse
    """
    store: JobStore = store_of(request)
    if (found := await store.read(identifier)) is None:
        return failure_response(
            HTTPStatus.NOT_FOUND, ErrorCode.UNKNOWN_JOB, JobMessage.unknown
        )
    job, result = found
    body: SegmentSchema | None = body_of(result)
    position: int | None = None if job.terminal else await store.depth()
    return JobSchema.of(job, result=body, queue_position=position)


@segmentation_router.delete(
    HttpRoute.job,
    response_model=JobSchema,
    responses=_failures(Documented.withdrawal),
    summary=RouteDocs.cancel_job.summary,
    description=RouteDocs.cancel_job.description,
)
async def cancel_job(
    request: Request, identifier: str
) -> JobSchema | JSONResponse:
    """Withdraw a job that has not started yet.

    :param request: Incoming request.
    :type request: fastapi.Request
    :param identifier: What acceptance handed back.
    :type identifier: str
    :returns: The cancelled job, or a failure response.
    :rtype: app.interfaces.http.schemas.JobSchema
        | fastapi.responses.JSONResponse
    """
    store: JobStore = store_of(request)
    if (found := await store.read(identifier)) is None:
        return failure_response(
            HTTPStatus.NOT_FOUND, ErrorCode.UNKNOWN_JOB, JobMessage.unknown
        )
    job, _ = found
    if job.state is not JobState.QUEUED:
        return failure_response(
            HTTPStatus.CONFLICT,
            ErrorCode.ALREADY_STARTED,
            JobMessage.already_started,
        )
    cancelled: Job = cancel(job, time())
    await store.write(cancelled)
    return JobSchema.of(cancelled)


__all__: list[str] = [
    "cancel_job",
    "create_job",
    "read_job",
    "segmentation_router",
]
