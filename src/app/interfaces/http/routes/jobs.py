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

from fastapi import APIRouter, Depends, Form, Header, Request
from fastapi.responses import JSONResponse

from app.application.capabilities import JobStore
from app.application.jobs import IdempotentRequest, JobPayload
from app.application.policies import WebhookPolicy
from app.bootstrap import Application
from app.domain import (
    Admission,
    Job,
    JobState,
    Prompt,
    UnknownBackend,
    cancel,
    queued,
)
from app.infrastructure.jobs.hashing import request_hash
from app.infrastructure.webhooks.notifier import acceptable
from app.interfaces.http.auth import (
    Credentials,
    principal_digest,
    require_credentials,
)
from app.interfaces.http.constants import (
    AuthRules,
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


def key_within_limit(idempotency_key: str | None) -> bool:
    """Report whether a caller's retry key is short enough to store.

    The key becomes part of a stored key, so its length is the
    service's to decide rather than something inherited from whatever
    header limit a deployment happens to run with.

    :param idempotency_key: The caller's retry key, when they sent one.
    :type idempotency_key: str | None
    :returns: ``True`` when there is no key, or it is within the bound.
    :rtype: bool
    """
    if idempotency_key is None:
        return True
    return len(idempotency_key) <= AuthRules.idempotency_key_max


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
    credentials: Credentials = None,
    idempotency_key: Annotated[
        str | None, Header(alias=HeaderName.idempotency_key)
    ] = None,
) -> JobSchema | JSONResponse:
    """Accept a segmentation and hand back something to poll.

    A caller that supplies ``Idempotency-Key`` gets the same job back for
    the same request, however many times a flaky connection makes them
    send it. The same key with a *different* request is refused rather
    than guessed at: whichever of the two the server picked would be
    wrong for the other one.

    :param request: Incoming request.
    :type request: fastapi.Request
    :param form: The whole multipart body, already validated.
    :type form: app.interfaces.http.schemas.SegmentForm
    :param credentials: Credentials the caller presented, used only to
        name whose idempotency key this is.
    :type credentials: fastapi.security.HTTPBasicCredentials | None
    :param idempotency_key: The caller's retry key, when they sent one.
    :type idempotency_key: str | None
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
    job_payload = JobPayload(
        image=payload,
        prompt=prompt.text,
        backend=backend,
        person_mode=form.person_mode,
        options=form.to_options(),
        callback_url=form.callback_url,
    )
    if not key_within_limit(idempotency_key):
        return failure_response(
            HTTPStatus.UNPROCESSABLE_CONTENT,
            ErrorCode.IDEMPOTENCY_KEY_TOO_LONG,
            JobMessage.idempotency_key_too_long,
        )
    admission: Admission = await store.enqueue(
        job,
        job_payload,
        idempotency=(
            None
            if idempotency_key is None
            else IdempotentRequest(
                key=idempotency_key,
                request_hash=request_hash(job_payload),
                # Who is claiming the key, so one caller's choice of
                # string cannot answer another caller's submission.
                scope=principal_digest(credentials),
            )
        ),
    )
    if admission.conflicted:
        return failure_response(
            HTTPStatus.CONFLICT,
            ErrorCode.IDEMPOTENCY_CONFLICT,
            JobMessage.idempotency_conflict,
        )
    if admission.replayed is not None:
        # The first submission's job, as it stands now. Read rather than
        # reconstructed: by the time a retry lands the job may already be
        # running or finished, and answering with a fresh ``queued`` would
        # tell the caller something that stopped being true.
        return await _replayed(store, admission.replayed)
    if admission.position is None:
        return failure_response(
            HTTPStatus.TOO_MANY_REQUESTS,
            ErrorCode.QUEUE_FULL,
            JobMessage.queue_full,
            headers={HeaderName.retry_after: str(RetryAfter.queue_full)},
        )
    return JobSchema.of(job, queue_position=admission.position)


async def _replayed(
    store: JobStore, identifier: str
) -> JobSchema | JSONResponse:
    """Answer a replay with the job the first submission created.

    :param store: Where jobs are kept.
    :type store: app.application.capabilities.JobStore
    :param identifier: The job the key was spent on.
    :type identifier: str
    :returns: That job, or a failure if it has since expired.
    :rtype: app.interfaces.http.schemas.JobSchema
        | fastapi.responses.JSONResponse
    """
    if (found := await store.read(identifier)) is None:
        # The claim outlived its job only if the job was purged early.
        # Reporting it unknown is honest; inventing a fresh one would
        # queue work the caller never asked for twice.
        return failure_response(
            HTTPStatus.NOT_FOUND, ErrorCode.UNKNOWN_JOB, JobMessage.unknown
        )
    job, result = found
    return JobSchema.of(
        job,
        result=body_of(result),
        queue_position=None if job.terminal else await store.depth(),
    )


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
