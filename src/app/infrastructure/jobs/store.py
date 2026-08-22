"""Redis-backed job store.

Implements :class:`~app.application.capabilities.JobStore`. The only
module that knows Redis exists.

Four keys per job, not one: the state changes on every transition, the
metadata is written once and read once, the image is megabytes that
nothing else wants dragged along, and the result is more of the same.
Splitting them means a poll reads a small document instead of the
pixels with it.

The image is stored as the bytes that arrived, so the client speaks
binary and the JSON documents are parsed straight from bytes. That is
what keeps a base64 round trip - and the third of a copy it adds - out
of the hot path.

Every key carries the same TTL, so an abandoned job disappears on its
own. That expiry is also what reclaims a job whose worker died mid-run:
the state machine deliberately offers no way out of ``RUNNING``.

coredis rather than redis-py: every command is typed exactly, so no
reply arrives as ``Awaitable[T] | T`` and nothing here needs a cast to
be read. Its connection pool has a lifetime, which is what :meth:`open`
and :meth:`close` exist to give it.

No server-side script: everything below is ordinary commands, and the
one thing that needed atomicity - admitting a job without letting two
callers past the same ceiling - gets it from a transaction and from
what ``RPUSH`` answers.
"""

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Self, final

from coredis import Redis
from coredis.exceptions import RedisError
from pydantic import BaseModel, ConfigDict

from app.application.jobs import JobPayload, JobResult
from app.application.jobs.results import SegmentSchema
from app.application.policies import JobPolicy, SegmentOptions
from app.domain import Job, JobState, JobStoreUnavailable
from app.infrastructure.imaging.types import TextEncoding
from app.infrastructure.jobs.constants import Claiming, JobKeys


@asynccontextmanager
async def _reachable() -> AsyncGenerator[None]:
    """Translate a Redis outage into something the transport knows.

    This module is the only one allowed to know Redis exists, so it is
    also the only one that can stop a ``RedisError`` from reaching the
    catch-all handler and being reported as an internal defect. An
    unreachable dependency is a 503, not a 500.

    :returns: A context manager guarding one exchange with the server.
    :rtype: collections.abc.AsyncGenerator[None]
    :raises app.domain.errors.JobStoreUnavailable: If Redis fails.
    """
    try:
        yield
    except RedisError as error:
        raise JobStoreUnavailable(detail=str(error)) from error


@final
class _PayloadDocument(BaseModel):
    """How a payload's metadata is written to and read from the store.

    Pydantic here for the same reason as at the HTTP boundary: what
    comes back out of a store is untrusted text until something has
    checked its shape. The image travels beside this, as bytes.
    """

    model_config = ConfigDict(frozen=True)

    prompt: str
    backend: str
    person_mode: bool
    options: SegmentOptions
    callback_url: str | None = None

    @classmethod
    def of(cls, payload: JobPayload) -> Self:
        """Project a payload onto its stored form, image excepted.

        :param payload: What the worker will need.
        :type payload: app.application.jobs.JobPayload
        :returns: The storable document.
        :rtype: _PayloadDocument
        """
        return cls(
            prompt=payload.prompt,
            backend=payload.backend,
            person_mode=payload.person_mode,
            options=payload.options,
            callback_url=payload.callback_url,
        )

    def payload(self, image: bytes) -> JobPayload:
        """Rebuild the payload as it was accepted.

        :param image: The bytes stored alongside this document.
        :type image: bytes
        :returns: The payload.
        :rtype: app.application.jobs.JobPayload
        """
        return JobPayload(
            image=image,
            prompt=self.prompt,
            backend=self.backend,
            person_mode=self.person_mode,
            options=self.options,
            callback_url=self.callback_url,
        )


@final
class _JobDocument(BaseModel):
    """How a job's state is written to and read from the store."""

    model_config = ConfigDict(frozen=True)

    identifier: str
    state: JobState
    created_at: float
    updated_at: float
    error: str | None = None

    @classmethod
    def of(cls, job: Job) -> Self:
        """Project a job onto its stored form.

        :param job: The job to record.
        :type job: app.domain.Job
        :returns: The storable document.
        :rtype: _JobDocument
        """
        return cls(
            identifier=job.identifier,
            state=job.state,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error=job.error,
        )

    def job(self) -> Job:
        """Rebuild the recorded job.

        :returns: The job.
        :rtype: app.domain.Job
        """
        return Job(
            identifier=self.identifier,
            state=self.state,
            created_at=self.created_at,
            updated_at=self.updated_at,
            error=self.error,
        )


@final
class _ResultDocument(BaseModel):
    """How a finished job's answer is stored.

    ``body`` is the model, so what comes back out of Redis is validated
    against the schema before anything reads a field of it - which is
    the whole reason a document sits between the store and the wire.
    """

    model_config = ConfigDict(frozen=True)

    body: SegmentSchema | None = None
    error: str | None = None


@final
class RedisJobStore:
    """Stores jobs, payloads and results in Redis."""

    def __init__(self, client: Redis[bytes], policy: JobPolicy) -> None:
        """Bind a client to its retention policy.

        :param client: Redis client, speaking bytes.
        :type client: coredis.Redis[bytes]
        :param policy: Retention and queue bounds.
        :type policy: app.application.policies.JobPolicy
        """
        self._client: Redis[bytes] = client
        self._policy: JobPolicy = policy
        self._pool: AsyncExitStack = AsyncExitStack()
        self._opened: bool = False

    async def open(self) -> None:
        """Enter the client's connection pool.

        coredis binds a pool to a scope rather than connecting lazily,
        so the store has to hold that scope for as long as it serves.
        """
        await self._pool.enter_async_context(self._client)
        self._opened = True

    async def close(self) -> None:
        """Release the connection pool."""
        self._opened = False
        await self._pool.aclose()

    # A method for symmetry with the rest of the store, which the
    # instance-free form would break for no gain.
    # pylint: disable-next=no-self-use
    def _keys(self, identifier: str) -> tuple[str, str, str, str]:
        """Return the four keys one job occupies.

        :param identifier: Job identity.
        :type identifier: str
        :returns: ``(job, payload, image, result)`` key names.
        :rtype: tuple[str, str, str, str]
        """
        return (
            f"{JobKeys.job}{identifier}",
            f"{JobKeys.payload}{identifier}",
            f"{JobKeys.image}{identifier}",
            f"{JobKeys.result}{identifier}",
        )

    async def ready(self) -> bool:
        """Report whether the server answers.

        Swallows the failure rather than raising: a probe asking "can
        you serve?" wants an answer, and "no" is one of them.

        :returns: ``True`` when Redis replies to a ping.
        :rtype: bool
        """
        # Before the pool is entered there is nothing to ask, and
        # coredis says so with a bare ``RuntimeError``. Answering from
        # the flag keeps that out of the probe, which wants a verdict
        # rather than a traceback.
        if not self._opened:
            return False
        try:
            await self._client.ping()
        # Anything from a refused connection to a failed auth means the
        # same thing to a caller: this store cannot take work now.
        except (RedisError, OSError):
            return False
        return True

    async def enqueue(self, job: Job, payload: JobPayload) -> int | None:
        """Store a job and put it in line, unless the queue is full.

        One transaction, and the queue entry is the last thing in it:
        a worker can never pop an identifier whose payload is not yet
        readable, because the documents and the push either all applied
        or none did.

        Admission is decided by what ``RPUSH`` answers rather than by a
        depth read beforehand, which two callers racing at the ceiling
        would both pass. Whoever lands past it takes itself back out.

        :param job: The freshly queued job.
        :type job: app.domain.Job
        :param payload: Everything the worker will need.
        :type payload: app.application.jobs.JobPayload
        :returns: The caller's place in line, or ``None`` when the queue
            was already at its configured depth.
        :rtype: int | None
        """
        job_key, payload_key, image_key, _ = self._keys(job.identifier)
        ttl: int = self._policy.retention_seconds
        async with _reachable():
            async with self._client.pipeline(transaction=True) as batch:
                batch.set(
                    job_key, _JobDocument.of(job).model_dump_json(), ex=ttl
                )
                batch.set(
                    payload_key,
                    _PayloadDocument.of(payload).model_dump_json(),
                    ex=ttl,
                )
                batch.set(image_key, payload.image, ex=ttl)
                pushed = batch.rpush(JobKeys.queue, [job.identifier])
            length: int = await pushed
        if (position := length - 1) < self._policy.max_queue_depth:
            return position
        return await self._withdraw(job.identifier, position)

    async def _withdraw(self, identifier: str, position: int) -> int | None:
        """Take back a job the queue turned out to have no room for.

        :param identifier: The job that landed past the ceiling.
        :type identifier: str
        :param position: Where it landed.
        :type position: int
        :returns: ``None`` once withdrawn, or ``position`` when a worker
            claimed it first.
        :rtype: int | None
        """
        job_key, payload_key, image_key, _ = self._keys(identifier)
        async with _reachable():
            removed: int = await self._client.lrem(
                JobKeys.queue, 1, identifier
            )
            if not removed:
                # A worker claimed it between the push and this call, so
                # the work is already under way. Refusing the caller now
                # would be a lie about a job that is going to run.
                return position
            # Removed before anything could claim it, so the documents
            # are ours to drop.
            await self._client.delete([job_key, payload_key, image_key])
        return None

    async def claim(self) -> tuple[Job, JobPayload] | None:
        """Take the next job off the queue, waiting briefly for one.

        A cancelled job is skipped rather than run: cancellation only
        rewrites the state, so its identifier is still in the queue.

        :returns: The claimed job and its payload, or ``None`` when the
            wait elapsed or the job vanished.
        :rtype: tuple[app.domain.Job, app.application.jobs.JobPayload]
            | None
        """
        async with _reachable():
            popped: list[bytes] | None = await self._client.blpop(
                [JobKeys.queue], timeout=Claiming.timeout_seconds
            )
        if popped is None:
            return None
        identifier: str = popped[Claiming.popped_value].decode(
            TextEncoding.UTF8
        )
        job_key, payload_key, image_key, _ = self._keys(identifier)
        # One round trip for the three documents: the worker is the
        # latency that matters least, but the connection is shared with
        # every poll the transport is serving.
        async with _reachable():
            stored: tuple[bytes | None, ...] = await self._client.mget(
                [job_key, payload_key, image_key]
            )
        job_document, payload_document, image = stored
        if job_document is None or payload_document is None or image is None:
            return None
        job: Job = _JobDocument.model_validate_json(job_document).job()
        if job.state is not JobState.QUEUED:
            return None
        document: _PayloadDocument = _PayloadDocument.model_validate_json(
            payload_document
        )
        return job, document.payload(image)

    async def read(
        self, identifier: str
    ) -> tuple[Job, JobResult | None] | None:
        """Look a job up by identity.

        Both keys are fetched together: the result is absent until the
        job is terminal, and an absent key costs nothing to ask for.

        :param identifier: What the caller was handed at acceptance.
        :type identifier: str
        :returns: The job and its result when finished, or ``None`` when
            no such job exists or it has expired.
        :rtype: tuple[app.domain.Job, app.application.jobs.JobResult
            | None] | None
        """
        job_key, _, _, result_key = self._keys(identifier)
        async with _reachable():
            stored: tuple[bytes | None, ...] = await self._client.mget(
                [job_key, result_key]
            )
        document, stored_result = stored
        if document is None:
            return None
        job: Job = _JobDocument.model_validate_json(document).job()
        if not job.terminal or stored_result is None:
            return job, None
        result: _ResultDocument = _ResultDocument.model_validate_json(
            stored_result
        )
        return job, JobResult(body=result.body, error=result.error)

    async def write(self, job: Job, result: JobResult | None = None) -> None:
        """Record a job's new state, and its result when it has one.

        Queued as one transaction, so a caller that reads a terminal
        state can never find the answer missing: the result is written
        before the state that advertises it.

        :param job: The moved job.
        :type job: app.domain.Job
        :param result: What it produced, when it produced anything.
        :type result: app.application.jobs.JobResult | None
        """
        job_key, payload_key, image_key, result_key = self._keys(
            job.identifier
        )
        ttl: int = self._policy.retention_seconds
        async with (
            _reachable(),
            self._client.pipeline(transaction=True) as batch,
        ):
            if result is not None:
                batch.set(
                    result_key,
                    _ResultDocument(
                        body=result.body, error=result.error
                    ).model_dump_json(),
                    ex=ttl,
                )
            batch.set(job_key, _JobDocument.of(job).model_dump_json(), ex=ttl)
            if job.terminal:
                # The image is the bulk of a job and is of no use once
                # the work is done, whatever the outcome.
                batch.delete([payload_key, image_key])

    async def depth(self) -> int:
        """Report how many jobs are waiting.

        Advisory: :meth:`enqueue` is what actually enforces the ceiling.
        This exists so a caller can be turned away before it pays for an
        upload, and so a poll can be told its place in line.

        :returns: Length of the queue, ignoring work already claimed.
        :rtype: int
        """
        async with _reachable():
            waiting: int = await self._client.llen(JobKeys.queue)
        return waiting


def connect(policy: JobPolicy) -> Redis[bytes]:
    """Open the Redis connection the store will use.

    Built, not connected: the pool opens when the store is opened, so
    an unreachable Redis surfaces on the readiness probe rather than
    preventing the process from starting and answering one.

    :param policy: Where Redis lives and how long it keeps things.
    :type policy: app.application.policies.JobPolicy
    :returns: A client speaking bytes, because one of the four values a
        job occupies is an image.
    :rtype: coredis.Redis[bytes]
    """
    return Redis.from_url(policy.url)


__all__: list[str] = ["RedisJobStore", "connect"]
