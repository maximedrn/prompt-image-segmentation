"""Process-local job store.

Implements :class:`~app.application.capabilities.JobStore` without a
server behind it, for the deployments that have none: a checkout being
developed against, the test suite, and the single-container Space.

Its limits are the reason it is selected explicitly rather than fallen
back to. Nothing here outlives the process, and nothing here is visible
to a second one: two uvicorn workers would each accept jobs the other
answers ``404`` for. ``JOB_BACKEND=redis`` is the answer to both, and
the operator is the one who picks.

Retention is honoured the way Redis honours it - an entry is dropped
once its window elapses - so a caller sees the same expiry behaviour
whichever backend is serving.
"""

from asyncio import Event, TimeoutError as AsyncTimeoutError, wait_for
from collections import deque
from dataclasses import dataclass
from time import monotonic, time
from typing import final

from app.application.jobs import IdempotentRequest, JobPayload, JobResult
from app.application.policies import JobPolicy
from app.domain import Admission, Job, JobState
from app.infrastructure.jobs.constants import Claiming


@final
@dataclass(slots=True)
class _Entry:
    """One job, everything it carries, and when it stops existing."""

    job: Job
    payload: JobPayload | None
    result: JobResult | None
    expires_at: float


def _claim_key(idempotency: IdempotentRequest) -> tuple[str, str]:
    """Name a claim by who made it as well as by what they called it.

    The key is a string the caller picks, so two callers picking the
    same one must not share an entry -- the second would be handed the
    first's job.

    :param idempotency: The caller's claim.
    :type idempotency: app.application.jobs.IdempotentRequest
    :returns: The scoped claim key.
    :rtype: tuple[str, str]
    """
    return (idempotency.scope, idempotency.key)


@final
@dataclass(slots=True)
class _Claim:
    """One spent idempotency key, and what it was spent on.

    Expires with the job it names, like every other entry here: a claim
    outliving its job would answer a replay with an identifier that has
    already been purged.
    """

    request_hash: str
    identifier: str
    expires_at: float


@final
class InMemoryJobStore:
    """Keeps jobs in this process, and only for as long as it runs."""

    def __init__(self, policy: JobPolicy) -> None:
        """Start empty, bound to the same retention Redis would apply.

        :param policy: Retention and queue bounds.
        :type policy: app.application.policies.JobPolicy
        """
        self._policy: JobPolicy = policy
        self._entries: dict[str, _Entry] = {}
        self._claims: dict[tuple[str, str], _Claim] = {}
        self._queue: deque[str] = deque()
        # Lets ``claim`` sleep until there is work rather than poll for
        # it, which is what keeps an idle worker off the CPU.
        self._arrived: Event = Event()

    def _purge(self) -> None:
        now: float = time()
        expired: set[str] = {
            identifier
            for identifier, entry in self._entries.items()
            if entry.expires_at <= now
        }
        if not expired:
            return
        for identifier in expired:
            del self._entries[identifier]
        self._queue = deque(
            identifier
            for identifier in self._queue
            if identifier not in expired
        )
        # Claims expire on their own clock rather than with the entries
        # above: a claim outlives nothing, but a job cancelled early
        # would otherwise take its claim with it and let the same key
        # queue a second job.
        self._claims = {
            key: claim
            for key, claim in self._claims.items()
            if claim.expires_at > now
        }

    def _expiry(self) -> float:
        return time() + self._policy.retention_seconds

    async def open(self) -> None:
        """Acquire nothing: the store is already here."""

    async def close(self) -> None:
        """Release nothing, for the same reason."""

    # A method because it implements the JobStore capability, which the
    # transport holds as an instance.
    # pylint: disable-next=no-self-use
    async def ready(self) -> bool:
        """Report that the store can serve.

        Always: there is nothing to reach. The store is this object, and
        if the process is answering at all then so is it.

        :returns: ``True``.
        :rtype: bool
        """
        return True

    async def enqueue(
        self,
        job: Job,
        payload: JobPayload,
        idempotency: IdempotentRequest | None = None,
    ) -> Admission:
        """Store a job and put it in line, unless something says not to.

        Admission and storage cannot interleave here: there is no await
        between the depth check and the append, so the event loop cannot
        hand another request the same place in line. The idempotency
        claim is decided in the same uninterrupted stretch, and for the
        same reason.

        :param job: The freshly queued job.
        :type job: app.domain.Job
        :param payload: Everything the worker will need.
        :type payload: app.application.jobs.JobPayload
        :param idempotency: The caller's key and request hash, when one
            was supplied.
        :type idempotency: app.application.jobs.IdempotentRequest | None
        :returns: What became of the submission.
        :rtype: app.domain.Admission
        """
        self._purge()
        if idempotency is not None:
            if (
                spent := self._claims.get(_claim_key(idempotency))
            ) is not None:
                return (
                    Admission.replay(spent.identifier)
                    if spent.request_hash == idempotency.request_hash
                    else Admission.conflict()
                )
        if (position := len(self._queue)) >= self._policy.max_queue_depth:
            return Admission.full()
        self._entries[job.identifier] = _Entry(
            job=job,
            payload=payload,
            result=None,
            expires_at=self._expiry(),
        )
        self._queue.append(job.identifier)
        if idempotency is not None:
            self._claims[_claim_key(idempotency)] = _Claim(
                request_hash=idempotency.request_hash,
                identifier=job.identifier,
                expires_at=self._expiry(),
            )
        self._arrived.set()
        return Admission.accepted(position)

    def _take(self) -> tuple[Job, JobPayload] | None:
        # A cancelled job is skipped rather than run: cancellation only
        # rewrites the state, so its identifier is still in the queue.
        while self._queue:  # pylint: disable=while-used
            entry: _Entry | None = self._entries.get(self._queue.popleft())
            if entry is None or entry.payload is None:
                continue
            if entry.job.state is JobState.QUEUED:
                return entry.job, entry.payload
        return None

    async def claim(self) -> tuple[Job, JobPayload] | None:
        """Take the next job off the queue, waiting briefly for one.

        :returns: The claimed job and its payload, or ``None`` when the
            wait elapsed with nothing to do.
        :rtype: tuple[app.domain.Job, app.application.jobs.JobPayload]
            | None
        """
        deadline: float = monotonic() + Claiming.timeout_seconds
        while True:  # pylint: disable=while-used
            self._purge()
            if (claimed := self._take()) is not None:
                return claimed
            self._arrived.clear()
            if (remaining := deadline - monotonic()) <= 0:
                return None
            try:
                await wait_for(self._arrived.wait(), remaining)
            # Nothing arrived in the window, which is how a shutdown
            # gets its chance to stop the loop.
            except AsyncTimeoutError:
                return None

    async def read(
        self, identifier: str
    ) -> tuple[Job, JobResult | None] | None:
        """Look a job up by identity.

        :param identifier: What the caller was handed at acceptance.
        :type identifier: str
        :returns: The job and its result when finished, or ``None`` when
            no such job exists or it has expired.
        :rtype: tuple[app.domain.Job, app.application.jobs.JobResult
            | None] | None
        """
        self._purge()
        if (entry := self._entries.get(identifier)) is None:
            return None
        if not entry.job.terminal:
            return entry.job, None
        return entry.job, entry.result

    async def write(self, job: Job, result: JobResult | None = None) -> None:
        """Record a job's new state, and its result when it has one.

        :param job: The moved job.
        :type job: app.domain.Job
        :param result: What it produced, when it produced anything.
        :type result: app.application.jobs.JobResult | None
        """
        self._purge()
        existing: _Entry | None = self._entries.get(job.identifier)
        payload: JobPayload | None = (
            None if existing is None else existing.payload
        )
        if job.terminal:
            # The image is the bulk of a job and is of no use once the
            # work is done, whatever the outcome.
            payload = None
        # A write without a result leaves the stored one alone, exactly
        # as the Redis backend does by not touching its result key.
        kept: JobResult | None = None if existing is None else existing.result
        self._entries[job.identifier] = _Entry(
            job=job,
            payload=payload,
            result=kept if result is None else result,
            expires_at=self._expiry(),
        )

    async def depth(self) -> int:
        """Report how many jobs are waiting.

        :returns: Length of the queue, ignoring work already claimed.
        :rtype: int
        """
        self._purge()
        return len(self._queue)


__all__: list[str] = ["InMemoryJobStore"]
