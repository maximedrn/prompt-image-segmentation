"""Capabilities the queue needs: somewhere to keep work, someone to tell."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from app.application.jobs import IdempotentRequest, JobPayload, JobResult
from app.domain import Admission, Job


@runtime_checkable
class JobStore(Protocol):
    """Keeps queued work, and hands it to whoever asks for it.

    Asynchronous throughout: the transport accepts jobs on the event
    loop while the worker waits on the queue, and neither should block
    the other. Inference itself is threaded elsewhere.
    """

    @abstractmethod
    async def open(self) -> None:
        """Acquire whatever the store needs for the process lifetime.

        A backing service is a scope, not a global: the client that
        talks to one has connections to open and to give back. Called
        once, by the composition root, before anything is served.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release what :meth:`open` acquired."""

    @abstractmethod
    async def ready(self) -> bool:
        """Report whether the store can be reached right now.

        Readiness asks this on every probe rather than once at startup,
        so an outage that ends is reflected without a restart.

        :returns: ``True`` when the store answers.
        :rtype: bool
        """

    @abstractmethod
    async def enqueue(
        self,
        job: Job,
        payload: JobPayload,
        idempotency: IdempotentRequest | None = None,
    ) -> Admission:
        """Store a job and put it in line, unless something says not to.

        Admission belongs here rather than in the caller: a depth read
        followed by a push lets two callers race past the same ceiling,
        so the store is the only place that can refuse one of them. The
        idempotency claim is the same shape of problem and belongs in
        the same place -- two identical submissions arriving together
        would both pass a check made before this call.

        :param job: The freshly queued job.
        :type job: app.domain.Job
        :param payload: Everything the worker will need.
        :type payload: app.application.jobs.JobPayload
        :param idempotency: The caller's key and request hash, when one
            was supplied.
        :type idempotency: app.application.jobs.IdempotentRequest | None
        :returns: What became of the submission: queued, full, a replay
            of an earlier one, or a conflict.
        :rtype: app.domain.Admission
        """

    @abstractmethod
    async def claim(self) -> tuple[Job, JobPayload] | None:
        """Take the next job off the queue, waiting briefly for one.

        :returns: The claimed job and its payload, or ``None`` when the
            wait elapsed with nothing to do.
        :rtype: tuple[app.domain.Job, app.application.jobs.JobPayload]
            | None
        """

    @abstractmethod
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

    @abstractmethod
    async def write(self, job: Job, result: JobResult | None = None) -> None:
        """Record a job's new state, and its result when it has one.

        :param job: The moved job.
        :type job: app.domain.Job
        :param result: What it produced, when it produced anything.
        :type result: app.application.jobs.JobResult | None
        """

    @abstractmethod
    async def depth(self) -> int:
        """Report how many jobs are waiting.

        Advisory only: :meth:`enqueue` is what enforces the ceiling.

        :returns: Length of the queue, ignoring work already claimed.
        :rtype: int
        """


@runtime_checkable
class JobNotifier(Protocol):
    """Tells a caller its job is done, without being asked."""

    @abstractmethod
    async def notify(
        self, job: Job, result: JobResult | None, callback_url: str
    ) -> None:
        """Deliver one terminal state to the caller's address.

        Best effort: polling and the socket remain, so a delivery that
        never lands must not fail the job that produced it.

        :param job: The finished job.
        :type job: app.domain.Job
        :param result: What it produced, if anything.
        :type result: app.application.jobs.JobResult | None
        :param callback_url: Where the caller asked to be told.
        :type callback_url: str
        """


__all__: list[str] = ["JobNotifier", "JobStore"]
