"""Transitions of the job lifecycle."""

from dataclasses import replace

from app.domain.errors import IllegalTransition
from app.domain.jobs.models import Job
from app.domain.jobs.types import JobState


def queued(identifier: str, at: float) -> Job:
    """Open a job in its initial state.

    :param identifier: Identity the caller will poll with.
    :type identifier: str
    :param at: Wall-clock seconds when it was accepted.
    :type at: float
    :returns: A job waiting for a worker.
    :rtype: Job
    """
    return Job(
        identifier=identifier,
        state=JobState.QUEUED,
        created_at=at,
        updated_at=at,
    )


def _moved(job: Job, state: JobState, at: float, error: str | None) -> Job:
    """Return the same job in a new state.

    :param job: Job being moved.
    :type job: Job
    :param state: State to move to.
    :type state: JobState
    :param at: Wall-clock seconds of the move.
    :type at: float
    :param error: Failure reason, when moving to ``FAILED``.
    :type error: str | None
    :returns: The moved job.
    :rtype: Job
    """
    return replace(job, state=state, updated_at=at, error=error)


def start(job: Job, at: float) -> Job:
    """Claim a queued job for a worker.

    :param job: Job to start.
    :type job: Job
    :param at: Wall-clock seconds of the move.
    :type at: float
    :returns: The job, now running.
    :rtype: Job
    :raises app.domain.errors.IllegalTransition: Unless the job is
        queued. A running job claimed twice would be run twice.
    """
    if job.state is not JobState.QUEUED:
        raise IllegalTransition(
            identifier=job.identifier,
            state=job.state,
            attempted=JobState.RUNNING,
        )
    return _moved(job, JobState.RUNNING, at, None)


def succeed(job: Job, at: float) -> Job:
    """Record that a running job produced its result.

    :param job: Job to complete.
    :type job: Job
    :param at: Wall-clock seconds of the move.
    :type at: float
    :returns: The job, now succeeded.
    :rtype: Job
    :raises app.domain.errors.IllegalTransition: Unless the job is
        running.
    """
    if job.state is not JobState.RUNNING:
        raise IllegalTransition(
            identifier=job.identifier,
            state=job.state,
            attempted=JobState.SUCCEEDED,
        )
    return _moved(job, JobState.SUCCEEDED, at, None)


def fail(job: Job, reason: str, at: float) -> Job:
    """Record that a job cannot produce a result.

    Reachable from ``QUEUED`` too: a payload can be rejected before any
    worker touches it.

    :param job: Job to fail.
    :type job: Job
    :param reason: Machine-readable failure code.
    :type reason: str
    :param at: Wall-clock seconds of the move.
    :type at: float
    :returns: The job, now failed.
    :rtype: Job
    :raises app.domain.errors.IllegalTransition: If the job already
        reached a terminal state.
    """
    if job.terminal:
        raise IllegalTransition(
            identifier=job.identifier,
            state=job.state,
            attempted=JobState.FAILED,
        )
    return _moved(job, JobState.FAILED, at, reason)


def cancel(job: Job, at: float) -> Job:
    """Withdraw a job that has not started.

    :param job: Job to cancel.
    :type job: Job
    :param at: Wall-clock seconds of the move.
    :type at: float
    :returns: The job, now cancelled.
    :rtype: Job
    :raises app.domain.errors.IllegalTransition: Unless the job is still
        queued. Cancelling a running job would leave the accelerator
        working for an answer nobody will read.
    """
    if job.state is not JobState.QUEUED:
        raise IllegalTransition(
            identifier=job.identifier,
            state=job.state,
            attempted=JobState.CANCELLED,
        )
    return _moved(job, JobState.CANCELLED, at, None)


__all__: list[str] = ["cancel", "fail", "queued", "start", "succeed"]
