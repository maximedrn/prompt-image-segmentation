"""Run one queued job, whatever its outcome.

The consumer side of the queue. It owns the state machine's moves and
nothing else: what to segment came from the payload, how to segment is
the wired application's business, and where to write is the store's.

Inference is synchronous and holds the device lock for its whole
duration, so it runs in a worker thread. The loop itself stays on the
event loop, which is what lets a shutdown interrupt it between jobs.
"""

from collections.abc import Callable
from logging import Logger, getLogger
from time import time
from typing import Final

from anyio import to_thread

from app.application.capabilities import JobNotifier, JobStore
from app.application.jobs import (
    JobFailure,
    JobPayload,
    JobResult,
    WorkerLog,
)
from app.domain import Job, fail, start, succeed

_logger: Final[Logger] = getLogger(__name__)

type Segmenter = Callable[[JobPayload], JobResult]
"""Turns one payload into the answer it deserves, failures included."""


async def _announce(
    notifier: JobNotifier | None,
    job: Job,
    result: JobResult | None,
    callback_url: str | None,
) -> None:
    """Push a terminal state, if the caller asked to be pushed to.

    A delivery that fails must not fail the job it announces: the
    answer is already stored, and polling still reaches it.

    :param notifier: Who delivers, when deliveries are configured.
    :type notifier: app.application.capabilities.JobNotifier | None
    :param job: The finished job.
    :type job: app.domain.Job
    :param result: What it produced, if anything.
    :type result: app.application.jobs.JobResult | None
    :param callback_url: Where the caller asked to be told.
    :type callback_url: str | None
    """
    if notifier is None or callback_url is None:
        return
    try:
        await notifier.notify(job, result, callback_url)
    except Exception:  # pylint: disable=broad-except
        _logger.exception(WorkerLog.webhook_failed, job.identifier)


async def process(
    store: JobStore,
    segmenter: Segmenter,
    job: Job,
    payload: JobPayload,
    notifier: JobNotifier | None = None,
) -> None:
    """Run one claimed job to a terminal state.

    Every outcome is a terminal write: a job left in ``RUNNING`` is a
    job a caller polls forever.

    :param store: Where the job's state lives.
    :type store: app.application.capabilities.JobStore
    :param segmenter: Runs the pipeline for one payload.
    :type segmenter: Segmenter
    :param job: The claimed job, still queued.
    :type job: app.domain.Job
    :param payload: What the caller submitted.
    :type payload: app.application.jobs.JobPayload
    :param notifier: Who pushes the outcome, when one is configured.
    :type notifier: app.application.capabilities.JobNotifier | None
    """
    running: Job = start(job, time())
    await store.write(running)
    try:
        result: JobResult = await to_thread.run_sync(segmenter, payload)
    # A defect still has to leave a terminal state behind, or the caller
    # polls a job that will never move again.
    except Exception:
        _logger.exception(WorkerLog.job_raised, job.identifier)
        defect: Job = fail(running, JobFailure.defect, time())
        broken: JobResult = JobResult(error=JobFailure.defect)
        await store.write(defect, broken)
        await _announce(notifier, defect, broken, payload.callback_url)
        raise
    finished: Job = (
        fail(running, result.error, time())
        if result.error is not None
        else succeed(running, time())
    )
    await store.write(finished, result)
    await _announce(notifier, finished, result, payload.callback_url)


async def consume(
    store: JobStore,
    segmenter: Segmenter,
    running: Callable[[], bool],
    notifier: JobNotifier | None = None,
) -> None:
    """Claim and run jobs until asked to stop.

    A failure to run one job must not end the loop: the next caller is
    entitled to its turn, so the fault is logged and the job is already
    terminal by then.

    :param store: Queue to consume.
    :type store: app.application.capabilities.JobStore
    :param segmenter: Runs the pipeline for one payload.
    :type segmenter: Segmenter
    :param running: Answers whether the worker should keep going.
    :type running: collections.abc.Callable[[], bool]
    :param notifier: Who pushes outcomes, when one is configured.
    :type notifier: app.application.capabilities.JobNotifier | None
    """
    # Deliberate: a plain loop, because the queue has no end condition
    # of its own; the flag is what a shutdown flips.
    while running():  # pylint: disable=while-used
        if (claimed := await store.claim()) is None:
            continue
        job, payload = claimed
        try:
            await process(store, segmenter, job, payload, notifier)
        # The worker outlives any single job; the alternative is a queue
        # that stops moving because one payload was pathological.
        except Exception:  # pylint: disable=broad-except
            _logger.exception(WorkerLog.job_defect, job.identifier)


__all__: list[str] = ["Segmenter", "consume", "process"]
