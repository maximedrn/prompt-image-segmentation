"""Job lifecycle: the state machine, and the worker that drives it.

Both run against stand-ins. The state machine has no dependencies at
all, and the worker only needs something shaped like a store - which is
the point of declaring the store as a capability.

The Redis adapter itself is exercised in ``test_store.py``, which needs
a server.
"""

# pytest collects the test functions by name; nothing in the module
# calls them, which is what ``allow-global-unused-variables`` flags.
# pylint: disable=unused-variable
# The stand-in store mirrors a protocol, so its methods stay
# methods whether or not they touch self.
# pylint: disable=no-self-use

from asyncio import run
from typing import Final, final

import pytest

from app.application.jobs import JobFailure, JobPayload, JobResult
from app.application.jobs.results import SegmentSchema
from app.application.policies import SegmentOptions
from app.application.use_cases.process_job import Segmenter, process
from app.domain import (
    IllegalTransition,
    Job,
    JobState,
    cancel,
    fail,
    queued,
    start,
    succeed,
)
from tests.conftest import segment_body

ACCEPTED_AT: Final[float] = 1000.0
STARTED_AT: Final[float] = 1001.0
FINISHED_AT: Final[float] = 1002.0
IDENTIFIER: Final[str] = "job-1"
#: Failure codes the transport would produce, mirrored here so the
#: assertions name them rather than repeating the strings.
INVALID_IMAGE: Final[str] = "invalid_image"
NO_DETECTION: Final[str] = "no_detection"


@final
class RecordingStore:
    """Remembers every write, so a test can read the whole history."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.writes: list[tuple[Job, JobResult | None]] = []

    async def enqueue(self, job: Job, payload: JobPayload) -> None:
        """Record an accepted job.

        :param job: The queued job.
        :type job: app.domain.Job
        :param payload: Ignored.
        :type payload: app.application.jobs.JobPayload
        """
        del payload
        self.writes.append((job, None))

    async def claim(self) -> tuple[Job, JobPayload] | None:
        """Never hand anything out; the tests drive ``process`` directly."""
        return None

    async def read(
        self, identifier: str
    ) -> tuple[Job, JobResult | None] | None:
        """Return the last write, if it is the one asked for.

        :param identifier: Job identity.
        :type identifier: str
        :returns: The last recorded state, or ``None``.
        :rtype: tuple[app.domain.Job, app.application.jobs.JobResult
            | None] | None
        """
        for job, result in reversed(self.writes):
            if job.identifier == identifier:
                return job, result
        return None

    async def write(self, job: Job, result: JobResult | None = None) -> None:
        """Record a transition.

        :param job: The moved job.
        :type job: app.domain.Job
        :param result: What it produced, if anything.
        :type result: app.application.jobs.JobResult | None
        """
        self.writes.append((job, result))

    async def depth(self) -> int:
        """Report an empty queue.

        :returns: Always zero.
        :rtype: int
        """
        return 0


def _payload() -> JobPayload:
    """Build a payload the fakes never actually look at.

    :returns: A minimal payload.
    :rtype: app.application.jobs.JobPayload
    """
    return JobPayload(
        image=b"not really an image",
        prompt="dog",
        backend="sam_dino",
        person_mode=False,
        options=SegmentOptions(),
    )


def _run_process(store: RecordingStore, segmenter: Segmenter) -> None:
    """Drive ``process`` once, synchronously.

    :param store: Fake store to record into.
    :type store: RecordingStore
    :param segmenter: What the worker will call.
    :type segmenter: app.application.use_cases.process_job.Segmenter
    """
    run(process(store, segmenter, queued(IDENTIFIER, ACCEPTED_AT), _payload()))


def test_a_job_walks_from_queued_to_succeeded() -> None:
    """The happy path moves through running and stops."""
    job: Job = queued(IDENTIFIER, ACCEPTED_AT)
    assert job.state is JobState.QUEUED
    assert job.terminal is False

    running: Job = start(job, STARTED_AT)
    assert running.state is JobState.RUNNING
    assert running.created_at == ACCEPTED_AT
    assert running.updated_at == STARTED_AT

    done: Job = succeed(running, FINISHED_AT)
    assert done.state is JobState.SUCCEEDED
    assert done.terminal is True


def test_a_job_cannot_be_claimed_twice() -> None:
    """Two workers must not run the same payload."""
    running: Job = start(queued(IDENTIFIER, ACCEPTED_AT), STARTED_AT)
    with pytest.raises(IllegalTransition) as raised:
        start(running, FINISHED_AT)
    assert raised.value.state == JobState.RUNNING
    assert raised.value.attempted == JobState.RUNNING


def test_only_a_queued_job_can_be_cancelled() -> None:
    """Cancelling running work would leave the device busy for nobody."""
    job: Job = queued(IDENTIFIER, ACCEPTED_AT)
    assert cancel(job, STARTED_AT).state is JobState.CANCELLED

    running: Job = start(job, STARTED_AT)
    with pytest.raises(IllegalTransition):
        cancel(running, FINISHED_AT)


def test_failure_is_reachable_before_and_during_a_run() -> None:
    """A payload can be refused before any worker touches it."""
    job: Job = queued(IDENTIFIER, ACCEPTED_AT)
    refused: Job = fail(job, INVALID_IMAGE, STARTED_AT)
    assert refused.state is JobState.FAILED
    assert refused.error == INVALID_IMAGE

    with pytest.raises(IllegalTransition):
        fail(refused, "internal", FINISHED_AT)


def test_success_is_recorded_with_its_body() -> None:
    """A produced body lands with the terminal state, not before."""
    store: RecordingStore = RecordingStore()
    body: SegmentSchema = segment_body()

    def segmenter(payload: JobPayload) -> JobResult:
        """Answer with a fixed body.

        :param payload: Ignored.
        :type payload: app.application.jobs.JobPayload
        :returns: The fixed body.
        :rtype: app.application.jobs.JobResult
        """
        del payload
        return JobResult(body=body)

    _run_process(store, segmenter)

    states: list[JobState] = [job.state for job, _ in store.writes]
    assert states == [JobState.RUNNING, JobState.SUCCEEDED]
    assert store.writes[0][1] is None
    assert store.writes[-1][1] is not None
    assert store.writes[-1][1].body == body


def test_a_named_failure_fails_the_job_rather_than_succeeding_it() -> None:
    """No detection is an outcome, and it is not a success."""
    store: RecordingStore = RecordingStore()

    def segmenter(payload: JobPayload) -> JobResult:
        """Answer with a named failure.

        :param payload: Ignored.
        :type payload: app.application.jobs.JobPayload
        :returns: The failure.
        :rtype: app.application.jobs.JobResult
        """
        del payload
        return JobResult(error=NO_DETECTION)

    _run_process(store, segmenter)

    final_job: Job = store.writes[-1][0]
    assert final_job.state is JobState.FAILED
    assert final_job.error == NO_DETECTION


def test_a_defect_still_leaves_a_terminal_state() -> None:
    """A raising worker must not leave a job polled forever."""
    store: RecordingStore = RecordingStore()

    def segmenter(payload: JobPayload) -> JobResult:
        """Fail the way a defect does.

        :param payload: Ignored.
        :type payload: app.application.jobs.JobPayload
        :raises RuntimeError: Always.
        """
        del payload
        raise RuntimeError("the accelerator caught fire")

    with pytest.raises(RuntimeError):
        _run_process(store, segmenter)

    final_job: Job = store.writes[-1][0]
    assert final_job.state is JobState.FAILED
    assert final_job.error == JobFailure.defect
