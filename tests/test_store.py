"""Both job stores, driven through the one capability they implement.

Every scenario runs twice: once in process memory, once against a real
Redis. The two are interchangeable by contract, and the only way to
keep them that way is to hold them to the same tests.

The Redis half is skipped when no server answers, so a checkout without
one still runs a green suite; CI provides the service. Database 15 by
convention, and flushed between tests: these write real keys, and they
should not meet anybody's data.

Everything else about jobs is covered in ``test_jobs.py``, with no
store at all.
"""

# pytest collects the test functions by name; nothing in the module
# calls them, which is what ``allow-global-unused-variables`` flags.
# pylint: disable=unused-variable

from asyncio import run
from dataclasses import replace
from collections.abc import Awaitable, Callable
from os import environ
from typing import Final

import pytest
from coredis import Redis
from coredis.exceptions import RedisError

from app.application.capabilities import JobStore
from app.application.jobs import IdempotentRequest, JobPayload, JobResult
from app.application.jobs.results import SegmentSchema
from app.application.policies import JobBackend, JobPolicy, SegmentOptions
from app.domain import Admission, Job, JobState, cancel, queued, start, succeed
from app.infrastructure.jobs import build_store
from app.infrastructure.jobs.hashing import request_hash
from app.infrastructure.jobs.store import RedisJobStore, connect
from tests.conftest import segment_body

URL: Final[str] = environ.get("REDIS_URL", "redis://localhost:6379/15")
RETENTION: Final[int] = 60
#: One accepted job, before anything claims it.
ONE_JOB: Final[int] = 1
#: Two, when a submission without a key is deliberately not deduplicated.
TWO_JOBS: Final[int] = 2
#: The front of the queue.
FIRST_IN_LINE: Final[int] = 0
FIRST_JOB: Final[str] = "first-job"
SECOND_JOB: Final[str] = "second-job"
DEPTH: Final[int] = 10
ACCEPTED_AT: Final[float] = 1000.0
STARTED_AT: Final[float] = 1001.0
FINISHED_AT: Final[float] = 1002.0
#: Memory first: it needs nothing, so a failure there is the store's
#: own and not the server's.
BACKENDS: Final[tuple[JobBackend, ...]] = (
    JobBackend.MEMORY,
    JobBackend.REDIS,
)
#: Applied to every scenario, so each one states both backends agree.
BOTH_BACKENDS = pytest.mark.parametrize(
    "backend", BACKENDS, ids=[backend.value for backend in BACKENDS]
)


def _payload() -> JobPayload:
    """Build a payload worth round-tripping.

    Every field differs from its default, so a serialisation that drops
    one is visible rather than accidentally correct.

    :returns: The payload.
    :rtype: app.application.jobs.JobPayload
    """
    return JobPayload(
        image=b"\x89PNG\r\n\x1a\n binary and not text",
        prompt="dog. cat",
        backend="sam_dino",
        person_mode=True,
        options=SegmentOptions(
            minimum_confidence=0.25,
            dilation_percentage=7.0,
            padding_percentage=3.0,
            split_masks=True,
            crop=False,
        ),
        callback_url="https://example.com/hook",
    )


def _scenario(
    body: Callable[[JobStore], Awaitable[None]], backend: JobBackend
) -> None:
    """Run one scenario against an empty store of the given backend.

    Everything happens in a single event loop: the Redis client binds
    its connection pool to the loop that first used it, so a test made
    of several ``run`` calls would talk to a closed one.

    :param body: What to do with the store.
    :type body: collections.abc.Callable[
        [app.application.capabilities.JobStore],
        collections.abc.Awaitable[None]]
    :param backend: Which implementation to hold to the scenario.
    :type backend: app.application.policies.JobBackend
    """
    policy: JobPolicy = JobPolicy(
        url=URL,
        retention_seconds=RETENTION,
        max_queue_depth=DEPTH,
        backend=backend,
    )

    if backend is JobBackend.MEMORY:
        # Nothing to reach and nothing to clean up: the store is born
        # empty and dies with the call.
        run(body(build_store(policy)))
        return

    async def main() -> None:
        """Connect, run the body, and leave nothing behind."""
        client: Redis[bytes] = connect(policy)
        # coredis binds its pool to a scope rather than connecting
        # lazily, so everything below happens inside it.
        async with client:
            try:
                await client.ping()
            # Both mean the same thing here: no server to test against,
            # which is a skip and not a failure.
            except (RedisError, OSError) as error:
                pytest.skip(f"no Redis at {URL}: {type(error).__name__}")
            await client.flushdb()
            try:
                await body(RedisJobStore(client, policy))
            finally:
                await client.flushdb()

    run(main())


@BOTH_BACKENDS
def test_a_payload_survives_the_round_trip(backend: JobBackend) -> None:
    """What the worker claims is what the caller submitted."""

    async def scenario(store: JobStore) -> None:
        """Drive the store.

        :param store: The store under test.
        :type store: app.application.capabilities.JobStore
        """
        sent: JobPayload = _payload()
        await store.enqueue(queued("round-trip", ACCEPTED_AT), sent)
        assert await store.depth() == ONE_JOB

        claimed: tuple[Job, JobPayload] | None = await store.claim()
        assert claimed is not None
        received: JobPayload = claimed[1]
        assert received == sent
        # Claiming is what removes it from the queue.
        assert not await store.depth()

    _scenario(scenario, backend)


@BOTH_BACKENDS
def test_the_result_appears_only_once_the_job_is_terminal(
    backend: JobBackend,
) -> None:
    """A poll mid-run reports progress, never a half-written answer."""

    async def scenario(store: JobStore) -> None:
        """Drive the store.

        :param store: The store under test.
        :type store: app.application.capabilities.JobStore
        """
        job: Job = queued("staged", ACCEPTED_AT)
        await store.enqueue(job, _payload())

        running: Job = start(job, STARTED_AT)
        await store.write(running)
        midway: tuple[Job, JobResult | None] | None = await store.read(
            "staged"
        )
        assert midway is not None
        assert midway[0].state is JobState.RUNNING
        assert midway[1] is None

        body: SegmentSchema = segment_body()
        await store.write(succeed(running, FINISHED_AT), JobResult(body=body))
        finished: tuple[Job, JobResult | None] | None = await store.read(
            "staged"
        )
        assert finished is not None
        assert finished[0].state is JobState.SUCCEEDED
        assert finished[1] is not None
        assert finished[1].body == body

    _scenario(scenario, backend)


@BOTH_BACKENDS
def test_an_unknown_job_is_absent_rather_than_empty(
    backend: JobBackend,
) -> None:
    """A caller asking for a job that expired gets nothing back."""

    async def scenario(store: JobStore) -> None:
        """Drive the store.

        :param store: The store under test.
        :type store: app.application.capabilities.JobStore
        """
        assert await store.read("never-existed") is None

    _scenario(scenario, backend)


@BOTH_BACKENDS
def test_a_cancelled_job_is_never_handed_to_a_worker(
    backend: JobBackend,
) -> None:
    """Cancelling rewrites the state; the queue entry outlives it."""

    async def scenario(store: JobStore) -> None:
        """Drive the store.

        :param store: The store under test.
        :type store: app.application.capabilities.JobStore
        """
        job: Job = queued("withdrawn", ACCEPTED_AT)
        await store.enqueue(job, _payload())
        await store.write(cancel(job, STARTED_AT))

        assert await store.claim() is None

    _scenario(scenario, backend)


@BOTH_BACKENDS
def test_a_finished_job_stops_holding_its_image(backend: JobBackend) -> None:
    """The bulk of a job is dropped as soon as it is of no use."""

    async def scenario(store: JobStore) -> None:
        """Drive the store.

        :param store: The store under test.
        :type store: app.application.capabilities.JobStore
        """
        job: Job = queued("bulky", ACCEPTED_AT)
        await store.enqueue(job, _payload())
        running: Job = start(job, STARTED_AT)
        await store.write(running)
        await store.write(
            succeed(running, FINISHED_AT), JobResult(body=segment_body())
        )

        # The job itself is still readable; only the payload is gone.
        assert await store.read("bulky") is not None
        assert await store.claim() is None

    _scenario(scenario, backend)


@BOTH_BACKENDS
def test_the_queue_ceiling_is_enforced_where_it_is_atomic(
    backend: JobBackend,
) -> None:
    """Admission refuses past the depth, and says where a caller lands.

    Reading the depth and pushing afterwards would let two callers
    racing at the ceiling both through, so the store decides. The
    position it answers is what the caller is told to expect.
    """

    async def scenario(store: JobStore) -> None:
        """Drive the store.

        :param store: The store under test.
        :type store: app.application.capabilities.JobStore
        """
        accepted: list[Admission] = [
            await store.enqueue(
                queued(f"job-{index}", ACCEPTED_AT), _payload()
            )
            for index in range(DEPTH + 2)
        ]

        # One place in line per accepted job, counted from the front.
        assert [one.position for one in accepted[:DEPTH]] == list(range(DEPTH))
        # Everything past the ceiling is refused rather than queued.
        assert [one.queued for one in accepted[DEPTH:]] == [False, False]
        assert await store.depth() == DEPTH

    _scenario(scenario, backend)


@BOTH_BACKENDS
def test_a_refused_job_leaves_nothing_behind(backend: JobBackend) -> None:
    """A job the queue had no room for is not readable either."""

    async def scenario(store: JobStore) -> None:
        """Drive the store.

        :param store: The store under test.
        :type store: app.application.capabilities.JobStore
        """
        for index in range(DEPTH):
            await store.enqueue(
                queued(f"filler-{index}", ACCEPTED_AT), _payload()
            )

        turned_away: Admission = await store.enqueue(
            queued("turned-away", ACCEPTED_AT), _payload()
        )

        assert not turned_away.queued
        assert await store.read("turned-away") is None

    _scenario(scenario, backend)


@BOTH_BACKENDS
def test_an_identical_submission_replays_the_first_job(
    backend: JobBackend,
) -> None:
    """The same key and the same request queue one job, not two."""

    async def scenario(store: JobStore) -> None:
        """Drive the store.

        :param store: The store under test.
        :type store: app.application.capabilities.JobStore
        """
        payload: JobPayload = _payload()
        claim = IdempotentRequest(
            key="retry-key", request_hash=request_hash(payload)
        )

        first: Admission = await store.enqueue(
            queued(FIRST_JOB, ACCEPTED_AT), payload, idempotency=claim
        )
        second: Admission = await store.enqueue(
            queued(SECOND_JOB, ACCEPTED_AT), payload, idempotency=claim
        )

        assert first.position == FIRST_IN_LINE
        # The second is answered with the first job, and nothing else
        # was queued: a retried POST must not double the work.
        assert second.replayed == FIRST_JOB
        assert not second.queued
        assert await store.depth() == ONE_JOB
        assert await store.read(SECOND_JOB) is None

    _scenario(scenario, backend)


@BOTH_BACKENDS
def test_a_spent_key_on_a_different_request_conflicts(
    backend: JobBackend,
) -> None:
    """Reusing a key for other work is refused, not guessed at."""

    async def scenario(store: JobStore) -> None:
        """Drive the store.

        :param store: The store under test.
        :type store: app.application.capabilities.JobStore
        """
        first: JobPayload = _payload()
        other: JobPayload = replace(first, prompt="something else entirely")

        await store.enqueue(
            queued(FIRST_JOB, ACCEPTED_AT),
            first,
            idempotency=IdempotentRequest(
                key="shared-key", request_hash=request_hash(first)
            ),
        )
        clash: Admission = await store.enqueue(
            queued("other-job", ACCEPTED_AT),
            other,
            idempotency=IdempotentRequest(
                key="shared-key", request_hash=request_hash(other)
            ),
        )

        # Whichever of the two the server picked would be wrong for the
        # other one, so it picks neither.
        assert clash.conflicted
        assert not clash.queued
        assert clash.replayed is None
        assert await store.depth() == ONE_JOB

    _scenario(scenario, backend)


@BOTH_BACKENDS
def test_a_submission_without_a_key_is_never_deduplicated(
    backend: JobBackend,
) -> None:
    """Idempotency is opt-in; two plain POSTs are two jobs."""

    async def scenario(store: JobStore) -> None:
        """Drive the store.

        :param store: The store under test.
        :type store: app.application.capabilities.JobStore
        """
        payload: JobPayload = _payload()

        await store.enqueue(queued("plain-one", ACCEPTED_AT), payload)
        await store.enqueue(queued("plain-two", ACCEPTED_AT), payload)

        assert await store.depth() == TWO_JOBS
        assert await store.read("plain-two") is not None

    _scenario(scenario, backend)
