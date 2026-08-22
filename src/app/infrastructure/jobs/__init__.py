"""Where queued work is kept, and how the choice is made.

Two implementations of the same capability: Redis for anything with
more than one process or more than one restart to survive, and process
memory for the deployments that have no server to talk to.

The choice is the operator's, taken once at composition time. Nothing
here probes Redis and quietly settles for memory instead: a queue that
does that hands a caller a ``202`` and an identifier that the next
worker, or the next restart, has never heard of.
"""

from app.application.capabilities import JobStore
from app.application.policies import JobBackend, JobPolicy
from app.infrastructure.jobs.memory import InMemoryJobStore
from app.infrastructure.jobs.store import RedisJobStore, connect


def build_store(policy: JobPolicy) -> JobStore:
    """Build the store the configuration asked for.

    :param policy: Where queued work lives, and how long it stays.
    :type policy: app.application.policies.JobPolicy
    :returns: The store the rest of the process will use.
    :rtype: app.application.capabilities.JobStore
    """
    if policy.backend is JobBackend.MEMORY:
        return InMemoryJobStore(policy)
    return RedisJobStore(connect(policy), policy)


__all__: list[str] = [
    "InMemoryJobStore",
    "RedisJobStore",
    "build_store",
    "connect",
]
