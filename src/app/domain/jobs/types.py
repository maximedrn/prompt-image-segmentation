"""The states one job can be in."""

from enum import StrEnum, unique


@unique
class JobState(StrEnum):
    """Where a job is in its life."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


__all__: list[str] = ["JobState"]
