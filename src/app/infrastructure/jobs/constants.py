"""Literals of the Redis job store."""

from dataclasses import dataclass
from typing import ClassVar, final


@final
@dataclass(frozen=True, slots=True)
class JobKeys:
    """Redis key prefixes, and the one list that orders the work."""

    job: ClassVar[str] = "job:"
    payload: ClassVar[str] = "job:payload:"
    image: ClassVar[str] = "job:image:"
    result: ClassVar[str] = "job:result:"
    queue: ClassVar[str] = "jobs:queue"


@final
@dataclass(frozen=True, slots=True)
class Claiming:
    """How a worker waits on an empty queue."""

    #: How long ``claim`` waits before answering ``None``, which is what
    #: lets a shutdown interrupt the worker promptly.
    timeout_seconds: ClassVar[int] = 5
    #: ``BLPOP`` answers ``(key, value)``; only the value is of interest.
    popped_value: ClassVar[int] = 1


__all__: list[str] = ["Claiming", "JobKeys"]
