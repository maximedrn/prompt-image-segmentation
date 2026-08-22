"""Literals of the job lifecycle."""

from dataclasses import dataclass
from typing import ClassVar, final

from app.domain.jobs.types import JobState


@final
@dataclass(frozen=True, slots=True)
class Lifecycle:
    """Which states end a job's life."""

    terminal: ClassVar[frozenset[JobState]] = frozenset({
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
    })


__all__: list[str] = ["Lifecycle"]
