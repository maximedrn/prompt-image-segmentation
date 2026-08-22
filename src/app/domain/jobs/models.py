"""What a queued job is, as a value."""

from dataclasses import dataclass

from app.domain.jobs.constants import Lifecycle
from app.domain.jobs.types import JobState


@dataclass(frozen=True, slots=True)
class Job:
    """One queued segmentation, and how far it has got.

    The result itself is not here: it is several megabytes of encoded
    pixels, and a caller polling for state should not carry them on
    every answer.
    """

    identifier: str
    state: JobState
    created_at: float
    updated_at: float
    #: Machine-readable reason, set only when ``state`` is ``FAILED``.
    error: str | None = None

    @property
    def terminal(self) -> bool:
        """Report whether this job can still change.

        :returns: ``True`` when no transition remains.
        :rtype: bool
        """
        return self.state in Lifecycle.terminal


__all__: list[str] = ["Job"]
