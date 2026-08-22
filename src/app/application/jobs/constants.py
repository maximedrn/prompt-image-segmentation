"""Literals the queue answers with."""

from dataclasses import dataclass
from typing import ClassVar, final


@final
@dataclass(frozen=True, slots=True)
class JobFailure:
    """Failure codes a job can record for itself."""

    #: What a job records when it died of a defect rather than a failure
    #: it could name. Mirrors the transport's ``ErrorCode.INTERNAL``,
    #: which is the value a caller will read back.
    defect: ClassVar[str] = "internal"


@final
@dataclass(frozen=True, slots=True)
class WorkerLog:
    """Format strings the queue worker logs with.

    ``%s`` placeholders rather than f-strings: logging interpolates
    lazily, so a suppressed record costs nothing to not format.
    """

    webhook_failed: ClassVar[str] = "Webhook for %s could not be sent"
    job_raised: ClassVar[str] = "Job %s raised"
    job_defect: ClassVar[str] = "Job %s ended in a defect"


__all__: list[str] = ["JobFailure", "WorkerLog"]
