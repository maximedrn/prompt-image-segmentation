"""What became of a submission, as a value.

``enqueue`` used to answer ``int | None``: a place in line, or a full
queue. An idempotency key adds two more outcomes that are neither -- the
same request arriving twice, and a different request arriving under a
key already spent -- and both have to be told apart from a fresh
admission by the caller that answers the client.

Three states in one frozen value rather than three classes: they are
mutually exclusive answers to one question, and the constructors below
are what callers use, so no caller ever builds an inconsistent one.
"""

from dataclasses import dataclass
from typing import Self, final


@final
@dataclass(frozen=True, slots=True)
class Admission:
    """The outcome of offering one job to the queue."""

    #: Jobs ahead of this one, when it was newly accepted.
    position: int | None = None
    #: The job an identical earlier submission created, when this one
    #: replayed it. Nothing was queued a second time.
    replayed: str | None = None
    #: Set when the idempotency key was already spent on a *different*
    #: request. Nothing was queued, and nothing was replayed either.
    conflicted: bool = False

    @classmethod
    def accepted(cls, position: int) -> Self:
        """Report a job that was newly queued.

        :param position: Jobs ahead of it.
        :type position: int
        :returns: The admission.
        :rtype: Admission
        """
        return cls(position=position)

    @classmethod
    def full(cls) -> Self:
        """Report a queue that was already at its configured depth.

        :returns: The admission.
        :rtype: Admission
        """
        return cls()

    @classmethod
    def replay(cls, identifier: str) -> Self:
        """Report an identical submission, answered with the first job.

        :param identifier: The job the first submission created.
        :type identifier: str
        :returns: The admission.
        :rtype: Admission
        """
        return cls(replayed=identifier)

    @classmethod
    def conflict(cls) -> Self:
        """Report a spent key offered with a different request.

        :returns: The admission.
        :rtype: Admission
        """
        return cls(conflicted=True)

    @property
    def queued(self) -> bool:
        """Report whether this submission put work in the queue.

        :returns: ``True`` only for a fresh admission.
        :rtype: bool
        """
        return self.position is not None


__all__: list[str] = ["Admission"]
