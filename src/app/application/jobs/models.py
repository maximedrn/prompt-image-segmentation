"""What a job carries in and what it carries out.

The payload is the request frozen at acceptance time: once ``POST /jobs``
answers, the connection is gone and nothing can be asked of the caller
again, so everything the worker needs has to be here.
"""

from dataclasses import dataclass
from typing import final

from app.application.jobs.results import SegmentSchema
from app.application.policies import SegmentOptions


@final
@dataclass(frozen=True, slots=True)
class JobPayload:
    """One accepted segmentation request, awaiting a worker.

    The image travels as the bytes that arrived rather than a decoded
    array: decoding is a validation boundary, and repeating it in the
    worker keeps the acceptance path cheap and the store small.
    """

    image: bytes
    prompt: str
    backend: str
    person_mode: bool
    options: SegmentOptions
    #: Where to post the outcome, when the caller asked for a push.
    callback_url: str | None = None


@final
@dataclass(frozen=True, slots=True)
class JobResult:
    """What a finished job leaves behind for the caller to collect.

    Either a body or a failure code, never both: the state on the job
    itself says which one to read.

    ``body`` is the model itself rather than a mapping of it: a queue
    that answered with a bare ``dict`` would leave every reader to
    guess, and would let a store hand back something no schema had
    checked.
    """

    body: SegmentSchema | None = None
    error: str | None = None


__all__: list[str] = ["JobPayload", "JobResult"]
