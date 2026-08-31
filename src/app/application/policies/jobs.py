"""Bounds of the queue, and of the deliveries it makes."""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import final


@unique
class JobBackend(StrEnum):
    """Where queued work is kept.

    Chosen by an operator, never inferred from whether Redis happens to
    answer. A queue that silently falls back to process memory hands a
    caller a ``202`` and an identifier that a second worker - or the
    next restart - knows nothing about.
    """

    REDIS = "redis"
    MEMORY = "memory"


@final
@dataclass(frozen=True, slots=True)
class JobPolicy:
    """Where queued work lives, and how long it stays there."""

    url: str
    retention_seconds: int
    max_queue_depth: int
    backend: JobBackend = JobBackend.REDIS


@final
@dataclass(frozen=True, slots=True)
class WebhookPolicy:
    """How an outbound delivery is signed, bounded and retried."""

    signing_secret: str
    timeout_seconds: float
    max_attempts: int
    initial_backoff_seconds: float
    #: Plain HTTP is refused unless an operator says otherwise, which is
    #: only reasonable inside a private network.
    allow_insecure: bool = False
    #: Internal addresses are refused unless an operator says otherwise.
    #: The service makes the callback request itself, so a caller who
    #: names the address chooses what it reaches. Turn this on only
    #: where every caller is one you control -- a local or CI stack
    #: whose receiver has no public name is the case it exists for.
    allow_private_hosts: bool = False

    @property
    def enabled(self) -> bool:
        """Report whether deliveries can be signed at all.

        :returns: ``True`` when a signing secret is configured.
        :rtype: bool
        """
        return bool(self.signing_secret)


__all__: list[str] = ["JobBackend", "JobPolicy", "WebhookPolicy"]
