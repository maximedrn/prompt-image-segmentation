"""Literals of an outbound delivery."""

from dataclasses import dataclass
from typing import ClassVar, Literal, final


@final
@dataclass(frozen=True, slots=True)
class WebhookRules:
    """Wire vocabulary of an outbound delivery."""

    algorithm: ClassVar[str] = "sha256"
    signature_header: ClassVar[str] = "X-Signature"
    timestamp_header: ClassVar[str] = "X-Timestamp"
    content_type_header: ClassVar[str] = "Content-Type"
    content_type: ClassVar[str] = "application/json"
    #: How a Pydantic body is rendered before it is signed: enums and
    #: floats resolved, because the receiver reads JSON.
    json_mode: ClassVar[Literal["json"]] = "json"
    https: ClassVar[str] = "https"
    http: ClassVar[str] = "http"
    #: Each retry waits twice as long as the one before.
    backoff_factor: ClassVar[float] = 2.0
    #: Index of the host inside a resolved socket address.
    host: ClassVar[int] = 0


@final
@dataclass(frozen=True, slots=True)
class DeliveryField:
    """Keys of the document a receiver is handed.

    Part of the published contract, and signed verbatim, so they live
    here rather than inline where a rename would go unnoticed.
    """

    #: Unique per terminal event, and the same across every attempt at
    #: delivering it. The receiver deduplicates on this, so a fresh one
    #: per attempt would turn a retry into new work in the very inbox it
    #: exists to feed.
    event_id: ClassVar[str] = "eventId"
    identifier: ClassVar[str] = "identifier"
    state: ClassVar[str] = "state"
    error: ClassVar[str] = "error"
    result: ClassVar[str] = "result"


@final
@dataclass(frozen=True, slots=True)
class DeliveryLog:
    """Format strings an outbound delivery logs with."""

    refused: ClassVar[str] = (
        "Webhook for %s refused: destination no longer public"
    )
    failed: ClassVar[str] = "Webhook for %s failed: %s (attempt %s)"
    answered: ClassVar[str] = "Webhook for %s answered %s (attempt %s)"
    gave_up: ClassVar[str] = "Webhook for %s gave up after %s attempts"


__all__: list[str] = ["DeliveryField", "DeliveryLog", "WebhookRules"]
