"""Ceilings the transport applies before any work is accepted."""

from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True, slots=True)
class UploadPolicy:
    """Ceilings applied to an inbound image before it is decoded."""

    max_upload_bytes: int
    max_image_pixels: int


@final
@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """Bounds of the per-client request budget."""

    max_requests: int
    window_seconds: float
    max_tracked_clients: int
    minimum_retry_after_seconds: int = 1


__all__: list[str] = ["RateLimitPolicy", "UploadPolicy"]
