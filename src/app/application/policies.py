"""Narrow policy value objects.

Each policy carries only what one component needs, so nothing has to
accept the whole settings object and become a service locator
(``SKILL.md`` section 27).
"""

from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True, slots=True)
class SegmentationPolicy:
    """Tuning the segmentation use case applies to its own output."""

    mask_padding_percentage: float
    dilation_percentage: float
    reliability_threshold: float


@final
@dataclass(frozen=True, slots=True)
class DetectionPolicy:
    """Thresholds the detector adapter filters its own output with."""

    score_threshold: float
    text_threshold: float


@final
@dataclass(frozen=True, slots=True)
class FacePolicy:
    """Tuning the face detector applies to its own output."""

    score_threshold: float


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


__all__: list[str] = [
    "DetectionPolicy",
    "FacePolicy",
    "RateLimitPolicy",
    "SegmentationPolicy",
    "UploadPolicy",
]
