"""Recoverable failure taxonomy.

Each error carries the data a caller needs to react, and nothing else.
There is deliberately no shared base class to catch: a handler that wants
to treat two failures alike must name both, which keeps the taxonomy
honest as it grows (``SKILL.md`` section 2).

Defects - a broken invariant, an impossible state - are not modelled
here. They raise normally and surface as 500 with telemetry.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InvalidPrompt(Exception):
    """The prompt is empty or otherwise unusable."""

    reason: str


@dataclass(frozen=True, slots=True)
class NoDetection(Exception):
    """The detector found nothing above the score threshold."""

    prompt: str


@dataclass(frozen=True, slots=True)
class UnknownBackend(Exception):
    """The caller asked for a segmentation backend that is not wired."""

    requested: str
    available: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImageDecodeFailed(Exception):
    """The uploaded bytes are not an image Pillow can open."""

    detail: str


@dataclass(frozen=True, slots=True)
class UploadTooLarge(Exception):
    """The upload exceeds the configured byte ceiling."""

    limit_bytes: int


@dataclass(frozen=True, slots=True)
class RateLimited(Exception):
    """The caller exhausted its request budget for the current window."""

    retry_after_seconds: int


@dataclass(frozen=True, slots=True)
class DeviceExhausted(Exception):
    """The accelerator ran out of memory for this input.

    Recoverable and retryable with a smaller image, which is why it is a
    typed failure rather than a defect.
    """

    detail: str


@dataclass(frozen=True, slots=True)
class ModelUnavailable(Exception):
    """A model could not be loaded, so the capability cannot serve."""

    model: str
    detail: str


@dataclass(frozen=True, slots=True)
class FaceAnalysisUnavailable(Exception):
    """Face analysis was requested but its optional extra is absent."""

    detail: str


__all__: list[str] = [
    "DeviceExhausted",
    "FaceAnalysisUnavailable",
    "ImageDecodeFailed",
    "InvalidPrompt",
    "ModelUnavailable",
    "NoDetection",
    "RateLimited",
    "UnknownBackend",
    "UploadTooLarge",
]
