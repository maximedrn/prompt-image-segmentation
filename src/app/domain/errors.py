"""Recoverable failure taxonomy."""

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
class IllegalTransition(Exception):
    """A job was asked to move somewhere it cannot go from here.

    A defect rather than a caller error in most cases - a worker
    claiming a job twice, say - but it carries the states so the log
    says which move was refused.
    """

    identifier: str
    state: str
    attempted: str


@dataclass(frozen=True, slots=True)
class FaceAnalysisUnavailable(Exception):
    """Face analysis was requested but its optional extra is absent."""

    detail: str


@dataclass(frozen=True, slots=True)
class JobStoreUnavailable(Exception):
    """The store queued work lives in cannot be reached.

    An outage of a dependency, not a defect and not the caller's doing,
    so it earns a 503 and a retry rather than an opaque 500.
    """

    detail: str


__all__: list[str] = [
    "DeviceExhausted",
    "FaceAnalysisUnavailable",
    "JobStoreUnavailable",
    "IllegalTransition",
    "ImageDecodeFailed",
    "InvalidPrompt",
    "ModelUnavailable",
    "NoDetection",
    "RateLimited",
    "UnknownBackend",
    "UploadTooLarge",
]
