"""Domain layer.

Value objects, recoverable failures and pure rules. Depends on nothing
but ``numpy``: importable without a model, a device or a web framework,
which is what makes the use cases testable without booting anything.
"""

from app.domain.constants import (
    MASK,
    PERCENTAGE,
    PERSON,
    SCORE,
    Gender,
    ImageFormat,
    ImageMode,
)
from app.domain.errors import (
    DeviceExhausted,
    FaceAnalysisUnavailable,
    ImageDecodeFailed,
    InvalidPrompt,
    ModelUnavailable,
    NoDetection,
    RateLimited,
    UnknownBackend,
    UploadTooLarge,
)
from app.domain.models import (
    BBox,
    Detection,
    MaskArray,
    PersonPayload,
    PixelBox,
    Prompt,
    SegmentationResult,
    SegmentedImage,
    SourceImage,
    clamp_score,
)
from app.domain.rules import (
    bbox_from_mask,
    binarize,
    clamp_percentage,
    crop_to_bbox,
    is_reliable,
)

__all__: list[str] = [
    "MASK",
    "PERCENTAGE",
    "PERSON",
    "SCORE",
    "BBox",
    "Detection",
    "DeviceExhausted",
    "FaceAnalysisUnavailable",
    "Gender",
    "ImageDecodeFailed",
    "ImageFormat",
    "ImageMode",
    "InvalidPrompt",
    "MaskArray",
    "ModelUnavailable",
    "NoDetection",
    "PersonPayload",
    "PixelBox",
    "Prompt",
    "RateLimited",
    "SegmentationResult",
    "SegmentedImage",
    "SourceImage",
    "UnknownBackend",
    "UploadTooLarge",
    "bbox_from_mask",
    "binarize",
    "clamp_percentage",
    "clamp_score",
    "crop_to_bbox",
    "is_reliable",
]
