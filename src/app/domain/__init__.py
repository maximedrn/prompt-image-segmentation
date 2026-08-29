"""Domain layer.

Value objects, recoverable failures and pure rules, grouped by concern:
``segmentation``, ``person`` and ``jobs``. Depends on nothing but
``numpy``, so it is importable without a model, a device or a web
framework - which is what makes the use cases testable without booting
anything.

This module is the layer's facade: everything above imports from here
rather than reaching into a concern, so moving a rule between files
costs nothing outside the domain.
"""

from app.domain.constants import MaskValues, PercentageBounds, ScoreBounds
from app.domain.errors import (
    DeviceExhausted,
    FaceAnalysisUnavailable,
    IllegalTransition,
    ImageDecodeFailed,
    InvalidPrompt,
    JobStoreUnavailable,
    ModelUnavailable,
    NoDetection,
    RateLimited,
    UnknownBackend,
    UploadTooLarge,
)
from app.domain.jobs.admission import Admission
from app.domain.jobs.constants import Lifecycle
from app.domain.jobs.models import Job
from app.domain.jobs.rules import cancel, fail, queued, start, succeed
from app.domain.jobs.types import JobState
from app.domain.person.constants import AgeBands, PersonRules
from app.domain.person.models import PersonPayload
from app.domain.person.rules import age_range, certainly_adult
from app.domain.segmentation.models import (
    BBox,
    Detection,
    PixelBox,
    Prompt,
    SegmentedImage,
    SegmentRegion,
    SourceImage,
    clamp_score,
)
from app.domain.segmentation.rules import (
    above_confidence,
    bbox_from_mask,
    binarize,
    clamp_percentage,
    crop_to_bbox,
    is_reliable,
    union_masks,
)
from app.domain.types import (
    AgeBand,
    AgeRange,
    Gender,
    ImageFormat,
    ImageMode,
    MaskArray,
)

__all__: list[str] = [
    "AgeBand",
    "AgeBands",
    "AgeRange",
    "BBox",
    "Detection",
    "DeviceExhausted",
    "FaceAnalysisUnavailable",
    "Gender",
    "IllegalTransition",
    "ImageDecodeFailed",
    "ImageFormat",
    "Admission",
    "ImageMode",
    "InvalidPrompt",
    "JobStoreUnavailable",
    "Job",
    "JobState",
    "Lifecycle",
    "MaskArray",
    "MaskValues",
    "ModelUnavailable",
    "NoDetection",
    "PercentageBounds",
    "PersonPayload",
    "PersonRules",
    "PixelBox",
    "Prompt",
    "RateLimited",
    "ScoreBounds",
    "SegmentRegion",
    "SegmentedImage",
    "SourceImage",
    "UnknownBackend",
    "UploadTooLarge",
    "above_confidence",
    "age_range",
    "bbox_from_mask",
    "binarize",
    "cancel",
    "certainly_adult",
    "clamp_percentage",
    "clamp_score",
    "crop_to_bbox",
    "fail",
    "is_reliable",
    "queued",
    "start",
    "succeed",
    "union_masks",
]
