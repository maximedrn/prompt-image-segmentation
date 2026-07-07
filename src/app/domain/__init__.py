"""Value objects (DTOs).

Pure data. No I/O, no framework imports, no business logic. Safe to
import from anywhere without triggering heavy dependencies.
"""

from typing import TypeAlias

from app.domain.bbox import BBox
from app.domain.person import PersonPayload
from app.domain.response import SegmentResponse
from app.domain.segmentation import (
    MaskArray,
    SegmentationResult,
)

JSONValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)
"""What ``model_dump()`` produces: any JSON-serializable value."""

__all__: list[str] = [
    "BBox",
    "JSONValue",
    "MaskArray",
    "PersonPayload",
    "SegmentResponse",
    "SegmentationResult",
]
