"""Segmenter output DTO."""

from typing import TypeAlias

from numpy import uint8
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

MaskArray: TypeAlias = NDArray[uint8]
"""Grayscale (H, W) uint8, values in {0, 255}."""


class SegmentationResult(BaseModel):
    """Raw output of a :class:`~app.segmenters.base.Segmenter`."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mask: MaskArray
    detections: int


__all__: list[str] = ["MaskArray", "SegmentationResult"]
