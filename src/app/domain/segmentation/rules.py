"""Pure geometry and decisions over masks."""

from numpy import (
    amax,
    amin,
    any as array_any,
    bool_,
    intp,
    nonzero,
    uint8,
)
from numpy.typing import NDArray

from app.domain.constants import DomainText, MaskValues, PercentageBounds
from app.domain.segmentation.models import BBox, Detection
from app.domain.types import MaskArray


def bbox_from_mask(mask: MaskArray, padding_percentage: float) -> BBox:
    """Return the padded bounding box of a mask's foreground.

    :param mask: 2D binary mask; non-zero pixels are foreground.
    :type mask: app.domain.MaskArray
    :param padding_percentage: Extra margin around the tight box, as a
        share of the mask's width and height. Clamped to the image.
    :type padding_percentage: float
    :returns: The padded box, zero-sized when the mask is blank.
    :rtype: app.domain.BBox
    """
    non_zero: tuple[NDArray[intp], ...] = nonzero(mask)
    if not non_zero[0].size:
        return BBox(x=0, y=0, width=0, height=0)
    height, width = mask.shape[:2]
    ratio: float = (
        clamp_percentage(padding_percentage) / PercentageBounds.whole
    )
    pad_x: int = int(width * ratio)
    pad_y: int = int(height * ratio)
    x_min: int = max(0, int(amin(non_zero[1])) - pad_x)
    x_max: int = min(width, int(amax(non_zero[1])) + pad_x)
    y_min: int = max(0, int(amin(non_zero[0])) - pad_y)
    y_max: int = min(height, int(amax(non_zero[0])) + pad_y)
    return BBox(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)


def crop_to_bbox(source: NDArray[uint8], bbox: BBox) -> NDArray[uint8]:
    """Restrict an array to a box, copying when the box is empty.

    :param source: 2D or 3D uint8 array (HxW or HxWxC).
    :type source: numpy.typing.NDArray[numpy.uint8]
    :param bbox: Region to keep.
    :type bbox: app.domain.BBox
    :returns: A view of the region, or a copy when the box is empty.
    :rtype: numpy.typing.NDArray[numpy.uint8]
    """
    if bbox.empty:
        return source.copy()
    return source[bbox.y : bbox.bottom, bbox.x : bbox.right]


def clamp_percentage(value: float) -> float:
    """Confine a percentage to the accepted range.

    :param value: Caller-supplied percentage.
    :type value: float
    :returns: The percentage confined to ``[0, 100]``.
    :rtype: float
    """
    return min(PercentageBounds.maximum, max(PercentageBounds.minimum, value))


def above_confidence(
    detections: tuple[Detection, ...], minimum: float
) -> tuple[Detection, ...]:
    """Keep the detections a caller is willing to trust.

    Filtering happens on the combined confidence rather than the
    detector's own score, because a confident box around a poor mask is
    exactly what a caller asking for a minimum wants gone.

    :param detections: Every scored detection, in detector order.
    :type detections: tuple[app.domain.Detection, ...]
    :param minimum: Lowest combined confidence to retain.
    :type minimum: float
    :returns: The retained detections, order preserved, possibly empty.
    :rtype: tuple[app.domain.Detection, ...]
    """
    return tuple(
        detection
        for detection in detections
        if detection.confidence >= minimum
    )


def union_masks(masks: tuple[MaskArray, ...]) -> MaskArray:
    """Merge per-detection masks into the single mask they cover.

    :param masks: One mask per detection, all the same shape.
    :type masks: tuple[app.domain.MaskArray, ...]
    :returns: The union, as a binary mask of the same shape.
    :rtype: app.domain.MaskArray
    :raises ValueError: If ``masks`` is empty, which no caller can
        usefully act on: a union of nothing has no shape to return.
    """
    if not masks:
        raise ValueError(DomainText.empty_union)
    stacked: NDArray[bool_] = array_any(
        [binarize(mask) for mask in masks], axis=0
    )
    return (stacked * MaskValues.foreground).astype(uint8)


def is_reliable(confidence: float, threshold: float) -> bool:
    """Decide whether a result clears the reliability bar.

    :param confidence: Combined confidence of the weakest detection.
    :type confidence: float
    :param threshold: Configured minimum.
    :type threshold: float
    :returns: ``True`` when the caller can trust the mask.
    :rtype: bool
    """
    return confidence >= threshold


def binarize(mask: MaskArray) -> NDArray[bool_]:
    """Return the boolean foreground of a mask that survived encoding.

    :param mask: Grayscale mask, possibly resampled or PNG round-tripped.
    :type mask: app.domain.MaskArray
    :returns: Foreground pixels as a boolean array.
    :rtype: numpy.typing.NDArray[numpy.bool_]
    """
    return mask > MaskValues.threshold


__all__: list[str] = [
    "above_confidence",
    "bbox_from_mask",
    "binarize",
    "clamp_percentage",
    "crop_to_bbox",
    "is_reliable",
    "union_masks",
]
