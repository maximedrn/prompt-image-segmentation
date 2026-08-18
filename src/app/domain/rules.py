"""Pure domain rules.

Deterministic geometry and decisions over masks. No I/O, no device, no
model: these stay ordinary Python functions rather than effects, because
they have no requirement and no recoverable failure (``SKILL.md``
sections 20 and 39).

``numpy`` is the only import: dilation lives in the imaging adapter
because it needs OpenCV, which the domain should not pull in.
"""

from numpy import amax, amin, bool_, intp, nonzero, uint8
from numpy.typing import NDArray

from app.domain.constants import (
    AGE_BAND_FLOOR,
    MASK,
    PERCENTAGE,
    PERSON,
    AgeBand,
)
from app.domain.models import BBox, MaskArray


def bbox_from_mask(mask: MaskArray, padding_percentage: float) -> BBox:
    """Return the padded bounding box of a mask's foreground.

    :param mask: 2D binary mask; non-zero pixels are foreground.
    :type mask: app.domain.models.MaskArray
    :param padding_percentage: Extra margin around the tight box, as a
        share of the mask's width and height. Clamped to the image.
    :type padding_percentage: float
    :returns: The padded box, zero-sized when the mask is blank.
    :rtype: app.domain.models.BBox
    """
    non_zero: tuple[NDArray[intp], ...] = nonzero(mask)
    if not non_zero[0].size:
        return BBox(x=0, y=0, width=0, height=0)
    height: int
    width: int
    height, width = mask.shape[:2]
    ratio: float = clamp_percentage(padding_percentage) / PERCENTAGE.whole
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
    :type bbox: app.domain.models.BBox
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
    return min(PERCENTAGE.maximum, max(PERCENTAGE.minimum, value))


def certainly_adult(bands: tuple[AgeBand, ...]) -> bool:
    """Decide whether every detected face is certainly an adult.

    A band certifies adulthood only when its *youngest* possible age
    already clears the threshold. The band spanning the threshold
    therefore never certifies, which makes the answer fail-safe rather
    than merely likely - a deliberate tightening over the numeric
    estimate this replaced.

    An image with no face is vacuously adult, which is the contract the
    API has always exposed.

    :param bands: One band per detected face.
    :type bands: tuple[app.domain.constants.AgeBand, ...]
    :returns: ``True`` when no face can be a minor.
    :rtype: bool
    """
    return all(AGE_BAND_FLOOR[band] >= PERSON.adult_age for band in bands)


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
    :type mask: app.domain.models.MaskArray
    :returns: Foreground pixels as a boolean array.
    :rtype: numpy.typing.NDArray[numpy.bool_]
    """
    return mask > MASK.threshold


__all__: list[str] = [
    "bbox_from_mask",
    "binarize",
    "certainly_adult",
    "clamp_percentage",
    "crop_to_bbox",
    "is_reliable",
]
