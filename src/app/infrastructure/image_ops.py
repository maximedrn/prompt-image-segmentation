"""Pure numpy / cv2 operations on masks and image arrays."""

from cv2 import dilate as cv2_dilate
from numpy import (
    amax,
    amin,
    clip,
    intp,
    nonzero,
    ones,
    uint8,
)
from numpy.typing import NDArray

from app.domain import BBox


def bbox_from_mask(mask: NDArray[uint8], padding_pct: int) -> BBox:
    """Padded bbox of non-zero pixels in ``mask`` (or empty bbox).

    :param mask: 2D binary mask (non-zero = foreground).
    :type mask: numpy.typing.NDArray[numpy.uint8]
    :param padding_pct: Extra margin around the tight bbox, as a
        percentage of the mask's width/height. Clamped to the image.
    :type padding_pct: int
    :returns: A :class:`~app.domain.bbox.BBox`. Zero-sized when
        the mask is entirely blank.
    :rtype: app.domain.bbox.BBox
    """
    non_zero: tuple[NDArray[intp], ...] = nonzero(mask)
    if not non_zero[0].size:
        return BBox(x=0, y=0, width=0, height=0)
    height, width = mask.shape[:2]
    pad_x: int = width * padding_pct // 100
    pad_y: int = height * padding_pct // 100
    x_min: int = max(0, int(amin(non_zero[1])) - pad_x)
    x_max: int = min(width, int(amax(non_zero[1])) + pad_x)
    y_min: int = max(0, int(amin(non_zero[0])) - pad_y)
    y_max: int = min(height, int(amax(non_zero[0])) + pad_y)
    return BBox(
        x=x_min,
        y=y_min,
        width=x_max - x_min,
        height=y_max - y_min,
    )


def crop_to_bbox(source: NDArray[uint8], bbox: BBox) -> NDArray[uint8]:
    """Return ``source`` restricted to ``bbox`` (copy if empty).

    :param source: 2D or 3D uint8 array (HxW or HxWxC).
    :type source: numpy.typing.NDArray[numpy.uint8]
    :param bbox: Region to keep.
    :type bbox: app.domain.bbox.BBox
    :returns: A view (or copy when the bbox is empty) of the region.
    :rtype: numpy.typing.NDArray[numpy.uint8]
    """
    if bbox.empty:
        return source.copy()
    return source[bbox.y : bbox.bottom, bbox.x : bbox.right]


def dilate_mask(source: NDArray[uint8], percentage: float) -> NDArray[uint8]:
    """Dilate ``source`` by ``percentage`` of its dimensions.

    :param source: 2D uint8 mask to dilate.
    :type source: numpy.typing.NDArray[numpy.uint8]
    :param percentage: Kernel radius as a share of the mask's width
        and height. Clamped to ``[0, 100]``.
    :type percentage: float
    :returns: The dilated mask (same shape as ``source``).
    :rtype: numpy.typing.NDArray[numpy.uint8]
    """
    pct: float = float(clip(percentage, 0, 100))
    height, width = source.shape[:2]
    pad_x: int = int(width * (pct / 100.0))
    pad_y: int = int(height * (pct / 100.0))
    kernel: NDArray[uint8] = ones((pad_y + 1, pad_x + 1), dtype=uint8)
    result: NDArray[uint8] = cv2_dilate(source, kernel, iterations=1).astype(
        uint8
    )
    return result


__all__: list[str] = ["bbox_from_mask", "crop_to_bbox", "dilate_mask"]
