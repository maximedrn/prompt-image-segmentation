"""Domain value objects.

Plain frozen dataclasses, not Pydantic models: validation belongs at the
boundary (``SKILL.md`` section 21), and the domain should not depend on a
serialization library to describe itself. The HTTP schemas that mirror
these live in ``app.interfaces.http.schemas``.
"""

from dataclasses import dataclass
from typing import Self

from numpy import uint8
from numpy.typing import NDArray

from app.domain.constants import SCORE, Gender
from app.domain.errors import InvalidPrompt

type MaskArray = NDArray[uint8]
"""Grayscale (H, W) uint8, values in ``{MASK.background, MASK.foreground}``."""


def clamp_score(value: float) -> float:
    """Confine a raw model score to the domain's score range.

    Score heads are linear layers, so a value can land marginally outside
    ``[0, 1]``. Clamping keeps a cosmetic excursion from turning a served
    request into a validation error.

    :param value: Raw score straight off a model.
    :type value: float
    :returns: The score confined to the domain range.
    :rtype: float
    """
    return min(SCORE.maximum, max(SCORE.minimum, value))


@dataclass(frozen=True, slots=True)
class Prompt:
    """A validated, detector-ready text prompt."""

    text: str

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Validate and normalise a caller-supplied prompt.

        :param raw: Untrusted prompt text.
        :type raw: str
        :returns: The validated prompt.
        :rtype: Prompt
        :raises app.domain.errors.InvalidPrompt: If ``raw`` is blank.
        """
        if not (stripped := raw.strip()):
            raise InvalidPrompt(reason="Prompt must not be empty.")
        return cls(text=stripped)


@dataclass(frozen=True, slots=True)
class SourceImage:
    """An RGB image as pixels, decoded and validated at the boundary.

    Holds a numpy array rather than a Pillow image so the domain and the
    capability interfaces depend on numpy alone; decoding lives in the
    imaging adapter.
    """

    pixels: NDArray[uint8]

    @property
    def height(self) -> int:
        """Image height in pixels.

        :returns: Number of rows.
        :rtype: int
        """
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        """Image width in pixels.

        :returns: Number of columns.
        :rtype: int
        """
        return int(self.pixels.shape[1])


@dataclass(frozen=True, slots=True)
class PixelBox:
    """A detector box in original-image pixels, corner form."""

    left: float
    top: float
    right: float
    bottom: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return the box in the ``xyxy`` order the models expect.

        :returns: ``(left, top, right, bottom)``.
        :rtype: tuple[float, float, float, float]
        """
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True, slots=True)
class BBox:
    """An integer bounding box in original-image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def empty(self) -> bool:
        """Report whether the box has zero or negative area.

        :returns: ``True`` if width or height is non-positive.
        :rtype: bool
        """
        return self.width <= 0 or self.height <= 0

    @property
    def right(self) -> int:
        """Right pixel, exclusive slice bound.

        :returns: ``x + width``.
        :rtype: int
        """
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Bottom pixel, exclusive slice bound.

        :returns: ``y + height``.
        :rtype: int
        """
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected instance and how much it can be trusted."""

    box: PixelBox
    detection_score: float
    mask_score: float

    @property
    def confidence(self) -> float:
        """Combined trust in this detection.

        The product, not the mean: a confident box around a poor mask is
        as useless as a good mask around the wrong object, so either weak
        factor has to drag the result down.

        :returns: ``detection_score * mask_score``.
        :rtype: float
        """
        return self.detection_score * self.mask_score


@dataclass(frozen=True, slots=True)
class PersonPayload:
    """Face-analysis summary for one image."""

    genders: tuple[Gender, ...]
    is_adult: bool


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """A union mask plus the scored detections that produced it."""

    mask: MaskArray
    detections: tuple[Detection, ...]

    @property
    def confidence(self) -> float:
        """Trust in the union mask as a whole.

        The weakest detection's, because the mask is their union: one bad
        member contaminates the result even if the others are perfect.

        :returns: Lowest per-detection confidence, ``SCORE.minimum`` when
            there is no detection.
        :rtype: float
        """
        if not self.detections:
            return SCORE.minimum
        return min(detection.confidence for detection in self.detections)


@dataclass(frozen=True, slots=True)
class SegmentedImage:
    """Everything a completed segmentation produces, still as pixels."""

    bbox: BBox
    cropped_mask: MaskArray
    cropped_image: MaskArray
    detections: tuple[Detection, ...]
    confidence: float
    reliable: bool
    person: PersonPayload | None


__all__: list[str] = [
    "BBox",
    "Detection",
    "MaskArray",
    "PersonPayload",
    "PixelBox",
    "Prompt",
    "SegmentationResult",
    "SegmentedImage",
    "SourceImage",
    "clamp_score",
]
