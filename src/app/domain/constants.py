"""Domain vocabulary.

Every literal the domain layer needs, grouped into named objects. Closed
sets are enumerations; bundles of related values are frozen singletons.
Nothing here is a lone constant.
"""

from dataclasses import dataclass
from enum import IntEnum, StrEnum, unique
from typing import Final, final


@unique
class ImageMode(StrEnum):
    """Pillow image modes this application produces or consumes."""

    RGB = "RGB"
    GRAYSCALE = "L"


@unique
class ImageFormat(StrEnum):
    """Encodings used on the wire."""

    PNG = "PNG"


@unique
class Gender(IntEnum):
    """Gender codes returned under ``response.person.genders``."""

    MALE = 0
    FEMALE = 1


@final
@dataclass(frozen=True, slots=True)
class MaskValues:
    """Pixel values of a binary mask, and the split between them."""

    background: int = 0
    foreground: int = 255
    #: Anything strictly above this counts as foreground when decoding a
    #: mask that has been through PNG encoding.
    threshold: int = 127


@final
@dataclass(frozen=True, slots=True)
class ScoreBounds:
    """Range every model score is clamped into before leaving the domain."""

    minimum: float = 0.0
    maximum: float = 1.0


@final
@dataclass(frozen=True, slots=True)
class PercentageBounds:
    """Range accepted by the padding and dilation ratios."""

    minimum: float = 0.0
    maximum: float = 100.0
    #: Divisor turning a percentage into a ratio.
    whole: float = 100.0


@final
@dataclass(frozen=True, slots=True)
class PersonRules:
    """Thresholds and labels of the face-analysis contract."""

    adult_age: int = 18
    #: Label the face estimator uses for male subjects; anything else is
    #: read as female, which is the estimator's own binary contract.
    male_label: str = "Male"
    #: A face stack is (count, height, width, channels); fewer dimensions
    #: than this means the detector found nothing.
    face_stack_dimensions: int = 4


MASK: Final[MaskValues] = MaskValues()
SCORE: Final[ScoreBounds] = ScoreBounds()
PERCENTAGE: Final[PercentageBounds] = PercentageBounds()
PERSON: Final[PersonRules] = PersonRules()


__all__: list[str] = [
    "MASK",
    "PERCENTAGE",
    "PERSON",
    "SCORE",
    "Gender",
    "ImageFormat",
    "ImageMode",
    "MaskValues",
    "PercentageBounds",
    "PersonRules",
    "ScoreBounds",
]
