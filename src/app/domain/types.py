"""Domain vocabulary: the closed sets and the aliases.

Enumerations and type aliases only. Values that happen to be literals
live in ``constants.py``; anything with behaviour lives with its
concern.
"""

from enum import IntEnum, StrEnum, unique
from typing import NamedTuple, final

from numpy import uint8
from numpy.typing import NDArray

type MaskArray = NDArray[uint8]
"""Grayscale (H, W) uint8, in ``{MaskValues.background, .foreground}``."""


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


@unique
class AgeBand(StrEnum):
    """Age ranges an estimator can report.

    Bands, not years: the estimators available under a permissive
    licence classify into ranges. That costs precision exactly where it
    matters, because the adult threshold falls *inside* ``TEEN``.
    """

    INFANT = "infant"
    CHILD = "child"
    TEEN = "teen"
    TWENTIES = "twenties"
    THIRTIES = "thirties"
    FORTIES = "forties"
    FIFTIES = "fifties"
    SIXTIES = "sixties"
    SEVENTIES_PLUS = "seventies-plus"


@final
class AgeRange(NamedTuple):
    """The ages a band can contain, both ends included."""

    youngest: int
    oldest: int


__all__: list[str] = [
    "AgeBand",
    "AgeRange",
    "Gender",
    "ImageFormat",
    "ImageMode",
    "MaskArray",
]
