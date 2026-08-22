"""Literals of the pixel plumbing."""

from dataclasses import dataclass
from typing import ClassVar, final


@final
@dataclass(frozen=True, slots=True)
class Dilation:
    """How the mask dilation kernel is built."""

    iterations: ClassVar[int] = 1
    #: A kernel needs at least one pixel, hence the offset on each axis.
    minimum_kernel_size: ClassVar[int] = 1


__all__: list[str] = ["Dilation"]
