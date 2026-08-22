"""Domain literals.

Grouped into ``final`` dataclasses whose fields are ``ClassVar``: they
are namespaces, never instantiated, so a caller reads
``MaskValues.foreground`` rather than a module-level singleton.

``ClassVar`` is not decoration. On a slotted dataclass, an ordinary
field read from the class returns the slot descriptor rather than its
default, which would compare an ``int`` against a ``member_descriptor``
and fail at runtime.
"""

from dataclasses import dataclass
from typing import ClassVar, final


@final
@dataclass(frozen=True, slots=True)
class MaskValues:
    """Pixel values of a binary mask, and the split between them."""

    background: ClassVar[int] = 0
    foreground: ClassVar[int] = 255
    #: Anything strictly above this counts as foreground when decoding a
    #: mask that has been through PNG encoding.
    threshold: ClassVar[int] = 127


@final
@dataclass(frozen=True, slots=True)
class ScoreBounds:
    """Range every model score is clamped into before leaving the domain."""

    minimum: ClassVar[float] = 0.0
    maximum: ClassVar[float] = 1.0


@final
@dataclass(frozen=True, slots=True)
class PercentageBounds:
    """Range accepted by the padding and dilation ratios."""

    minimum: ClassVar[float] = 0.0
    maximum: ClassVar[float] = 100.0
    #: Divisor turning a percentage into a ratio.
    whole: ClassVar[float] = 100.0


@final
@dataclass(frozen=True, slots=True)
class DomainText:
    """What the domain says when it refuses.

    Here rather than inline for the same reason as everywhere else:
    a message a caller can read is part of the contract, and the
    contract is not written in the middle of a function.
    """

    #: A caller error: the prompt arrived empty.
    empty_prompt: ClassVar[str] = "Prompt must not be empty."
    #: A defect: the use case filters before it merges, so an empty set
    #: means the caller of this rule skipped that step.
    empty_union: ClassVar[str] = "Cannot union an empty set of masks."


__all__: list[str] = [
    "DomainText",
    "MaskValues",
    "PercentageBounds",
    "ScoreBounds",
]
