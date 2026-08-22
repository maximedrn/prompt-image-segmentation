"""Literals of the face-analysis contract."""

from dataclasses import dataclass
from typing import ClassVar, final

from app.domain.types import AgeBand, AgeRange


@final
@dataclass(frozen=True, slots=True)
class PersonRules:
    """Thresholds the face-analysis contract is written against."""

    adult_age: ClassVar[int] = 18


@final
@dataclass(frozen=True, slots=True)
class AgeBands:
    """What each band contains, both ends included.

    One table for both bounds: ``youngest`` decides adulthood, and the
    pair is what ``age_bands_digits`` reports. ``TEEN`` spans the adult
    threshold, which is why it can never certify adulthood.
    """

    ranges: ClassVar[dict[AgeBand, AgeRange]] = {
        AgeBand.INFANT: AgeRange(youngest=0, oldest=2),
        AgeBand.CHILD: AgeRange(youngest=3, oldest=9),
        AgeBand.TEEN: AgeRange(youngest=10, oldest=19),
        AgeBand.TWENTIES: AgeRange(youngest=20, oldest=29),
        AgeBand.THIRTIES: AgeRange(youngest=30, oldest=39),
        AgeBand.FORTIES: AgeRange(youngest=40, oldest=49),
        AgeBand.FIFTIES: AgeRange(youngest=50, oldest=59),
        AgeBand.SIXTIES: AgeRange(youngest=60, oldest=69),
        AgeBand.SEVENTIES_PLUS: AgeRange(youngest=70, oldest=120),
    }


__all__: list[str] = ["AgeBands", "PersonRules"]
