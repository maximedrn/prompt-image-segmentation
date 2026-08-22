"""What a face analysis produces."""

from dataclasses import dataclass

from app.domain.types import AgeBand, Gender


@dataclass(frozen=True, slots=True)
class PersonPayload:
    """Face-analysis summary for one image.

    ``age_bands`` is index-aligned with ``genders`` and is exposed so a
    caller can apply its own policy: ``is_adult`` deliberately refuses to
    certify the band that straddles the threshold.
    """

    genders: tuple[Gender, ...]
    age_bands: tuple[AgeBand, ...]
    is_adult: bool


__all__: list[str] = ["PersonPayload"]
