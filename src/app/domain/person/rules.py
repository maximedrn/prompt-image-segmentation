"""Pure rules over a face analysis."""

from app.domain.person.constants import AgeBands, PersonRules
from app.domain.types import AgeBand, AgeRange


def age_range(band: AgeBand) -> AgeRange:
    """Return the youngest and oldest age a band can contain.

    Bounds rather than a single number: the estimators classify into
    ranges, and collapsing that to one age would invent a precision the
    model does not have.

    :param band: Band an estimator reported.
    :type band: app.domain.types.AgeBand
    :returns: The band's bounds, both inclusive.
    :rtype: app.domain.types.AgeRange
    """
    return AgeBands.ranges[band]


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
    :type bands: tuple[app.domain.types.AgeBand, ...]
    :returns: ``True`` when no face can be a minor.
    :rtype: bool
    """
    return all(
        age_range(band).youngest >= PersonRules.adult_age for band in bands
    )


__all__: list[str] = ["age_range", "certainly_adult"]
