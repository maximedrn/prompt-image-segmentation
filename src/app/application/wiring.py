"""Capability wiring.

Separate from :mod:`app.application.effects` so the effect vocabulary
stays importable by the use cases without a cycle: this module knows
both, neither knows this one.
"""

from stateless.handler import Handler
from stateless.need import supply

from app.application.capabilities import (
    MaskDilator,
    MaskRefiner,
    ObjectDetector,
)
from app.application.effects import (
    CaughtSegment,
    SegmentAbilities,
    WiredSegment,
    catch_segment_failures,
)
from app.application.policies import SegmentationPolicy
from app.application.use_cases.segment import segment


def wire_segment(
    detector: ObjectDetector,
    refiner: MaskRefiner,
    dilator: MaskDilator,
    policy: SegmentationPolicy,
) -> CaughtSegment:
    """Supply every capability and turn failures into returned values.

    The single place ``stateless`` composition happens, so the bootstrap
    constructs adapters without importing the effect library at all.

    :param detector: The detection capability.
    :type detector: app.application.capabilities.ObjectDetector
    :param refiner: The mask refinement capability.
    :type refiner: app.application.capabilities.MaskRefiner
    :param dilator: The mask dilation capability.
    :type dilator: app.application.capabilities.MaskDilator
    :param policy: Segmentation tuning.
    :type policy: app.application.policies.SegmentationPolicy
    :returns: The use case, ready to call and returning typed outcomes.
    :rtype: CaughtSegment
    """
    # pyright cannot see that supply() eliminates the Need abilities,
    # because the match is structural rather than nominal.
    handler: Handler[SegmentAbilities] = supply(
        detector, refiner, dilator, policy
    )
    supplied: WiredSegment = handler(
        segment
    )  # pyright: ignore[reportArgumentType]
    return catch_segment_failures(supplied)


__all__: list[str] = ["wire_segment"]
