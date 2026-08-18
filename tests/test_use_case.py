"""The use case, driven entirely by fake capabilities.

No model, no device, no monkeypatching: the whole orchestration runs
against hand-written stand-ins supplied through ``stateless``. If this
needs a patch to work, the dependency injection did not land
(``SKILL.md`` sections 35 and 36).
"""

from typing import Final, final

import numpy as np
import pytest
from stateless.effect import run
from stateless.handler import Handler
from stateless.need import supply

from app.application.capabilities import MaskRefiner, ObjectDetector
from app.application.effects import (
    CaughtSegment,
    SegmentAbilities,
    SegmentFailure,
    catch_segment_failures,
)
from app.application.policies import SegmentationPolicy
from app.application.use_cases.segment import segment
from app.domain import (
    MASK,
    AgeBand,
    DeviceExhausted,
    InvalidPrompt,
    MaskArray,
    NoDetection,
    PixelBox,
    Prompt,
    SegmentedImage,
    SourceImage,
    certainly_adult,
)

# The fakes implement capability protocols, which the bootstrap supplies
# as instances, so their methods stay methods.
# pylint: disable=no-self-use
# pytest collects the test functions by name; nothing in the module
# calls them, which is what ``allow-global-unused-variables`` flags.
# pylint: disable=unused-variable

IMAGE_SIZE: Final[int] = 100
PROMPT: Final[str] = "dog"
BOX: Final[PixelBox] = PixelBox(left=10.0, top=10.0, right=50.0, bottom=50.0)
DETECTION_SCORE: Final[float] = 0.9
MASK_SCORE: Final[float] = 0.8


@final
class FakeDetector:
    """Returns a fixed set of boxes with a fixed confidence."""

    def __init__(self, boxes: tuple[PixelBox, ...]) -> None:
        """Record what this detector will always return.

        :param boxes: Boxes to hand back.
        :type boxes: tuple[app.domain.models.PixelBox, ...]
        """
        self._boxes: tuple[PixelBox, ...] = boxes

    def detect(
        self, image: SourceImage, prompt: Prompt
    ) -> tuple[tuple[PixelBox, ...], tuple[float, ...]]:
        """Return the recorded boxes.

        :param image: Ignored.
        :type image: app.domain.models.SourceImage
        :param prompt: Ignored.
        :type prompt: app.domain.models.Prompt
        :returns: The recorded boxes and their scores.
        :rtype: tuple[tuple[app.domain.models.PixelBox, ...],
            tuple[float, ...]]
        """
        del image, prompt
        return self._boxes, tuple(DETECTION_SCORE for _ in self._boxes)


@final
class FakeRefiner:
    """Paints a fixed rectangle as the mask."""

    def refine(
        self, image: SourceImage, boxes: tuple[PixelBox, ...]
    ) -> tuple[MaskArray, tuple[float, ...]]:
        """Return a rectangular mask and a fixed quality score.

        :param image: Used only for its dimensions.
        :type image: app.domain.models.SourceImage
        :param boxes: Used only for its length.
        :type boxes: tuple[app.domain.models.PixelBox, ...]
        :returns: The mask and one score per box.
        :rtype: tuple[app.domain.models.MaskArray, tuple[float, ...]]
        """
        mask: MaskArray = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[10:50, 10:50] = MASK.foreground
        return mask, tuple(MASK_SCORE for _ in boxes)


@final
class ExhaustedRefiner:
    """Stands in for an accelerator that ran out of memory."""

    def refine(
        self, image: SourceImage, boxes: tuple[PixelBox, ...]
    ) -> tuple[MaskArray, tuple[float, ...]]:
        """Always fail the way the real adapter does.

        :param image: Ignored.
        :type image: app.domain.models.SourceImage
        :param boxes: Ignored.
        :type boxes: tuple[app.domain.models.PixelBox, ...]
        :raises app.domain.errors.DeviceExhausted: Always.
        """
        del image, boxes
        raise DeviceExhausted(detail="simulated")


@final
class PassthroughDilator:
    """Leaves the mask untouched, so assertions stay exact."""

    def dilate(self, mask: MaskArray, percentage: float) -> MaskArray:
        """Return the mask unchanged.

        :param mask: Mask to pass through.
        :type mask: app.domain.models.MaskArray
        :param percentage: Ignored.
        :type percentage: float
        :returns: The same mask.
        :rtype: app.domain.models.MaskArray
        """
        del percentage
        return mask


def _run(
    detector: ObjectDetector, refiner: MaskRefiner
) -> SegmentedImage | SegmentFailure:
    """Wire the fakes and run one segmentation to completion.

    :param detector: Object satisfying the detector capability.
    :type detector: app.application.capabilities.ObjectDetector
    :param refiner: Object satisfying the refiner capability.
    :type refiner: app.application.capabilities.MaskRefiner
    :returns: The result, or the typed failure that stopped it.
    :rtype: app.domain.models.SegmentedImage
        | app.application.effects.SegmentFailure
    """
    policy: SegmentationPolicy = SegmentationPolicy(
        mask_padding_percentage=0.0,
        dilation_percentage=0.0,
        reliability_threshold=0.4,
    )
    handler: Handler[SegmentAbilities] = supply(
        detector, refiner, PassthroughDilator(), policy
    )
    wired: CaughtSegment = catch_segment_failures(handler(segment))
    image: SourceImage = SourceImage(
        pixels=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    )
    return run(wired(image, Prompt.parse(PROMPT), None))


def test_success_path_needs_no_model() -> None:
    """The orchestration runs end to end against stand-ins."""
    outcome: SegmentedImage | SegmentFailure = _run(
        FakeDetector((BOX,)), FakeRefiner()
    )
    assert isinstance(outcome, SegmentedImage)
    assert outcome.confidence == pytest.approx(DETECTION_SCORE * MASK_SCORE)
    assert outcome.reliable is True
    assert outcome.bbox.width > 0 and outcome.bbox.height > 0


def test_no_detection_is_returned_not_raised() -> None:
    """An empty detection comes back as a value the caller matches on."""
    outcome: SegmentedImage | SegmentFailure = _run(
        FakeDetector(()), FakeRefiner()
    )
    assert isinstance(outcome, NoDetection)
    assert outcome.prompt == PROMPT


def test_adapter_failure_is_returned_not_raised() -> None:
    """An adapter's typed failure travels the error channel as a value."""
    outcome: SegmentedImage | SegmentFailure = _run(
        FakeDetector((BOX,)), ExhaustedRefiner()
    )
    assert isinstance(outcome, DeviceExhausted)


def test_prompt_validation_happens_at_the_boundary() -> None:
    """A blank prompt never reaches a capability."""
    with pytest.raises(InvalidPrompt):
        Prompt.parse("   ")


def test_age_band_spanning_the_threshold_never_certifies() -> None:
    """The band containing 18 must not certify adulthood.

    The estimator that replaced facelib classifies into ranges rather
    than years, so the band from ten to nineteen straddles the adult
    threshold. Certifying it would be a guess dressed as a fact.
    """
    assert certainly_adult(()) is True
    assert certainly_adult((AgeBand.TWENTIES,)) is True
    assert certainly_adult((AgeBand.TEEN,)) is False
    assert certainly_adult((AgeBand.FORTIES, AgeBand.CHILD)) is False
