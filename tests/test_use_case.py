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
from stateless.need import supply

from app.application.effects import catch_segment_failures
from app.application.policies import SegmentationPolicy
from app.application.use_cases.segment import segment
from app.domain import (
    MASK,
    DeviceExhausted,
    NoDetection,
    PixelBox,
    Prompt,
    SegmentedImage,
    SourceImage,
)

IMAGE_SIZE: Final[int] = 100
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
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        """Return a rectangular mask and a fixed quality score.

        :param image: Used only for its dimensions.
        :type image: app.domain.models.SourceImage
        :param boxes: Used only for its length.
        :type boxes: tuple[app.domain.models.PixelBox, ...]
        :returns: The mask and one score per box.
        :rtype: tuple[numpy.ndarray, tuple[float, ...]]
        """
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[10:50, 10:50] = MASK.foreground
        return mask, tuple(MASK_SCORE for _ in boxes)


@final
class ExhaustedRefiner:
    """Stands in for an accelerator that ran out of memory."""

    def refine(
        self, image: SourceImage, boxes: tuple[PixelBox, ...]
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        """Always fail the way the real adapter does.

        :param image: Ignored.
        :type image: app.domain.models.SourceImage
        :param boxes: Ignored.
        :type boxes: tuple[app.domain.models.PixelBox, ...]
        :returns: Never returns.
        :rtype: tuple[numpy.ndarray, tuple[float, ...]]
        :raises app.domain.errors.DeviceExhausted: Always.
        """
        del image, boxes
        raise DeviceExhausted(detail="simulated")


@final
class PassthroughDilator:
    """Leaves the mask untouched, so assertions stay exact."""

    def dilate(self, mask: np.ndarray, percentage: float) -> np.ndarray:
        """Return the mask unchanged.

        :param mask: Mask to pass through.
        :type mask: numpy.ndarray
        :param percentage: Ignored.
        :type percentage: float
        :returns: The same mask.
        :rtype: numpy.ndarray
        """
        del percentage
        return mask


def _run(detector: object, refiner: object) -> object:
    """Wire the fakes and run one segmentation to completion.

    :param detector: Object satisfying the detector capability.
    :type detector: object
    :param refiner: Object satisfying the refiner capability.
    :type refiner: object
    :returns: The result, or the typed failure that stopped it.
    :rtype: object
    """
    policy = SegmentationPolicy(
        mask_padding_percentage=0.0,
        dilation_percentage=0.0,
        reliability_threshold=0.4,
    )
    handler = supply(detector, refiner, PassthroughDilator(), policy)
    wired = catch_segment_failures(handler(segment))
    image = SourceImage(
        pixels=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    )
    return run(wired(image, Prompt.parse("dog"), None))


def test_success_path_needs_no_model() -> None:
    """The orchestration runs end to end against stand-ins."""
    outcome = _run(FakeDetector((BOX,)), FakeRefiner())
    assert isinstance(outcome, SegmentedImage)
    assert outcome.confidence == pytest.approx(DETECTION_SCORE * MASK_SCORE)
    assert outcome.reliable is True
    assert outcome.bbox.width > 0 and outcome.bbox.height > 0


def test_no_detection_is_returned_not_raised() -> None:
    """An empty detection comes back as a value the caller matches on."""
    outcome = _run(FakeDetector(()), FakeRefiner())
    assert isinstance(outcome, NoDetection)
    assert outcome.prompt == "dog"


def test_adapter_failure_is_returned_not_raised() -> None:
    """An adapter's typed failure travels the error channel as a value."""
    outcome = _run(FakeDetector((BOX,)), ExhaustedRefiner())
    assert isinstance(outcome, DeviceExhausted)


def test_prompt_validation_happens_at_the_boundary() -> None:
    """A blank prompt never reaches a capability."""
    from app.domain import InvalidPrompt

    with pytest.raises(InvalidPrompt):
        Prompt.parse("   ")
