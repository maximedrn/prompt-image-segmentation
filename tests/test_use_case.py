"""The use case, driven entirely by fake capabilities.

No model, no device, no monkeypatching: the whole orchestration runs
against hand-written stand-ins supplied through ``stateless``. If this
needs a patch to work, the dependency injection did not land.
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
from app.application.policies import (
    ResolvedOptions,
    SegmentationPolicy,
    SegmentOptions,
    resolve_options,
)
from app.application.use_cases.segment import segment
from app.domain import (
    AgeBand,
    AgeRange,
    DeviceExhausted,
    InvalidPrompt,
    MaskArray,
    MaskValues,
    NoDetection,
    PersonRules,
    PixelBox,
    Prompt,
    SegmentedImage,
    SegmentRegion,
    SourceImage,
    age_range,
    binarize,
    certainly_adult,
)
from app.infrastructure.imaging.imaging import encode_png
from app.interfaces.http.schemas import RegionSchema, SegmentSchema

# The fakes implement capability protocols, which the bootstrap supplies
# as instances, so their methods stay methods.
# pylint: disable=no-self-use
# pytest collects the test functions by name; nothing in the module
# calls them, which is what ``allow-global-unused-variables`` flags.
# pylint: disable=unused-variable

IMAGE_SIZE: Final[int] = 100
PROMPT: Final[str] = "dog"
BOX: Final[PixelBox] = PixelBox(left=10.0, top=10.0, right=50.0, bottom=50.0)
OTHER_BOX: Final[PixelBox] = PixelBox(
    left=10.0, top=30.0, right=50.0, bottom=70.0
)
DETECTION_SCORE: Final[float] = 0.9
MASK_SCORE: Final[float] = 0.8
#: Rows between two fake masks, so a union is visibly larger.
MASK_OFFSET: Final[int] = 20
RELIABILITY: Final[float] = 0.4
#: What a caller gets when it asks for nothing in particular.
DEFAULT_OPTIONS: Final[SegmentOptions] = SegmentOptions()
#: Two boxes in, two regions out, when the caller asks for a split.
EXPECTED_SPLIT_REGIONS: Final[int] = 2
#: The band that straddles the adult threshold.
TEEN_YEARS: Final[tuple[int, int]] = (10, 19)


@final
class FakeDetector:
    """Returns a fixed set of boxes with a fixed confidence."""

    def __init__(self, boxes: tuple[PixelBox, ...]) -> None:
        """Record what this detector will always return.

        :param boxes: Boxes to hand back.
        :type boxes: tuple[app.domain.PixelBox, ...]
        """
        self._boxes: tuple[PixelBox, ...] = boxes

    def detect(
        self, image: SourceImage, prompt: Prompt
    ) -> tuple[tuple[PixelBox, ...], tuple[float, ...]]:
        """Return the recorded boxes.

        :param image: Ignored.
        :type image: app.domain.SourceImage
        :param prompt: Ignored.
        :type prompt: app.domain.Prompt
        :returns: The recorded boxes and their scores.
        :rtype: tuple[tuple[app.domain.PixelBox, ...],
            tuple[float, ...]]
        """
        del image, prompt
        return self._boxes, tuple(DETECTION_SCORE for _ in self._boxes)


@final
class FakeRefiner:
    """Paints one rectangle per box, offset so they stay distinct."""

    def refine(
        self, image: SourceImage, boxes: tuple[PixelBox, ...]
    ) -> tuple[tuple[MaskArray, ...], tuple[float, ...]]:
        """Return one mask per box and a fixed quality score.

        :param image: Used only for its dimensions.
        :type image: app.domain.SourceImage
        :param boxes: One mask is painted per entry.
        :type boxes: tuple[app.domain.PixelBox, ...]
        :returns: The masks and one score per box.
        :rtype: tuple[tuple[app.domain.MaskArray, ...],
            tuple[float, ...]]
        """
        masks: list[MaskArray] = []
        for index in range(len(boxes)):
            mask: MaskArray = np.zeros(
                (image.height, image.width), dtype=np.uint8
            )
            offset: int = index * MASK_OFFSET
            mask[10 + offset : 50 + offset, 10:50] = MaskValues.foreground
            masks.append(mask)
        return tuple(masks), tuple(MASK_SCORE for _ in boxes)


@final
class ExhaustedRefiner:
    """Stands in for an accelerator that ran out of memory."""

    def refine(
        self, image: SourceImage, boxes: tuple[PixelBox, ...]
    ) -> tuple[tuple[MaskArray, ...], tuple[float, ...]]:
        """Always fail the way the real adapter does.

        :param image: Ignored.
        :type image: app.domain.SourceImage
        :param boxes: Ignored.
        :type boxes: tuple[app.domain.PixelBox, ...]
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
        :type mask: app.domain.MaskArray
        :param percentage: Ignored.
        :type percentage: float
        :returns: The same mask.
        :rtype: app.domain.MaskArray
        """
        del percentage
        return mask


def _run(
    detector: ObjectDetector,
    refiner: MaskRefiner,
    options: SegmentOptions = DEFAULT_OPTIONS,
) -> SegmentedImage | SegmentFailure:
    """Wire the fakes and run one segmentation to completion.

    :param detector: Object satisfying the detector capability.
    :type detector: app.application.capabilities.ObjectDetector
    :param refiner: Object satisfying the refiner capability.
    :type refiner: app.application.capabilities.MaskRefiner
    :param options: What this call asks of the segmentation.
    :type options: app.application.policies.SegmentOptions
    :returns: The result, or the typed failure that stopped it.
    :rtype: app.domain.SegmentedImage
        | app.application.effects.SegmentFailure
    """
    policy: SegmentationPolicy = SegmentationPolicy(
        mask_padding_percentage=0.0,
        dilation_percentage=0.0,
        reliability_threshold=RELIABILITY,
        minimum_confidence=0.0,
    )
    handler: Handler[SegmentAbilities] = supply(
        detector, refiner, PassthroughDilator(), policy
    )
    wired: CaughtSegment = catch_segment_failures(handler(segment))
    image: SourceImage = SourceImage(
        pixels=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    )
    return run(wired(image, Prompt.parse(PROMPT), None, options))


def test_success_path_needs_no_model() -> None:
    """The orchestration runs end to end against stand-ins."""
    outcome: SegmentedImage | SegmentFailure = _run(
        FakeDetector((BOX,)), FakeRefiner()
    )
    assert isinstance(outcome, SegmentedImage)
    assert outcome.confidence == pytest.approx(DETECTION_SCORE * MASK_SCORE)
    assert outcome.reliable is True
    assert len(outcome.regions) == 1
    region: SegmentRegion = outcome.regions[0]
    assert region.bbox.width > 0 and region.bbox.height > 0
    assert region.image is not None


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


def test_masks_are_merged_unless_a_split_is_asked_for() -> None:
    """One region by default, one per detection on request."""
    detector: FakeDetector = FakeDetector((BOX, OTHER_BOX))

    merged: SegmentedImage | SegmentFailure = _run(detector, FakeRefiner())
    assert isinstance(merged, SegmentedImage)
    assert len(merged.regions) == 1
    assert len(merged.detections) == EXPECTED_SPLIT_REGIONS

    split: SegmentedImage | SegmentFailure = _run(
        detector, FakeRefiner(), SegmentOptions(split_masks=True)
    )
    assert isinstance(split, SegmentedImage)
    assert len(split.regions) == EXPECTED_SPLIT_REGIONS
    # The union covers both rectangles, so it cannot be the smaller one.
    assert int(binarize(merged.regions[0].mask).sum()) > int(
        binarize(split.regions[0].mask).sum()
    )


def test_uncropped_masks_keep_the_source_size_and_carry_no_image() -> None:
    """Declining the crop returns the mask alone, full size."""
    outcome: SegmentedImage | SegmentFailure = _run(
        FakeDetector((BOX,)), FakeRefiner(), SegmentOptions(crop=False)
    )
    assert isinstance(outcome, SegmentedImage)
    region: SegmentRegion = outcome.regions[0]
    assert region.image is None
    assert region.mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    # The box is still reported, so a caller can crop it itself.
    assert region.bbox.width > 0


def test_minimum_confidence_drops_the_weak_and_can_empty_the_result() -> None:
    """The floor filters detections, and refuses when none survive."""
    kept: SegmentedImage | SegmentFailure = _run(
        FakeDetector((BOX,)),
        FakeRefiner(),
        SegmentOptions(minimum_confidence=DETECTION_SCORE * MASK_SCORE),
    )
    assert isinstance(kept, SegmentedImage)

    dropped: SegmentedImage | SegmentFailure = _run(
        FakeDetector((BOX,)),
        FakeRefiner(),
        SegmentOptions(minimum_confidence=1.0),
    )
    assert isinstance(dropped, NoDetection)


def test_request_options_override_the_configured_defaults() -> None:
    """A blank option falls back to the policy, a set one wins."""
    policy: SegmentationPolicy = SegmentationPolicy(
        mask_padding_percentage=5.0,
        dilation_percentage=1.0,
        reliability_threshold=RELIABILITY,
        minimum_confidence=0.25,
    )
    fallback: ResolvedOptions = resolve_options(SegmentOptions(), policy)
    assert fallback.minimum_confidence == pytest.approx(0.25)
    assert fallback.padding_percentage == pytest.approx(5.0)
    assert fallback.crop is True

    override: ResolvedOptions = resolve_options(
        SegmentOptions(minimum_confidence=0.9, crop=False), policy
    )
    assert override.minimum_confidence == pytest.approx(0.9)
    assert override.dilation_percentage == pytest.approx(1.0)
    assert override.crop is False


def test_age_bands_carry_inclusive_numeric_bounds() -> None:
    """Every band reports the ages it can contain, both ends included."""
    assert age_range(AgeBand.TEEN) == TEEN_YEARS
    assert not age_range(AgeBand.INFANT).youngest
    # The lower bound is what decides adulthood, so it has to straddle.
    teen: AgeRange = age_range(AgeBand.TEEN)
    assert teen.youngest < PersonRules.adult_age < teen.oldest


def test_wire_shape_mirrors_what_the_caller_asked_for() -> None:
    """The response carries one region per mask, image only if cropped."""
    outcome: SegmentedImage | SegmentFailure = _run(
        FakeDetector((BOX, OTHER_BOX)),
        FakeRefiner(),
        SegmentOptions(split_masks=True, crop=False),
    )
    assert isinstance(outcome, SegmentedImage)

    body: SegmentSchema = SegmentSchema.of(
        result=outcome,
        prompt=PROMPT,
        segmenter="sam_dino",
        regions=RegionSchema.of_all(outcome.regions, encode_png),
    )
    # Asserted on the model rather than on a dump of it: the schema is
    # the contract, and reading it by attribute is what checks it.
    assert len(body.regions) == EXPECTED_SPLIT_REGIONS
    region: RegionSchema
    for region in body.regions:
        assert region.image is None
        assert region.mask
        assert region.bbox.width > 0
