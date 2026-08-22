"""Segment an image from a text prompt."""

from typing import Final

from stateless.effect import throw, throws

from app.application.capabilities import (
    MaskDilator,
    MaskRefiner,
    ObjectDetector,
)
from app.application.effects import (
    Detect,
    Refine,
    SegmentEffect,
    need_mask_dilator,
    need_mask_refiner,
    need_object_detector,
    need_segmentation_policy,
)
from app.application.policies import (
    ResolvedOptions,
    SegmentationPolicy,
    SegmentOptions,
    resolve_options,
)
from app.domain import (
    BBox,
    Detection,
    DeviceExhausted,
    MaskArray,
    NoDetection,
    PersonPayload,
    PixelBox,
    Prompt,
    SegmentedImage,
    SegmentRegion,
    SourceImage,
    above_confidence,
    bbox_from_mask,
    clamp_score,
    crop_to_bbox,
    is_reliable,
    union_masks,
)


def _score_detections(
    boxes: tuple[PixelBox, ...],
    detection_scores: tuple[float, ...],
    mask_scores: tuple[float, ...],
) -> tuple[Detection, ...]:
    """Zip boxes with their two scores into domain detections.

    Pure: no requirement, no recoverable failure, so it stays an ordinary
    function rather than an effect.

    :param boxes: Boxes the detector returned.
    :type boxes: tuple[app.domain.PixelBox, ...]
    :param detection_scores: Detector confidence per box.
    :type detection_scores: tuple[float, ...]
    :param mask_scores: Refiner self-assessed IoU per box.
    :type mask_scores: tuple[float, ...]
    :returns: One scored detection per box, truncated to the shortest
        input so a length mismatch degrades instead of raising.
    :rtype: tuple[app.domain.Detection, ...]
    """
    return tuple(
        Detection(
            box=box,
            detection_score=clamp_score(detection_score),
            mask_score=clamp_score(mask_score),
        )
        for box, detection_score, mask_score in zip(
            boxes, detection_scores, mask_scores
        )
    )


#: What a caller gets when it asks for nothing in particular.
_DEFAULT_OPTIONS: Final[SegmentOptions] = SegmentOptions()


def _region(
    mask: MaskArray,
    image: SourceImage,
    detection: Detection,
    dilator: MaskDilator,
    options: ResolvedOptions,
) -> SegmentRegion:
    """Shape one mask the way the caller asked for it.

    :param mask: Full-size mask for this region.
    :type mask: app.domain.MaskArray
    :param image: Source image the mask belongs to.
    :type image: app.domain.SourceImage
    :param detection: Detection this region answers for.
    :type detection: app.domain.Detection
    :param dilator: The dilation capability.
    :type dilator: app.application.capabilities.MaskDilator
    :param options: Resolved per-request options.
    :type options: app.application.policies.ResolvedOptions
    :returns: The region, cropped and dilated as requested.
    :rtype: app.domain.SegmentRegion
    """
    bbox: BBox = bbox_from_mask(mask, options.padding_percentage)
    if not options.crop:
        return SegmentRegion(
            bbox=bbox,
            mask=dilator.dilate(mask, options.dilation_percentage),
            image=None,
            detection=detection,
        )
    return SegmentRegion(
        bbox=bbox,
        mask=dilator.dilate(
            crop_to_bbox(mask, bbox), options.dilation_percentage
        ),
        image=crop_to_bbox(image.pixels, bbox),
        detection=detection,
    )


def _regions(
    retained: tuple[tuple[MaskArray, Detection], ...],
    image: SourceImage,
    dilator: MaskDilator,
    options: ResolvedOptions,
) -> tuple[SegmentRegion, ...]:
    """Shape the retained masks into the regions the caller asked for.

    :param retained: Surviving masks paired with their detection.
    :type retained: tuple[tuple[app.domain.MaskArray,
        app.domain.Detection], ...]
    :param image: Source image the masks belong to.
    :type image: app.domain.SourceImage
    :param dilator: The dilation capability.
    :type dilator: app.application.capabilities.MaskDilator
    :param options: Resolved per-request options.
    :type options: app.application.policies.ResolvedOptions
    :returns: One region per detection when split, otherwise a single
        region for their union.
    :rtype: tuple[app.domain.SegmentRegion, ...]
    """
    if options.split_masks:
        return tuple(
            _region(mask, image, detection, dilator, options)
            for mask, detection in retained
        )
    # The union answers for its weakest member: one poor mask in the
    # merge contaminates all of it.
    weakest: Detection = min(
        (detection for _, detection in retained),
        key=lambda detection: detection.confidence,
    )
    return (
        _region(
            union_masks(tuple(mask for mask, _ in retained)),
            image,
            weakest,
            dilator,
            options,
        ),
    )


def segment(
    image: SourceImage,
    prompt: Prompt,
    person: PersonPayload | None = None,
    options: SegmentOptions = _DEFAULT_OPTIONS,
) -> SegmentEffect:
    """Detect, refine, filter, shape and score one image.

    :param image: Decoded source image.
    :type image: app.domain.SourceImage
    :param prompt: Validated prompt.
    :type prompt: app.domain.Prompt
    :param person: Face summary to attach, when one was requested.
    :type person: app.domain.PersonPayload | None
    :param options: What this caller asked of this segmentation.
    :type options: app.application.policies.SegmentOptions
    :returns: The effect producing the shaped, scored result.
    :rtype: app.application.effects.SegmentEffect
    """
    detector: ObjectDetector = yield from need_object_detector()
    refiner: MaskRefiner = yield from need_mask_refiner()
    dilator: MaskDilator = yield from need_mask_dilator()
    policy: SegmentationPolicy = yield from need_segmentation_policy()
    resolved: ResolvedOptions = resolve_options(options, policy)

    detect: Detect = throws(DeviceExhausted)(detector.detect)
    refine: Refine = throws(DeviceExhausted)(refiner.refine)

    boxes, detection_scores = yield from detect(image, prompt)
    if not boxes:
        yield from throw(NoDetection(prompt=prompt.text))

    masks, mask_scores = yield from refine(image, boxes)
    scored: tuple[Detection, ...] = _score_detections(
        boxes, detection_scores, mask_scores
    )

    # Masks are index-aligned with `scored`, so the rule decides which
    # detections survive and the pairing carries their masks along.
    kept: frozenset[Detection] = frozenset(
        above_confidence(scored, resolved.minimum_confidence)
    )
    retained: tuple[tuple[MaskArray, Detection], ...] = tuple(
        (mask, detection)
        for mask, detection in zip(masks, scored)
        if detection in kept
    )
    if not retained:
        yield from throw(NoDetection(prompt=prompt.text))

    detections: tuple[Detection, ...] = tuple(
        detection for _, detection in retained
    )
    confidence: float = min(detection.confidence for detection in detections)
    return SegmentedImage(
        regions=_regions(retained, image, dilator, resolved),
        detections=detections,
        confidence=confidence,
        reliable=is_reliable(confidence, policy.reliability_threshold),
        person=person,
    )


__all__: list[str] = ["segment"]
