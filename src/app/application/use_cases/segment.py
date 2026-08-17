"""Segment an image from a text prompt.

The one orchestration in this application: it coordinates three
capabilities, fails in two recoverable ways, and returns a cropped
result. Everything it delegates to is either a capability or a pure
domain rule, so the whole use case runs against fakes with no model
loaded (``SKILL.md`` section 35).
"""

from stateless.effect import throw, throws

from app.application.capabilities import (
    MaskDilator,
    MaskRefiner,
    ObjectDetector,
)
from app.application.effects import (
    SegmentEffect,
    need_mask_dilator,
    need_mask_refiner,
    need_object_detector,
    need_segmentation_policy,
)
from app.application.policies import SegmentationPolicy
from app.domain import (
    BBox,
    DeviceExhausted,
    Detection,
    MaskArray,
    NoDetection,
    PersonPayload,
    PixelBox,
    Prompt,
    SegmentedImage,
    SourceImage,
    bbox_from_mask,
    clamp_score,
    crop_to_bbox,
    is_reliable,
)


def _score_detections(
    boxes: tuple[PixelBox, ...],
    detection_scores: tuple[float, ...],
    mask_scores: tuple[float, ...],
) -> tuple[Detection, ...]:
    """Zip boxes with their two scores into domain detections.

    Pure: no requirement, no recoverable failure, so it stays an ordinary
    function rather than an effect (``SKILL.md`` section 39).

    :param boxes: Boxes the detector returned.
    :type boxes: tuple[app.domain.models.PixelBox, ...]
    :param detection_scores: Detector confidence per box.
    :type detection_scores: tuple[float, ...]
    :param mask_scores: Refiner self-assessed IoU per box.
    :type mask_scores: tuple[float, ...]
    :returns: One scored detection per box, truncated to the shortest
        input so a length mismatch degrades instead of raising.
    :rtype: tuple[app.domain.models.Detection, ...]
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


def segment(
    image: SourceImage,
    prompt: Prompt,
    person: PersonPayload | None = None,
) -> SegmentEffect:
    """Detect, refine, crop and score one image.

    :param image: Decoded source image.
    :type image: app.domain.models.SourceImage
    :param prompt: Validated prompt.
    :type prompt: app.domain.models.Prompt
    :param person: Face summary to attach, when one was requested.
    :type person: app.domain.models.PersonPayload | None
    :returns: The effect producing the cropped, scored result.
    :rtype: app.application.effects.SegmentEffect
    """
    detector: ObjectDetector = yield from need_object_detector()
    refiner: MaskRefiner = yield from need_mask_refiner()
    dilator: MaskDilator = yield from need_mask_dilator()
    policy: SegmentationPolicy = yield from need_segmentation_policy()

    detect = throws(DeviceExhausted)(detector.detect)
    refine = throws(DeviceExhausted)(refiner.refine)

    boxes, detection_scores = yield from detect(image, prompt)
    if not boxes:
        yield from throw(NoDetection(prompt=prompt.text))

    mask, mask_scores = yield from refine(image, boxes)
    detections: tuple[Detection, ...] = _score_detections(
        boxes, detection_scores, mask_scores
    )

    bbox: BBox = bbox_from_mask(mask, policy.mask_padding_percentage)
    cropped_mask: MaskArray = dilator.dilate(
        crop_to_bbox(mask, bbox), policy.dilation_percentage
    )
    cropped_image: MaskArray = crop_to_bbox(image.pixels, bbox)

    confidence: float = (
        min(detection.confidence for detection in detections)
        if detections
        else clamp_score(0.0)
    )
    return SegmentedImage(
        bbox=bbox,
        cropped_mask=cropped_mask,
        cropped_image=cropped_image,
        detections=detections,
        confidence=confidence,
        reliable=is_reliable(confidence, policy.reliability_threshold),
        person=person,
    )


__all__: list[str] = ["segment"]
