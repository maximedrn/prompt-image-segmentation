"""Effect vocabulary for this application."""

from collections.abc import Callable
from typing import Never

from stateless.effect import Catch, Effect, Success, Try, catch
from stateless.need import Need, need

from app.application.capabilities import (
    FaceAnalyser,
    MaskDilator,
    MaskRefiner,
    ObjectDetector,
)
from app.application.policies import SegmentationPolicy, SegmentOptions
from app.domain import (
    DeviceExhausted,
    FaceAnalysisUnavailable,
    MaskArray,
    NoDetection,
    PersonPayload,
    PixelBox,
    Prompt,
    SegmentedImage,
    SourceImage,
)

type SegmentAbilities = (
    Need[ObjectDetector]
    | Need[MaskRefiner]
    | Need[MaskDilator]
    | Need[SegmentationPolicy]
)
"""What segmenting an image requires: three models and its own tuning."""

type SegmentFailure = NoDetection | DeviceExhausted
"""How segmenting an image can fail in a way a caller can act on."""

type SegmentEffect = Effect[SegmentAbilities, SegmentFailure, SegmentedImage]
"""Segment one image: needs models, may fail, yields a cropped result."""

type PersonAbilities = Need[FaceAnalyser]
"""What analysing faces requires."""

type PersonFailure = FaceAnalysisUnavailable
"""How analysing faces can fail recoverably."""

type PersonEffect = Effect[PersonAbilities, PersonFailure, PersonPayload]
"""Analyse the faces in one image."""


type Detect = Callable[
    [SourceImage, Prompt],
    Try[DeviceExhausted, tuple[tuple[PixelBox, ...], tuple[float, ...]]],
]
"""Detection with its device failure moved into the error channel."""

type Refine = Callable[
    [SourceImage, tuple[PixelBox, ...]],
    Try[DeviceExhausted, tuple[tuple[MaskArray, ...], tuple[float, ...]]],
]
"""Refinement with its device failure moved into the error channel."""


type WiredSegment = Callable[
    [SourceImage, Prompt, PersonPayload | None, SegmentOptions],
    Effect[Never, SegmentFailure, SegmentedImage],
]
"""The segmentation use case once every capability has been supplied."""

type CaughtSegment = Callable[
    [SourceImage, Prompt, PersonPayload | None, SegmentOptions],
    Success[SegmentedImage | SegmentFailure],
]
"""The same, with the error channel turned into a returned value."""


def catch_segment_failures(wired: WiredSegment) -> CaughtSegment:
    """Turn segmentation's error channel into an ordinary return value.

    The transport layer then maps the outcome with a ``match`` over
    concrete types instead of a ``try`` block, which is what keeps
    ``except`` out of the interfaces layer.

    Annotating the decorator as ``Catch[SegmentFailure]`` is what keeps
    the result union narrow: inferred, ``catch`` widens it to bare
    ``Exception``.

    :param wired: Use case with all capabilities already supplied.
    :type wired: WiredSegment
    :returns: The same callable, returning failures instead of yielding
        them.
    :rtype: CaughtSegment
    """
    handle: Catch[SegmentFailure] = catch(NoDetection, DeviceExhausted)
    caught: CaughtSegment = handle(wired)
    return caught


def need_object_detector() -> (
    Effect[Need[ObjectDetector], Never, ObjectDetector]
):
    """Request the object detector capability.

    :returns: An effect yielding the supplied detector.
    :rtype: stateless.effect.Effect
    """
    detector: ObjectDetector = yield from need(
        ObjectDetector  # type: ignore[type-abstract]
    )
    return detector


def need_mask_refiner() -> Effect[Need[MaskRefiner], Never, MaskRefiner]:
    """Request the mask refiner capability.

    :returns: An effect yielding the supplied refiner.
    :rtype: stateless.effect.Effect
    """
    refiner: MaskRefiner = yield from need(
        MaskRefiner  # type: ignore[type-abstract]
    )
    return refiner


def need_mask_dilator() -> Effect[Need[MaskDilator], Never, MaskDilator]:
    """Request the mask dilator capability.

    :returns: An effect yielding the supplied dilator.
    :rtype: stateless.effect.Effect
    """
    dilator: MaskDilator = yield from need(
        MaskDilator  # type: ignore[type-abstract]
    )
    return dilator


def need_face_analyser() -> Effect[Need[FaceAnalyser], Never, FaceAnalyser]:
    """Request the face analyser capability.

    :returns: An effect yielding the supplied analyser.
    :rtype: stateless.effect.Effect
    """
    analyser: FaceAnalyser = yield from need(
        FaceAnalyser  # type: ignore[type-abstract]
    )
    return analyser


def need_segmentation_policy() -> (
    Effect[Need[SegmentationPolicy], Never, SegmentationPolicy]
):
    """Request the segmentation tuning value.

    Concrete, so no concession is needed here.

    :returns: An effect yielding the supplied policy.
    :rtype: stateless.effect.Effect
    """
    policy: SegmentationPolicy = yield from need(SegmentationPolicy)
    return policy


__all__: list[str] = [
    "CaughtSegment",
    "Detect",
    "PersonAbilities",
    "PersonEffect",
    "PersonFailure",
    "Refine",
    "SegmentAbilities",
    "SegmentEffect",
    "SegmentFailure",
    "WiredSegment",
    "catch_segment_failures",
    "need_face_analyser",
    "need_mask_dilator",
    "need_mask_refiner",
    "need_object_detector",
    "need_segmentation_policy",
]
