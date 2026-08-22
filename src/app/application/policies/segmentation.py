"""Tuning of one segmentation: the operator's, and the caller's."""

from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True, slots=True)
class SegmentationPolicy:
    """Tuning the segmentation use case applies to its own output."""

    mask_padding_percentage: float
    dilation_percentage: float
    reliability_threshold: float
    minimum_confidence: float


@final
@dataclass(frozen=True, slots=True)
class SegmentOptions:
    """What one caller asked of one segmentation.

    Every numeric field is optional: ``None`` means "whatever the
    operator configured", so the policy stays the default and the
    request only overrides what it names. The two flags shape the
    response rather than tune it, so they carry their own defaults.
    """

    minimum_confidence: float | None = None
    dilation_percentage: float | None = None
    padding_percentage: float | None = None
    split_masks: bool = False
    crop: bool = True


@final
@dataclass(frozen=True, slots=True)
class ResolvedOptions:
    """A caller's options with every blank filled from the policy."""

    minimum_confidence: float
    dilation_percentage: float
    padding_percentage: float
    split_masks: bool
    crop: bool


def resolve_options(
    options: SegmentOptions, policy: SegmentationPolicy
) -> ResolvedOptions:
    """Merge a request's options over the operator's defaults.

    :param options: What the caller asked for, blanks included.
    :type options: SegmentOptions
    :param policy: Configured defaults.
    :type policy: SegmentationPolicy
    :returns: The options the use case will act on.
    :rtype: ResolvedOptions
    """
    return ResolvedOptions(
        minimum_confidence=(
            policy.minimum_confidence
            if options.minimum_confidence is None
            else options.minimum_confidence
        ),
        dilation_percentage=(
            policy.dilation_percentage
            if options.dilation_percentage is None
            else options.dilation_percentage
        ),
        padding_percentage=(
            policy.mask_padding_percentage
            if options.padding_percentage is None
            else options.padding_percentage
        ),
        split_masks=options.split_masks,
        crop=options.crop,
    )


@final
@dataclass(frozen=True, slots=True)
class DetectionPolicy:
    """Thresholds the detector adapter filters its own output with."""

    score_threshold: float
    text_threshold: float


@final
@dataclass(frozen=True, slots=True)
class FacePolicy:
    """Tuning the face detector applies to its own output."""

    score_threshold: float


__all__: list[str] = [
    "DetectionPolicy",
    "FacePolicy",
    "ResolvedOptions",
    "SegmentOptions",
    "SegmentationPolicy",
    "resolve_options",
]
