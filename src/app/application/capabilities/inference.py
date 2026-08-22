"""Capabilities served by a model on an accelerator."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from app.domain import (
    MaskArray,
    PersonPayload,
    PixelBox,
    Prompt,
    SourceImage,
)


@runtime_checkable
class ObjectDetector(Protocol):
    """Locates the regions of an image matching a text prompt."""

    @abstractmethod
    def detect(
        self, image: SourceImage, prompt: Prompt
    ) -> tuple[tuple[PixelBox, ...], tuple[float, ...]]:
        """Return the boxes matching ``prompt`` and their confidences.

        :param image: Source image.
        :type image: app.domain.SourceImage
        :param prompt: Validated prompt.
        :type prompt: app.domain.Prompt
        :returns: ``(boxes, scores)``, index-aligned and possibly empty.
        :rtype: tuple[tuple[app.domain.PixelBox, ...],
            tuple[float, ...]]
        :raises app.domain.errors.DeviceExhausted: If the accelerator
            runs out of memory for this input.
        """


@runtime_checkable
class MaskRefiner(Protocol):
    """Turns coarse boxes into pixel-accurate masks."""

    @abstractmethod
    def refine(
        self, image: SourceImage, boxes: tuple[PixelBox, ...]
    ) -> tuple[tuple[MaskArray, ...], tuple[float, ...]]:
        """Return one mask per box, and how good each one is.

        Per box rather than merged: the caller decides whether to union
        them, and it cannot un-merge what this returns already fused.

        :param image: Source image.
        :type image: app.domain.SourceImage
        :param boxes: Non-empty boxes to segment.
        :type boxes: tuple[app.domain.PixelBox, ...]
        :returns: ``(masks, mask_scores)``, both index-aligned with
            ``boxes``.
        :rtype: tuple[tuple[app.domain.MaskArray, ...],
            tuple[float, ...]]
        :raises app.domain.errors.DeviceExhausted: If the accelerator
            runs out of memory for this input.
        """


@runtime_checkable
class FaceAnalyser(Protocol):
    """Summarises the faces present in an image."""

    @abstractmethod
    def analyse(self, image: SourceImage) -> PersonPayload:
        """Return the gender codes and adulthood of every detected face.

        :param image: Source image.
        :type image: app.domain.SourceImage
        :returns: The face summary; empty when no face is found.
        :rtype: app.domain.PersonPayload
        :raises app.domain.errors.FaceAnalysisUnavailable: If the
            optional face-analysis extra is not installed.
        """


@runtime_checkable
class MaskDilator(Protocol):
    """Grows a mask by a share of its own dimensions."""

    @abstractmethod
    def dilate(self, mask: MaskArray, percentage: float) -> MaskArray:
        """Return ``mask`` grown by ``percentage`` of its dimensions.

        :param mask: Mask to grow.
        :type mask: app.domain.MaskArray
        :param percentage: Kernel radius as a share of width and height.
        :type percentage: float
        :returns: The dilated mask, same shape as the input.
        :rtype: app.domain.MaskArray
        """


__all__: list[str] = [
    "FaceAnalyser",
    "MaskDilator",
    "MaskRefiner",
    "ObjectDetector",
]
