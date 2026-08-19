"""Capability interfaces.

Each Protocol describes what a use case needs, not what an SDK offers
(``SKILL.md`` section 3). They are the seam that lets the application run
against a fake in a test and against a 300 MB model in production, with
no monkeypatching either way.

Implementations live in ``app.infrastructure`` and are the only place
allowed to import ``transformers`` or ``facelib``.
"""

# The ``...`` bodies are required: pyright treats a Protocol method
# with only a docstring as falling off the end without returning a
# value. pylint then reads the ellipsis as the whole body and calls
# the documented return redundant. pyright's complaint is the
# substantive one, so the ellipsis stays and pylint yields here.
# pylint: disable=unnecessary-ellipsis,redundant-returns-doc
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

    def detect(
        self, image: SourceImage, prompt: Prompt
    ) -> tuple[tuple[PixelBox, ...], tuple[float, ...]]:
        """Return the boxes matching ``prompt`` and their confidences.

        :param image: Source image.
        :type image: app.domain.models.SourceImage
        :param prompt: Validated prompt.
        :type prompt: app.domain.models.Prompt
        :returns: ``(boxes, scores)``, index-aligned and possibly empty.
        :rtype: tuple[tuple[app.domain.models.PixelBox, ...],
            tuple[float, ...]]
        :raises app.domain.errors.DeviceExhausted: If the accelerator
            runs out of memory for this input.
        """
        ...


@runtime_checkable
class MaskRefiner(Protocol):
    """Turns coarse boxes into a pixel-accurate union mask."""

    def refine(
        self, image: SourceImage, boxes: tuple[PixelBox, ...]
    ) -> tuple[MaskArray, tuple[float, ...]]:
        """Return the union mask over ``boxes`` and its per-box quality.

        :param image: Source image.
        :type image: app.domain.models.SourceImage
        :param boxes: Non-empty boxes to segment.
        :type boxes: tuple[app.domain.models.PixelBox, ...]
        :returns: ``(union_mask, mask_scores)``, the scores index-aligned
            with ``boxes``.
        :rtype: tuple[app.domain.models.MaskArray, tuple[float, ...]]
        :raises app.domain.errors.DeviceExhausted: If the accelerator
            runs out of memory for this input.
        """
        ...


@runtime_checkable
class FaceAnalyser(Protocol):
    """Summarises the faces present in an image."""

    def analyse(self, image: SourceImage) -> PersonPayload:
        """Return the gender codes and adulthood of every detected face.

        :param image: Source image.
        :type image: app.domain.models.SourceImage
        :returns: The face summary; empty when no face is found.
        :rtype: app.domain.models.PersonPayload
        :raises app.domain.errors.FaceAnalysisUnavailable: If the
            optional face-analysis extra is not installed.
        """
        ...


@runtime_checkable
class MaskDilator(Protocol):
    """Grows a mask by a share of its own dimensions."""

    def dilate(self, mask: MaskArray, percentage: float) -> MaskArray:
        """Return ``mask`` grown by ``percentage`` of its dimensions.

        :param mask: Mask to grow.
        :type mask: app.domain.models.MaskArray
        :param percentage: Kernel radius as a share of width and height.
        :type percentage: float
        :returns: The dilated mask, same shape as the input.
        :rtype: app.domain.models.MaskArray
        """
        ...


__all__: list[str] = [
    "FaceAnalyser",
    "MaskDilator",
    "MaskRefiner",
    "ObjectDetector",
]
