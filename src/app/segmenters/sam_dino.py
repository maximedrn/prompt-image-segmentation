"""GroundingDINO + SAM ViT-H segmenter (default backend)."""

from typing import ClassVar, Final

from PIL.Image import Image
from numpy import maximum, uint8, zeros_like
from numpy.typing import NDArray
from segment_anything.utils.transforms import ResizeLongestSide
from torch import Tensor, no_grad

from app.config import get_device, get_settings
from app.core import NoDetectionError
from app.domain import SegmentationResult
from app.infrastructure.image_io import (
    pil_to_array,
    pil_to_dino_tensor,
    tensor_to_uint8_array,
)
from app.managers import ModelManager
from app.segmenters.base import Segmenter
from app.segmenters.factory import SEGMENTER_FACTORY

BACKEND_NAME: Final[str] = "sam_dino"


def _format_caption(prompt: str) -> str:
    """Normalize a caption to what GroundingDINO expects.

    Lowercased, stripped, and terminated with a trailing period.

    :param prompt: Raw user prompt.
    :type prompt: str
    :returns: Caption suitable for GroundingDINO inference.
    :rtype: str
    """
    formatted: str = prompt.lower().strip()
    if not formatted.endswith("."):
        formatted += "."
    return formatted


@SEGMENTER_FACTORY.register(BACKEND_NAME)
class SamDinoSegmenter(Segmenter):
    """GroundingDINO for boxes + SAM ViT-H for mask refinement."""

    name: ClassVar[str] = BACKEND_NAME

    def __init__(self) -> None:
        """Bind to the process-wide ModelManager and settings."""
        self._models: ModelManager = ModelManager()
        self._score_threshold: float = get_settings().detection_score_threshold

    def _detect_boxes(self, prompt: str, image: Image) -> Tensor:
        """Run GroundingDINO and filter boxes above the score threshold.

        :param prompt: Raw user prompt.
        :type prompt: str
        :param image: Source RGB image.
        :type image: PIL.Image.Image
        :returns: A ``Nx4`` tensor of centre-form boxes (``cxcywh``).
        :rtype: torch.Tensor
        """
        tensor: Tensor = pil_to_dino_tensor(image)
        caption: str = _format_caption(prompt)
        with no_grad():
            outputs = self._models.grounding_dino(
                tensor[None], captions=[caption]
            )
        logits: Tensor = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes: Tensor = outputs["pred_boxes"].cpu()[0]
        keep: Tensor = logits.max(dim=1)[0] > self._score_threshold
        return boxes[keep]

    @staticmethod
    def _boxes_to_pixels(boxes: Tensor, size: tuple[int, int]) -> Tensor:
        """Convert ``cxcywh`` normalized boxes to pixel-space xyxy.

        :param boxes: ``Nx4`` tensor in centre-form, normalized to
            ``[0, 1]``.
        :type boxes: torch.Tensor
        :param size: Image ``(width, height)`` in pixels.
        :type size: tuple[int, int]
        :returns: ``Nx4`` xyxy tensor in pixel coordinates.
        :rtype: torch.Tensor
        """
        width, height = size
        scale: Tensor = Tensor([width, height, width, height])
        pixels: Tensor = boxes * scale
        pixels[:, :2] -= pixels[:, 2:] / 2
        pixels[:, 2:] += pixels[:, :2]
        return pixels

    def segment(self, image: Image, prompt: str) -> SegmentationResult:
        """Run GroundingDINO -> SAM and return the union mask.

        :param image: Source RGB image.
        :type image: PIL.Image.Image
        :param prompt: Text prompt (dot- or comma-separated labels).
        :type prompt: str
        :returns: The union mask + detection count.
        :rtype: app.domain.segmentation.SegmentationResult
        :raises app.core.exceptions.NoDetectionError: If
            GroundingDINO returns no boxes above the score threshold.
        """
        size: tuple[int, int] = image.size
        boxes: Tensor = self._detect_boxes(prompt, image)
        if not boxes.size(0):
            raise NoDetectionError(prompt)
        pixel_boxes: Tensor = self._boxes_to_pixels(boxes, size).to(
            get_device()
        )
        transform: ResizeLongestSide = self._models.sam.transform
        transformed: Tensor = transform.apply_boxes_torch(pixel_boxes, size)
        image_array: NDArray[uint8] = pil_to_array(image)
        self._models.sam.set_image(image_array)
        masks: Tensor = self._models.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )[0]
        final: NDArray[uint8] = zeros_like(image_array[..., 0], dtype=uint8)
        for tensor_mask in masks:
            mask: NDArray[uint8] = tensor_to_uint8_array(tensor_mask)
            mask = (mask.squeeze() * 255).astype(uint8)
            final = maximum(final, mask)
        return SegmentationResult(mask=final, detections=int(boxes.size(0)))


__all__: list[str] = ["BACKEND_NAME", "SamDinoSegmenter"]
