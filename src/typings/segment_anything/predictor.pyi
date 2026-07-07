"""High-level SAM inference wrapper."""

# pylint: disable=locally-disabled
# pylint: disable=suppressed-message,useless-suppression
# pylint: disable=unused-variable,unused-argument

from numpy import uint8
from numpy.typing import NDArray
from torch import Tensor

from segment_anything.modeling.sam import Sam
from segment_anything.utils.transforms import ResizeLongestSide

class SamPredictor:
    """Cache one image and answer prompt-based mask queries."""

    transform: ResizeLongestSide

    def __init__(self, sam_model: Sam) -> None:
        """Bind the predictor to a loaded :class:`Sam` model."""

    def set_image(
        self,
        image: NDArray[uint8],
        image_format: str = ...,
    ) -> None:
        """Encode ``image`` and cache the embedding for later queries."""

    def predict_torch(  # pylint: disable=too-many-positional-arguments
        self,
        point_coords: Tensor | None,
        point_labels: Tensor | None,
        boxes: Tensor | None = ...,
        mask_input: Tensor | None = ...,
        multimask_output: bool = ...,
        return_logits: bool = ...,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(masks, iou_predictions, low_res_masks)``."""
