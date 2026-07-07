"""Coordinate transforms shared by the SAM inference wrapper."""

# pylint: disable=locally-disabled
# pylint: disable=suppressed-message,useless-suppression
# pylint: disable=unused-variable,unused-argument

from torch import Tensor

class ResizeLongestSide:
    """Resize the longest side to ``target_length`` and back."""

    def __init__(self, target_length: int) -> None:
        """Store the target length used to scale boxes/points."""

    def apply_boxes_torch(
        self, boxes: Tensor, original_size: tuple[int, int]
    ) -> Tensor:
        """Rescale ``boxes`` from ``original_size`` to model coords."""
