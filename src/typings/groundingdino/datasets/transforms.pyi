"""Image transforms used by the GroundingDINO inference pipeline."""

# pylint: disable=locally-disabled
# pylint: disable=suppressed-message,useless-suppression
# pylint: disable=unused-variable

from PIL.Image import Image
from torch import Tensor

class _Transform:
    """Base class for callable image transforms."""

    def __call__(
        self,
        image: Image,
        target: dict[str, Tensor] | None = ...,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Apply the transform. Returns ``(tensor, updated_target)``."""

class Compose(_Transform):
    """Chain several transforms in order."""

    def __init__(self, transforms: list[_Transform]) -> None:
        """Wrap ``transforms`` into a single pipeline."""

class Normalize(_Transform):
    """Per-channel normalisation."""

    def __init__(self, mean: list[float], std: list[float]) -> None:
        """Store ``(mean, std)`` used to normalise every channel."""

class RandomResize(_Transform):
    """Resize to a random size drawn from ``sizes``."""

    def __init__(self, sizes: list[int], max_size: int | None = ...) -> None:
        """Pick a random resize target within ``sizes`` (capped)."""

class ToTensor(_Transform):
    """Convert a PIL image to a torch tensor."""

    def __init__(self) -> None:
        """Instantiate the stateless converter."""
