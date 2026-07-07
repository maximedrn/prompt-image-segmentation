"""Load SAM ViT-H into a :class:`SamPredictor`."""

from typing import ClassVar

from segment_anything import build_sam_vit_h
from segment_anything.modeling.sam import Sam
from segment_anything.predictor import SamPredictor

from app.config import SAM_CHECKPOINT, SAM_URL, get_device
from app.infrastructure.checkpoints import download_file
from app.models.base import ModelLoader


class SamLoader(ModelLoader[SamPredictor]):
    """Download + wrap the SAM ViT-H checkpoint into a predictor."""

    identifier: ClassVar[str] = "sam_vit_h"

    def load(self) -> SamPredictor:
        """Return a device-placed :class:`SamPredictor`.

        :returns: A predictor wrapping the SAM ViT-H model, moved onto
            the process-wide torch device.
        :rtype: segment_anything.predictor.SamPredictor
        """
        download_file(SAM_URL, SAM_CHECKPOINT)
        model: Sam = build_sam_vit_h(str(SAM_CHECKPOINT))
        return SamPredictor(model.to(get_device()))


__all__: list[str] = ["SamLoader"]
