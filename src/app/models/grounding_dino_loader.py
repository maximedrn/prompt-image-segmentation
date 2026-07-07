"""Load GroundingDINO SwinT from the HuggingFace Hub."""

from typing import ClassVar

from groundingdino.config import GroundingDINO_SwinT_OGC
from groundingdino.models import build_model
from groundingdino.models.GroundingDINO.groundingdino import (
    GroundingDINO,
)
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from torch import load

from app.config import (
    GROUNDING_DINO_FILE,
    GROUNDING_DINO_REPO,
    get_device,
)
from app.models.base import ModelLoader


class GroundingDinoLoader(ModelLoader[GroundingDINO]):
    """Download + wire the SwinT GroundingDINO weights on demand."""

    identifier: ClassVar[str] = "grounding_dino_swint"

    def load(self) -> GroundingDINO:
        """Fetch the checkpoint, build the model, move it to device.

        :returns: A GroundingDINO SwinT model ready for inference on
            the process-wide torch device.
        :rtype: groundingdino.models.GroundingDINO.groundingdino.GroundingDINO
        """
        path: str = hf_hub_download(
            repo_id=GROUNDING_DINO_REPO,
            filename=GROUNDING_DINO_FILE,
        )
        checkpoint = load(path, map_location=get_device())
        weights = clean_state_dict(checkpoint["model"])
        model: GroundingDINO = build_model(GroundingDINO_SwinT_OGC)
        model.load_state_dict(weights, strict=False)
        model.eval()
        return model.to(get_device())


__all__: list[str] = ["GroundingDinoLoader"]
