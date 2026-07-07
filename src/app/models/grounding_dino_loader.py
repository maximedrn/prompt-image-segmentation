"""Load GroundingDINO SwinT from the HuggingFace Hub."""

from typing import ClassVar

from groundingdino.config import GroundingDINO_SwinT_OGC
from groundingdino.models import build_model
from groundingdino.models.GroundingDINO.groundingdino import (
    GroundingDINO,
)
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from torch import Tensor, device as torch_device, dtype as torch_dtype, load
from transformers import BertModel

from app.config import (
    GROUNDING_DINO_FILE,
    GROUNDING_DINO_REPO,
    get_device,
)
from app.models.base import ModelLoader

# ``BertModel.get_head_mask``, dropped in transformers 5.x. Re-inject
# the 4.x no-op branch (head_mask=None → list of Nones). Ceiling:
# actual head masking is unsupported; the model always runs with no
# masking, which matches GroundingDINO's default inference path.
if not hasattr(BertModel, "get_head_mask"):

    def _get_head_mask(
        self: BertModel,
        head_mask: object,
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> list[None]:
        """Return the no-mask default (list of Nones) used at inference.

        :param head_mask: Ignored; always returns a list of Nones.
        :type head_mask: object
        :param num_hidden_layers: Number of hidden layers in the model.
        :type num_hidden_layers: int
        :param is_attention_chunked: Ignored; always returns a list of Nones.
        :type is_attention_chunked: bool
        :returns: A list of Nones, one for each hidden layer.
        :rtype: list[None]
        """
        del self, head_mask, is_attention_chunked
        return [None] * num_hidden_layers

    BertModel.get_head_mask = _get_head_mask  # type: ignore[attr-defined]


# transformers 5 renamed ``get_extended_attention_mask``'s 3rd positional arg
# from ``device`` to ``dtype``. GroundingDINO's bertwarper still passes
# ``device``. Shim to accept the 4.x layout and forward to the 5.x method
# (which deduces device from the mask).
_original_get_extended_attention_mask = BertModel.get_extended_attention_mask


def _get_extended_attention_mask(
    self: BertModel,
    attention_mask: Tensor,
    input_shape: tuple[int, ...],
    device: torch_device | None = None,
    dtype: torch_dtype | None = None,
) -> Tensor:
    """Adapt the 4.x signature onto the 5.x implementation."""
    del device
    return _original_get_extended_attention_mask(
        self, attention_mask, input_shape, dtype
    )


BertModel.get_extended_attention_mask = _get_extended_attention_mask  # type: ignore[assignment,method-assign]


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
        checkpoint: dict[str, dict[str, Tensor]] = load(
            path, map_location=get_device()
        )
        weights: dict[str, Tensor] = clean_state_dict(checkpoint["model"])
        model: GroundingDINO = build_model(GroundingDINO_SwinT_OGC)
        model.load_state_dict(weights, strict=False)
        model.eval()
        return model.to(get_device())


__all__: list[str] = ["GroundingDinoLoader"]
