"""GroundingDINO model class."""

# pylint: disable=locally-disabled
# pylint: disable=suppressed-message,useless-suppression
# pylint: disable=unused-variable,unused-argument

from collections.abc import Mapping
from typing import NamedTuple

from torch import Tensor
from torch.nn import Module

class _IncompatibleKeys(NamedTuple):
    """What ``torch.nn.Module.load_state_dict`` returns at runtime."""

    missing_keys: list[str]
    unexpected_keys: list[str]

class GroundingDINO(Module):  # pylint: disable=abstract-method
    """Open-set detection model. Inherits ``to``/``eval`` from Module."""

    def __call__(
        self,
        samples: Tensor,
        captions: list[str],
    ) -> dict[str, Tensor]:
        """Forward pass. Returns per-caption prediction tensors."""

    def load_state_dict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = ...,
        assign: bool = ...,
    ) -> _IncompatibleKeys:
        """Load ``state_dict`` into the model, returning any diff."""
