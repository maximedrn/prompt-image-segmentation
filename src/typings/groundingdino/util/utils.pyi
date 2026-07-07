"""Shared helpers for GroundingDINO weight loading."""

# pylint: disable=locally-disabled
# pylint: disable=suppressed-message,useless-suppression
# pylint: disable=unused-variable,unused-argument

from collections.abc import Mapping

from torch import Tensor

def clean_state_dict(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Strip DDP prefixes so ``state_dict`` loads on a plain module."""
