"""GroundingDINO model builders."""

# pylint: disable=locally-disabled
# pylint: disable=suppressed-message,useless-suppression
# pylint: disable=unused-variable,unused-argument

from types import ModuleType

from groundingdino.models.GroundingDINO.groundingdino import (
    GroundingDINO,
)

def build_model(args: ModuleType) -> GroundingDINO:
    """Instantiate a GroundingDINO model from a config module."""
