"""Public API of the ``segment-anything-py`` package."""

# pylint: disable=locally-disabled
# pylint: disable=suppressed-message,useless-suppression
# pylint: disable=unused-variable,unused-argument

from segment_anything.modeling.sam import Sam

def build_sam_vit_h(checkpoint: str) -> Sam:
    """Build a SAM ViT-H model and load ``checkpoint``."""
