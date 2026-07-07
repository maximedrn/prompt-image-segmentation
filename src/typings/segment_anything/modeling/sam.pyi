"""Segment-Anything model class."""

# pylint: disable=locally-disabled
# pylint: disable=suppressed-message,useless-suppression
# pylint: disable=unused-variable,unused-argument

from torch.nn import Module

class Sam(Module):  # pylint: disable=abstract-method
    """SAM ViT-H. Inherits ``to``, ``eval``, etc. from ``Module``."""
