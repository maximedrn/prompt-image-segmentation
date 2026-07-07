"""Segmenter backends.

Every module in this package is imported here for its
``@SEGMENTER_FACTORY.register(...)`` side effect: importing
``app.segmenters`` guarantees the factory is fully populated.
"""

from app.segmenters.base import Segmenter
from app.segmenters.factory import SEGMENTER_FACTORY
from app.segmenters.sam_dino import SamDinoSegmenter

__all__: list[str] = ["SEGMENTER_FACTORY", "Segmenter", "SamDinoSegmenter"]
