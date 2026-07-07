"""ML model loaders.

Each concrete loader implements :class:`ModelLoader` and is responsible
for fetching / caching / instantiating one model weight file. They are
consumed by :class:`~app.managers.ModelManager`, never directly by
services or segmenters.
"""

from app.models.base import ModelLoader
from app.models.grounding_dino_loader import GroundingDinoLoader
from app.models.sam_loader import SamLoader

__all__: list[str] = ["GroundingDinoLoader", "ModelLoader", "SamLoader"]
