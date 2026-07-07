"""Model lifecycle manager (singleton).

Responsibilities:

* keep exactly one instance of each loaded model in memory,
* load lazily on first access,
* expose a warmup entry point for the API startup hook.
"""

from threading import Lock

from segment_anything.predictor import SamPredictor
from groundingdino.models.GroundingDINO.groundingdino import (
    GroundingDINO,
)

from app.core.singleton import SingletonMeta
from app.models import GroundingDinoLoader, SamLoader


class ModelManager(metaclass=SingletonMeta):
    """Central holder of every loaded ML model."""

    def __init__(self) -> None:
        """Build the manager with the built-in SAM + GroundingDINO loaders.

        Called exactly once per process (singleton via
        :class:`~app.core.singleton.SingletonMeta`).
        """
        self._sam_loader: SamLoader = SamLoader()
        self._dino_loader: GroundingDinoLoader = GroundingDinoLoader()
        self._sam: SamPredictor | None = None
        self._dino: GroundingDINO | None = None
        self._load_lock: Lock = Lock()

    @property
    def sam(self) -> SamPredictor:
        """Lazy-loaded :class:`SamPredictor` singleton.

        :returns: The SAM ViT-H predictor, built on first access.
        :rtype: segment_anything.predictor.SamPredictor
        """
        if self._sam is not None:
            return self._sam
        with self._load_lock:
            if self._sam is None:
                self._sam = self._sam_loader.load()
            return self._sam

    @property
    def grounding_dino(self) -> GroundingDINO:
        """Lazy-loaded :class:`GroundingDINO` singleton.

        :returns: The GroundingDINO SwinT model, built on first access.
        :rtype: groundingdino.models.GroundingDINO.groundingdino.GroundingDINO
        """
        if self._dino is not None:
            return self._dino
        with self._load_lock:
            if self._dino is None:
                self._dino = self._dino_loader.load()
            return self._dino

    def warmup(self) -> None:
        """Force every registered model to load. Call at startup."""
        _ = self.sam
        _ = self.grounding_dino


__all__: list[str] = ["ModelManager"]
