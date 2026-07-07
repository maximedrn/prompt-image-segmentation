"""Static & environment-driven configuration.

* :mod:`paths` - filesystem / URL constants (immutable).
* :mod:`prompts` - default prompt bundles.
* :mod:`device` - torch device selection.
* :mod:`settings` - env-backed pydantic Settings (singleton).
"""

from app.config.device import get_device
from app.config.paths import (
    CHECKPOINTS_DIR,
    GROUNDING_DINO_FILE,
    GROUNDING_DINO_REPO,
    PROJECT_ROOT,
    SAM_CHECKPOINT,
    SAM_URL,
)
from app.config.settings import Settings, get_settings

__all__: list[str] = [
    "CHECKPOINTS_DIR",
    "GROUNDING_DINO_FILE",
    "GROUNDING_DINO_REPO",
    "PROJECT_ROOT",
    "SAM_CHECKPOINT",
    "SAM_URL",
    "Settings",
    "get_device",
    "get_settings",
]
