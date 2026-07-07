"""Filesystem paths and remote asset URLs."""

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
CHECKPOINTS_DIR: Final[Path] = PROJECT_ROOT / "checkpoints"

SAM_URL: Final[str] = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
SAM_CHECKPOINT: Final[Path] = CHECKPOINTS_DIR / "sam_vit_h_4b8939.pth"

GROUNDING_DINO_REPO: Final[str] = "ShilongLiu/GroundingDINO"
GROUNDING_DINO_FILE: Final[str] = "groundingdino_swint_ogc.pth"


__all__: list[str] = [
    "PROJECT_ROOT",
    "CHECKPOINTS_DIR",
    "SAM_URL",
    "SAM_CHECKPOINT",
    "GROUNDING_DINO_REPO",
    "GROUNDING_DINO_FILE",
]
