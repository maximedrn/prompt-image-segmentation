"""Runtime settings driven by environment variables.

Access through :func:`get_settings` (LRU-cached: one instance per
process).
"""

from functools import lru_cache
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

GROUNDING_DINO_SCORE_THRESHOLD: Final[float] = 0.3
DEFAULT_MASK_PADDING_PCT: Final[int] = 10
DEFAULT_DILATION_PCT: Final[float] = 3.0


class Settings(BaseSettings):
    """Environment-backed settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    segmentation_username: str | None = Field(
        default=None, alias="SEGMENTATION_USERNAME"
    )
    segmentation_password: str | None = Field(
        default=None, alias="SEGMENTATION_PASSWORD"
    )
    default_segmenter: str = Field(
        default="sam_dino", alias="DEFAULT_SEGMENTER"
    )
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=7860, alias="PORT")
    ui_mount_path: str = Field(default="/", alias="UI_MOUNT_PATH")
    mask_padding_pct: int = Field(
        default=DEFAULT_MASK_PADDING_PCT, alias="MASK_PADDING_PCT"
    )
    dilation_pct: float = Field(
        default=DEFAULT_DILATION_PCT, alias="DILATION_PCT"
    )
    detection_score_threshold: float = Field(
        default=GROUNDING_DINO_SCORE_THRESHOLD,
        alias="DETECTION_SCORE_THRESHOLD",
    )

    @property
    def auth_enabled(self) -> bool:
        """``True`` when both credentials are present.

        :returns: Whether HTTP Basic Auth should be enforced by the
            JSON API and the Gradio UI.
        :rtype: bool
        """
        return bool(self.segmentation_username and self.segmentation_password)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    :returns: The cached environment-backed settings instance.
    :rtype: Settings
    """
    return Settings()


__all__: list[str] = [
    "GROUNDING_DINO_SCORE_THRESHOLD",
    "DEFAULT_MASK_PADDING_PCT",
    "DEFAULT_DILATION_PCT",
    "Settings",
    "get_settings",
]
