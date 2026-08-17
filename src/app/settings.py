"""Configuration boundary.

Environment variables are untrusted input, so they are validated once,
here, and then handed downward as narrow immutable policies rather than
as a settings object every component can reach into (``SKILL.md``
sections 21 and 27).
"""

from enum import StrEnum, unique
from typing import Final, Self, final

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.application.policies import (
    DetectionPolicy,
    RateLimitPolicy,
    SegmentationPolicy,
    UploadPolicy,
)


@unique
class AuthMode(StrEnum):
    """How the API authenticates callers."""

    BASIC = "basic"
    NONE = "none"


@final
class Defaults:
    """Every default value the environment can override.

    Grouped rather than scattered as module constants, so the whole
    tunable surface of the service reads as one object.
    """

    AUTH_MODE: Final[AuthMode] = AuthMode.BASIC
    HOST: Final[str] = "0.0.0.0"
    PORT: Final[int] = 7860
    ENABLE_UI: Final[bool] = False
    UI_MOUNT_PATH: Final[str] = "/"
    DEFAULT_SEGMENTER: Final[str] = "sam_dino"

    DETECTION_SCORE_THRESHOLD: Final[float] = 0.3
    TEXT_SCORE_THRESHOLD: Final[float] = 0.25
    MASK_PADDING_PERCENTAGE: Final[float] = 10.0
    DILATION_PERCENTAGE: Final[float] = 3.0
    #: Deliberately low: it flags a doubtful mask, it does not reject
    #: one. Raise it once the production distribution is known.
    RELIABILITY_THRESHOLD: Final[float] = 0.4

    #: 20 MiB of encoded image. Uvicorn has no body limit of its own, so
    #: this is the only thing between an upload and the heap.
    MAX_UPLOAD_BYTES: Final[int] = 20 * 1024 * 1024
    #: A 40 Mpx image decodes to roughly 120 MB as RGB uint8. Past that
    #: it is a decompression bomb, not a photograph.
    MAX_IMAGE_PIXELS: Final[int] = 40_000_000

    RATE_LIMIT_REQUESTS: Final[int] = 60
    RATE_LIMIT_WINDOW_SECONDS: Final[float] = 60.0
    MAX_TRACKED_CLIENTS: Final[int] = 10_000


@final
class Settings(BaseSettings):
    """Validated environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    auth_mode: AuthMode = Field(default=Defaults.AUTH_MODE, alias="AUTH_MODE")
    segmentation_username: str | None = Field(
        default=None, alias="SEGMENTATION_USERNAME"
    )
    segmentation_password: str | None = Field(
        default=None, alias="SEGMENTATION_PASSWORD"
    )
    host: str = Field(default=Defaults.HOST, alias="HOST")
    port: int = Field(default=Defaults.PORT, alias="PORT")
    enable_ui: bool = Field(default=Defaults.ENABLE_UI, alias="ENABLE_UI")
    ui_mount_path: str = Field(
        default=Defaults.UI_MOUNT_PATH, alias="UI_MOUNT_PATH"
    )
    default_segmenter: str = Field(
        default=Defaults.DEFAULT_SEGMENTER, alias="DEFAULT_SEGMENTER"
    )
    detection_score_threshold: float = Field(
        default=Defaults.DETECTION_SCORE_THRESHOLD,
        alias="DETECTION_SCORE_THRESHOLD",
    )
    text_score_threshold: float = Field(
        default=Defaults.TEXT_SCORE_THRESHOLD, alias="TEXT_SCORE_THRESHOLD"
    )
    mask_padding_percentage: float = Field(
        default=Defaults.MASK_PADDING_PERCENTAGE, alias="MASK_PADDING_PCT"
    )
    dilation_percentage: float = Field(
        default=Defaults.DILATION_PERCENTAGE, alias="DILATION_PCT"
    )
    reliability_threshold: float = Field(
        default=Defaults.RELIABILITY_THRESHOLD, alias="RELIABILITY_THRESHOLD"
    )
    max_upload_bytes: int = Field(
        default=Defaults.MAX_UPLOAD_BYTES, alias="MAX_UPLOAD_BYTES"
    )
    max_image_pixels: int = Field(
        default=Defaults.MAX_IMAGE_PIXELS, alias="MAX_IMAGE_PIXELS"
    )
    rate_limit_requests: int = Field(
        default=Defaults.RATE_LIMIT_REQUESTS, alias="RATE_LIMIT_REQUESTS"
    )
    rate_limit_window_seconds: float = Field(
        default=Defaults.RATE_LIMIT_WINDOW_SECONDS,
        alias="RATE_LIMIT_WINDOW_SECONDS",
    )
    max_tracked_clients: int = Field(
        default=Defaults.MAX_TRACKED_CLIENTS, alias="MAX_TRACKED_CLIENTS"
    )

    @model_validator(mode="after")
    def _credentials_required_for_basic(self) -> Self:
        """Refuse to start in ``basic`` mode without credentials.

        Deriving "is auth on?" from whether credentials happen to be set
        means an empty environment silently serves an open API. Failing
        loudly is the only safe reading of a misconfigured secret.

        :returns: The validated settings.
        :rtype: Settings
        :raises ValueError: When ``basic`` is selected but a credential
            is missing.
        """
        if self.auth_mode is AuthMode.BASIC and not (
            self.segmentation_username and self.segmentation_password
        ):
            raise ValueError(
                "AUTH_MODE=basic requires SEGMENTATION_USERNAME and "
                "SEGMENTATION_PASSWORD. Set AUTH_MODE=none to serve "
                "without authentication, deliberately."
            )
        return self

    @property
    def auth_enabled(self) -> bool:
        """Whether callers must authenticate.

        :returns: ``True`` unless auth was explicitly turned off.
        :rtype: bool
        """
        return self.auth_mode is AuthMode.BASIC

    def segmentation_policy(self) -> SegmentationPolicy:
        """Build the tuning the segmentation use case needs.

        :returns: The segmentation policy.
        :rtype: app.application.policies.SegmentationPolicy
        """
        return SegmentationPolicy(
            mask_padding_percentage=self.mask_padding_percentage,
            dilation_percentage=self.dilation_percentage,
            reliability_threshold=self.reliability_threshold,
        )

    def detection_policy(self) -> DetectionPolicy:
        """Build the thresholds the detector adapter needs.

        :returns: The detection policy.
        :rtype: app.application.policies.DetectionPolicy
        """
        return DetectionPolicy(
            score_threshold=self.detection_score_threshold,
            text_threshold=self.text_score_threshold,
        )

    def upload_policy(self) -> UploadPolicy:
        """Build the ceilings the HTTP boundary applies.

        :returns: The upload policy.
        :rtype: app.application.policies.UploadPolicy
        """
        return UploadPolicy(
            max_upload_bytes=self.max_upload_bytes,
            max_image_pixels=self.max_image_pixels,
        )

    def rate_limit_policy(self) -> RateLimitPolicy:
        """Build the per-client request budget.

        :returns: The rate-limit policy.
        :rtype: app.application.policies.RateLimitPolicy
        """
        return RateLimitPolicy(
            max_requests=self.rate_limit_requests,
            window_seconds=self.rate_limit_window_seconds,
            max_tracked_clients=self.max_tracked_clients,
        )


__all__: list[str] = ["AuthMode", "Defaults", "Settings"]
