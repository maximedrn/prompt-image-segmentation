"""Configuration boundary."""

from typing import Self, final

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.application.policies import (
    DetectionPolicy,
    FacePolicy,
    JobBackend,
    JobPolicy,
    RateLimitPolicy,
    SegmentationPolicy,
    UploadPolicy,
    WebhookPolicy,
)
from app.settings.constants import Defaults, EnvVar
from app.settings.types import AuthMode


@final
class Settings(BaseSettings):
    """Validated environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    auth_mode: AuthMode = Field(
        default=Defaults.AUTH_MODE, alias=EnvVar.AUTH_MODE
    )
    segmentation_username: str | None = Field(
        default=None, alias=EnvVar.SEGMENTATION_USERNAME
    )
    segmentation_password: str | None = Field(
        default=None, alias=EnvVar.SEGMENTATION_PASSWORD
    )
    host: str = Field(default=Defaults.HOST, alias=EnvVar.HOST)
    port: int = Field(default=Defaults.PORT, alias=EnvVar.PORT)
    enable_ui: bool = Field(default=Defaults.ENABLE_UI, alias=EnvVar.ENABLE_UI)
    ui_mount_path: str = Field(
        default=Defaults.UI_MOUNT_PATH, alias=EnvVar.UI_MOUNT_PATH
    )
    default_segmenter: str = Field(
        default=Defaults.DEFAULT_SEGMENTER, alias=EnvVar.DEFAULT_SEGMENTER
    )
    detection_score_threshold: float = Field(
        default=Defaults.DETECTION_SCORE_THRESHOLD,
        alias=EnvVar.DETECTION_SCORE_THRESHOLD,
    )
    text_score_threshold: float = Field(
        default=Defaults.TEXT_SCORE_THRESHOLD,
        alias=EnvVar.TEXT_SCORE_THRESHOLD,
    )
    mask_padding_percentage: float = Field(
        default=Defaults.MASK_PADDING_PERCENTAGE,
        alias=EnvVar.MASK_PADDING_PERCENTAGE,
    )
    dilation_percentage: float = Field(
        default=Defaults.DILATION_PERCENTAGE, alias=EnvVar.DILATION_PERCENTAGE
    )
    reliability_threshold: float = Field(
        default=Defaults.RELIABILITY_THRESHOLD,
        alias=EnvVar.RELIABILITY_THRESHOLD,
    )
    minimum_confidence: float = Field(
        default=Defaults.MINIMUM_CONFIDENCE, alias=EnvVar.MINIMUM_CONFIDENCE
    )
    face_score_threshold: float = Field(
        default=Defaults.FACE_SCORE_THRESHOLD,
        alias=EnvVar.FACE_SCORE_THRESHOLD,
    )
    job_backend: JobBackend = Field(
        default=Defaults.JOB_BACKEND, alias=EnvVar.JOB_BACKEND
    )
    redis_url: str = Field(default=Defaults.REDIS_URL, alias=EnvVar.REDIS_URL)
    job_retention_seconds: int = Field(
        default=Defaults.JOB_RETENTION_SECONDS,
        alias=EnvVar.JOB_RETENTION_SECONDS,
    )
    job_max_queue_depth: int = Field(
        default=Defaults.JOB_MAX_QUEUE_DEPTH, alias=EnvVar.JOB_MAX_QUEUE_DEPTH
    )
    webhook_signing_secret: str = Field(
        default=Defaults.WEBHOOK_SIGNING_SECRET,
        alias=EnvVar.WEBHOOK_SIGNING_SECRET,
    )
    webhook_timeout_seconds: float = Field(
        default=Defaults.WEBHOOK_TIMEOUT_SECONDS,
        alias=EnvVar.WEBHOOK_TIMEOUT_SECONDS,
    )
    webhook_max_attempts: int = Field(
        default=Defaults.WEBHOOK_MAX_ATTEMPTS,
        alias=EnvVar.WEBHOOK_MAX_ATTEMPTS,
    )
    webhook_initial_backoff_seconds: float = Field(
        default=Defaults.WEBHOOK_INITIAL_BACKOFF_SECONDS,
        alias=EnvVar.WEBHOOK_INITIAL_BACKOFF_SECONDS,
    )
    webhook_allow_insecure: bool = Field(
        default=Defaults.WEBHOOK_ALLOW_INSECURE,
        alias=EnvVar.WEBHOOK_ALLOW_INSECURE,
    )
    webhook_allow_private_hosts: bool = Field(
        default=Defaults.WEBHOOK_ALLOW_PRIVATE_HOSTS,
        alias=EnvVar.WEBHOOK_ALLOW_PRIVATE_HOSTS,
    )
    max_upload_bytes: int = Field(
        default=Defaults.MAX_UPLOAD_BYTES, alias=EnvVar.MAX_UPLOAD_BYTES
    )
    max_image_pixels: int = Field(
        default=Defaults.MAX_IMAGE_PIXELS, alias=EnvVar.MAX_IMAGE_PIXELS
    )
    rate_limit_requests: int = Field(
        default=Defaults.RATE_LIMIT_REQUESTS, alias=EnvVar.RATE_LIMIT_REQUESTS
    )
    rate_limit_window_seconds: float = Field(
        default=Defaults.RATE_LIMIT_WINDOW_SECONDS,
        alias=EnvVar.RATE_LIMIT_WINDOW_SECONDS,
    )
    max_tracked_clients: int = Field(
        default=Defaults.MAX_TRACKED_CLIENTS, alias=EnvVar.MAX_TRACKED_CLIENTS
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

    @model_validator(mode="after")
    def _signing_secret_is_worth_signing_with(self) -> Self:
        """Refuse a webhook secret too short to resist a brute force.

        Empty is a legitimate answer - it turns deliveries off, and a
        callback URL is then refused rather than sent unsigned. A short
        secret is the bad middle: it advertises a signature the receiver
        is meant to trust, from a key that a single captured delivery is
        enough to recover offline.

        :returns: The validated settings.
        :rtype: Settings
        :raises ValueError: When a secret is set but too short.
        """
        length: int = len(self.webhook_signing_secret)
        if 0 < length < Defaults.MINIMUM_WEBHOOK_SECRET_LENGTH:
            raise ValueError(
                "WEBHOOK_SIGNING_SECRET must be at least "
                f"{Defaults.MINIMUM_WEBHOOK_SECRET_LENGTH} characters. "
                "Leave it empty to serve without webhooks."
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
            minimum_confidence=self.minimum_confidence,
        )

    def job_policy(self) -> JobPolicy:
        """Build the bounds the job store needs.

        :returns: The job policy.
        :rtype: app.application.policies.JobPolicy
        """
        return JobPolicy(
            url=self.redis_url,
            retention_seconds=self.job_retention_seconds,
            max_queue_depth=self.job_max_queue_depth,
            backend=self.job_backend,
        )

    def webhook_policy(self) -> WebhookPolicy:
        """Build the rules outbound deliveries follow.

        :returns: The webhook policy.
        :rtype: app.application.policies.WebhookPolicy
        """
        return WebhookPolicy(
            signing_secret=self.webhook_signing_secret,
            timeout_seconds=self.webhook_timeout_seconds,
            max_attempts=self.webhook_max_attempts,
            initial_backoff_seconds=self.webhook_initial_backoff_seconds,
            allow_insecure=self.webhook_allow_insecure,
            allow_private_hosts=self.webhook_allow_private_hosts,
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

    def face_policy(self) -> FacePolicy:
        """Build the tuning the face detector needs.

        :returns: The face policy.
        :rtype: app.application.policies.FacePolicy
        """
        return FacePolicy(score_threshold=self.face_score_threshold)

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
