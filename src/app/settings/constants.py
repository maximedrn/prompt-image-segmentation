"""Every default value the environment can override.

``ClassVar`` rather than ``Final``: this is a namespace, never an
instance, and the annotation is what says so.
"""

from typing import ClassVar, final

from app.application.policies import JobBackend
from app.settings.types import AuthMode


@final
class EnvVar:
    """Name of every variable the environment is read through.

    Written down once because they are spelled in three places at
    least: the ``alias`` of each setting, the template an operator
    copies, and the tests that build a configuration by hand.
    """

    AUTH_MODE: ClassVar[str] = "AUTH_MODE"
    SEGMENTATION_USERNAME: ClassVar[str] = "SEGMENTATION_USERNAME"
    SEGMENTATION_PASSWORD: ClassVar[str] = "SEGMENTATION_PASSWORD"
    HOST: ClassVar[str] = "HOST"
    PORT: ClassVar[str] = "PORT"
    ENABLE_UI: ClassVar[str] = "ENABLE_UI"
    UI_MOUNT_PATH: ClassVar[str] = "UI_MOUNT_PATH"
    DEFAULT_SEGMENTER: ClassVar[str] = "DEFAULT_SEGMENTER"
    DETECTION_SCORE_THRESHOLD: ClassVar[str] = "DETECTION_SCORE_THRESHOLD"
    TEXT_SCORE_THRESHOLD: ClassVar[str] = "TEXT_SCORE_THRESHOLD"
    MASK_PADDING_PERCENTAGE: ClassVar[str] = "MASK_PADDING_PERCENTAGE"
    DILATION_PERCENTAGE: ClassVar[str] = "DILATION_PERCENTAGE"
    RELIABILITY_THRESHOLD: ClassVar[str] = "RELIABILITY_THRESHOLD"
    MINIMUM_CONFIDENCE: ClassVar[str] = "MINIMUM_CONFIDENCE"
    FACE_SCORE_THRESHOLD: ClassVar[str] = "FACE_SCORE_THRESHOLD"
    JOB_BACKEND: ClassVar[str] = "JOB_BACKEND"
    REDIS_URL: ClassVar[str] = "REDIS_URL"
    JOB_RETENTION_SECONDS: ClassVar[str] = "JOB_RETENTION_SECONDS"
    JOB_MAX_QUEUE_DEPTH: ClassVar[str] = "JOB_MAX_QUEUE_DEPTH"
    WEBHOOK_SIGNING_SECRET: ClassVar[str] = "WEBHOOK_SIGNING_SECRET"
    WEBHOOK_TIMEOUT_SECONDS: ClassVar[str] = "WEBHOOK_TIMEOUT_SECONDS"
    WEBHOOK_MAX_ATTEMPTS: ClassVar[str] = "WEBHOOK_MAX_ATTEMPTS"
    WEBHOOK_INITIAL_BACKOFF_SECONDS: ClassVar[str] = (
        "WEBHOOK_INITIAL_BACKOFF_SECONDS"
    )
    WEBHOOK_ALLOW_INSECURE: ClassVar[str] = "WEBHOOK_ALLOW_INSECURE"
    WEBHOOK_ALLOW_PRIVATE_HOSTS: ClassVar[str] = "WEBHOOK_ALLOW_PRIVATE_HOSTS"
    MAX_UPLOAD_BYTES: ClassVar[str] = "MAX_UPLOAD_BYTES"
    MAX_IMAGE_PIXELS: ClassVar[str] = "MAX_IMAGE_PIXELS"
    RATE_LIMIT_REQUESTS: ClassVar[str] = "RATE_LIMIT_REQUESTS"
    RATE_LIMIT_WINDOW_SECONDS: ClassVar[str] = "RATE_LIMIT_WINDOW_SECONDS"
    MAX_TRACKED_CLIENTS: ClassVar[str] = "MAX_TRACKED_CLIENTS"


@final
class Defaults:
    """Every default value the environment can override.

    Grouped rather than scattered as module constants, so the whole
    tunable surface of the service reads as one object.
    """

    AUTH_MODE: ClassVar[AuthMode] = AuthMode.BASIC
    HOST: ClassVar[str] = "0.0.0.0"
    PORT: ClassVar[int] = 7860
    ENABLE_UI: ClassVar[bool] = False
    UI_MOUNT_PATH: ClassVar[str] = "/"
    DEFAULT_SEGMENTER: ClassVar[str] = "sam_dino"

    DETECTION_SCORE_THRESHOLD: ClassVar[float] = 0.3
    TEXT_SCORE_THRESHOLD: ClassVar[float] = 0.25
    MASK_PADDING_PERCENTAGE: ClassVar[float] = 10.0
    DILATION_PERCENTAGE: ClassVar[float] = 3.0
    #: Deliberately low: it flags a doubtful mask, it does not reject
    #: one. Raise it once the production distribution is known.
    RELIABILITY_THRESHOLD: ClassVar[float] = 0.4
    #: Zero keeps every detection the detector already cleared. Unlike
    #: the threshold above, this one drops parts of the answer, so the
    #: default has to be the one that discards nothing.
    MINIMUM_CONFIDENCE: ClassVar[float] = 0.0

    #: Where queued work lives. A job API without it has nowhere to
    #: put anything, so this has no sensible silent default beyond
    #: the conventional local address.
    REDIS_URL: ClassVar[str] = "redis://localhost:6379/0"
    #: Redis unless an operator says otherwise. Memory is for a single
    #: process that can afford to lose its queue on restart, and is
    #: chosen rather than fallen back to.
    JOB_BACKEND: ClassVar[JobBackend] = JobBackend.REDIS
    #: An hour is long enough for a caller to come back for its
    #: answer, short enough that abandoned megabytes do not pile up.
    JOB_RETENTION_SECONDS: ClassVar[int] = 3600
    #: Past this, acceptance refuses rather than promising work the
    #: single accelerator cannot reach in any reasonable time.
    JOB_MAX_QUEUE_DEPTH: ClassVar[int] = 100

    #: Signs outbound webhooks. Empty means the feature is off: an
    #: unsigned delivery is one a receiver cannot tell from a forgery,
    #: so a callback URL is refused rather than sent unsigned.
    WEBHOOK_SIGNING_SECRET: ClassVar[str] = ""
    #: Shortest secret worth signing with. A captured delivery gives an
    #: attacker the body, the timestamp and the digest, which is
    #: everything needed to brute-force a short key offline.
    MINIMUM_WEBHOOK_SECRET_LENGTH: ClassVar[int] = 32
    WEBHOOK_TIMEOUT_SECONDS: ClassVar[float] = 10.0
    WEBHOOK_MAX_ATTEMPTS: ClassVar[int] = 3
    WEBHOOK_INITIAL_BACKOFF_SECONDS: ClassVar[float] = 1.0
    #: Only reasonable inside a private network, where the receiver
    #: has no certificate of its own.
    WEBHOOK_ALLOW_INSECURE: ClassVar[bool] = False
    #: Off. The service makes the callback request itself, so a caller
    #: naming the address chooses what it reaches. Only a stack where
    #: every caller is one you control should turn this on.
    WEBHOOK_ALLOW_PRIVATE_HOSTS: ClassVar[bool] = False

    #: 20 MiB of encoded image. Uvicorn has no body limit of its own, so
    #: this is the only thing between an upload and the heap.
    MAX_UPLOAD_BYTES: ClassVar[int] = 20 * 1024 * 1024
    #: A 40 Mpx image decodes to roughly 120 MB as RGB uint8. Past that
    #: it is a decompression bomb, not a photograph.
    MAX_IMAGE_PIXELS: ClassVar[int] = 40_000_000

    #: YuNet answers on animal faces too. On the bundled examples a
    #: human scores 0.94 while a dog and a cat score 0.71 and 0.67, so
    #: this sits between them with margin. Re-measure on real traffic.
    FACE_SCORE_THRESHOLD: ClassVar[float] = 0.8

    RATE_LIMIT_REQUESTS: ClassVar[int] = 60
    RATE_LIMIT_WINDOW_SECONDS: ClassVar[float] = 60.0
    MAX_TRACKED_CLIENTS: ClassVar[int] = 10_000


__all__: list[str] = ["Defaults", "EnvVar"]
