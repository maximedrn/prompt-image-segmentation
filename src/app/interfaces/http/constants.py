"""HTTP transport vocabulary.

Routes, headers, error codes and caller-facing messages, grouped so no
handler ever spells a protocol literal inline.
"""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, final


@unique
class ErrorCode(StrEnum):
    """Machine-readable code every failure body carries.

    Clients branch on this and never parse ``message``.
    """

    UNAUTHORIZED = "unauthorized"
    INVALID_PROMPT = "invalid_prompt"
    NO_DETECTION = "no_detection"
    UNKNOWN_BACKEND = "unknown_backend"
    INVALID_IMAGE = "invalid_image"
    UPLOAD_TOO_LARGE = "upload_too_large"
    RATE_LIMITED = "rate_limited"
    OUT_OF_MEMORY = "out_of_memory"
    UNAVAILABLE_FEATURE = "unavailable_feature"
    NOT_READY = "not_ready"
    INTERNAL = "internal"


@unique
class HealthState(StrEnum):
    """Values the two probes report."""

    OK = "ok"
    READY = "ready"


@final
@dataclass(frozen=True, slots=True)
class HttpRoute:
    """Every public path, in one place."""

    health: str = "/healthz"
    ready: str = "/readyz"
    segmenters: str = "/segmenters"
    segment: str = "/segment"


@final
@dataclass(frozen=True, slots=True)
class HeaderName:
    """Response headers this service sets."""

    retry_after: str = "Retry-After"
    authenticate: str = "WWW-Authenticate"


@final
@dataclass(frozen=True, slots=True)
class AuthScheme:
    """Authentication scheme presented in a challenge."""

    basic: str = "Basic"


@final
@dataclass(frozen=True, slots=True)
class FormField:
    """Multipart field names of ``POST /segment``."""

    image: str = "image"
    prompt: str = "prompt"
    person_mode: str = "person_mode"
    segmenter: str = "segmenter"


@final
@dataclass(frozen=True, slots=True)
class Message:
    """Caller-facing text. Deliberately free of internal detail."""

    authentication_required: str = "Authentication required."
    invalid_credentials: str = "Invalid credentials."
    rate_limited: str = "Rate limit exceeded."
    empty_upload: str = "Empty upload."
    models_loading: str = "Models are still loading."
    internal: str = "Internal server error."
    feature_unavailable: str = "This feature is not installed on the server."
    out_of_memory: str = (
        "Not enough device memory for this image. Retry with a smaller one."
    )


@final
@dataclass(frozen=True, slots=True)
class OpenApiTag:
    """Groups the routes are documented under."""

    meta: str = "meta"
    segmentation: str = "segmentation"


@final
@dataclass(frozen=True, slots=True)
class ApiMetadata:
    """What the generated OpenAPI document announces."""

    title: str = "Prompt image segmentation"
    description: str = (
        "Prompt-driven segmentation via GroundingDINO + SAM 2.1. "
        "Layered architecture: domain / application / infrastructure / "
        "interfaces, wired in a single bootstrap."
    )


ROUTE: Final[HttpRoute] = HttpRoute()
HEADER: Final[HeaderName] = HeaderName()
SCHEME: Final[AuthScheme] = AuthScheme()
FIELD: Final[FormField] = FormField()
MESSAGE: Final[Message] = Message()
TAG: Final[OpenApiTag] = OpenApiTag()
API: Final[ApiMetadata] = ApiMetadata()


__all__: list[str] = [
    "API",
    "FIELD",
    "HEADER",
    "MESSAGE",
    "ROUTE",
    "SCHEME",
    "TAG",
    "ApiMetadata",
    "AuthScheme",
    "ErrorCode",
    "FormField",
    "HeaderName",
    "HealthState",
    "HttpRoute",
    "Message",
    "OpenApiTag",
]
