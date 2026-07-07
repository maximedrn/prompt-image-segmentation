"""Cross-cutting primitives (singleton meta, generic factory, errors).

Nothing in ``core`` depends on any other package layer: it is the
foundation everything else can freely import.
"""

from app.core.exceptions import (
    BackendUnavailableError,
    InvalidPromptError,
    NoDetectionError,
    SegmenterError,
)
from app.core.factory import Factory
from app.core.singleton import SingletonMeta

__all__: list[str] = [
    "BackendUnavailableError",
    "Factory",
    "InvalidPromptError",
    "NoDetectionError",
    "SegmenterError",
    "SingletonMeta",
]
