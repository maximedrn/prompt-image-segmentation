"""Domain exceptions.

Every exception is a subclass of :class:`SegmenterError` so callers
can catch the whole domain with one ``except``. HTTP layer maps them
to status codes centrally in :mod:`app.api.errors`.
"""


class SegmenterError(Exception):
    """Base class for every domain-level error."""


class InvalidPromptError(SegmenterError):
    """Raised when the prompt is empty or otherwise malformed."""

    def __init__(self, reason: str) -> None:
        """Build the error with a caller-facing ``reason``.

        :param reason: Short explanation carried to the HTTP layer.
        :type reason: str
        """
        super().__init__(reason)
        self.reason: str = reason


class NoDetectionError(SegmenterError):
    """Raised when the detector returns zero boxes for a prompt."""

    def __init__(self, prompt: str) -> None:
        """Build the error, embedding the offending prompt.

        :param prompt: The user prompt that produced no detection.
        :type prompt: str
        """
        super().__init__(f"No detection for prompt {prompt!r}.")
        self.prompt: str = prompt


class BackendUnavailableError(SegmenterError):
    """Raised when a requested backend name is not registered."""

    def __init__(self, kind: str, name: str, available: list[str]) -> None:
        """Build the error with context about the missing backend.

        :param kind: Factory kind (e.g. ``"segmenter"``).
        :type kind: str
        :param name: Backend name the caller requested.
        :type name: str
        :param available: All names currently registered under ``kind``.
        :type available: list[str]
        """
        super().__init__(f"Unknown {kind} {name!r}. Available: {available}")
        self.kind: str = kind
        self.name: str = name
        self.available: list[str] = available


__all__: list[str] = [
    "SegmenterError",
    "InvalidPromptError",
    "NoDetectionError",
    "BackendUnavailableError",
]
