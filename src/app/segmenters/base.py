"""Segmenter abstract base class."""

from abc import ABC, abstractmethod
from typing import ClassVar

from PIL.Image import Image

from app.domain import SegmentationResult


class Segmenter(ABC):
    """Prompt-driven segmenter contract.

    Implementations are stateless per call and thread-safe within a
    single Python process (models are loaded once by the
    :class:`~app.managers.ModelManager`).
    """

    name: ClassVar[str] = ""
    """Registry key set by ``@SEGMENTER_FACTORY.register(...)``."""

    @abstractmethod
    def segment(self, image: Image, prompt: str) -> SegmentationResult:
        """Return the mask matching ``prompt`` for ``image``.

        :param image: Source RGB image.
        :type image: PIL.Image.Image
        :param prompt: Free-form text prompt (dot/comma separated).
        :type prompt: str
        :returns: The mask + detection count wrapped in a DTO.
        :rtype: app.domain.segmentation.SegmentationResult
        :raises app.core.exceptions.NoDetectionError: If nothing
            is detected for ``prompt``.
        :raises NotImplementedError: If a subclass forgets to override.
        """
        raise NotImplementedError


__all__: list[str] = ["Segmenter"]
