"""Top-level segmentation use case."""

from PIL.Image import Image
from numpy import uint8
from numpy.typing import NDArray

from app.config import get_settings
from app.config.settings import Settings
from app.core import InvalidPromptError, SingletonMeta
from app.core.factory import Factory
from app.domain import (
    BBox,
    PersonPayload,
    SegmentResponse,
    SegmentationResult,
)
from app.infrastructure.image_io import (
    array_to_pil,
    pil_to_array,
    pil_to_base64,
)
from app.infrastructure.image_ops import (
    bbox_from_mask,
    crop_to_bbox,
    dilate_mask,
)
from app.segmenters import SEGMENTER_FACTORY, Segmenter


class SegmentationService(metaclass=SingletonMeta):
    """Orchestrates: backend -> bbox -> optional person -> encode."""

    def __init__(self) -> None:
        """Wire up the service with the segmenter factory + settings."""
        self._factory: Factory[Segmenter] = SEGMENTER_FACTORY
        settings: Settings = get_settings()
        self._default_backend: str = settings.default_segmenter
        self._padding_pct: int = settings.mask_padding_pct
        self._dilation_pct: float = settings.dilation_pct

    def available_backends(self) -> list[str]:
        """Names of every registered segmentation backend.

        :returns: A sorted list of backend keys.
        :rtype: list[str]
        """
        return self._factory.names()

    def run(
        self,
        image: Image,
        prompt: str,
        person_mode: bool = False,
        segmenter_name: str | None = None,
    ) -> SegmentResponse:
        """End-to-end: segment, crop, optionally person-analyse, encode.

        :param image: Source RGB image.
        :type image: PIL.Image.Image
        :param prompt: Free-form text prompt (dot- or comma-separated
            labels for GroundingDINO).
        :type prompt: str
        :param person_mode: When ``True``, also run face detection.
        :type person_mode: bool
        :param segmenter_name: Optional backend override. Defaults to
            :attr:`Settings.default_segmenter`.
        :type segmenter_name: str | None
        :returns: The base64-encoded response DTO.
        :rtype: app.domain.response.SegmentResponse
        :raises app.core.exceptions.InvalidPromptError: If the
            prompt is empty.
        :raises app.core.exceptions.NoDetectionError: If the
            backend detects nothing.
        :raises app.core.exceptions.BackendUnavailableError: If
            ``segmenter_name`` is unknown.
        """
        if not prompt.strip():
            raise InvalidPromptError("Prompt must not be empty.")
        backend_name: str = segmenter_name or self._default_backend
        backend: Segmenter = self._factory.get(backend_name)
        result: SegmentationResult = backend.segment(image, prompt)
        bbox: BBox = bbox_from_mask(result.mask, padding_pct=self._padding_pct)
        person_payload: PersonPayload | None = (
            self._run_person(image) if person_mode else None
        )
        mask_b64, image_b64 = self._crop_and_encode(image, result.mask, bbox)
        return SegmentResponse(
            prompt=prompt,
            mask=mask_b64,
            image=image_b64,
            bbox=bbox,
            detections=result.detections,
            segmenter=backend_name,
            person=person_payload,
        )

    def _crop_and_encode(
        self,
        original: Image,
        mask: NDArray[uint8],
        bbox: BBox,
    ) -> tuple[str, str]:
        """Crop, dilate, base64-encode the mask + cropped source.

        :param original: Full-frame image.
        :type original: PIL.Image.Image
        :param mask: 2D uint8 mask covering the full frame.
        :type mask: numpy.typing.NDArray[numpy.uint8]
        :param bbox: Padded bbox produced by :func:`bbox_from_mask`.
        :type bbox: app.domain.bbox.BBox
        :returns: ``(mask_b64_png, image_b64_png)`` both cropped to
            ``bbox``.
        :rtype: tuple[str, str]
        """
        original_array: NDArray[uint8] = pil_to_array(original)
        cropped_original: NDArray[uint8] = crop_to_bbox(original_array, bbox)
        cropped_mask: NDArray[uint8] = crop_to_bbox(mask, bbox)
        dilated: NDArray[uint8] = dilate_mask(cropped_mask, self._dilation_pct)
        return (
            pil_to_base64(array_to_pil(dilated)),
            pil_to_base64(array_to_pil(cropped_original)),
        )

    @staticmethod
    def _run_person(image: Image) -> PersonPayload:
        """Run face analysis.

        :param image: Full-frame RGB PIL image.
        :type image: PIL.Image.Image
        :param mask: 2D uint8 mask (full-frame) from the app.
        :type mask: numpy.typing.NDArray[numpy.uint8]
        :returns: The person payload, including genders and adult status.
        :rtype: app.domain.person.PersonPayload
        """
        # Both live behind the ``person`` extra; keep them off the base
        # import path so segmenter still imports without those deps.
        # pylint: disable=locally-disabled
        # pylint: disable=suppressed-message,useless-suppression
        # pylint: disable=import-outside-toplevel
        from app.services.person_service import PersonService

        image_array: NDArray[uint8] = pil_to_array(image)
        genders, is_adult = PersonService().analyse(image_array)
        payload: PersonPayload = PersonPayload(
            genders=genders,
            is_adult=is_adult,
        )
        return payload


__all__: list[str] = ["SegmentationService"]
