"""Gradio Blocks UI. Thin adapter over :class:`SegmentationService`."""

from base64 import b64decode
from io import BytesIO
from typing import cast

from PIL import Image as PilImage
from PIL.Image import Image
from gradio import (
    Blocks,
    Button,
    Checkbox,
    Column,
    Dropdown,
    Image as GradioImage,
    JSON,
    Markdown,
    Row,
    Textbox,
)

from app.core import NoDetectionError, SegmenterError
from app.domain import JSONValue, SegmentResponse
from app.services import SegmentationService

_TITLE: str = "Prompt image segmentation"


def _decode_b64_png(payload: str) -> Image:
    """Decode a base64-encoded PNG string into a PIL image.

    :param payload: Base64 ASCII string of a PNG file.
    :type payload: str
    :returns: The decoded PIL image.
    :rtype: PIL.Image.Image
    """
    return PilImage.open(BytesIO(b64decode(payload)))


def _handle_submit(
    image: Image | None,
    prompt: str,
    person_mode: bool,
    backend: str,
) -> tuple[Image | None, Image | None, dict[str, JSONValue]]:
    """Wire the Gradio submit button to :meth:`SegmentationService.run`.

    :param image: Uploaded image (``None`` if the field is empty).
    :type image: PIL.Image.Image | None
    :param prompt: Text prompt for GroundingDINO.
    :type prompt: str
    :param person_mode: Whether to run face detection.
    :type person_mode: bool
    :param backend: Backend name selected from the dropdown.
    :type backend: str
    :returns: ``(cropped_image, mask_image, json_payload)`` - the first
        two are ``None`` on any error, and the JSON payload carries
        either the response fields or ``{"error": ..., ...}``.
    :rtype: tuple[
        PIL.Image.Image | None,
        PIL.Image.Image | None,
        dict[str, JSONValue],
    ]
    """
    if image is None:
        return None, None, {"error": "missing_image"}
    try:
        response: SegmentResponse = SegmentationService().run(
            image=image.convert("RGB"),
            prompt=prompt,
            person_mode=person_mode,
            segmenter_name=backend or None,
        )
    except NoDetectionError as error:
        return (
            None,
            None,
            {"error": "no_detection", "message": str(error)},
        )
    except SegmenterError as error:
        return (
            None,
            None,
            {"error": "invalid_input", "message": str(error)},
        )
    payload: dict[str, JSONValue] = response.to_dict()
    cropped_image: Image = _decode_b64_png(cast(str, payload.pop("image")))
    mask_image: Image = _decode_b64_png(cast(str, payload.pop("mask")))
    return cropped_image, mask_image, payload


def build_gradio_blocks() -> Blocks:
    """Assemble the Gradio UI mounted under FastAPI.

    :returns: The Blocks instance to hand to
        :func:`gradio.mount_gradio_app`.
    :rtype: gradio.Blocks
    """
    service: SegmentationService = SegmentationService()
    backends: list[str] = service.available_backends()
    default_backend: str = backends[0] if backends else ""
    with Blocks(title=_TITLE, analytics_enabled=False) as blocks:
        Markdown(f"# {_TITLE}")
        Markdown(
            "Upload an image, describe what to segment "
            "(e.g. `dog`, `license plate`)."
        )
        with Row():
            with Column(scale=1):
                image_in: GradioImage = GradioImage(label="Image", type="pil")
                prompt_in: Textbox = Textbox(
                    label="Prompt",
                    placeholder="dog. cat. bicycle.",
                    lines=2,
                )
                backend_in: Dropdown = Dropdown(
                    label="Backend",
                    choices=backends,
                    value=default_backend,
                    interactive=True,
                )
                person_in: Checkbox = Checkbox(
                    label="person_mode (face)",
                    value=False,
                )
                submit: Button = Button("Segment", variant="primary")
            with Column(scale=1):
                image_out: GradioImage = GradioImage(label="Image (cropped)")
                mask_out: GradioImage = GradioImage(label="Mask")
                json_out: JSON = JSON(label="Response")
        # metaclass; astroid can't see it.
        submit.click(
            fn=_handle_submit,
            inputs=[image_in, prompt_in, person_in, backend_in],
            outputs=[image_out, mask_out, json_out],
        )
    return blocks


__all__: list[str] = ["build_gradio_blocks"]
