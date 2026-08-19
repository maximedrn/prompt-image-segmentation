"""Gradio UI, mounted onto the transport when enabled.

A second adapter over the same wired use case: it holds no logic the
JSON API does not, and it reaches for the application through the same
state the routes use.
"""

from io import BytesIO
from typing import final

from fastapi import FastAPI
from gradio import (
    JSON,
    Blocks,
    Button,
    Checkbox,
    Column,
    Dropdown,
    Image as GradioImage,
    Markdown,
    Row,
    Textbox,
    mount_gradio_app,
)
from numpy import array, uint8
from PIL import Image as PilImage
from PIL.Image import Image

from app.bootstrap import Application, SegmentOutcome
from app.domain import (
    DeviceExhausted,
    ImageMode,
    NoDetection,
    PersonPayload,
    Prompt,
    SegmentedImage,
    SourceImage,
)
from app.infrastructure.imaging import encode_png
from app.interfaces.http.schemas import SegmentSchema
from app.interfaces.ui.constants import (
    COMPONENT,
    LABEL,
    LAYOUT,
    UI_MESSAGE,
)
from app.settings import Settings

type UiOutcome = tuple[Image | None, Image | None, dict[str, object]]


def _decode(payload: str) -> Image:
    """Decode a base64 PNG string into a displayable image.

    :param payload: Base64 text of a PNG payload.
    :type payload: str
    :returns: The decoded image.
    :rtype: PIL.Image.Image
    """
    from base64 import b64decode  # pylint: disable=import-outside-toplevel

    return PilImage.open(BytesIO(b64decode(payload)))


@final
class SegmentationUi:
    """Binds the Gradio controls to the wired application."""

    def __init__(self, application: FastAPI) -> None:
        """Record the transport the UI borrows its state from.

        :param application: The FastAPI application holding the state.
        :type application: fastapi.FastAPI
        """
        self._transport: FastAPI = application

    def submit(
        self,
        image: Image | None,
        prompt: str,
        person_mode: bool,
        backend: str,
    ) -> UiOutcome:
        """Run one segmentation and shape it for the widgets.

        :param image: Uploaded image, ``None`` when the field is empty.
        :type image: PIL.Image.Image | None
        :param prompt: Prompt text.
        :type prompt: str
        :param person_mode: Whether to attach a face summary.
        :type person_mode: bool
        :param backend: Backend selected in the dropdown.
        :type backend: str
        :returns: ``(cropped, mask, payload)``; the images are ``None``
            on any failure and the payload carries the reason.
        :rtype: UiOutcome
        """
        wired: Application | None = getattr(
            self._transport.state, "application", None
        )
        if wired is None:
            return None, None, {"error": UI_MESSAGE.not_ready}
        if image is None:
            return None, None, {"error": UI_MESSAGE.missing_image}

        source: SourceImage = SourceImage(
            pixels=array(image.convert(ImageMode.RGB), dtype=uint8)
        )
        parsed: Prompt = Prompt.parse(prompt)
        person: PersonPayload | None = (
            wired.analyse_faces(source) if person_mode else None
        )
        outcome: SegmentOutcome = wired.segment(
            backend, source, parsed, person
        )
        match outcome:
            case NoDetection():
                return None, None, {"error": f"No detection for {prompt!r}."}
            case DeviceExhausted():
                return None, None, {"error": UI_MESSAGE.not_ready}
            case SegmentedImage():
                body: SegmentSchema = SegmentSchema.of(
                    result=outcome,
                    prompt=parsed.text,
                    segmenter=backend,
                    mask=encode_png(outcome.cropped_mask),
                    image=encode_png(outcome.cropped_image),
                )
                payload: dict[str, object] = body.model_dump(mode="json")
                cropped: Image = _decode(str(payload.pop("image")))
                mask: Image = _decode(str(payload.pop("mask")))
                return cropped, mask, payload

    def blocks(self, backends: list[str]) -> Blocks:
        """Assemble the Gradio layout.

        :param backends: Backend names offered in the dropdown.
        :type backends: list[str]
        :returns: The Blocks instance to mount.
        :rtype: gradio.Blocks
        """
        default: str = backends[0] if backends else ""
        blocks: Blocks
        with Blocks(title=LABEL.title, analytics_enabled=False) as blocks:
            Markdown(f"# {LABEL.title}")
            Markdown(LABEL.instructions)
            with Row():
                with Column(scale=LAYOUT.column_scale):
                    image_in: GradioImage = GradioImage(
                        label=LABEL.image_input,
                        type=COMPONENT.pil_image,
                    )
                    prompt_in: Textbox = Textbox(
                        label=LABEL.prompt_input,
                        placeholder=LABEL.prompt_placeholder,
                        lines=LAYOUT.prompt_lines,
                    )
                    backend_in: Dropdown = Dropdown(
                        label=LABEL.backend_input,
                        choices=backends,
                        value=default,
                        interactive=True,
                    )
                    person_in: Checkbox = Checkbox(
                        label=LABEL.person_input, value=False
                    )
                    submit: Button = Button(
                        LABEL.submit, variant=COMPONENT.primary_button
                    )
                with Column(scale=LAYOUT.column_scale):
                    image_out: GradioImage = GradioImage(
                        label=LABEL.image_output
                    )
                    mask_out: GradioImage = GradioImage(
                        label=LABEL.mask_output
                    )
                    json_out: JSON = JSON(label=LABEL.json_output)
            submit.click(
                fn=self.submit,
                inputs=[image_in, prompt_in, person_in, backend_in],
                outputs=[image_out, mask_out, json_out],
            )
        return blocks


def mount_ui(application: FastAPI, settings: Settings) -> None:
    """Mount the Gradio UI onto the transport.

    :param application: The application to mount onto.
    :type application: fastapi.FastAPI
    :param settings: Wired configuration.
    :type settings: app.settings.Settings
    """
    credentials: tuple[str, str] | None = None
    if (
        settings.auth_enabled
        and settings.segmentation_username
        and settings.segmentation_password
    ):
        credentials = (
            settings.segmentation_username,
            settings.segmentation_password,
        )
    ui: SegmentationUi = SegmentationUi(application)
    mount_gradio_app(
        application,
        ui.blocks([settings.default_segmenter]),
        path=settings.ui_mount_path,
        auth=credentials,
    )


__all__: list[str] = ["SegmentationUi", "mount_ui"]
