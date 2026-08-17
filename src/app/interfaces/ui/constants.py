"""Gradio vocabulary: every label and placeholder the UI shows."""

from dataclasses import dataclass
from typing import Final, Literal, final


@final
@dataclass(frozen=True, slots=True)
class ComponentKind:
    """Gradio component options this UI selects.

    Typed as ``Literal`` rather than ``StrEnum`` because gradio
    annotates these arguments with literal unions, which a string
    enumeration member does not satisfy.
    """

    pil_image: Literal["pil"] = "pil"
    primary_button: Literal["primary"] = "primary"


@final
@dataclass(frozen=True, slots=True)
class Label:
    """Text shown next to each control."""

    title: str = "Prompt image segmentation"
    instructions: str = (
        "Upload an image and describe what to segment, for example "
        "`dog` or `license plate`."
    )
    image_input: str = "Image"
    prompt_input: str = "Prompt"
    prompt_placeholder: str = "dog. cat. bicycle."
    backend_input: str = "Backend"
    person_input: str = "Face analysis"
    submit: str = "Segment"
    image_output: str = "Image (cropped)"
    mask_output: str = "Mask"
    json_output: str = "Response"


@final
@dataclass(frozen=True, slots=True)
class Layout:
    """Proportions of the two-column layout."""

    column_scale: int = 1
    prompt_lines: int = 2


@final
@dataclass(frozen=True, slots=True)
class UiMessage:
    """What the UI reports when it cannot call the service."""

    missing_image: str = "Upload an image first."
    not_ready: str = "The models are still loading. Try again shortly."


COMPONENT: Final[ComponentKind] = ComponentKind()
LABEL: Final[Label] = Label()
LAYOUT: Final[Layout] = Layout()
UI_MESSAGE: Final[UiMessage] = UiMessage()


__all__: list[str] = [
    "COMPONENT",
    "LABEL",
    "LAYOUT",
    "UI_MESSAGE",
    "ComponentKind",
    "Label",
    "Layout",
    "UiMessage",
]
