"""Gradio vocabulary: every label and placeholder the UI shows."""

from dataclasses import dataclass
from typing import ClassVar, Literal, final


@final
@dataclass(frozen=True, slots=True)
class ComponentKind:
    """Gradio component options this UI selects.

    Typed as ``Literal`` rather than ``StrEnum`` because gradio
    annotates these arguments with literal unions, which a string
    enumeration member does not satisfy.
    """

    pil_image: ClassVar[Literal["pil"]] = "pil"
    primary_button: ClassVar[Literal["primary"]] = "primary"


@final
@dataclass(frozen=True, slots=True)
class Label:
    """Text shown next to each control."""

    title: ClassVar[str] = "Prompt image segmentation"
    instructions: ClassVar[str] = (
        "Upload an image and describe what to segment, for example "
        "`dog` or `license plate`."
    )
    image_input: ClassVar[str] = "Image"
    prompt_input: ClassVar[str] = "Prompt"
    prompt_placeholder: ClassVar[str] = "dog. cat. bicycle."
    backend_input: ClassVar[str] = "Backend"
    person_input: ClassVar[str] = "Face analysis"
    submit: ClassVar[str] = "Segment"
    image_output: ClassVar[str] = "Image (cropped)"
    mask_output: ClassVar[str] = "Mask"
    json_output: ClassVar[str] = "Response"


@final
@dataclass(frozen=True, slots=True)
class Layout:
    """Proportions of the two-column layout."""

    column_scale: ClassVar[int] = 1
    prompt_lines: ClassVar[int] = 2
    #: The UI never asks for a split, so the sole region is the first.
    first_region: ClassVar[int] = 0


@final
@dataclass(frozen=True, slots=True)
class UiMessage:
    """What the UI reports when it cannot call the service."""

    missing_image: ClassVar[str] = "Upload an image first."
    not_ready: ClassVar[str] = (
        "The models are still loading. Try again shortly."
    )
    no_detection: ClassVar[str] = "No detection for {prompt!r}."


@final
@dataclass(frozen=True, slots=True)
class PanelKey:
    """Keys of the JSON panel the UI fills."""

    error: ClassVar[str] = "error"


@final
@dataclass(frozen=True, slots=True)
class Hidden:
    """What the JSON panel leaves out of a successful response.

    The encoded pixels are unreadable and dwarf everything beside them,
    so the panel shows the answer without them. Spelled as the schema
    spells them, because that is what ``exclude`` matches on.
    """

    regions: ClassVar[str] = "regions"
    #: Pydantic's marker for "every item of this collection".
    every_region: ClassVar[str] = "__all__"
    mask: ClassVar[str] = "mask"
    image: ClassVar[str] = "image"


__all__: list[str] = [
    "ComponentKind",
    "Hidden",
    "Label",
    "Layout",
    "PanelKey",
    "UiMessage",
]
