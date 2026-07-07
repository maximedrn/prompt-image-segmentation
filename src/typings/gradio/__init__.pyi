"""Type stubs for gradio's public re-exports used by the app."""

# pylint: disable=locally-disabled
# pylint: disable=suppressed-message,useless-suppression
# pylint: disable=unused-variable,unused-argument

from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Literal, ParamSpec, TypeVar

from fastapi import FastAPI

_ClickParams = ParamSpec("_ClickParams")
_ClickReturn = TypeVar("_ClickReturn")

class Component:
    """Base class for gradio input/output leaf components."""

class _Container:
    """Base for context-manager containers (Blocks, Row, Column)."""

    def __enter__(self) -> "_Container":
        """Enter the container context."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the container context (never suppresses)."""

class Blocks(_Container):
    """Top-level Blocks UI container."""

    def __init__(
        self,
        *,
        title: str = ...,
        analytics_enabled: bool | None = ...,
    ) -> None:
        """Configure the Blocks page title and analytics opt-in."""

    def __enter__(self) -> "Blocks":
        """Enter the Blocks context (narrowed return type)."""

class Row(_Container):
    """Horizontal layout row."""

    def __init__(self) -> None:
        """Instantiate an empty row."""

    def __enter__(self) -> "Row":
        """Enter the Row context (narrowed return type)."""

class Column(_Container):
    """Vertical layout column."""

    def __init__(self, *, scale: int = ...) -> None:
        """Instantiate a column with an optional flex ``scale``."""

    def __enter__(self) -> "Column":
        """Enter the Column context (narrowed return type)."""

class Markdown(Component):
    """Static Markdown block."""

    def __init__(self, value: str | None = ...) -> None:
        """Render ``value`` as Markdown."""

class Textbox(Component):
    """Multiline text input."""

    def __init__(
        self,
        value: str | None = ...,
        *,
        lines: int = ...,
        placeholder: str | None = ...,
        label: str | None = ...,
    ) -> None:
        """Configure the text field."""

class Checkbox(Component):
    """Boolean checkbox."""

    def __init__(
        self,
        value: bool = ...,
        *,
        label: str | None = ...,
    ) -> None:
        """Configure the checkbox default and label."""

class Dropdown(Component):
    """Single-select dropdown."""

    def __init__(
        self,
        choices: Sequence[str] | None = ...,
        *,
        value: str | None = ...,
        label: str | None = ...,
        interactive: bool | None = ...,
    ) -> None:
        """Configure the dropdown options and default."""

class Image(Component):
    """Image input / output component."""

    def __init__(
        self,
        *,
        label: str | None = ...,
        # pylint: disable=redefined-builtin
        type: Literal["numpy", "pil", "filepath"] = ...,
    ) -> None:
        """Configure the label and marshalled image type."""

class JSON(Component):
    """JSON output block."""

    def __init__(
        self,
        *,
        label: str | None = ...,
    ) -> None:
        """Configure the JSON viewer label."""

class Button(Component):
    """Clickable button component."""

    def __init__(
        self,
        value: str = ...,
        *,
        variant: Literal["primary", "secondary", "stop", "huggingface"] = ...,
    ) -> None:
        """Configure the button label and visual variant."""

    def click(
        self,
        fn: Callable[_ClickParams, _ClickReturn],
        inputs: list[Component] | None = ...,
        outputs: list[Component] | None = ...,
        api_name: str | bool | None = ...,
    ) -> None:
        """Bind ``fn`` to the click event.

        ``fn`` is invoked with the values of ``inputs`` and its return
        is dispatched to ``outputs`` positionally. Set ``api_name=False``
        to skip this handler in ``/gradio_api/info``.
        """

def mount_gradio_app(
    app: FastAPI,
    blocks: Blocks,
    path: str,
    *,
    auth: tuple[str, str] | None = ...,
) -> FastAPI:
    """Mount ``blocks`` under ``path`` on ``app``. Returns ``app``."""
