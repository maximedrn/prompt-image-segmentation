"""Gradio user interface, mounted only when ENABLE_UI is set."""

from app.interfaces.ui.gradio_app import mount_ui

__all__: list[str] = ["mount_ui"]
