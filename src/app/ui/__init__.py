"""Gradio user interface. Mounted on the FastAPI app at startup."""

from app.ui.gradio_app import build_gradio_blocks

__all__: list[str] = ["build_gradio_blocks"]
