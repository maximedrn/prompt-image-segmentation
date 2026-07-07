"""HTTP layer: FastAPI + Gradio mount + auth + error handlers."""

from app.api.main import app, create_app

__all__: list[str] = ["app", "create_app"]
