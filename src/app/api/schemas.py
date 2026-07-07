"""API-only schemas (no domain counterpart).

Response bodies mirroring domain models live directly in
:mod:`app.domain`; FastAPI uses them as ``response_model``.
"""

from pydantic import BaseModel


class ErrorSchema(BaseModel):
    """Uniform error envelope returned by the API handlers."""

    error: str
    message: str


class HealthSchema(BaseModel):
    """``/healthz`` payload."""

    status: str


__all__: list[str] = ["ErrorSchema", "HealthSchema"]
