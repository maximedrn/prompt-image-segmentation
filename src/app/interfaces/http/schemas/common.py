"""Schemas every route can answer with, and what JSON is."""

from typing import final

from pydantic import BaseModel, ConfigDict

from app.interfaces.http.constants import ErrorCode, HealthState

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)
"""Anything ``model_dump(mode="json")`` can produce.

Narrower than ``object``, and it says what it is: a decoded JSON value,
not an arbitrary Python one.
"""


@final
class ErrorSchema(BaseModel):
    """Uniform failure envelope shared by every route."""

    model_config = ConfigDict(frozen=True)

    error: ErrorCode
    message: str


@final
class HealthSchema(BaseModel):
    """Payload of both probes."""

    model_config = ConfigDict(frozen=True)

    status: HealthState


@final
class SegmentersSchema(BaseModel):
    """Payload of the backend listing."""

    model_config = ConfigDict(frozen=True)

    available: list[str]


__all__: list[str] = [
    "JsonValue",
    "ErrorSchema",
    "HealthSchema",
    "SegmentersSchema",
]
