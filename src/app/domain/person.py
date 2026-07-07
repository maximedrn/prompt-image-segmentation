"""Person analysis DTO (face)."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from app.domain import JSONValue


class PersonPayload(BaseModel):
    """Face-detection summary. Returned under ``response.person``."""

    model_config = ConfigDict(frozen=True)

    genders: list[int] = Field(
        ..., description="0=Male, 1=Female. One entry per face."
    )
    is_adult: bool = Field(..., description="True iff every face is >= 14yo.")

    def to_dict(self) -> dict[str, "JSONValue"]:
        """Serialize to a JSON-compatible dict.

        :returns: ``{"genders": [...], "is_adult": bool}``.
        :rtype: dict[str, "JSONValue"]
        """
        return self.model_dump()


__all__: list[str] = ["PersonPayload"]
