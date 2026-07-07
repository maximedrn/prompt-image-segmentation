"""Top-level segmentation response DTO."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from app.domain.bbox import BBox
from app.domain.person import PersonPayload

if TYPE_CHECKING:
    from app.domain import JSONValue


class SegmentResponse(BaseModel):
    """Complete pipeline output, serializable via :meth:`to_dict`."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    mask: str = Field(
        ..., description="Base64 PNG grayscale mask, cropped to bbox."
    )
    image: str = Field(
        ..., description="Base64 PNG original image, cropped to bbox."
    )
    bbox: BBox
    detections: int
    segmenter: str
    person: PersonPayload | None = None

    def to_dict(self) -> dict[str, "JSONValue"]:
        """Serialize to a JSON-compatible dict.

        The ``person`` key is only present when person-mode ran and
        produced a payload.

        :returns: Flat JSON-compatible dict.
        :rtype: dict[str, "JSONValue"]
        """
        return self.model_dump(exclude_none=True)


__all__: list[str] = ["SegmentResponse"]
