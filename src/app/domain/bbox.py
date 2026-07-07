"""Bounding box value object."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from app.domain import JSONValue


class BBox(BaseModel):
    """Bounding box in original-image pixel coordinates."""

    model_config = ConfigDict(frozen=True)

    x: int
    y: int
    width: int
    height: int

    @property
    def empty(self) -> bool:
        """Report whether the box has zero (or negative) area.

        :returns: ``True`` if ``width`` or ``height`` is non-positive.
        :rtype: bool
        """
        return self.width <= 0 or self.height <= 0

    @property
    def right(self) -> int:
        """Right pixel (exclusive slice bound).

        :returns: ``x + width``.
        :rtype: int
        """
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Bottom pixel (exclusive slice bound).

        :returns: ``y + height``.
        :rtype: int
        """
        return self.y + self.height

    def to_dict(self) -> dict[str, "JSONValue"]:
        """Serialize to a plain dict (for JSON responses).

        :returns: ``{"x": ..., "y": ..., "width": ..., "height": ...}``.
        :rtype: dict[str, "JSONValue"]
        """
        return self.model_dump()


__all__: list[str] = ["BBox"]
