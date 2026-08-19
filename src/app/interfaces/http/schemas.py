"""Wire schemas.

Pydantic lives here and nowhere deeper: the domain describes itself with
plain dataclasses, and these models exist only to validate and serialise
at the transport boundary (``SKILL.md`` section 21).
"""

from typing import Self, final

from pydantic import BaseModel, ConfigDict, Field

from app.domain import BBox, Detection, PersonPayload, SegmentedImage
from app.interfaces.http.constants import ErrorCode, HealthState


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


@final
class BBoxSchema(BaseModel):
    """A bounding box in original-image pixel coordinates."""

    model_config = ConfigDict(frozen=True)

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def of(cls, bbox: BBox) -> Self:
        """Project a domain box onto the wire.

        :param bbox: Domain bounding box.
        :type bbox: app.domain.models.BBox
        :returns: The wire representation.
        :rtype: BBoxSchema
        """
        return cls(x=bbox.x, y=bbox.y, width=bbox.width, height=bbox.height)


@final
class DetectionSchema(BaseModel):
    """One detection and the two scores behind its confidence."""

    model_config = ConfigDict(frozen=True)

    detection_score: float = Field(
        ..., description="Detector confidence that the box matches."
    )
    mask_score: float = Field(
        ..., description="IoU the refiner predicts for its own mask."
    )
    confidence: float = Field(
        ..., description="Product of the two scores above."
    )

    @classmethod
    def of(cls, detection: Detection) -> Self:
        """Project a domain detection onto the wire.

        :param detection: Domain detection.
        :type detection: app.domain.models.Detection
        :returns: The wire representation.
        :rtype: DetectionSchema
        """
        return cls(
            detection_score=detection.detection_score,
            mask_score=detection.mask_score,
            confidence=detection.confidence,
        )


@final
class PersonSchema(BaseModel):
    """Face-analysis summary, present only when it was requested."""

    model_config = ConfigDict(frozen=True)

    genders: list[int] = Field(
        ..., description="0 = Male, 1 = Female. One entry per face."
    )
    age_bands: list[str] = Field(
        ...,
        description=(
            "Estimated age range per face, index-aligned with genders."
        ),
    )
    is_adult: bool = Field(
        ...,
        description=(
            "True only when no face can be a minor. The band spanning "
            "the threshold never certifies adulthood."
        ),
    )

    @classmethod
    def of(cls, person: PersonPayload) -> Self:
        """Project a domain payload onto the wire.

        :param person: Domain face summary.
        :type person: app.domain.models.PersonPayload
        :returns: The wire representation.
        :rtype: PersonSchema
        """
        return cls(
            genders=[int(gender) for gender in person.genders],
            age_bands=[band.value for band in person.age_bands],
            is_adult=person.is_adult,
        )


@final
class SegmentSchema(BaseModel):
    """Complete response of a successful segmentation."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    mask: str = Field(
        ..., description="Base64 PNG grayscale mask, cropped to bbox."
    )
    image: str = Field(
        ..., description="Base64 PNG original image, cropped to bbox."
    )
    bbox: BBoxSchema
    detections: list[DetectionSchema]
    confidence: float = Field(
        ..., description="Weakest detection's confidence."
    )
    reliable: bool = Field(
        ..., description="Whether confidence clears the threshold."
    )
    segmenter: str
    person: PersonSchema | None = None

    @classmethod
    def of(
        cls,
        result: SegmentedImage,
        prompt: str,
        segmenter: str,
        mask: str,
        image: str,
    ) -> Self:
        """Assemble the response from a domain result.

        :param result: What the use case produced.
        :type result: app.domain.models.SegmentedImage
        :param prompt: Prompt the caller supplied.
        :type prompt: str
        :param segmenter: Backend that served the request.
        :type segmenter: str
        :param mask: Base64 PNG of the cropped mask.
        :type mask: str
        :param image: Base64 PNG of the cropped source.
        :type image: str
        :returns: The wire representation.
        :rtype: SegmentSchema
        """
        return cls(
            prompt=prompt,
            mask=mask,
            image=image,
            bbox=BBoxSchema.of(result.bbox),
            detections=[
                DetectionSchema.of(detection)
                for detection in result.detections
            ],
            confidence=result.confidence,
            reliable=result.reliable,
            segmenter=segmenter,
            person=(
                None
                if result.person is None
                else PersonSchema.of(result.person)
            ),
        )


__all__: list[str] = [
    "BBoxSchema",
    "DetectionSchema",
    "ErrorSchema",
    "HealthSchema",
    "PersonSchema",
    "SegmentSchema",
    "SegmentersSchema",
]
