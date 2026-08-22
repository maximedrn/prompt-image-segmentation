"""Pydantic projection of one segmentation result.

In the application layer rather than beside the transport, because it
is what a *job* produces: :class:`~app.application.jobs.JobResult`
names this type, and a queue that answered with an untyped mapping
would be handing its caller something nothing had checked.

The HTTP layer re-exports these, so they remain the response models the
OpenAPI document is generated from.
"""

from collections.abc import Callable
from typing import Self, final

from pydantic import BaseModel, ConfigDict, Field

from app.domain import (
    BBox,
    Detection,
    MaskArray,
    PersonPayload,
    SegmentedImage,
    SegmentRegion,
    age_range,
)


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
        :type bbox: app.domain.BBox
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
        :type detection: app.domain.Detection
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
    age_bands_digits: list[tuple[int, int]] = Field(
        ...,
        description=(
            "Youngest and oldest age of each band, inclusive, "
            "index-aligned with age_bands."
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
        :type person: app.domain.PersonPayload
        :returns: The wire representation.
        :rtype: PersonSchema
        """
        return cls(
            genders=[int(gender) for gender in person.genders],
            age_bands=[band.value for band in person.age_bands],
            age_bands_digits=[age_range(band) for band in person.age_bands],
            is_adult=person.is_adult,
        )


@final
class RegionSchema(BaseModel):
    """One returned mask, its box, and the detection behind it."""

    model_config = ConfigDict(frozen=True)

    bbox: BBoxSchema
    mask: str = Field(..., description="Base64 PNG grayscale mask.")
    image: str | None = Field(
        default=None,
        description=(
            "Base64 PNG of the source under bbox. Null when the caller "
            "declined cropping, since the uncropped image is the one it "
            "already holds."
        ),
    )
    detection: DetectionSchema

    @classmethod
    def of_all(
        cls,
        regions: tuple[SegmentRegion, ...],
        encode: Callable[[MaskArray], str],
    ) -> list[Self]:
        """Project every region, encoding the pixels on the way out.

        The encoder is injected rather than imported: this module knows
        the wire, not the codec, and both callers already hold one.

        :param regions: What the use case produced.
        :type regions: tuple[app.domain.SegmentRegion, ...]
        :param encode: Turns a pixel array into its wire text.
        :type encode: collections.abc.Callable[[MaskArray], str]
        :returns: One wire region per domain region, in order.
        :rtype: list[RegionSchema]
        """
        return [
            cls.of(
                region,
                mask=encode(region.mask),
                image=(None if region.image is None else encode(region.image)),
            )
            for region in regions
        ]

    @classmethod
    def of(cls, region: SegmentRegion, mask: str, image: str | None) -> Self:
        """Project a domain region onto the wire.

        :param region: What the use case produced for one mask.
        :type region: app.domain.SegmentRegion
        :param mask: Base64 PNG of the region's mask.
        :type mask: str
        :param image: Base64 PNG of the cropped source, if cropped.
        :type image: str | None
        :returns: The wire representation.
        :rtype: RegionSchema
        """
        return cls(
            bbox=BBoxSchema.of(region.bbox),
            mask=mask,
            image=image,
            detection=DetectionSchema.of(region.detection),
        )


@final
class SegmentSchema(BaseModel):
    """Complete response of a successful segmentation."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    regions: list[RegionSchema] = Field(
        ...,
        description=(
            "One entry when the masks were merged, one per retained "
            "detection when the caller asked for a split."
        ),
    )
    detections: list[DetectionSchema] = Field(
        ..., description="Every detection retained after filtering."
    )
    confidence: float = Field(
        ..., description="Weakest retained detection's confidence."
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
        regions: list[RegionSchema],
    ) -> Self:
        """Assemble the response from a domain result.

        :param result: What the use case produced.
        :type result: app.domain.SegmentedImage
        :param prompt: Prompt the caller supplied.
        :type prompt: str
        :param segmenter: Backend that served the request.
        :type segmenter: str
        :param regions: Encoded regions, in the order produced.
        :type regions: list[RegionSchema]
        :returns: The wire representation.
        :rtype: SegmentSchema
        """
        return cls(
            prompt=prompt,
            regions=regions,
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
    "PersonSchema",
    "RegionSchema",
    "SegmentSchema",
]
