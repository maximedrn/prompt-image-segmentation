"""Schema of what one segmentation asks for.

The result models live in :mod:`app.application.jobs.results`, because
a job produces them; this module keeps what is genuinely HTTP, which is
the multipart body.
"""

from typing import final

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.application.policies import SegmentOptions
from app.domain import PercentageBounds, ScoreBounds


@final
class SegmentForm(BaseModel):
    """Everything ``POST /jobs`` reads from its multipart body.

    One model rather than a dozen parameters: FastAPI only flattens a
    form model when it is the sole body parameter, so splitting the
    request between a model and loose fields would silently ignore the
    model. It also gives the request one documented schema.

    Every numeric field is optional and falls back to the operator's
    configuration; the flags shape the response and carry their own
    defaults.
    """

    model_config = ConfigDict(frozen=True)

    image: UploadFile = Field(..., description="Image to segment.")
    prompt: str = Field(
        ..., description="Text prompt (comma or dot separated)."
    )
    person_mode: bool = Field(
        default=False, description="Enable face analysis."
    )
    segmenter: str | None = Field(default=None, description="Backend name.")
    callback_url: str | None = Field(
        default=None,
        description=(
            "Where to POST the outcome once the job is terminal. "
            "Signed with X-Signature; https and a public address only."
        ),
    )
    minimum_confidence: float | None = Field(
        default=None,
        ge=ScoreBounds.minimum,
        le=ScoreBounds.maximum,
        description=(
            "Drop detections whose combined confidence falls below this. "
            "Defaults to the configured minimum."
        ),
    )
    dilation_percentage: float | None = Field(
        default=None,
        ge=PercentageBounds.minimum,
        le=PercentageBounds.maximum,
        description="Grow the returned mask by this share of its size.",
    )
    padding_percentage: float | None = Field(
        default=None,
        ge=PercentageBounds.minimum,
        le=PercentageBounds.maximum,
        description="Margin around the crop. Ignored when crop is false.",
    )
    split_masks: bool = Field(
        default=False,
        description="Return one mask per detection instead of one union.",
    )
    crop: bool = Field(
        default=True,
        description=(
            "Crop the mask and image to the subject. When false, masks keep "
            "the source dimensions and no image is returned."
        ),
    )

    def to_options(self) -> SegmentOptions:
        """Hand the request's wishes to the application layer.

        :returns: The application-level options.
        :rtype: app.application.policies.SegmentOptions
        """
        return SegmentOptions(
            minimum_confidence=self.minimum_confidence,
            dilation_percentage=self.dilation_percentage,
            padding_percentage=self.padding_percentage,
            split_masks=self.split_masks,
            crop=self.crop,
        )


__all__: list[str] = ["SegmentForm"]
