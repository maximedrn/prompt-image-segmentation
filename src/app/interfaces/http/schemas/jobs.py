"""Schema of a queued job, and of the answer it eventually carries."""

from typing import Final, Self, final

from pydantic import BaseModel, ConfigDict, Field

from app.application.jobs.results import SegmentSchema
from app.domain import Job, JobState
from app.interfaces.http.constants import JobField, OpenApiKey

_EXAMPLE_IDENTIFIER: Final[str] = "0d9a1c1e-3f4b-4e2a-9c8d-7b6a5e4f3d2c"
_EXAMPLE_ACCEPTED_AT: Final[float] = 1755600000.0
_EXAMPLE_POSITION: Final[int] = 3


@final
class JobSchema(BaseModel):
    """What a caller polls for, and collects."""

    # One example rather than a generated one: the generated body fills
    # every field at once, which is a shape no real answer ever has.
    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            OpenApiKey.examples: [{
                JobField.identifier: _EXAMPLE_IDENTIFIER,
                JobField.state: JobState.QUEUED,
                JobField.created_at: _EXAMPLE_ACCEPTED_AT,
                JobField.updated_at: _EXAMPLE_ACCEPTED_AT,
                JobField.queue_position: _EXAMPLE_POSITION,
            }]
        },
    )

    identifier: str = Field(..., description="Poll this back on /jobs.")
    state: JobState = Field(
        ...,
        description=(
            "queued, running, succeeded, failed or cancelled. Only the "
            "last three are terminal."
        ),
    )
    created_at: float
    updated_at: float
    error: str | None = Field(
        default=None, description="Failure code, when the state is failed."
    )
    result: SegmentSchema | None = Field(
        default=None, description="The segmentation, once it succeeded."
    )
    queue_position: int | None = Field(
        default=None,
        description="Jobs ahead of this one, while it is still queued.",
    )

    @classmethod
    def of(
        cls,
        job: Job,
        result: SegmentSchema | None = None,
        queue_position: int | None = None,
    ) -> Self:
        """Project a job onto the wire.

        :param job: The job as stored.
        :type job: app.domain.Job
        :param result: Its answer, when it has one.
        :type result: SegmentSchema | None
        :param queue_position: Jobs ahead of it, while queued.
        :type queue_position: int | None
        :returns: The wire representation.
        :rtype: JobSchema
        """
        return cls(
            identifier=job.identifier,
            state=job.state,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error=job.error,
            result=result,
            queue_position=queue_position,
        )


__all__: list[str] = ["JobSchema"]
