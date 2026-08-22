"""Wire schemas.

The domain describes itself with plain dataclasses; these models exist
to validate and serialise at a boundary. The result half lives one
layer down, in :mod:`app.application.jobs.results`, because a queued
job produces it before any transport sees it - and is re-exported here,
so the routes and the OpenAPI document read one name for it.
"""

from app.application.jobs.results import (
    BBoxSchema,
    DetectionSchema,
    PersonSchema,
    RegionSchema,
    SegmentSchema,
)
from app.interfaces.http.schemas.common import (
    ErrorSchema,
    HealthSchema,
    JsonValue,
    SegmentersSchema,
)
from app.interfaces.http.schemas.jobs import JobSchema
from app.interfaces.http.schemas.segmentation import SegmentForm

__all__: list[str] = [
    "BBoxSchema",
    "DetectionSchema",
    "ErrorSchema",
    "HealthSchema",
    "JobSchema",
    "JsonValue",
    "PersonSchema",
    "RegionSchema",
    "SegmentForm",
    "SegmentSchema",
    "SegmentersSchema",
]
