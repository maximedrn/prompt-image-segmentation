"""Business services.

Services are the single point of entry for use cases. They orchestrate
segmenters, managers, and infrastructure into complete flows; they
never leak model / framework types to the API or UI.
"""

from app.services.person_service import PersonService
from app.services.segmentation_service import (
    SegmentationService,
)

__all__: list[str] = ["PersonService", "SegmentationService"]
