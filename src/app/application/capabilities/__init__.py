"""What the application needs, declared as protocols."""

from app.application.capabilities.inference import (
    FaceAnalyser,
    MaskDilator,
    MaskRefiner,
    ObjectDetector,
)
from app.application.capabilities.jobs import JobNotifier, JobStore

__all__: list[str] = [
    "FaceAnalyser",
    "JobNotifier",
    "JobStore",
    "MaskDilator",
    "MaskRefiner",
    "ObjectDetector",
]
