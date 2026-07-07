"""Type stubs for the ``facelib`` optional extra."""

# pylint: disable=locally-disabled
# pylint: disable=suppressed-message,useless-suppression
# pylint: disable=unused-variable,unused-argument

from numpy import uint8
from numpy.typing import NDArray

class FaceDetector:
    """Detect and align faces in an RGB uint8 image."""

    def __init__(self) -> None:
        """Instantiate the detector (loads weights lazily)."""

    def detect_align(self, image: NDArray[uint8]) -> tuple[
        NDArray[uint8],
        NDArray[uint8],
        NDArray[uint8],
        NDArray[uint8],
    ]:
        """Return ``(bboxes, landmarks, faces, warped_faces)`` arrays."""

class AgeGenderEstimator:
    """Predict per-face age and gender labels."""

    def __init__(self) -> None:
        """Instantiate the estimator (loads weights lazily)."""

    def detect(self, faces: NDArray[uint8]) -> tuple[list[str], list[int]]:
        """Return ``(gender_labels, ages)`` for the stacked faces."""
