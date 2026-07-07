"""Face-based gender / age estimation (person extra required).

Lazy-loaded on first instantiation to avoid pulling ``facelib`` into
processes that never enable ``person_mode``.
"""

from typing import Final

from numpy import uint8
from numpy.typing import NDArray

from app.core.singleton import SingletonMeta

ADULT_AGE_THRESHOLD: Final[int] = 18
GENDER_MALE: Final[int] = 0
GENDER_FEMALE: Final[int] = 1
_MALE_LABEL: Final[str] = "Male"
_FACE_STACK_NDIM: Final[int] = 4


class PersonService(metaclass=SingletonMeta):
    """Wraps :mod:`facelib` face detection + age/gender estimation."""

    def __init__(self) -> None:
        """Instantiate the ``facelib`` detector + age/gender estimator.

        Imports ``facelib`` lazily so processes that never touch
        person-mode can skip installing the ``person`` extra.

        :raises ImportError: If the ``person`` extra is not installed.
        """
        # facelib is an optional extra; pulling it in at module import
        # time would break installs that skip the ``person`` group.
        # pylint: disable=locally-disabled
        # pylint: disable=suppressed-message,useless-suppression
        # pylint: disable=import-outside-toplevel
        from facelib import (
            AgeGenderEstimator,
            FaceDetector,
        )

        self._detector = FaceDetector()
        self._estimator = AgeGenderEstimator()

    def analyse(self, image: NDArray[uint8]) -> tuple[list[int], bool]:
        """Return ``(gender_codes, every_face_is_adult)``.

        :param image: Full-frame RGB image (HxWx3 uint8).
        :type image: numpy.typing.NDArray[numpy.uint8]
        :returns: A 2-tuple; first element is the per-face gender code
            list (:data:`GENDER_MALE` / :data:`GENDER_FEMALE`), second
            element is ``True`` when every detected face is at least
            :data:`ADULT_AGE_THRESHOLD` years old (or when no face is
            detected).
        :rtype: tuple[list[int], bool]
        """
        faces, _, _, _ = self._detector.detect_align(image)
        if len(faces.shape) < _FACE_STACK_NDIM:
            return [], True
        raw_genders, raw_ages = self._estimator.detect(faces)
        genders: list[int] = [
            GENDER_MALE if label == _MALE_LABEL else GENDER_FEMALE
            for label in raw_genders
        ]
        is_adult: bool = all(
            int(age) >= ADULT_AGE_THRESHOLD for age in raw_ages
        )
        return genders, is_adult


__all__: list[str] = [
    "ADULT_AGE_THRESHOLD",
    "GENDER_MALE",
    "GENDER_FEMALE",
    "PersonService",
]
