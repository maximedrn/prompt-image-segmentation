"""Composition root."""

from contextlib import ExitStack
from types import TracebackType
from typing import Self, final

from app.application.capabilities import (
    FaceAnalyser,
    MaskDilator,
    MaskRefiner,
    ObjectDetector,
)
from app.application.effects import CaughtSegment
from app.application.policies import SegmentationPolicy
from app.application.wiring import wire_segment
from app.bootstrap.models import Application
from app.infrastructure.imaging.imaging import OpenCvMaskDilator
from app.infrastructure.inference.dino import load_detector
from app.infrastructure.inference.faces import load_face_analyser
from app.infrastructure.inference.sam2 import load_refiner
from app.settings import Settings


@final
class ApplicationScope:
    """Owns every long-lived resource for the process lifetime.

    A context manager rather than module state: the resources close
    deterministically, and a test can build a scope, use it and drop it
    without touching a global.
    """

    def __init__(self, settings: Settings) -> None:
        """Record the configuration the scope will build from.

        :param settings: Validated configuration.
        :type settings: app.settings.Settings
        """
        self._settings: Settings = settings
        self._stack: ExitStack = ExitStack()

    def __enter__(self) -> Application:
        """Construct the adapters and wire the use cases.

        :returns: The assembled application.
        :rtype: Application
        """
        self._stack.__enter__()
        return build(self._settings)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release everything the scope opened.

        :param exception_type: Type of the in-flight exception, if any.
        :type exception_type: type[BaseException] | None
        :param exception: The in-flight exception, if any.
        :type exception: BaseException | None
        :param traceback: Traceback of the in-flight exception, if any.
        :type traceback: types.TracebackType | None
        :returns: ``None``; exceptions are never suppressed.
        :rtype: None
        """
        self._stack.__exit__(exception_type, exception, traceback)

    def __call__(self) -> Self:
        """Return the scope itself, so it reads well inline.

        :returns: This scope.
        :rtype: ApplicationScope
        """
        return self


def build(settings: Settings) -> Application:
    """Construct adapters and supply them to the use cases.

    :param settings: Validated configuration.
    :type settings: app.settings.Settings
    :returns: The assembled application.
    :rtype: Application
    :raises app.domain.errors.ModelUnavailable: If a model cannot be
        loaded, which makes the whole service unable to serve.
    """
    # Annotated as the capability, not the concrete class: supply()
    # infers Need[T] from the declared type, and a structural match
    # would otherwise degrade to Need[object].
    detector: ObjectDetector = load_detector(settings.detection_policy())
    refiner: MaskRefiner = load_refiner()
    dilator: MaskDilator = OpenCvMaskDilator()
    policy: SegmentationPolicy = settings.segmentation_policy()

    wired: CaughtSegment = wire_segment(
        detector=detector, refiner=refiner, dilator=dilator, policy=policy
    )

    # No longer optional: the three artefacts total under 400 MB and
    # ride the same hub cache as the rest, so there is nothing left to
    # make optional.
    analyser: FaceAnalyser = load_face_analyser(settings.face_policy())

    return Application(
        settings=settings,
        backends={settings.default_segmenter: wired},
        face_analyser=analyser,
    )


__all__: list[str] = ["ApplicationScope", "build"]
