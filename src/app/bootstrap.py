"""Composition root.

The one place that constructs concrete adapters and binds them to the
capabilities a use case declares. Everything above this module receives
what it needs; nothing above it reaches for a global (``SKILL.md``
sections 14 and 26).

Model loading is eager rather than lazy-on-first-request: ownership is
explicit, and the readiness probe already encodes the cold-start window
that eager loading needs.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from types import TracebackType
from typing import Self, final

from stateless.effect import run

from app.application.capabilities import (
    FaceAnalyser,
    MaskDilator,
    MaskRefiner,
    ObjectDetector,
)
from app.application.effects import CaughtSegment, SegmentFailure
from app.application.wiring import wire_segment
from app.application.policies import SegmentationPolicy
from app.domain import (
    FaceAnalysisUnavailable,
    PersonPayload,
    Prompt,
    SegmentedImage,
    SourceImage,
)
from app.infrastructure.dino import load_detector
from app.infrastructure.facelib import load_face_analyser
from app.infrastructure.imaging import OpenCvMaskDilator
from app.infrastructure.sam2 import load_refiner
from app.settings import Settings

type SegmentOutcome = SegmentedImage | SegmentFailure
"""What a wired segmentation returns: the result, or a typed failure."""


@final
@dataclass(frozen=True, slots=True)
class Application:
    """Everything the interfaces layer is allowed to reach for."""

    settings: Settings
    #: Backend name to the wired use case. One entry today; the mapping
    #: exists because the name is public API, not because a registry is.
    backends: dict[str, CaughtSegment]
    face_analyser: FaceAnalyser | None

    def segment(
        self,
        backend: str,
        image: SourceImage,
        prompt: Prompt,
        person: PersonPayload | None,
    ) -> SegmentOutcome:
        """Run a wired segmentation to completion.

        :param backend: Registered backend name.
        :type backend: str
        :param image: Decoded source image.
        :type image: app.domain.models.SourceImage
        :param prompt: Validated prompt.
        :type prompt: app.domain.models.Prompt
        :param person: Face summary to attach, if any.
        :type person: app.domain.models.PersonPayload | None
        :returns: The segmented image, or the typed failure that stopped
            it. Failures are values here, so the transport layer maps
            them with a match instead of a try block.
        :rtype: SegmentOutcome
        :raises KeyError: If ``backend`` is not registered. Callers
            validate the name against :attr:`backends` first, so reaching
            this is a defect.
        """
        return run(self.backends[backend](image, prompt, person))

    def analyse_faces(self, image: SourceImage) -> PersonPayload:
        """Summarise the faces in an image.

        :param image: Decoded source image.
        :type image: app.domain.models.SourceImage
        :returns: The face summary.
        :rtype: app.domain.models.PersonPayload
        :raises app.domain.errors.FaceAnalysisUnavailable: If the
            optional extra was not installed at startup.
        """
        if self.face_analyser is None:
            raise FaceAnalysisUnavailable(
                detail="The 'person' extra is not installed."
            )
        return self.face_analyser.analyse(image)


@final
class ApplicationScope:
    """Owns every long-lived resource for the process lifetime.

    A context manager rather than module state: the resources close
    deterministically, and a test can build a scope, use it and drop it
    without touching a global (``SKILL.md`` section 16).
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

    analyser: FaceAnalyser | None = None
    try:
        analyser = load_face_analyser()
    except FaceAnalysisUnavailable:
        # Optional by design: the API answers 501 for face requests and
        # serves every other route normally.
        analyser = None

    return Application(
        settings=settings,
        backends={settings.default_segmenter: wired},
        face_analyser=analyser,
    )


__all__: list[str] = [
    "Application",
    "ApplicationScope",
    "SegmentOutcome",
    "build",
]
