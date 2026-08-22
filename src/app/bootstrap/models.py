"""What the composition root hands to the layers above."""

from dataclasses import dataclass
from typing import Final, final

from stateless.effect import run

from app.application.capabilities import (
    FaceAnalyser,
)
from app.application.effects import CaughtSegment, SegmentFailure
from app.application.policies import SegmentOptions
from app.domain import (
    PersonPayload,
    Prompt,
    SegmentedImage,
    SourceImage,
)
from app.settings import Settings

type SegmentOutcome = SegmentedImage | SegmentFailure
"""What a wired segmentation returns: the result, or a typed failure."""

#: What a caller gets when it asks for nothing in particular.
_DEFAULT_OPTIONS: Final[SegmentOptions] = SegmentOptions()


@final
@dataclass(frozen=True, slots=True)
class Application:
    """Everything the interfaces layer is allowed to reach for."""

    settings: Settings
    #: Backend name to the wired use case. One entry today; the mapping
    #: exists because the name is public API, not because a registry is.
    backends: dict[str, CaughtSegment]
    face_analyser: FaceAnalyser

    def segment(
        self,
        backend: str,
        image: SourceImage,
        prompt: Prompt,
        person: PersonPayload | None,
        options: SegmentOptions = _DEFAULT_OPTIONS,
    ) -> SegmentOutcome:
        """Run a wired segmentation to completion.

        :param backend: Registered backend name.
        :type backend: str
        :param image: Decoded source image.
        :type image: app.domain.SourceImage
        :param prompt: Validated prompt.
        :type prompt: app.domain.Prompt
        :param person: Face summary to attach, if any.
        :type person: app.domain.PersonPayload | None
        :param options: What this caller asked of this segmentation.
        :type options: app.application.policies.SegmentOptions
        :returns: The segmented image, or the typed failure that stopped
            it. Failures are values here, so the transport layer maps
            them with a match instead of a try block.
        :rtype: SegmentOutcome
        :raises KeyError: If ``backend`` is not registered. Callers
            validate the name against :attr:`backends` first, so reaching
            this is a defect.
        """
        return run(self.backends[backend](image, prompt, person, options))

    def analyse_faces(self, image: SourceImage) -> PersonPayload:
        """Summarise the faces in an image.

        :param image: Decoded source image.
        :type image: app.domain.SourceImage
        :returns: The face summary.
        :rtype: app.domain.PersonPayload
        """
        return self.face_analyser.analyse(image)


__all__: list[str] = ["Application", "SegmentOutcome"]
