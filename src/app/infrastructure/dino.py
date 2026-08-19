"""GroundingDINO adapter: open-vocabulary detection from a text prompt.

Implements :class:`~app.application.capabilities.ObjectDetector`. One of
only two modules allowed to import ``transformers``.
"""

from functools import partial
from typing import final

from torch import Tensor, device as TorchDevice, dtype as TorchDtype
from transformers import (
    BatchFeature,
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
)
from transformers.models.grounding_dino.modeling_grounding_dino import (
    GroundingDinoObjectDetectionOutput,
)

from app.application.policies import DetectionPolicy
from app.domain import PixelBox, Prompt, SourceImage
from app.infrastructure.constants import (
    DETECTOR,
    GROUNDING_DINO,
    TensorFormat,
)
from app.infrastructure.device import get_device, get_model_dtype
from app.infrastructure.execution import exclusive_device, fetching, place

_OPERATION: str = "detect"
#: ``post_process_grounded_object_detection`` returns one entry per image
#: and this adapter always submits exactly one.
_FIRST_IMAGE: int = 0


def _caption(prompt: Prompt) -> str:
    """Normalise a prompt into the caption GroundingDINO expects.

    :param prompt: Validated prompt.
    :type prompt: app.domain.models.Prompt
    :returns: Lowercased caption with a terminating period.
    :rtype: str
    """
    lowered: str = prompt.text.lower()
    if lowered.endswith(DETECTOR.caption_terminator):
        return lowered
    return f"{lowered}{DETECTOR.caption_terminator}"


@final
class GroundingDinoDetector:
    """Detects prompt-matching regions with GroundingDINO."""

    def __init__(
        self,
        model: GroundingDinoForObjectDetection,
        processor: GroundingDinoProcessor,
        policy: DetectionPolicy,
    ) -> None:
        """Bind an already-loaded model to its processor and thresholds.

        Construction takes the model rather than loading it, so ownership
        stays in the bootstrap (``SKILL.md`` section 26).

        :param model: Device-placed detection model.
        :type model: transformers.GroundingDinoForObjectDetection
        :param processor: Matching processor.
        :type processor: transformers.GroundingDinoProcessor
        :param policy: Score thresholds applied to the raw output.
        :type policy: app.application.policies.DetectionPolicy
        """
        self._model: GroundingDinoForObjectDetection = model
        self._processor: GroundingDinoProcessor = processor
        self._policy: DetectionPolicy = policy

    def detect(
        self, image: SourceImage, prompt: Prompt
    ) -> tuple[tuple[PixelBox, ...], tuple[float, ...]]:
        """Return the boxes matching ``prompt`` and their confidences.

        :param image: Source image.
        :type image: app.domain.models.SourceImage
        :param prompt: Validated prompt.
        :type prompt: app.domain.models.Prompt
        :returns: ``(boxes, scores)``, index-aligned and possibly empty.
        :rtype: tuple[tuple[app.domain.models.PixelBox, ...],
            tuple[float, ...]]
        :raises app.domain.errors.DeviceExhausted: On accelerator
            out-of-memory.
        """
        # The whole device pass is one critical section, post-processing
        # included: it runs torch ops on device tensors too, and letting
        # it overlap another thread's forward aborts the process on MPS.
        with exclusive_device(_OPERATION):
            # transformers declares the processor's extra arguments
            # through **kwargs upstream, so pyright cannot see them.
            fmt: TensorFormat = TensorFormat.PYTORCH
            inputs: BatchFeature = self._processor(
                images=image.pixels,
                text=_caption(prompt),
                return_tensors=fmt,  # pyright: ignore[reportCallIssue]
            )
            device: TorchDevice = get_device()
            dtype: TorchDtype = get_model_dtype()
            inputs = inputs.to(device, dtype)
            outputs: GroundingDinoObjectDetectionOutput = self._model(**inputs)
            detected: dict[str, Tensor] = (
                self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=self._policy.score_threshold,
                    text_threshold=self._policy.text_threshold,
                    target_sizes=[(image.height, image.width)],
                )[_FIRST_IMAGE]
            )
            box_values: list[list[float]] = (
                detected["boxes"].float().cpu().tolist()
            )
            score_values: list[float] = (
                detected["scores"].float().cpu().tolist()
            )
        boxes: tuple[PixelBox, ...] = tuple(
            PixelBox(left=left, top=top, right=right, bottom=bottom)
            for left, top, right, bottom in box_values
        )
        return boxes, tuple(score_values)


def load_detector(policy: DetectionPolicy) -> GroundingDinoDetector:
    """Build the detector, downloading its weights if needed.

    :param policy: Score thresholds for the adapter.
    :type policy: app.application.policies.DetectionPolicy
    :returns: A ready detector bound to the process-wide device.
    :rtype: GroundingDinoDetector
    :raises app.domain.errors.ModelUnavailable: If the weights cannot be
        fetched or the model cannot be built.
    """
    processor: GroundingDinoProcessor = fetching(
        GROUNDING_DINO.repository,
        partial(
            GroundingDinoProcessor.from_pretrained,
            GROUNDING_DINO.repository,
            revision=GROUNDING_DINO.revision,
        ),
    )
    model: GroundingDinoForObjectDetection = fetching(
        GROUNDING_DINO.repository,
        partial(
            GroundingDinoForObjectDetection.from_pretrained,
            GROUNDING_DINO.repository,
            revision=GROUNDING_DINO.revision,
            dtype=get_model_dtype(),
            disable_custom_kernels=DETECTOR.disable_custom_kernels,
        ),
    )
    placed: GroundingDinoForObjectDetection = place(model)
    return GroundingDinoDetector(
        model=placed, processor=processor, policy=policy
    )


__all__: list[str] = ["GroundingDinoDetector", "load_detector"]
