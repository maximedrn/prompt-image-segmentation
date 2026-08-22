"""SAM 2.1 adapter: coarse boxes to a pixel-accurate union mask.

Implements :class:`~app.application.capabilities.MaskRefiner`.

The checkpoint is published as a video model; loading it into the image
model keeps only the image subset, which is both what this service needs
and several hundred megabytes cheaper than the memory bank.
"""

from collections.abc import Callable
from functools import partial
from typing import final

from numpy import any as array_any, bool_, uint8
from numpy.typing import NDArray
from torch import Tensor, device as TorchDevice, dtype as TorchDtype
from transformers import BatchFeature, Sam2Model, Sam2Processor
from transformers.models.sam2.modeling_sam2 import (
    Sam2ImageSegmentationOutput,
)

from app.domain import (
    MaskArray,
    MaskValues,
    ModelUnavailable,
    PixelBox,
    SourceImage,
)
from app.infrastructure.inference.constants import (
    Checkpoints,
    RefinerDefaults,
    TensorKey,
)
from app.infrastructure.inference.device import get_device, get_model_dtype
from app.infrastructure.inference.execution import (
    exclusive_device,
    fetching,
    place,
)
from app.infrastructure.inference.types import TensorFormat


@final
class Sam2MaskRefiner:
    """Refines boxes into one union mask with SAM 2.1 Hiera-Tiny."""

    def __init__(self, model: Sam2Model, processor: Sam2Processor) -> None:
        """Bind an already-loaded model to its processor.

        :param model: Device-placed segmentation model.
        :type model: transformers.Sam2Model
        :param processor: Matching processor.
        :type processor: transformers.Sam2Processor
        """
        self._model: Sam2Model = model
        self._processor: Sam2Processor = processor

    def refine(
        self, image: SourceImage, boxes: tuple[PixelBox, ...]
    ) -> tuple[tuple[MaskArray, ...], tuple[float, ...]]:
        """Return one mask per box, and how good each one is.

        :param image: Source image.
        :type image: app.domain.SourceImage
        :param boxes: Non-empty boxes to segment.
        :type boxes: tuple[app.domain.PixelBox, ...]
        :returns: ``(masks, mask_scores)``, both index-aligned with
            the input boxes.
        :rtype: tuple[tuple[app.domain.MaskArray, ...],
            tuple[float, ...]]
        :raises app.domain.errors.DeviceExhausted: On accelerator
            out-of-memory.
        :raises app.domain.errors.ModelUnavailable: If the checkpoint
            answers without masks, which makes its output meaningless.
        """
        # One critical section for the whole device pass, post-processing
        # included: it runs torch ops on device tensors too, and letting
        # it overlap another thread's forward aborts the process on MPS.
        with exclusive_device(RefinerDefaults.operation):
            # transformers declares the processor's extra arguments
            # through **kwargs upstream, so pyright cannot see them.
            boxed: list[list[list[float]]] = [
                [list(box.as_tuple()) for box in boxes]
            ]
            # Declared as ``BatchEncoding`` upstream, but the image
            # processor it delegates to returns a ``BatchFeature`` -
            # which is the class whose ``to()`` takes a dtype below.
            inputs: BatchFeature = (
                self._processor(  # pyright: ignore[reportAssignmentType]
                    images=image.pixels,
                    input_boxes=boxed,  # pyright: ignore[reportCallIssue]
                    return_tensors=TensorFormat.PYTORCH,
                )
            )
            # BatchFeature.to(device, dtype) casts only the floating
            # point entries, leaving input ids intact. The declared
            # overload takes one positional argument.
            device: TorchDevice = get_device()
            dtype: TorchDtype = get_model_dtype()
            inputs = inputs.to(
                device, dtype  # pyright: ignore[reportCallIssue]
            )
            outputs: Sam2ImageSegmentationOutput = self._model(
                **inputs, multimask_output=RefinerDefaults.multimask_output
            )
            predicted: Tensor | None = outputs.pred_masks
            quality: Tensor | None = outputs.iou_scores
            if predicted is None or quality is None:
                raise ModelUnavailable(
                    model=Checkpoints.sam.repository,
                    detail="The refiner returned no mask.",
                )
            # Carries no annotations upstream.
            restore: Callable[[Tensor, Tensor], list[Tensor]] = (
                self._processor.post_process_masks
            )
            restored: list[Tensor] = restore(
                predicted, inputs[TensorKey.original_sizes]
            )
            stacked: NDArray[bool_] = (
                restored[RefinerDefaults.first_image]
                .cpu()
                .numpy()
                .astype(bool)
            )
            scores: tuple[float, ...] = tuple(
                quality.float().flatten().tolist()
            )
        # (num_boxes, num_masks, H, W): SAM proposes several masks per
        # box, so only that axis collapses. Merging the box axis too is
        # the caller's decision, and it would not be reversible here.
        merged: NDArray[bool_] = array_any(stacked, axis=1)
        masks: tuple[MaskArray, ...] = tuple(
            (mask * MaskValues.foreground).astype(uint8) for mask in merged
        )
        return masks, scores


def load_refiner() -> Sam2MaskRefiner:
    """Build the refiner, downloading its weights if needed.

    :returns: A ready refiner bound to the process-wide device.
    :rtype: Sam2MaskRefiner
    :raises app.domain.errors.ModelUnavailable: If the weights cannot be
        fetched or the model cannot be built.
    """
    processor: Sam2Processor = fetching(
        Checkpoints.sam.repository,
        partial(
            Sam2Processor.from_pretrained,
            Checkpoints.sam.repository,
            revision=Checkpoints.sam.revision,
        ),
    )
    model: Sam2Model = fetching(
        Checkpoints.sam.repository,
        partial(
            Sam2Model.from_pretrained,
            Checkpoints.sam.repository,
            revision=Checkpoints.sam.revision,
            dtype=get_model_dtype(),
        ),
    )
    placed: Sam2Model = place(model)
    return Sam2MaskRefiner(model=placed, processor=processor)


__all__: list[str] = ["Sam2MaskRefiner", "load_refiner"]
