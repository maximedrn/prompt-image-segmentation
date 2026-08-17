"""SAM 2.1 adapter: coarse boxes to a pixel-accurate union mask.

Implements :class:`~app.application.capabilities.MaskRefiner`.

The checkpoint is published as a video model; loading it into the image
model keeps only the image subset, which is both what this service needs
and several hundred megabytes cheaper than the memory bank.
"""

from functools import partial
from typing import final

from numpy import any as array_any, uint8
from numpy.typing import NDArray
from torch import Tensor
from transformers import Sam2Model, Sam2Processor

from app.domain import MASK, MaskArray, PixelBox, SourceImage
from app.infrastructure.constants import REFINER, SAM, TensorFormat
from app.infrastructure.device import get_device, get_model_dtype
from app.infrastructure.execution import exclusive_device, fetching, place

_OPERATION: str = "refine"
#: The processor batches images; this adapter always submits exactly one.
_FIRST_IMAGE: int = 0


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
    ) -> tuple[MaskArray, tuple[float, ...]]:
        """Return the union mask over ``boxes`` and its per-box quality.

        :param image: Source image.
        :type image: app.domain.models.SourceImage
        :param boxes: Non-empty boxes to segment.
        :type boxes: tuple[app.domain.models.PixelBox, ...]
        :returns: ``(union_mask, mask_scores)``, index-aligned with the
            input boxes.
        :rtype: tuple[app.domain.models.MaskArray, tuple[float, ...]]
        :raises app.domain.errors.DeviceExhausted: On accelerator
            out-of-memory.
        """
        # One critical section for the whole device pass, post-processing
        # included: it runs torch ops on device tensors too, and letting
        # it overlap another thread's forward aborts the process on MPS.
        with exclusive_device(_OPERATION):
            # transformers declares the processor's extra arguments
            # through **kwargs upstream, so pyright cannot see them.
            boxed = [[list(box.as_tuple()) for box in boxes]]
            inputs = self._processor(
                images=image.pixels,
                input_boxes=boxed,  # pyright: ignore[reportCallIssue]
                return_tensors=TensorFormat.PYTORCH,
            )
            # BatchFeature.to(device, dtype) casts only the floating
            # point entries, leaving input ids intact. The declared
            # overload takes one positional argument.
            device, dtype = get_device(), get_model_dtype()
            inputs = inputs.to(
                device, dtype  # pyright: ignore[reportCallIssue]
            )
            outputs = self._model(
                **inputs, multimask_output=REFINER.multimask_output
            )
            # Carries no annotations upstream.
            restore = self._processor.post_process_masks
            restored: list[Tensor] = restore(  # type: ignore[no-untyped-call]
                outputs.pred_masks, inputs["original_sizes"]
            )
            stacked = restored[_FIRST_IMAGE].cpu().numpy().astype(bool)
            scores: tuple[float, ...] = tuple(
                outputs.iou_scores.float().flatten().tolist()
            )
        # (num_boxes, num_masks, H, W) collapses to a single (H, W) union.
        height, width = stacked.shape[-2:]
        union: NDArray[uint8] = (
            array_any(stacked.reshape(-1, height, width), axis=0)
            * MASK.foreground
        ).astype(uint8)
        return union, scores


def load_refiner() -> Sam2MaskRefiner:
    """Build the refiner, downloading its weights if needed.

    :returns: A ready refiner bound to the process-wide device.
    :rtype: Sam2MaskRefiner
    :raises app.domain.errors.ModelUnavailable: If the weights cannot be
        fetched or the model cannot be built.
    """
    processor: Sam2Processor = fetching(
        SAM.repository,
        partial(
            Sam2Processor.from_pretrained,
            SAM.repository,
            revision=SAM.revision,
        ),
    )
    model: Sam2Model = fetching(
        SAM.repository,
        partial(
            Sam2Model.from_pretrained,
            SAM.repository,
            revision=SAM.revision,
            dtype=get_model_dtype(),
        ),
    )
    placed: Sam2Model = place(model)
    return Sam2MaskRefiner(model=placed, processor=processor)


__all__: list[str] = ["Sam2MaskRefiner", "load_refiner"]
