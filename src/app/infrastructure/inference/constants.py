"""Fixed arguments and pinned checkpoints of the model adapters."""

from dataclasses import dataclass
from typing import ClassVar, final

from app.domain import AgeBand
from app.infrastructure.inference.types import FaceModelSource, ModelSource


@final
@dataclass(frozen=True, slots=True)
class Checkpoints:
    """Every model this service loads, pinned to a revision."""

    grounding_dino: ClassVar[ModelSource] = ModelSource(
        repository="IDEA-Research/grounding-dino-tiny",
        revision="a2bb814dd30d776dcf7e30523b00659f4f141c71",
    )
    sam: ClassVar[ModelSource] = ModelSource(
        repository="facebook/sam2.1-hiera-tiny",
        revision="de431c4043854a71d8101e17995dfe596bf101a5",
    )
    face_detector: ClassVar[FaceModelSource] = FaceModelSource(
        repository="opencv/face_detection_yunet",
        revision="3cc26e7f1014a5ee5d74a42acee58bafc9d0a310",
        filename="face_detection_yunet_2023mar.onnx",
    )
    age_estimator: ClassVar[ModelSource] = ModelSource(
        repository="dima806/fairface_age_image_detection",
        revision="4e02ab8057ea7fd74b1670940995c5dfda3e6ec0",
    )
    gender_estimator: ClassVar[ModelSource] = ModelSource(
        repository="prithivMLmods/Realistic-Gender-Classification",
        revision="f2651e15953b3779aea922a73f6b016c66e8b05b",
    )


@final
@dataclass(frozen=True, slots=True)
class TensorKey:
    """Keys the ``transformers`` processors label their outputs with.

    Part of the library's wire format rather than ours, which is exactly
    why they are written down once: an upgrade that renames one should
    fail in a single place.
    """

    input_ids: ClassVar[str] = "input_ids"
    boxes: ClassVar[str] = "boxes"
    scores: ClassVar[str] = "scores"
    original_sizes: ClassVar[str] = "original_sizes"


@final
@dataclass(frozen=True, slots=True)
class DetectorDefaults:
    """Fixed arguments of the GroundingDINO adapter."""

    #: Keeps multi-scale deformable attention on the pure-PyTorch path,
    #: which is what lets ROCm, MPS and CPU run without a compiled
    #: extension.
    disable_custom_kernels: ClassVar[bool] = True
    #: GroundingDINO expects a lowercase caption terminated by a period.
    caption_terminator: ClassVar[str] = "."
    #: ``post_process_grounded_object_detection`` returns one entry per
    #: image and the adapter always submits exactly one.
    first_image: ClassVar[int] = 0
    operation: ClassVar[str] = "detect"


@final
@dataclass(frozen=True, slots=True)
class RefinerDefaults:
    """Fixed arguments of the SAM 2.1 adapter."""

    #: One mask per box: the union is built by the use case, so asking
    #: for alternatives would only cost memory.
    multimask_output: ClassVar[bool] = False
    #: The processor batches images; the adapter submits exactly one.
    first_image: ClassVar[int] = 0
    operation: ClassVar[str] = "refine"


@final
@dataclass(frozen=True, slots=True)
class FaceDetectorDefaults:
    """Fixed arguments of the YuNet detector.

    ``score_threshold`` is the calibration knob: YuNet answers on animal
    faces too, and on the bundled examples a human scores 0.94 while a
    dog and a cat score 0.71 and 0.67. This default sits between them
    with margin, and is worth re-measuring on real traffic.
    """

    score_threshold: ClassVar[float] = 0.8
    nms_threshold: ClassVar[float] = 0.3
    top_k: ClassVar[int] = 50
    #: YuNet is constructed with a placeholder size, then told the real
    #: one per image through ``setInputSize``.
    initial_size: ClassVar[tuple[int, int]] = (320, 320)
    #: A crop smaller than this carries no usable facial detail.
    minimum_crop_pixels: ClassVar[int] = 24
    #: The detection row is (x, y, w, h, ...landmarks..., score).
    score_index: ClassVar[int] = -1
    #: The winning class of a classifier answer.
    first_result: ClassVar[int] = 0
    operation: ClassVar[str] = "analyse-faces"


@final
@dataclass(frozen=True, slots=True)
class Labels:
    """What the estimators publish, mapped onto the domain's vocabulary.

    So no model string ever reaches the application layer.
    """

    male: ClassVar[str] = "male portrait"
    age_bands: ClassVar[dict[str, AgeBand]] = {
        "0-2": AgeBand.INFANT,
        "3-9": AgeBand.CHILD,
        "10-19": AgeBand.TEEN,
        "20-29": AgeBand.TWENTIES,
        "30-39": AgeBand.THIRTIES,
        "40-49": AgeBand.FORTIES,
        "50-59": AgeBand.FIFTIES,
        "60-69": AgeBand.SIXTIES,
        "more than 70": AgeBand.SEVENTIES_PLUS,
    }


@final
@dataclass(frozen=True, slots=True)
class InferenceText:
    """Detail an adapter attaches to a failure it re-raises."""

    #: Names the operation that ran out, then what torch said.
    exhausted: ClassVar[str] = "{operation}: {error}"


@final
@dataclass(frozen=True, slots=True)
class CudaAllocator:
    """Environment tuning applied before the first CUDA allocation."""

    variable: ClassVar[str] = "PYTORCH_CUDA_ALLOC_CONF"
    expandable_segments: ClassVar[str] = "expandable_segments:True"


__all__: list[str] = [
    "InferenceText",
    "TensorKey",
    "Checkpoints",
    "CudaAllocator",
    "DetectorDefaults",
    "FaceDetectorDefaults",
    "Labels",
    "RefinerDefaults",
]
