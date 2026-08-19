"""Infrastructure vocabulary.

Model coordinates, device names and the literal arguments the machine
learning libraries expect. Grouped into named objects so no adapter ever
spells one of them inline.
"""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, final


@unique
class DeviceType(StrEnum):
    """Torch device families this application resolves between."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@unique
class TextEncoding(StrEnum):
    """Character encodings used when serialising to the wire."""

    UTF8 = "utf-8"


@unique
class TensorFormat(StrEnum):
    """The ``return_tensors`` values the processors accept."""

    PYTORCH = "pt"


@final
@dataclass(frozen=True, slots=True)
class ModelSource:
    """Where one model comes from, and at which exact revision.

    Revisions are pinned: an unpinned ``from_pretrained`` follows the
    branch head silently, which makes a build impossible to reproduce and
    hands the upstream account a supply-chain foothold.
    """

    repository: str
    revision: str


@final
@dataclass(frozen=True, slots=True)
class DetectorDefaults:
    """Fixed arguments of the GroundingDINO adapter."""

    #: Keeps multi-scale deformable attention on the pure-PyTorch path,
    #: which is what lets ROCm, MPS and CPU run without a compiled
    #: extension.
    disable_custom_kernels: bool = True
    #: GroundingDINO expects a lowercase caption terminated by a period.
    caption_terminator: str = "."


@final
@dataclass(frozen=True, slots=True)
class RefinerDefaults:
    """Fixed arguments of the SAM 2.1 adapter."""

    #: One mask per box: the union is built by the use case, so asking
    #: for alternatives would only cost memory.
    multimask_output: bool = False


@final
@dataclass(frozen=True, slots=True)
class FaceModelSource:
    """A face-detection model published as a single ONNX file."""

    repository: str
    revision: str
    filename: str


@final
@dataclass(frozen=True, slots=True)
class FaceDetectorDefaults:
    """Fixed arguments of the YuNet detector.

    ``score_threshold`` is the calibration knob: YuNet answers on animal
    faces too, and on the bundled examples a human scores 0.94 while a
    dog and a cat score 0.71 and 0.67. This default sits between them
    with margin, and is worth re-measuring on real traffic.
    """

    score_threshold: float = 0.8
    nms_threshold: float = 0.3
    top_k: int = 50
    #: YuNet is constructed with a placeholder size, then told the real
    #: one per image through ``setInputSize``.
    initial_size: tuple[int, int] = (320, 320)
    #: A crop smaller than this carries no usable facial detail.
    minimum_crop_pixels: int = 24
    #: The detection row is (x, y, w, h, ...landmarks..., score).
    score_index: int = -1


@final
@dataclass(frozen=True, slots=True)
class CudaAllocator:
    """Environment tuning applied before the first CUDA allocation."""

    variable: str = "PYTORCH_CUDA_ALLOC_CONF"
    expandable_segments: str = "expandable_segments:True"


GROUNDING_DINO: Final[ModelSource] = ModelSource(
    repository="IDEA-Research/grounding-dino-tiny",
    revision="a2bb814dd30d776dcf7e30523b00659f4f141c71",
)
SAM: Final[ModelSource] = ModelSource(
    repository="facebook/sam2.1-hiera-tiny",
    revision="de431c4043854a71d8101e17995dfe596bf101a5",
)
FACE_DETECTOR: Final[FaceModelSource] = FaceModelSource(
    repository="opencv/face_detection_yunet",
    revision="3cc26e7f1014a5ee5d74a42acee58bafc9d0a310",
    filename="face_detection_yunet_2023mar.onnx",
)
AGE_ESTIMATOR: Final[ModelSource] = ModelSource(
    repository="dima806/fairface_age_image_detection",
    revision="4e02ab8057ea7fd74b1670940995c5dfda3e6ec0",
)
GENDER_ESTIMATOR: Final[ModelSource] = ModelSource(
    repository="prithivMLmods/Realistic-Gender-Classification",
    revision="f2651e15953b3779aea922a73f6b016c66e8b05b",
)
DETECTOR: Final[DetectorDefaults] = DetectorDefaults()
FACE: Final[FaceDetectorDefaults] = FaceDetectorDefaults()
REFINER: Final[RefinerDefaults] = RefinerDefaults()
ALLOCATOR: Final[CudaAllocator] = CudaAllocator()


__all__: list[str] = [
    "AGE_ESTIMATOR",
    "ALLOCATOR",
    "DETECTOR",
    "FACE",
    "FACE_DETECTOR",
    "GENDER_ESTIMATOR",
    "GROUNDING_DINO",
    "REFINER",
    "SAM",
    "CudaAllocator",
    "DetectorDefaults",
    "DeviceType",
    "FaceDetectorDefaults",
    "FaceModelSource",
    "ModelSource",
    "RefinerDefaults",
    "TensorFormat",
    "TextEncoding",
]
