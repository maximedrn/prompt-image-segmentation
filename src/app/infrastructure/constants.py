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
DETECTOR: Final[DetectorDefaults] = DetectorDefaults()
REFINER: Final[RefinerDefaults] = RefinerDefaults()
ALLOCATOR: Final[CudaAllocator] = CudaAllocator()


__all__: list[str] = [
    "ALLOCATOR",
    "DETECTOR",
    "GROUNDING_DINO",
    "REFINER",
    "SAM",
    "CudaAllocator",
    "DetectorDefaults",
    "DeviceType",
    "ModelSource",
    "RefinerDefaults",
    "TensorFormat",
    "TextEncoding",
]
