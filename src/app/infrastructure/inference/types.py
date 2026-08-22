"""Closed sets and value objects the model adapters exchange."""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import final


@unique
class DeviceType(StrEnum):
    """Torch device families this application resolves between."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


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
class FaceModelSource:
    """A face-detection model published as a single ONNX file."""

    repository: str
    revision: str
    filename: str


__all__: list[str] = [
    "DeviceType",
    "FaceModelSource",
    "ModelSource",
    "TensorFormat",
]
