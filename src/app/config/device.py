"""Torch device resolution (CUDA / ROCm / MPS / CPU).

Precedence: ``cuda:0`` -> ``mps`` -> ``cpu``. ROCm builds of PyTorch
expose their GPUs through the CUDA API too, so ``cuda:0`` catches
both NVIDIA and AMD backends transparently.

Wrapped in a function so tests / scripts can freely monkeypatch, and so
importing this module never triggers CUDA initialization as a side
effect of any other import in the tree.
"""

from functools import lru_cache

from torch import device as TorchDevice
from torch.backends import mps as torch_mps
from torch.cuda import is_available as cuda_is_available


def _mps_is_available() -> bool:
    """MPS is only usable on Apple Silicon builds of torch.

    :returns: ``True`` if MPS is both built into torch and available on
        this machine.
    :rtype: bool
    """
    return bool(torch_mps.is_built()) and bool(torch_mps.is_available())


@lru_cache(maxsize=1)
def get_device() -> TorchDevice:
    """Return the process-wide torch device.

    Precedence: ``cuda:0`` -> ``mps`` -> ``cpu``. Result is cached for
    the life of the process (LRU size 1).

    :returns: The best available torch device.
    :rtype: torch.device
    """
    if cuda_is_available():
        return TorchDevice("cuda:0")
    if _mps_is_available():
        return TorchDevice("mps")
    return TorchDevice("cpu")


__all__: list[str] = ["get_device"]
