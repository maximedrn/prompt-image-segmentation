"""Torch device and precision resolution (CUDA / ROCm / XPU / MPS / CPU).

Precedence: CUDA, then XPU, then MPS, then CPU. ROCm builds of PyTorch
expose their GPUs through the CUDA API too, so the CUDA branch catches
both NVIDIA and AMD transparently. Intel GPUs do not: they have their own
namespace, and only the XPU builds of torch carry it.
"""

from contextlib import AbstractContextManager, nullcontext
from functools import lru_cache

from torch import (
    autocast,
    bfloat16,
    device as TorchDevice,
    dtype as TorchDtype,
    float16,
    float32,
)
from torch.backends import mps as torch_mps
from torch.cuda import (
    is_available as cuda_is_available,
    is_bf16_supported as cuda_is_bf16_supported,
)
from torch.xpu import (
    is_available as xpu_is_available,
    is_bf16_supported as xpu_is_bf16_supported,
)

from app.infrastructure.inference.types import DeviceType

_FIRST_DEVICE_INDEX: int = 0


def _mps_is_available() -> bool:
    """MPS is only usable on Apple Silicon builds of torch.

    :returns: ``True`` if MPS is both built into torch and available.
    :rtype: bool
    """
    return bool(torch_mps.is_built()) and bool(torch_mps.is_available())


@lru_cache(maxsize=1)
def get_device() -> TorchDevice:
    """Return the process-wide torch device.

    :returns: The best available torch device.
    :rtype: torch.device
    """
    if cuda_is_available():
        return TorchDevice(f"{DeviceType.CUDA}:{_FIRST_DEVICE_INDEX}")
    if xpu_is_available():
        return TorchDevice(f"{DeviceType.XPU}:{_FIRST_DEVICE_INDEX}")
    if _mps_is_available():
        return TorchDevice(DeviceType.MPS)
    return TorchDevice(DeviceType.CPU)


@lru_cache(maxsize=1)
def get_model_dtype() -> TorchDtype:
    """Return the dtype model weights are loaded in.

    This is what halves the memory footprint: ``autocast`` only casts
    activations, so weights stay single precision unless they are loaded
    in half precision to begin with.

    Precision is not portable across backends:

    * CUDA / ROCm: ``bfloat16`` where the hardware supports it, else
      ``float16``.
    * XPU: same rule, asked of Intel's own probe. Arc and Xe answer yes;
      the older integrated parts answer no and get ``float16``.
    * MPS: ``float16`` only - ``bfloat16`` raises ``TypeError`` there.
    * CPU: ``float32`` - ``float16`` has no native CPU kernels.

    :returns: The dtype to hand to ``from_pretrained(dtype=...)``.
    :rtype: torch.dtype
    """
    device: TorchDevice = get_device()
    if device.type == DeviceType.CUDA:
        return bfloat16 if cuda_is_bf16_supported() else float16
    if device.type == DeviceType.XPU:
        return bfloat16 if xpu_is_bf16_supported() else float16
    if device.type == DeviceType.MPS:
        return float16
    return float32


def autocast_context() -> AbstractContextManager[autocast | None, bool | None]:
    """Return the mixed-precision context every forward pass runs in.

    Loading the weights in half precision is not sufficient on its own:
    ``transformers`` hard-codes ``float32`` in several places inside
    GroundingDINO (the reference-point grid, the sine position
    embeddings), so a half-precision value meets a single-precision
    sampling grid and ``grid_sample`` raises ``expected input and grid to
    have the same type``.

    ``autocast`` fixes that by inserting per-operation casts, and it was
    measured to leave detection scores unchanged. Half-precision weights
    *plus* this context is the combination that works; neither alone does.

    CPU opts out: weights stay single precision there, so there is no
    mismatch to reconcile.

    The yield type is the union the two branches actually produce:
    torch hands its own context back, ``nullcontext`` hands ``None``.
    No caller binds the ``as`` clause either way - both are entered for
    the side effect - but naming it beats widening it to ``object``.

    :returns: An autocast context on GPU backends, a no-op on CPU.
    :rtype: contextlib.AbstractContextManager[torch.autocast | None,
        bool | None]
    """
    device: TorchDevice = get_device()
    if device.type == DeviceType.CPU:
        return nullcontext()
    return autocast(device.type, dtype=get_model_dtype())


__all__: list[str] = ["autocast_context", "get_device", "get_model_dtype"]
