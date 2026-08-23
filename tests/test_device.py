"""Device precedence and the precision each backend is given.

CUDA first, then MPS, then CPU. No machine can exercise more than one of
those branches, and the machine that runs this suite usually exercises
none of them, so every probe is stubbed here: what follows states the
resolution rules rather than the hardware of the test run.

The resolvers are cached for the process lifetime - one device, decided
once - so each scenario clears them on the way in and on the way out.
"""

# pytest collects the test functions by name; nothing in the module
# calls them, which is what ``allow-global-unused-variables`` flags.
# pylint: disable=unused-variable

from collections.abc import Iterator
from contextlib import nullcontext
from typing import Final

import pytest
from torch import autocast, bfloat16, dtype as TorchDtype, float16, float32

from app.infrastructure.inference import device as resolution
from app.infrastructure.inference.device import (
    autocast_context,
    get_device,
    get_model_dtype,
)
from app.infrastructure.inference.types import DeviceType

#: The availability probe standing behind each accelerator, by name, so
#: a scenario can answer for all of them at once.
PROBES: Final[dict[DeviceType, str]] = {
    DeviceType.CUDA: "cuda_is_available",
    DeviceType.MPS: "_mps_is_available",
}


@pytest.fixture(autouse=True)
def _forget_resolution() -> Iterator[None]:
    """Clear the cached device and dtype around every scenario.

    :returns: A fixture yielding once, with clean caches either side.
    :rtype: collections.abc.Iterator[None]
    """
    get_device.cache_clear()
    get_model_dtype.cache_clear()
    yield
    get_device.cache_clear()
    get_model_dtype.cache_clear()


def _present(
    monkeypatch: pytest.MonkeyPatch,
    *backends: DeviceType,
    bf16: bool = True,
) -> None:
    """Answer the availability probes as if only ``backends`` existed.

    :param monkeypatch: The patcher, so every stub is undone after.
    :type monkeypatch: pytest.MonkeyPatch
    :param backends: The accelerators to report as available. None of
        them means a host with no accelerator at all.
    :type backends: app.infrastructure.inference.types.DeviceType
    :param bf16: What the ``bfloat16`` probes answer.
    :type bf16: bool
    """
    for backend, probe in PROBES.items():
        monkeypatch.setattr(
            resolution, probe, lambda backend=backend: backend in backends
        )
    monkeypatch.setattr(resolution, "is_bf16_supported", lambda: bf16)


def test_cuda_is_taken_before_anything_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host answering for every backend still resolves to CUDA.

    ROCm reports itself through the CUDA probe, so this covers both.
    """
    _present(monkeypatch, *PROBES)
    assert get_device() == resolution.TorchDevice("cuda:0")


def test_mps_is_taken_when_no_cuda_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPS is reached only once CUDA has declined, and carries no index."""
    _present(monkeypatch, DeviceType.MPS)
    assert get_device() == resolution.TorchDevice("mps")


def test_cpu_is_the_fallback_when_nothing_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No accelerator at all resolves, without raising, to the CPU."""
    _present(monkeypatch)
    assert get_device() == resolution.TorchDevice("cpu")


@pytest.mark.parametrize(
    ("backend", "bf16", "expected"),
    [
        (DeviceType.CUDA, True, bfloat16),
        (DeviceType.CUDA, False, float16),
        # Neither of these asks a probe: MPS raises on bfloat16, and the
        # CPU has no half-precision kernels to run.
        (DeviceType.MPS, True, float16),
        (None, True, float32),
    ],
    ids=["cuda-bf16", "cuda-fp16", "mps-fp16", "cpu-fp32"],
)
def test_each_backend_loads_weights_in_its_own_precision(
    backend: DeviceType | None,
    bf16: bool,
    expected: TorchDtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half precision where the backend has it, single precision else."""
    _present(monkeypatch, *([backend] if backend else []), bf16=bf16)
    assert get_model_dtype() is expected


# Building a CUDA autocast on a host without one makes torch say it is
# disabling itself. Expected here: the probes are stubbed, the hardware
# is not, and the object under assertion is built either way.
@pytest.mark.filterwarnings("ignore:CUDA is not available")
def test_accelerator_work_runs_under_an_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mixed-precision context follows the device it resolved to."""
    _present(monkeypatch, DeviceType.CUDA)
    context = autocast_context()
    assert isinstance(context, autocast)
    assert context.device == DeviceType.CUDA


def test_cpu_work_runs_under_no_autocast_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPU keeps single precision, so it has nothing to reconcile."""
    _present(monkeypatch)
    assert isinstance(autocast_context(), nullcontext)
