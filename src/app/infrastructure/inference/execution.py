"""How device work is entered: serialised, half precision, OOM-typed."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import Lock
from typing import Final

from torch import OutOfMemoryError, inference_mode
from torch.cuda import empty_cache as cuda_empty_cache
from torch.xpu import empty_cache as xpu_empty_cache
from transformers import PreTrainedModel

from app.domain import DeviceExhausted, ModelUnavailable
from app.infrastructure.inference.constants import InferenceText
from app.infrastructure.inference.device import autocast_context, get_device
from app.infrastructure.inference.types import DeviceType


def fetching[ArtefactT](
    model: str, build: Callable[[], ArtefactT]
) -> ArtefactT:
    """Build one model artefact, normalising every fault into one failure.

    :param model: Repository identifier, for the failure detail.
    :type model: str
    :param build: Zero-argument builder, usually a bound
        ``from_pretrained`` closed over its arguments.
    :type build: collections.abc.Callable[[], ArtefactT]
    :returns: Whatever the builder produced.
    :rtype: ArtefactT
    :raises app.domain.errors.ModelUnavailable: On any fetch or build
        fault - network, disk or checkpoint - because they all mean
        the same thing here: this capability cannot serve.
    """
    try:
        return build()
    except Exception as error:
        raise ModelUnavailable(model=model, detail=str(error)) from error


def place[ModelT: PreTrainedModel](model: ModelT) -> ModelT:
    """Move a freshly built model onto the device, in inference mode.

    The ignores cover a ``transformers`` stub artefact: ``to()`` and
    ``eval()`` are wrapped by decorators that mypy resolves as unbound
    functions, so it reads the device argument as ``self``.

    :param model: A freshly built model, still on the host.
    :type model: ModelT
    :returns: The same model, device-placed and frozen for inference.
    :rtype: ModelT
    """
    placed: ModelT = model.to(get_device())  # type: ignore[arg-type]
    # torch's evaluation-mode switch, which disables dropout and freezes
    # batch-norm statistics. Unrelated to the Python builtin of the same
    # name: it evaluates no code.
    placed.eval()  # type: ignore[no-untyped-call]
    return placed


# Deliberate: one process-wide lock. The GPU serialises the work anyway, so
# a per-device semaphore only becomes worthwhile with a second device.
_DEVICE_LOCK: Final[Lock] = Lock()

# Which allocator to drain after an out-of-memory, per backend. Each one
# lives in its own namespace and only exists for its own device, so this
# cannot collapse into a single call. Backends absent from the table -
# CPU above all - allocate through the host and have nothing to release.
_CACHE_RELEASE: Final[dict[str, Callable[[], None]]] = {
    DeviceType.CUDA: cuda_empty_cache,
    DeviceType.XPU: xpu_empty_cache,
}


@contextmanager
def exclusive_device(operation: str) -> Generator[None]:
    """Run a block as the only device user, in half precision.

    :param operation: Short name of the work, used in the failure detail.
    :type operation: str
    :returns: A context manager yielding once.
    :rtype: collections.abc.Generator[None]
    :raises app.domain.errors.DeviceExhausted: If the accelerator runs
        out of memory. The allocator cache is released first, otherwise
        it stays fragmented and every later request fails too.
    """
    with _DEVICE_LOCK, inference_mode(), autocast_context():
        try:
            yield
        except OutOfMemoryError as error:
            release: Callable[[], None] | None = _CACHE_RELEASE.get(
                get_device().type
            )
            if release is not None:
                release()
            raise DeviceExhausted(
                detail=InferenceText.exhausted.format(
                    operation=operation, error=error
                )
            ) from error


__all__: list[str] = ["exclusive_device", "fetching", "place"]
