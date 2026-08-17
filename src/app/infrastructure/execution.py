"""How device work is entered: serialised, half precision, OOM-typed.

Three concerns that every model call shares, in one place instead of
repeated per adapter:

* **serialisation** - one shared model instance cannot serve two
  inferences at once, and unbounded concurrency is how this service runs
  out of memory;
* **precision** - the half-precision context that makes the sub-gigabyte
  budget reachable;
* **failure normalisation** - a driver ``OutOfMemoryError`` becomes the
  domain's :class:`~app.domain.errors.DeviceExhausted`, so no torch
  exception class ever reaches the application layer
  (``SKILL.md`` section 4).
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import Lock
from typing import Final, TypeVar

from torch import OutOfMemoryError, inference_mode
from torch.cuda import empty_cache, is_available as cuda_is_available
from transformers import PreTrainedModel

from app.domain import DeviceExhausted, ModelUnavailable
from app.infrastructure.device import autocast_context, get_device

_ModelT = TypeVar("_ModelT", bound=PreTrainedModel)
_ArtefactT = TypeVar("_ArtefactT")


def fetching(model: str, build: Callable[[], _ArtefactT]) -> _ArtefactT:
    """Build one model artefact, normalising every fault into one failure.

    :param model: Repository identifier, for the failure detail.
    :type model: str
    :param build: Zero-argument builder, usually a bound
        ``from_pretrained`` closed over its arguments.
    :type build: collections.abc.Callable[[], _ArtefactT]
    :returns: Whatever the builder produced.
    :rtype: _ArtefactT
    :raises app.domain.errors.ModelUnavailable: On any fetch or build
        fault - network, disk or checkpoint - because they all mean
        the same thing here: this capability cannot serve.
    """
    try:
        return build()
    except Exception as error:
        raise ModelUnavailable(model=model, detail=str(error)) from error


def place(model: _ModelT) -> _ModelT:
    """Move a freshly built model onto the device, in inference mode.

    The ignores cover a ``transformers`` stub artefact: ``to()`` and
    ``eval()`` are wrapped by decorators that mypy resolves as unbound
    functions, so it reads the device argument as ``self``.

    :param model: A freshly built model, still on the host.
    :type model: _ModelT
    :returns: The same model, device-placed and frozen for inference.
    :rtype: _ModelT
    """
    placed: _ModelT = model.to(get_device())  # type: ignore[arg-type]
    # torch's evaluation-mode switch, which disables dropout and freezes
    # batch-norm statistics. Unrelated to the Python builtin of the same
    # name: it evaluates no code.
    placed.eval()  # type: ignore[no-untyped-call]
    return placed


# ponytail: one process-wide lock. The GPU serialises the work anyway, so
# a per-device semaphore only becomes worthwhile with a second device.
_DEVICE_LOCK: Final[Lock] = Lock()


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
            if cuda_is_available():
                empty_cache()
            raise DeviceExhausted(detail=f"{operation}: {error}") from error


__all__: list[str] = ["exclusive_device", "fetching", "place"]
