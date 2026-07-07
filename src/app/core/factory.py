"""Generic registry-based factory.

Any collection of interchangeable implementations that share a common
base class can get a factory by instantiating ``Factory[Base]``. The
factory owns:

* a registry of ``name -> class`` (via :meth:`register` decorator);
* a lazy cache of ``name -> instance`` (via :meth:`get`).

Registering a duplicate raises; requesting an unknown name raises
:class:`~app.core.exceptions.BackendUnavailableError`.
"""

from threading import Lock
from typing import Generic, TypeVar
from collections.abc import Callable

from app.core.exceptions import BackendUnavailableError

T = TypeVar("T")


class Factory(Generic[T]):
    """Thread-safe factory + registry for a family of ``T`` classes."""

    def __init__(self, kind: str) -> None:
        """Create an empty factory tagged with ``kind``.

        :param kind: Human-readable label for the family (used in
            error messages).
        :type kind: str
        """
        self._kind: str = kind
        self._registry: dict[str, type[T]] = {}
        self._instances: dict[str, T] = {}
        self._lock: Lock = Lock()

    @property
    def kind(self) -> str:
        """Human-readable label for the family (used in errors).

        :returns: The ``kind`` label supplied at construction.
        :rtype: str
        """
        return self._kind

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a class decorator that registers ``cls`` under ``name``.

        :param name: Registry key; must be unique within this factory.
        :type name: str
        :returns: A decorator taking ``cls`` and returning it unchanged
            after inserting it into the registry.
        :rtype: collections.abc.Callable[[type[T]], type[T]]
        :raises ValueError: If ``name`` is already registered.
        """

        def _decorator(cls: type[T]) -> type[T]:
            with self._lock:
                if name in self._registry:
                    raise ValueError(
                        f"{self._kind} {name!r} already registered."
                    )
                self._registry[name] = cls
            return cls

        return _decorator

    def get(self, name: str) -> T:
        """Return the cached instance, building it on first call.

        :param name: Registry key to resolve.
        :type name: str
        :returns: The singleton instance associated with ``name``.
        :rtype: T
        :raises app.core.exceptions.BackendUnavailableError: If
            ``name`` has never been registered.
        """
        with self._lock:
            if (instance := self._instances.get(name)) is not None:
                return instance
            if (cls := self._registry.get(name)) is None:
                raise BackendUnavailableError(
                    kind=self._kind,
                    name=name,
                    available=sorted(self._registry),
                )
            instance = cls()
            self._instances[name] = instance
            return instance

    def names(self) -> list[str]:
        """Return the sorted list of registered names.

        :returns: A snapshot of registry keys, alphabetically ordered.
        :rtype: list[str]
        """
        with self._lock:
            return sorted(self._registry)

    def is_registered(self, name: str) -> bool:
        """Report whether ``name`` has been registered.

        :param name: Registry key to test.
        :type name: str
        :returns: ``True`` if ``name`` maps to a registered class.
        :rtype: bool
        """
        with self._lock:
            return name in self._registry


__all__: list[str] = ["T", "Factory"]
