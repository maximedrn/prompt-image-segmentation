"""Thread-safe singleton metaclass.

Any class using :class:`SingletonMeta` as its metaclass will yield the
same instance on every construction, even under concurrent access.
Singletons must expose a zero-argument ``__init__`` - construction
arguments would require ParamSpec typing that pyright cannot bind
through metaclass ``__call__``.
"""

from threading import Lock
from typing import Final, TypeVar, cast

_T = TypeVar("_T")

_INSTANCE_ATTR: Final[str] = "_singleton_instance"


class SingletonMeta(type):
    """Metaclass that memoizes exactly one instance per subclass.

    The cached instance lives on each singleton class itself (under
    :data:`_INSTANCE_ATTR`), so no shared heterogeneous registry is
    needed and every attribute has a concrete type.
    """

    _lock: Lock = Lock()

    def __call__(cls: type[_T]) -> _T:
        """Return the cached instance, building it on first call.

        Uses a double-checked lock so concurrent constructions collapse
        to a single instance.

        :returns: The memoised instance for ``cls``.
        :rtype: _T
        """
        if _INSTANCE_ATTR in cls.__dict__:
            return cast(_T, cls.__dict__[_INSTANCE_ATTR])
        with SingletonMeta._lock:
            if _INSTANCE_ATTR not in cls.__dict__:
                setattr(cls, _INSTANCE_ATTR, type.__call__(cls))
            return cast(_T, cls.__dict__[_INSTANCE_ATTR])


__all__: list[str] = ["SingletonMeta"]
