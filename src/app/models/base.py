"""ModelLoader interface."""

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

ModelT_co = TypeVar("ModelT_co", covariant=True)


class ModelLoader(ABC, Generic[ModelT_co]):
    """Load one specific model. Stateless: :meth:`load` returns fresh.

    ``ModelT_co`` is covariant so ``SamLoader(ModelLoader[SamPredictor])``
    is a subtype of ``ModelLoader[SamPredictor]`` and each concrete
    loader exposes its own model type through :meth:`load`.
    """

    identifier: ClassVar[str] = ""
    """Human-readable identifier used in logs and errors."""

    @abstractmethod
    def load(self) -> ModelT_co:
        """Instantiate and return the model on device.

        :returns: The freshly built model, ready for inference.
        :rtype: ModelT_co
        :raises NotImplementedError: If a subclass forgets to override.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a compact identifier suitable for logs.

        :returns: ``<ClassName 'identifier'>``.
        :rtype: str
        """
        return f"<{type(self).__name__} {self.identifier!r}>"


__all__: list[str] = ["ModelLoader", "ModelT_co"]
