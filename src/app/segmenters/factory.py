"""Segmenter factory (single shared instance).

Concrete backends decorate themselves with
``@SEGMENTER_FACTORY.register("name")`` and become available through
:meth:`~app.core.factory.Factory.get`.
"""

from typing import Final

from app.core.factory import Factory
from app.segmenters.base import Segmenter

SEGMENTER_FACTORY: Final[Factory[Segmenter]] = Factory[Segmenter](
    kind="segmenter"
)


__all__: list[str] = ["SEGMENTER_FACTORY"]
