"""Composition root.

The facade the interfaces layer imports from, so the split between what
is built and how it is built stays internal.
"""

from app.bootstrap.build import ApplicationScope, build
from app.bootstrap.models import Application, SegmentOutcome

__all__: list[str] = [
    "Application",
    "ApplicationScope",
    "SegmentOutcome",
    "build",
]
