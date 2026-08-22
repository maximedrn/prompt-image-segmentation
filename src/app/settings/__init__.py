"""Configuration layer.

The facade the rest of the application imports from, so the split
between defaults, closed sets and the validated model stays internal.
"""

from app.settings.constants import Defaults, EnvVar
from app.settings.settings import Settings
from app.settings.types import AuthMode

__all__: list[str] = ["AuthMode", "Defaults", "EnvVar", "Settings"]
