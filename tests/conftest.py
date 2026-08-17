"""Test environment.

Settings are read once per process and refuse to start in ``basic`` mode
without credentials, so the environment has to be set before anything
under ``app`` is imported.
"""

import os
from typing import Final

USERNAME: Final[str] = "tester"
PASSWORD: Final[str] = "secret"

os.environ.update({
    "AUTH_MODE": "basic",
    "SEGMENTATION_USERNAME": USERNAME,
    "SEGMENTATION_PASSWORD": PASSWORD,
    "ENABLE_UI": "false",
    # Small enough that a test payload can cross it cheaply.
    "MAX_UPLOAD_BYTES": "2048",
    "RATE_LIMIT_REQUESTS": "5",
    "RATE_LIMIT_WINDOW_SECONDS": "60",
})
