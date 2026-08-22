"""Test environment, and the fixtures every module builds on.

``Settings`` refuses to start in ``basic`` mode without credentials, so
the environment has to be set before anything *builds* one. Importing
the settings module is safe and happens here: it declares the fields,
it does not read them.
"""

# Imported by name from the test modules; nothing in here calls it,
# which is what ``allow-global-unused-variables`` flags.
# pylint: disable=unused-variable

import os
from typing import TYPE_CHECKING, Final

from app.settings import AuthMode, EnvVar

if TYPE_CHECKING:
    from app.application.jobs.results import SegmentSchema

USERNAME: Final[str] = "tester"
PASSWORD: Final[str] = "secret"
#: Small enough that a test payload can cross it cheaply.
UPLOAD_LIMIT: Final[int] = 2048
BUDGET: Final[int] = 5
WINDOW_SECONDS: Final[int] = 60

os.environ.update({
    EnvVar.AUTH_MODE: AuthMode.BASIC,
    EnvVar.SEGMENTATION_USERNAME: USERNAME,
    EnvVar.SEGMENTATION_PASSWORD: PASSWORD,
    EnvVar.ENABLE_UI: str(False).lower(),
    EnvVar.MAX_UPLOAD_BYTES: str(UPLOAD_LIMIT),
    EnvVar.RATE_LIMIT_REQUESTS: str(BUDGET),
    EnvVar.RATE_LIMIT_WINDOW_SECONDS: str(WINDOW_SECONDS),
})


def segment_body(prompt: str = "dog") -> "SegmentSchema":
    """Build the smallest answer a finished job can carry.

    ``JobResult.body`` is a model, not a mapping, so a test that wants a
    finished job has to build one that validates - which is the point of
    typing it.

    :param prompt: Prompt the imagined request carried.
    :type prompt: str
    :returns: A valid, detection-free result body.
    :rtype: app.application.jobs.results.SegmentSchema
    """
    # Imported late: the environment above has to be set first.
    # pylint: disable-next=import-outside-toplevel
    from app.application.jobs.results import SegmentSchema

    return SegmentSchema(
        prompt=prompt,
        regions=[],
        detections=[],
        confidence=1.0,
        reliable=True,
        segmenter="sam_dino",
    )
