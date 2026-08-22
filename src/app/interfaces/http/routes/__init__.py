"""Route handlers.

Three routers, because they are guarded differently: the probes are
open, the job routes carry credentials and a rate limit, and the event
socket checks its own credentials because a WebSocket upgrade has no
``Request`` for a dependency to read.
"""

from app.interfaces.http.routes.events import events_router, job_events
from app.interfaces.http.routes.jobs import (
    cancel_job,
    create_job,
    read_job,
    segmentation_router,
)
from app.interfaces.http.routes.meta import (
    list_segmenters,
    liveness,
    meta_router,
    readiness,
)

__all__: list[str] = [
    "cancel_job",
    "create_job",
    "events_router",
    "job_events",
    "list_segmenters",
    "liveness",
    "meta_router",
    "read_job",
    "readiness",
    "segmentation_router",
]
