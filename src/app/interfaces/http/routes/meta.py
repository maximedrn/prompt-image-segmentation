"""Probes and the backend listing.

No credentials and no rate limit: a probe has to answer before any
secret is configured, and none of these exposes image data.
"""

from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.interfaces.http.constants import (
    ErrorCode,
    HealthState,
    HttpRoute,
    JobMessage,
    Message,
    OpenApiKey,
    OpenApiTag,
    RouteDocs,
    StateKey,
)
from app.interfaces.http.errors import failure_response
from app.interfaces.http.routes.dependencies import application_of, store_of
from app.interfaces.http.schemas import (
    ErrorSchema,
    HealthSchema,
    SegmentersSchema,
)

meta_router: APIRouter = APIRouter(tags=[OpenApiTag.meta])


@meta_router.get(
    HttpRoute.health,
    response_model=HealthSchema,
    summary=RouteDocs.liveness.summary,
    description=RouteDocs.liveness.description,
)
def liveness() -> HealthSchema:
    """Report that the process is up, before the models are.

    Answers immediately so a container runtime can tell a cold start
    from a wedged process.

    :returns: The liveness payload.
    :rtype: app.interfaces.http.schemas.HealthSchema
    """
    return HealthSchema(status=HealthState.OK)


@meta_router.get(
    HttpRoute.ready,
    response_model=HealthSchema,
    responses={
        HTTPStatus.SERVICE_UNAVAILABLE: {OpenApiKey.model: ErrorSchema}
    },
    summary=RouteDocs.readiness.summary,
    description=RouteDocs.readiness.description,
)
async def readiness(request: Request) -> HealthSchema | JSONResponse:
    """Report that the models are resident and the queue is reachable.

    Both are asked on every probe rather than once at startup, so a
    dependency that comes back is reflected without a restart, and a
    queue that goes away stops traffic instead of collecting jobs
    nothing will ever claim.

    :param request: Incoming request.
    :type request: fastapi.Request
    :returns: The readiness payload, or a 503 while something is
        missing.
    :rtype: app.interfaces.http.schemas.HealthSchema
        | fastapi.responses.JSONResponse
    """
    if getattr(request.app.state, StateKey.application, None) is None:
        return failure_response(
            HTTPStatus.SERVICE_UNAVAILABLE,
            ErrorCode.NOT_READY,
            Message.models_loading,
        )
    if not await store_of(request).ready():
        return failure_response(
            HTTPStatus.SERVICE_UNAVAILABLE,
            ErrorCode.STORE_UNAVAILABLE,
            JobMessage.store_unavailable,
        )
    return HealthSchema(status=HealthState.READY)


@meta_router.get(
    HttpRoute.segmenters,
    response_model=SegmentersSchema,
    summary=RouteDocs.segmenters.summary,
    description=RouteDocs.segmenters.description,
)
def list_segmenters(request: Request) -> SegmentersSchema:
    """Enumerate the wired segmentation backends.

    :param request: Incoming request.
    :type request: fastapi.Request
    :returns: The backend listing.
    :rtype: app.interfaces.http.schemas.SegmentersSchema
    """
    return SegmentersSchema(available=sorted(application_of(request).backends))


__all__: list[str] = [
    "list_segmenters",
    "liveness",
    "meta_router",
    "readiness",
]
