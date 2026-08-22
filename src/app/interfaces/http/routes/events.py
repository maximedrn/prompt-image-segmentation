"""The event socket: one job's transitions, pushed as they happen.

A router of its own: the job router's dependencies expect a ``Request``,
which a WebSocket upgrade does not carry. Its credentials are checked in
the handler instead, and a refusal closes the socket rather than
answering a status a socket cannot carry.
"""

from asyncio import sleep

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.application.capabilities import JobStore
from app.domain import JobState
from app.interfaces.http.auth import socket_authorised
from app.interfaces.http.constants import (
    HttpRoute,
    OpenApiTag,
    Serialisation,
    SocketRules,
)
from app.interfaces.http.rate_limit import within_rate_limit
from app.interfaces.http.routes.dependencies import body_of
from app.interfaces.http.schemas import JobSchema
from app.settings import Settings

events_router: APIRouter = APIRouter(tags=[OpenApiTag.segmentation])


async def _stream(
    websocket: WebSocket, store: JobStore, identifier: str
) -> None:
    """Push each new state of one job until it stops moving.

    :param websocket: The accepted connection.
    :type websocket: fastapi.WebSocket
    :param store: Where the job's state lives.
    :type store: app.application.capabilities.JobStore
    :param identifier: Job to follow.
    :type identifier: str
    """
    reported: JobState | None = None
    # Deliberate: see SocketRules.poll_seconds - the server polls so the client
    # does not have to.
    while True:  # pylint: disable=while-used
        if (found := await store.read(identifier)) is None:
            await websocket.close(code=SocketRules.policy_violation)
            return
        job, result = found
        if job.state is not reported:
            await websocket.send_json(
                JobSchema.of(job, result=body_of(result)).model_dump(
                    mode=Serialisation.json_mode
                )
            )
            reported = job.state
        if job.terminal:
            await websocket.close(code=SocketRules.normal_close)
            return
        await sleep(SocketRules.poll_seconds)


@events_router.websocket(HttpRoute.job_events)
async def job_events(websocket: WebSocket, identifier: str) -> None:
    """Stream one job's transitions until it stops moving.

    Exists so a caller does not have to poll: the same states, pushed
    as they happen, and the socket closes itself on a terminal one.
    Polling remains for anything that cannot hold a connection.

    :param websocket: The upgraded connection.
    :type websocket: fastapi.WebSocket
    :param identifier: What acceptance handed back.
    :type identifier: str
    """
    settings: Settings = websocket.app.state.settings
    if not socket_authorised(websocket.headers, settings):
        await websocket.close(code=SocketRules.policy_violation)
        return
    # Budgeted like any other request: without this a caller can hold
    # open as many subscriptions as it likes, each one polling the store
    # four times a second for as long as the job lives.
    if not within_rate_limit(websocket):
        await websocket.close(code=SocketRules.try_again_later)
        return
    await websocket.accept()
    try:
        await _stream(websocket, websocket.app.state.store, identifier)
    except WebSocketDisconnect:
        # The caller left, which is an ordinary end to a subscription.
        return


__all__: list[str] = ["events_router", "job_events"]
