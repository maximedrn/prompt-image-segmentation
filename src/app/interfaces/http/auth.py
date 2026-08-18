"""HTTP Basic authentication, with constant-time comparison.

Transport-level: authentication is not a domain concept here, so its
failure type lives with the transport rather than in
``app.domain.errors``.
"""

from dataclasses import dataclass
from secrets import compare_digest
from typing import Annotated, final

from fastapi import Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.interfaces.http.constants import MESSAGE
from app.settings import Settings

_SECURITY: HTTPBasic = HTTPBasic(auto_error=False)

type Credentials = Annotated[HTTPBasicCredentials | None, Depends(_SECURITY)]
"""What FastAPI parses out of the ``Authorization`` header."""


@final
@dataclass(frozen=True, slots=True)
class Unauthorized(Exception):
    """The caller presented absent or invalid credentials."""

    message: str


def require_credentials(
    request: Request,
    credentials: Credentials,
) -> None:
    """Validate Basic credentials; a no-op when auth is disabled.

    Both comparisons always run: short-circuiting on the username would
    leak, through timing, whether a username exists.

    :param request: Incoming request, carrying the wired settings.
    :type request: fastapi.Request
    :param credentials: Credentials FastAPI parsed, if the client sent
        an ``Authorization`` header.
    :type credentials: fastapi.security.HTTPBasicCredentials | None
    :raises Unauthorized: When authentication is on and the credentials
        are absent or wrong.
    """
    # Read from the transport, not the wired application: rejecting an
    # unauthenticated caller must not require the models to be loaded.
    settings: Settings = request.app.state.settings
    if not settings.auth_enabled:
        return
    if credentials is None:
        raise Unauthorized(message=MESSAGE.authentication_required)
    user_matches: bool = compare_digest(
        credentials.username, settings.segmentation_username or ""
    )
    password_matches: bool = compare_digest(
        credentials.password, settings.segmentation_password or ""
    )
    if not (user_matches and password_matches):
        raise Unauthorized(message=MESSAGE.invalid_credentials)


__all__: list[str] = ["Unauthorized", "require_credentials"]
