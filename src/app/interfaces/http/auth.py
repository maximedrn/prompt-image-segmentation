"""HTTP Basic authentication, with constant-time comparison.

Transport-level: authentication is not a domain concept here, so its
failure type lives with the transport rather than in
``app.domain.errors``.

Credentials are compared as UTF-8 bytes rather than as text:
``compare_digest`` raises on a non-ASCII ``str``, and a caller chooses
what it puts in the header.
"""

from base64 import b64decode
from collections.abc import Mapping
from dataclasses import dataclass
from secrets import compare_digest
from typing import Annotated, final

from fastapi import Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.infrastructure.imaging.types import TextEncoding
from app.interfaces.http.constants import BasicScheme, Message
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
        raise Unauthorized(message=Message.authentication_required)
    user_matches: bool = compare_digest(
        credentials.username.encode(TextEncoding.UTF8),
        (settings.segmentation_username or "").encode(TextEncoding.UTF8),
    )
    password_matches: bool = compare_digest(
        credentials.password.encode(TextEncoding.UTF8),
        (settings.segmentation_password or "").encode(TextEncoding.UTF8),
    )
    if not (user_matches and password_matches):
        raise Unauthorized(message=Message.invalid_credentials)


def socket_authorised(headers: Mapping[str, str], settings: Settings) -> bool:
    """Check Basic credentials carried on a WebSocket upgrade.

    A socket cannot answer 401 and be asked again, so this returns a
    verdict rather than raising: the caller closes the connection with
    a policy-violation code.

    Browsers cannot set headers on a WebSocket handshake, which is why
    this suits server-to-server callers. A signed short-lived ticket
    would cover browsers, and is not built until something needs it.

    :param headers: Headers of the upgrade request.
    :type headers: collections.abc.Mapping[str, str]
    :param settings: Wired configuration.
    :type settings: app.settings.Settings
    :returns: ``True`` when the socket may proceed.
    :rtype: bool
    """
    if not settings.auth_enabled:
        return True
    header: str | None = headers.get(BasicScheme.header)
    if header is None or not header.startswith(BasicScheme.prefix):
        return False
    try:
        decoded: str = b64decode(
            header.removeprefix(BasicScheme.prefix), validate=True
        ).decode(TextEncoding.UTF8)
    # UnicodeDecodeError is a ValueError; both mean the header was
    # not what it claimed to be.
    except ValueError:
        return False
    username, _, password = decoded.partition(BasicScheme.separator)
    # Both comparisons always run, for the same reason as above.
    user_matches: bool = compare_digest(
        username.encode(TextEncoding.UTF8),
        (settings.segmentation_username or "").encode(TextEncoding.UTF8),
    )
    password_matches: bool = compare_digest(
        password.encode(TextEncoding.UTF8),
        (settings.segmentation_password or "").encode(TextEncoding.UTF8),
    )
    return user_matches and password_matches


__all__: list[str] = [
    "Unauthorized",
    "require_credentials",
    "socket_authorised",
]
