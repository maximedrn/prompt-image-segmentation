"""HTTP Basic authentication dependency (constant-time compare).

Applied to the JSON API. Gradio uses its own ``auth=`` argument, wired
in :func:`app.api.main.create_app`.
"""

from secrets import compare_digest

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import get_settings
from app.config.settings import Settings

_security: HTTPBasic = HTTPBasic(auto_error=False)


def require_basic_auth(
    credentials: HTTPBasicCredentials | None = Depends(_security),
) -> None:
    """Validate Basic Auth; no-op if auth is disabled in settings.

    :param credentials: FastAPI-injected credentials (via ``Depends``).
        ``None`` when the client sends no ``Authorization`` header.
    :type credentials: fastapi.security.HTTPBasicCredentials | None
    :raises fastapi.HTTPException: 401 when auth is required and the
        credentials are absent or invalid.
    """
    settings: Settings = get_settings()
    if not settings.auth_enabled:
        return
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Basic"},
        )
    user_ok: bool = compare_digest(
        credentials.username,
        settings.segmentation_username or "",
    )
    pass_ok: bool = compare_digest(
        credentials.password,
        settings.segmentation_password or "",
    )
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )


__all__: list[str] = ["require_basic_auth"]
