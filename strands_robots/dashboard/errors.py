"""Turn a refusal into its HTTP status: the one place every route maps an exception to a response.

A leaf module (``fastapi`` only), so a route module reaches it without importing ``app`` back.
"""

from __future__ import annotations

from fastapi import HTTPException


def http_error(exc: Exception) -> HTTPException:
    from ..robot.safety.estop import EStopped

    if isinstance(exc, EStopped):
        return HTTPException(409, str(exc))
    if isinstance(exc, PermissionError):
        return HTTPException(403, str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(404, str(exc.args[0]) if exc.args else "not found")
    if isinstance(exc, (ValueError, TypeError)):
        return HTTPException(400, str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(409, str(exc))
    return HTTPException(500, f"{type(exc).__name__}: {exc}")
