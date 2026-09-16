"""The dashboard app: ``create_app()`` and nothing that is not routing.

Every decision that matters is made by a module that landed on its own with its
own tests - :mod:`.auth` (who may sign in), :mod:`.access` (who may call),
:mod:`.settings` (what may be configured), :mod:`.log_redaction` (what a log may
say), :mod:`.safety_state` (what an e-stop means). This file wires them to
paths. A route that does not take ``access.require_session`` is public on
purpose, and there are exactly three: the login screen's ``/api/auth/status``,
the ceremonies it drives, and ``/api/health`` which says only that the process
is up and what version it is.

The static UI is plain files under ``static/`` served by this process: no build
step, so it ships in the wheel and works offline. Later slices add a router each
(fleet, sim, agent) and register it in :func:`create_app`.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from strands_robots.dashboard import access, fleet, log_redaction, routes_auth, routes_sim, settings
from strands_robots.dashboard.sim_session import SessionStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# What a settings read may return. `security.auth_token` is a secret: the UI
# only needs to know whether one is set.
_SECRET_KEYS = {("security", "auth_token")}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("strands-robots")
    except Exception:  # pragma: no cover - source checkout without metadata
        return "unknown"


def redacted_settings(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """A copy of *data* with every secret replaced by whether it is set."""
    out: dict[str, dict[str, Any]] = {}
    for section, values in data.items():
        out[section] = {}
        for key, value in values.items():
            if (section, key) in _SECRET_KEYS:
                out[section][key] = bool(value)
            else:
                out[section][key] = value
    return out


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    app.state.safety.store.shutdown()


def create_app() -> FastAPI:
    """Build the dashboard application. Safe to call more than once (tests do)."""
    log_redaction.install_redaction()
    app = FastAPI(
        title="strands-robots dashboard", version=_version(), docs_url=None, redoc_url=None, lifespan=_lifespan
    )

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        # One shape for every refusal so the UI has one place to render them.
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.middleware("http")
    async def _refuse_cross_origin_writes(request: Request, call_next: Any) -> Any:
        # A page from another origin can make the browser send a write here
        # (a no-preflight simple request) but cannot hide where it came from.
        # Whatever credential rides along, the Origin decides first.
        if request.method in access.UNSAFE_METHODS and not access.origin_is_self(request):
            return JSONResponse({"error": "cross-origin write refused"}, status_code=403)
        return await call_next(request)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "version": _version(), "service": "strands-robots dashboard"}

    app.include_router(routes_auth.router)
    app.include_router(fleet.router)
    app.include_router(routes_sim.router)
    app.state.safety = routes_sim.Safety(SessionStore())

    @app.get("/api/settings")
    async def get_settings(_: dict = Depends(access.require_session)) -> dict[str, Any]:
        return {"settings": redacted_settings(settings.load(refresh=True)), "file": str(settings.SETTINGS_FILE)}

    @app.post("/api/settings")
    async def post_settings(request: Request, who: dict = Depends(access.require_session)) -> JSONResponse:
        if who.get("via") == "loopback" and not access.peer_is_loopback(request):
            raise HTTPException(401, "sign in required")
        body = await routes_auth._json_body(request)
        unknown = settings.unknown_keys(body)
        if unknown:
            raise HTTPException(400, f"unknown settings: {', '.join(unknown)}")
        changed, errors = settings.update_strict(body)
        return JSONResponse({"changed": changed, "errors": errors}, status_code=422 if errors else 200)

    @app.get("/api/whoami")
    async def whoami(who: dict = Depends(access.require_session)) -> dict[str, Any]:
        return {"via": who.get("via"), "name": who.get("name"), "sub": who.get("sub")}

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app
