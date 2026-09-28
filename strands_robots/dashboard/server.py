"""The dashboard app: ``create_app()`` and nothing that is not routing.

Every decision that matters is made by a module that landed on its own -
:mod:`.auth` (who may sign in), :mod:`.access` (who may call), :mod:`.settings`
(what may be configured), :mod:`.log_redaction` (what a log may say),
:mod:`.safety_state` (what an e-stop means), :mod:`.consent` (what a refusal
offers). This file wires them to paths. A route that does not take
``access.require_session`` is public on purpose, and there are exactly three: the
login screen's ``/api/auth/status``, the ceremonies it drives, and
``/api/health`` which says only that the process is up and what version it is.

One router per concern, each in its own module, each with an optional
``attach(app)`` that puts its state on ``app.state`` and appends the work it
needs at startup and shutdown to ``app.state.startup_hooks`` /
``app.state.shutdown_hooks``; the lifespan below runs those lists in order.
Order matters once: :mod:`.routes_mesh` goes before :mod:`.fleet` so the
literal ``/api/robots/registry`` outranks ``/api/robots/{name}``, and before
:mod:`.routes_devices` / :mod:`.routes_record` so the bridge exists when they
attach.

The UI is the built operator SPA committed under ``static/`` (source in
``frontend/``; rebuild with ``cd strands_robots/dashboard/frontend && npm ci &&
npm run build``). No node at runtime: it ships in the wheel and works offline.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from strands_robots.dashboard import (
    access,
    build_info,
    fleet,
    log_redaction,
    routes_agent,
    routes_auth,
    routes_config,
    routes_consent,
    routes_devices,
    routes_mesh,
    routes_record,
    routes_sim,
    routes_train,
    routes_voice,
    settings,
)
from strands_robots.dashboard.sim_session import SessionStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# What a settings read may return. `security.auth_token` is a secret: the UI
# only needs to know whether one is set.
_SECRET_KEYS = {("security", "auth_token")}
_SPA_PATH = re.compile(r"(?!\.)[A-Za-z0-9._-]+(?:/(?!\.)[A-Za-z0-9._-]+)*")


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


async def _run_hooks(app: FastAPI, name: str) -> None:
    """Run every callable a router left in ``app.state.<name>``; one failing hook is logged, not fatal."""
    for hook in list(getattr(app.state, name, None) or []):
        try:
            result = hook()
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 - the dashboard stays up without that subsystem
            logger.exception("%s hook %s failed", name, getattr(hook, "__qualname__", hook))


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    await _run_hooks(app, "startup_hooks")
    try:
        yield
    finally:
        await _run_hooks(app, "shutdown_hooks")
        app.state.safety.store.shutdown()


def create_app() -> FastAPI:
    """Build the dashboard application. Safe to call more than once (tests do)."""
    log_redaction.install_redaction()
    app = FastAPI(
        title="strands-robots dashboard", version=_version(), docs_url=None, redoc_url=None, lifespan=_lifespan
    )

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        if exc.status_code == 401:
            # A refused credential is a fact about the network the dashboard sits on;
            # /api/health reports the tally so the page can say "someone is knocking".
            tally = getattr(request.app.state, "refusals", None)
            if tally is not None:
                tally.record(
                    client=(request.client.host if request.client else "?"), path=request.url.path, now=time.time()
                )
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
    async def health(request: Request) -> dict[str, Any]:
        # Public by design (the LAN hint and the login screen poll it before any sign-in), so the
        # refusal block names WHO is being refused only to a caller who is signed in.
        now = time.time()
        out: dict[str, Any] = {
            "ok": True,
            "status": "ok",
            "version": _version(),
            "service": "strands-robots dashboard",
            "build": build_info.build_info(),
            "t": now,
        }
        tally = getattr(app.state, "refusals", None)
        if tally is not None:
            trusted = access.session_claims(request) is not None or access.open_posture(request)
            summary = tally.summary(now, detailed=trusted)
            if summary is not None:
                out["refused_handshakes"] = summary
        return out

    app.state.startup_hooks = []
    app.state.shutdown_hooks = []
    app.include_router(routes_auth.router)
    app.include_router(routes_mesh.router)
    routes_mesh.attach(app)
    app.include_router(fleet.router)
    app.include_router(routes_sim.router)
    app.include_router(routes_agent.router)
    app.state.safety = routes_sim.Safety(SessionStore())
    app.include_router(routes_devices.router)
    routes_devices.attach(app)
    app.include_router(routes_record.router)
    routes_record.attach(app)
    app.include_router(routes_train.router)
    routes_train.attach(app)
    app.include_router(routes_config.router)
    app.include_router(routes_consent.router)
    app.include_router(routes_voice.router)
    # The bridge tells the fleet apart from processes this dashboard spawned itself.
    devices = app.state.devices
    app.state.bridge.protected_peer_ids = lambda: {pid for pid, m in list(devices.robots.items()) if m.alive()}
    app.state.bridge.peer_annotations = devices.annotations_by_peer
    app.state.bridge.managed_children = devices.managed_children

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
        index_html = STATIC_DIR / "index.html"

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(index_html)

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            # Client-side routes reload to the same document. The request path is
            # only compared, never joined to a filesystem path: anything that is
            # not a plain SPA segment list, or that names an API, is a 404.
            if not _SPA_PATH.fullmatch(path) or path.split("/", 1)[0] in ("api", "ws", "static"):
                raise HTTPException(404, "not found")
            return FileResponse(index_html)

    return app
