"""What counts as the operator's yes on a dashboard request, and which ``/api/world`` calls need it.

``X-Confirm: yes`` is a header any client can type, so it is the operator's yes only on a request that also carries a
signed-in session (the passkey cookie or ``Authorization: Bearer``, checked by ``auth/guard.py``). With sign-in off
it counts for nothing: a real robot then asks its own ``confirm`` callback, as it does for a script.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: ``/api/world`` verbs that only read the world; every other verb (including one added later) needs the yes.
WORLD_READS = frozenset({"describe", "pose", "contacts", "render"})
OBJECTS_ENV = "STRANDS_ROBOTS_DASHBOARD_OBJECTS"


def operator_yes(request: Any) -> bool:
    """``X-Confirm: yes`` on a signed-in request (a starlette ``Request`` or ``WebSocket``)."""
    said = request.headers.get("x-confirm", "").lower() == "yes"
    return said and request.scope.get("state", {}).get("session") is not None


def object_dir(configured: str | Path | None) -> Path | None:
    """The folder the page may load object files from: ``Dashboard(object_dir=)``, else ``$OBJECTS_ENV``."""
    raw = configured if configured is not None else os.environ.get(OBJECTS_ENV) or None
    return None if raw is None else Path(raw).expanduser().resolve()


def check_world_call(call: Any, *, confirmed: bool, objects: Path | None) -> None:
    """Refuse a page's ``{"rpc": verb, verb: {...}}`` that changes the world without the yes, or loads a file outside
    ``objects``."""
    if not isinstance(call, dict):
        raise ValueError('body must be {"rpc": <verb>, <verb>: {arguments}}')
    verb = call.get("rpc")
    if verb in WORLD_READS:
        return
    if not confirmed:
        raise PermissionError(f"world {verb!r} changes the world: sign in and send header X-Confirm: yes")
    args = call.get(verb) if isinstance(verb, str) else None
    path = args.get("path") if verb == "add_object" and isinstance(args, dict) else None
    if path is None:
        return
    if objects is None:
        raise PermissionError(f"no objects folder for this page: set Dashboard(object_dir=...) or ${OBJECTS_ENV}")
    file = Path(str(path)).expanduser().resolve()
    if not file.is_relative_to(objects):
        raise PermissionError(f"{path!r} is outside the objects folder {str(objects)!r}")


def world(sim: Any, call: Any, *, confirmed: bool, objects: Path | None) -> Any:
    """Run a page's world call as a dashboard's call (``from_agent``): no policy at a network address, every robot
    write through the safety chain, after :func:`check_world_call`."""
    from ..robot.run.source import from_agent
    from ..tools.call import dispatch
    from ..tools.wire import json_safe
    from ..tools.world import WorldDevice

    check_world_call(call, confirmed=confirmed, objects=objects)
    with from_agent(confirmed=confirmed):
        return json_safe(dispatch(WorldDevice(sim), call))
