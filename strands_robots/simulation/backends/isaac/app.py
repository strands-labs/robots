"""The one ``SimulationApp`` of a process, and the one open Isaac world it may hold.

Isaac allows a single ``SimulationApp`` per process and a single stage inside it, so the app is created on
first use and kept until exit, and a second ``Simulation("isaac")`` is refused while the first is open.
Closing a world frees the stage; the next ``Simulation("isaac")`` reuses the same app.
"""

from __future__ import annotations

import atexit
import importlib.util
import logging
import os
from typing import Any

from ...._extras import hint

HINT = f"Simulation('isaac') needs Isaac Sim: {hint('isaac')} (and OMNI_KIT_ACCEPT_EULA=YES)"

EULA = "Simulation('isaac'): accept the NVIDIA Isaac Sim EULA yourself by setting OMNI_KIT_ACCEPT_EULA=YES"

WARM_UP = (
    "Isaac: the first scene change after a fresh install warms Kit's shader and extension caches; "
    "it can take minutes, once per machine"
)

_APP: Any = None
_OPEN: object | None = None


def simulation_app(headless: bool) -> Any:
    """The process-wide SimulationApp, created on first use and closed at exit."""
    global _APP
    if _APP is None:
        if importlib.util.find_spec("isaacsim") is None:
            raise ImportError(HINT)
        if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in ("YES", "Y", "1", "TRUE"):
            raise RuntimeError(EULA)  # before the import: importing isaacsim asks for the EULA on stdin
        from isaacsim import SimulationApp

        _APP = SimulationApp({"headless": headless})
        logging.getLogger(__name__).warning(WARM_UP)
        atexit.register(_close_app)
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.asset.importer.mjcf")
        enable_extension("isaacsim.asset.importer.urdf")
    return _APP


def _close_app() -> None:
    if _APP is not None:
        _APP.close()


def claim(owner: object) -> None:
    """Mark ``owner`` as the open Isaac world; refuse if another one is still open."""
    global _OPEN
    if _OPEN is not None and _OPEN is not owner:
        raise RuntimeError(
            "an Isaac world is already open in this process: Isaac runs one SimulationApp with one stage, "
            "so add robots to that Simulation('isaac') (one world, many robots) or close() it first"
        )
    _OPEN = owner


def release(owner: object) -> None:
    global _OPEN
    if _OPEN is owner:
        _OPEN = None
