"""Reachy Mini hardware layer - shared pieces for the native driver.

The Reachy Mini speaks two protocols at once, and which one carries real-time
data depends on the hardware variant the daemon reports:

* REST on ``:8000`` (``/api/daemon/status``, ``/api/move/...``) for
  reachability, variant detection, recorded moves and the motion stop.
* A real-time link for joints and IMU - a WebSocket straight to the daemon on a
  **Lite** (no onboard computer), or Zenoh on a **Wireless** (onboard CM4).

Both live in :mod:`strands_robots.device_connect.reachy_transport`, which the
Device Connect driver already ships and which
:class:`~strands_robots.drivers.reachy.ReachyDriver` reuses rather than
re-implements. This package holds only what the *two* Reachy consumers must
agree on and neither owns: the motion envelope.

The agent ``@tool``s that sit on the same daemon live in two sibling modules -
:mod:`~strands_robots.tools.reachy.reachy_actions` (the execution verbs) and
:mod:`~strands_robots.tools.reachy.reachy_reads` (the read-only pair) - and are
re-exported here lazily so importing the package costs nothing until a verb is
actually used. They import
:func:`~strands_robots.tools.reachy._reachy_common.envelope_error` machinery
from here so a limit is defined once for the robot rather than once per caller.

Nothing here imports a transport, a daemon client or the driver, so this package
is importable and fully testable on a machine with no Reachy attached.
"""

import importlib as _importlib

from strands_robots.tools.reachy._reachy_common import (
    HEAD_BODY_YAW_DELTA_LIMIT_DEG,
    MOTION_ENVELOPE_DEG,
    envelope_error,
    live_handle_refusal,
)

#: verb -> (relative module, attribute). Lazy so ``import strands_robots.tools.reachy``
#: stays free of the ``strands`` import the verb modules need.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "reachy_look": (".reachy_actions", "reachy_look"),
    "reachy_antennas": (".reachy_actions", "reachy_antennas"),
    "reachy_body_turn": (".reachy_actions", "reachy_body_turn"),
    "reachy_home": (".reachy_actions", "reachy_home"),
    "reachy_stop": (".reachy_actions", "reachy_stop"),
    "reachy_wake": (".reachy_actions", "reachy_wake"),
    "reachy_express": (".reachy_actions", "reachy_express"),
    "reachy_motors": (".reachy_actions", "reachy_motors"),
    "reachy_play_sound": (".reachy_actions", "reachy_play_sound"),
    "reachy_volume": (".reachy_actions", "reachy_volume"),
    "reachy_camera": (".reachy_actions", "reachy_camera"),
    "reachy_look_at": (".reachy_actions", "reachy_look_at"),
    "reachy_get_state": (".reachy_reads", "reachy_get_state"),
    "reachy_list_emotions": (".reachy_reads", "reachy_list_emotions"),
}

__all__ = [
    "HEAD_BODY_YAW_DELTA_LIMIT_DEG",
    "MOTION_ENVELOPE_DEG",
    "envelope_error",
    "live_handle_refusal",
    *_LAZY_IMPORTS.keys(),
]


def __getattr__(name: str):  # noqa: N807
    if name in _LAZY_IMPORTS:
        rel_module, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(rel_module, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
