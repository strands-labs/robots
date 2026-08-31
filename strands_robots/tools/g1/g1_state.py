"""Agent-facing wrapper for ``G1Driver.get_status``.

``G1Driver.get_status`` returns a JSON envelope naming the driver's
connection state, the last-observed FSM id, ``mode_machine``,
``battery_pct`` and the motion-switcher wire diagnostics
(``fsm_mode_name`` / ``fsm_refusal`` / ``motion_switcher_open_error``).
That envelope is what the mesh publishes on its status wire, but it is
not what an agent asks for: a caller planning a write wants to know
whether the arm-SDK gate would admit today, and the driver's answer is
"here is ``fsm_id``, compare it yourself against ``HANDSHAKE_FSMS`` /
``WALK_FSMS``". This verb closes that gap: it calls
:meth:`~strands_robots.drivers.g1.G1Driver.get_status`, then decides the
``admits_arm`` / ``admits_loco`` membership against the same driver
constants :mod:`~strands_robots.tools.g1.g1_motion_gates` names.

The verb takes a :class:`~strands_robots.drivers.g1.G1Driver` instance,
which is the first driver-instance-taking verb in this package. Every
earlier verb here (:mod:`~strands_robots.tools.g1.g1_joints`,
:mod:`~strands_robots.tools.g1.g1_motion_gates`) is a pure reader over
module-level constants and takes no argument; this one is a live read
against a wired driver and cannot answer without one. The driver
argument is typed :class:`~typing.Any` at runtime rather than as
``G1Driver``: the driver module imports ``ensure_dds`` from this package
at load, so a runtime import of ``G1Driver`` here would close a cycle,
and ``@tool`` calls :func:`typing.get_type_hints` at decoration time so
a string forward reference cannot resolve without pulling the driver at
import. The verb is duck-typed on ``get_status`` (any object with an
``async get_status`` returning the driver's envelope answers), which is
also how the tests hand it a hand-rolled double. ``import
strands_robots.tools.g1.g1_state`` still pulls no ``unitree_sdk2py``
submodule (the package's SDK-load-hygiene contract, refs
strands-labs/robots#358).

What this module does not do.

* Subscribe DDS. The driver's own subscribers deliver every field this
  verb returns; adding a second subscriber path would compete for the
  same topic and duplicate the bus load ``strands-labs/robots#358``'s
  singleton lock is meant to prevent.
* Rebuild a wedged loco RPC client. ``get_status`` reports what the
  driver's FSM refresher already read; a wedged RPC surfaces here as an
  ``fsm_id`` of ``None`` and an ``fsm_refusal`` string, decidably, and
  the recovery is on the driver's refresh loop rather than this read.
* Decode ``mode_machine`` into a posture label. Every posture label the
  neon bundle used to compute here read the same eight-integer set the
  driver already carries; adding a table on this side would be a second
  source of truth for a driver-side domain that will not stay in sync
  without a wire-level test - which the driver's tests are the right
  home for.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import HANDSHAKE_FSMS, WALK_FSMS

#: The status fields only a G1 envelope carries.
#:
#: ``G1Driver.get_status`` reports these five alongside the ``tool_name`` /
#: ``connected`` / ``battery_pct`` triple every native driver reports. The
#: triple is the shared :data:`~strands_robots.drivers.base.DRIVER_SURFACE`
#: contract, so it cannot tell one driver's envelope from another's; these
#: five are the FSM and motion-switcher fields this verb's whole answer is
#: computed from, and no other shipped driver's envelope declares any of
#: them. A handle whose envelope is missing them is not a G1.
_G1_STATUS_CONTRACT_KEYS = (
    "fsm_id",
    "mode_machine",
    "fsm_mode_name",
    "fsm_refusal",
    "motion_switcher_open_error",
)


def _err(text: str) -> dict[str, Any]:
    """Wrap ``text`` in the error envelope every ``@tool`` owes a caller."""
    return {"status": "error", "content": [{"text": text}]}


def _inner_json(envelope: Any) -> dict[str, Any] | None:
    """Return the envelope's inner ``content[0]["json"]`` dict, or ``None``.

    The three subscripts this replaces are the driver's own envelope shape,
    which every shipped driver honours - but ``driver`` is typed
    :class:`~typing.Any`, so the object that arrives need not be a driver at
    all. Reading the chain defensively lets the verb refuse such a handle by
    name instead of surfacing a ``KeyError`` past the tool boundary.
    """
    if not isinstance(envelope, dict):
        return None
    blocks = envelope.get("content")
    if not isinstance(blocks, list) or not blocks or not isinstance(blocks[0], dict):
        return None
    inner = blocks[0].get("json")
    return inner if isinstance(inner, dict) else None


@tool
async def g1_get_state(driver: Any) -> dict[str, Any]:
    """Return the driver's status plus the arm / loco gate membership answers.

    Read-only. Calls :meth:`~strands_robots.drivers.g1.G1Driver.get_status`
    once, then decides membership of the reported ``fsm_id`` against
    :data:`~strands_robots.tools.g1._g1_common.HANDSHAKE_FSMS` (the arm-SDK
    gate) and :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` (the
    locomotion gate). The membership answer is the same one
    :func:`~strands_robots.tools.g1.g1_motion_gates.g1_fsm_admits` would
    compute for the given ``fsm_id``; this verb saves the caller a second
    tool call by carrying it alongside the state read.

    Args:
        driver: An object with an ``async get_status`` method returning
            the driver's status envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`). The driver may
            be connected or not; ``get_status`` reports which, and every
            field it does not have yet comes back ``None`` rather than
            raising. Typed :class:`~typing.Any` rather than as ``G1Driver``
            to keep this module out of the import cycle the driver's own
            ``ensure_dds`` reach into this package would close - see the
            module docstring's SDK-load-hygiene note.

    Returns:
        A dict with ``status``, the driver's ``tool_name`` and ``connected``
        flag, its last-observed ``fsm_id`` / ``mode_machine`` /
        ``battery_pct``, the motion-switcher diagnostics
        (``fsm_mode_name`` / ``fsm_refusal`` / ``motion_switcher_open_error``),
        two decided ``admits_arm`` / ``admits_loco`` booleans, and the
        ``handshake_fsms`` / ``walk_fsms`` id sets the answers were
        computed against (sorted, as lists) so a caller can quote them in
        its own voice. An ``fsm_id`` of ``None`` reports both admit
        booleans as ``False`` - the gate cannot open on a read that never
        arrived.
    """
    if driver is None:
        return _err(
            "g1_get_state: `driver` is required. Pass the live G1Driver handle the "
            "orchestrator constructed - an agent cannot synthesize it, because every "
            "field this verb returns comes from that driver's own status read."
        )
    if not callable(getattr(driver, "get_status", None)):
        return _err(
            f"g1_get_state: `driver` of type {type(driver).__name__!r} does not expose "
            "get_status(). Pass a strands_robots G1Driver, or an object with an async "
            "get_status() returning the driver's status envelope."
        )

    envelope = await driver.get_status()
    inner = _inner_json(envelope)
    if inner is None:
        return _err(
            f"g1_get_state: get_status() on a `driver` of type {type(driver).__name__!r} "
            "did not return the driver status envelope shape (a dict whose "
            'content[0]["json"] is a dict). Pass a strands_robots G1Driver.'
        )

    absent = [key for key in _G1_STATUS_CONTRACT_KEYS if key not in inner]
    if absent:
        return _err(
            f"g1_get_state: the status envelope from a `driver` of type "
            f"{type(driver).__name__!r} declares none of this verb's FSM fields "
            f"(absent: {absent}). Every native driver reports tool_name / connected / "
            "battery_pct, so those cannot identify a G1; the arm and loco gate answers "
            "this verb decides are computed from fsm_id, which a driver with no FSM "
            "never reports. Pass a strands_robots G1Driver."
        )

    fsm_id = inner.get("fsm_id")
    admits_arm = isinstance(fsm_id, int) and not isinstance(fsm_id, bool) and fsm_id in HANDSHAKE_FSMS
    admits_loco = isinstance(fsm_id, int) and not isinstance(fsm_id, bool) and fsm_id in WALK_FSMS

    return {
        "status": envelope["status"],
        "tool_name": inner.get("tool_name"),
        "connected": inner.get("connected"),
        "connect_error": inner.get("connect_error"),
        "port": inner.get("port"),
        "network_interface": inner.get("network_interface"),
        "fsm_id": fsm_id,
        "mode_machine": inner.get("mode_machine"),
        "battery_pct": inner.get("battery_pct"),
        "fsm_mode_name": inner.get("fsm_mode_name"),
        "fsm_refusal": inner.get("fsm_refusal"),
        "motion_switcher_open_error": inner.get("motion_switcher_open_error"),
        "admits_arm": admits_arm,
        "admits_loco": admits_loco,
        "handshake_fsms": sorted(HANDSHAKE_FSMS),
        "walk_fsms": sorted(WALK_FSMS),
    }
