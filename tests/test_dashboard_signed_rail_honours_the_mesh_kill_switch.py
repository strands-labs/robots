"""The signed safety rail asks the mesh kill switch before it opens a session.

``STRANDS_MESH=false`` is a hard kill switch, and
:func:`strands_robots.mesh.core.mesh_disabled_by_env` states the rule in its own
docstring: an operator who sets the switch asked for no Zenoh session and no
presence on the fleet, so EVERY path which can open a session asks the same
question -- #2515 closed the same gap for the robot-less gateway in
``tools/robot_mesh`` after a direct ``Mesh(...)`` construction bypassed the
inline test and put a live ``gateway-*`` peer on the fleet.

``MeshBridge.start()`` honours it. ``MeshBridge._safety_mesh()`` is the second
site in the same file that constructs and starts a ``Mesh``, and it did not:
with the switch set, the first e-stop still put a ``<peer>-safety`` gateway peer
on the live fleet. That is the ghost-peer case the switch was added to close,
arriving on the one path where an operator is least likely to be watching the
peer list, by way of the action they are least likely to want to debug
afterwards.

The switch has two consequences and they need separate pins. One is about the
SESSION: no Zenoh peer may be opened. The other is about the ANSWER: a verb that
reports "there is no rail" must say which of the two reasons applies, because an
operator acts on them differently -- one is the switch they set, the other is a
fault to chase.

``signed_estop`` got both. ``signed_resume`` -- the same rail, the same
operator, 38 lines down the same file -- got only the first, and answered
``"safety mesh unavailable"`` in the one state this file spends 30 lines proving
is not a fault. ``docs/dashboard/troubleshooting.md`` lists exactly two causes
for a refused resume, both about ``override_code``, so "unavailable" sends the
operator to a code that is fine rather than to the switch they set.

These pin, for both consequences:

* the switch is asked BEFORE a Mesh is constructed -- not after, because
  constructing one is what joins the fleet;
* BOTH rail verbs say the rail was switched OFF rather than "unavailable", so
  the operator is pointed at the switch they set and not at a fault;
* with the switch clear, both verbs still reach the rail -- the guard refuses a
  session, it does not disable the feature;
* a future third construction site is graded from the source, since a new site
  added without the predicate is exactly how this defect arrived;
* and so is a future third *answer*: the construction-site guard keys on
  ``Mesh(...)`` and is structurally blind to a verb that constructs nothing,
  which is why the resume half could drift. The wording has one owner and that
  owner is what gets graded.

Parametrized over ``_mesh_switch.NEGATIVE`` directly, per that module's own
note: the vocabulary has one owner and a test that restates it would pass while
the product and the switch disagreed.

Run with --no-cov.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from strands_robots._mesh_switch import NEGATIVE
from strands_robots.dashboard.mesh_bridge import MeshBridge

#: Every kill spelling, plus the normalisation ``mesh_env_request`` documents
#: (case and surrounding whitespace), derived from the owner rather than typed.
KILLED = (*NEGATIVE, *(v.upper() for v in NEGATIVE), *(f"  {v}  " for v in NEGATIVE))

#: Values that leave the switch clear: an opt-in, and "said nothing".
ALLOWED = ("true", "1", "")

#: The wording that means *a fault*, pinned by
#: ``test_a_broken_rail_still_reports_unavailable_not_switched_off`` below. The
#: source-derived guard reads it from here so the two cannot drift apart.
FAULT_WORDING = "safety mesh unavailable"


class RecordingMesh:
    """Stands in for Mesh and records that it was constructed/started at all.

    Construction is recorded separately from ``start()`` because constructing a
    Mesh is already asking for one: a guard placed after the constructor must
    not pass this.
    """

    events: list[tuple[str, str | None]] = []
    alive = True

    def __init__(self, robot: Any, peer_id: str | None = None, peer_type: str = "robot") -> None:
        self.peer_id = peer_id
        RecordingMesh.events.append(("constructed", peer_id))

    def start(self) -> None:
        RecordingMesh.events.append(("started", self.peer_id))

    def emergency_stop(self) -> list[dict[str, Any]]:
        RecordingMesh.events.append(("emergency_stop", self.peer_id))
        return []

    def _resume_lockout(self, override_code: str) -> dict[str, Any]:
        RecordingMesh.events.append(("resume_lockout", self.peer_id))
        return {"resumed": True}


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[RecordingMesh]:
    """Patch the Mesh class the bridge imports, and reset the event log."""
    import strands_robots.mesh.core as core

    RecordingMesh.events = []
    monkeypatch.setattr(core, "Mesh", RecordingMesh)
    return RecordingMesh


@pytest.mark.parametrize("value", KILLED)
def test_the_kill_switch_refuses_the_safety_rail_a_session(value, recorder, monkeypatch):
    monkeypatch.setenv("STRANDS_MESH", value)
    bridge = MeshBridge(peer_id="dash")

    assert bridge._safety_mesh() is None
    assert recorder.events == [], f"STRANDS_MESH={value!r} still opened a session: {recorder.events}"


@pytest.mark.parametrize("value", KILLED)
def test_an_estop_under_the_kill_switch_creates_no_ghost_peer(value, recorder, monkeypatch):
    """The regression: the FIRST e-stop was what put the ghost peer on the fleet."""
    monkeypatch.setenv("STRANDS_MESH", value)
    bridge = MeshBridge(peer_id="dash")

    out = bridge.signed_estop()

    assert out["signed"] is False
    assert recorder.events == [], f"e-stop opened a session under STRANDS_MESH={value!r}"
    # Switched off, not broken - the operator is pointed at the switch.
    assert "STRANDS_MESH" in out["error"], out["error"]
    assert "unavailable" not in out["error"], out["error"]


def test_a_broken_rail_still_reports_unavailable_not_switched_off(recorder, monkeypatch):
    """The two answers stay distinguishable: this one really is a fault."""
    monkeypatch.setenv("STRANDS_MESH", "true")
    bridge = MeshBridge(peer_id="dash")

    with mock.patch.object(MeshBridge, "_safety_mesh", return_value=None):
        out = bridge.signed_estop()

    assert out == {"signed": False, "error": "safety mesh unavailable"}


@pytest.mark.parametrize("value", ALLOWED)
def test_the_guard_refuses_a_session_it_does_not_disable_the_rail(value, recorder, monkeypatch):
    """With the switch clear the rail starts, so the guard is a gate not a mute."""
    monkeypatch.setenv("STRANDS_MESH", value)
    bridge = MeshBridge(peer_id="dash")

    m = bridge._safety_mesh()

    assert m is not None
    assert recorder.events == [("constructed", "dash-safety"), ("started", "dash-safety")]


@pytest.mark.parametrize("value", KILLED)
def test_a_resume_under_the_kill_switch_reports_the_switch_not_a_fault(value, recorder, monkeypatch):
    """The regression: the resume half named a fault for a switch that was set.

    Same rail, same operator and the same two answers as the e-stop half, which
    has pointed at the switch since it was written. Reading "unavailable" here
    sends the operator to the two ``override_code`` causes the troubleshooting
    sheet documents for a refused resume -- and to a code that is fine.
    """
    monkeypatch.setenv("STRANDS_MESH", value)
    bridge = MeshBridge(peer_id="dash")

    out = bridge.signed_resume("operator-code")

    assert out["signed"] is False
    assert recorder.events == [], f"resume opened a session under STRANDS_MESH={value!r}"
    assert "STRANDS_MESH" in out["error"], out["error"]
    assert "unavailable" not in out["error"], out["error"]


def test_a_broken_rail_still_reports_unavailable_on_the_resume_path(recorder, monkeypatch):
    """The other half of the distinction, so the fix is not one string for both.

    With the switch clear a rail that will not start IS a fault, and the resume
    half must keep saying so -- otherwise "point at the switch" would have been
    bought by describing every failure as the switch.
    """
    monkeypatch.setenv("STRANDS_MESH", "true")
    bridge = MeshBridge(peer_id="dash")

    with mock.patch.object(MeshBridge, "_safety_mesh", return_value=None):
        out = bridge.signed_resume("operator-code")

    assert out == {"signed": False, "error": FAULT_WORDING}


@pytest.mark.parametrize("value", ALLOWED)
def test_with_the_switch_clear_a_resume_reaches_the_rail(value, recorder, monkeypatch):
    """A gate, not a mute: the resume verb still does its work when allowed."""
    monkeypatch.setenv("STRANDS_MESH", value)
    bridge = MeshBridge(peer_id="dash")

    out = bridge.signed_resume("operator-code")

    assert out["signed"] is True
    assert ("resume_lockout", "dash-safety") in recorder.events, recorder.events


def test_every_rail_unavailable_answer_asks_the_kill_switch():
    """The invariant that actually failed, graded from the source.

    The construction-site guard below keys on ``Mesh(...)``. That is the right
    key for "did we open a session" and it is structurally blind to this defect:
    ``signed_resume`` constructs nothing, so it was never in that guard's
    population, and it still answered on the rail's behalf. The rule that broke
    is about the ANSWER, so this grades every function that can emit the fault
    wording -- which after the fix is the single owner both verbs call.

    A second copy of the wording anywhere that does not ask the predicate fails
    here, whether or not it opens a session.
    """
    import ast
    import inspect

    import strands_robots.dashboard.mesh_bridge as bridge_mod

    tree = ast.parse(inspect.getsource(bridge_mod))
    emitters: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        strings = {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        if any(FAULT_WORDING in s for s in strings):
            called = {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            emitters[node.name] = "mesh_disabled_by_env" in called

    assert emitters, f"nothing emits {FAULT_WORDING!r} any more - has the wording moved?"
    ungated = sorted(name for name, gated in emitters.items() if not gated)
    assert not ungated, (
        f"these answer that the rail is unavailable without asking mesh_disabled_by_env(): {ungated}. "
        "A switched-off rail is not a fault, and the operator acts on the difference."
    )


def test_every_mesh_construction_site_in_the_bridge_asks_the_predicate():
    """Derived from the source, so a NEW site cannot skip the gate silently.

    Grading behaviour alone would leave a third construction site untested
    until someone wrote a case for it, and the absence of that case is how
    ``_safety_mesh`` stayed ungated while ``start`` was gated.
    """
    import ast
    import inspect

    import strands_robots.dashboard.mesh_bridge as bridge_mod

    tree = ast.parse(inspect.getsource(bridge_mod))
    opens_session: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        called = {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        if "Mesh" in called:  # constructing one is what joins the fleet
            opens_session[node.name] = "mesh_disabled_by_env" in called

    assert opens_session, "no Mesh construction site found - has the class been renamed?"
    ungated = sorted(name for name, gated in opens_session.items() if not gated)
    assert not ungated, (
        f"these open a mesh session without asking mesh_disabled_by_env(): {ungated}. "
        "STRANDS_MESH=false must be honoured at every construction site."
    )
