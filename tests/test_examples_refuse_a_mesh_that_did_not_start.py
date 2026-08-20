"""A fleet example refuses a mesh that did not start instead of awaiting presence.

``init_mesh`` returns ``None`` only when the mesh is switched off on purpose.
When the mesh is *enabled* but no session opens -- ``eclipse-zenoh`` absent, or
any of the other paths on which ``get_session`` yields nothing -- it returns a
``Mesh`` whose ``alive`` is ``False``.  That peer publishes no presence and
discovers none, so it is not a slow peer: every wait below it can only expire.

``mesh.alive`` is the documented observable for that state (docs/troubleshooting.md),
and ``examples/fleet/dashboard.py`` already refuses on it with the remedy named.
The four live-fleet builders checked only ``is None``, so a dead peer passed the
guard.  Measured with ``import zenoh`` failing, before this branch:

* 02 built both zones, then died 15 s later in "timed out after 15s waiting for
  presence discovery of both zone peers" -- a message naming no dependency.
* 03 and 04 did the same against their own presence waits.
* 05 has no presence wait at all: every RPC came back ``{'status': 'error',
  'error': 'mesh not running'}``, so it ran the whole queue, wrote four
  ``work_order_failed`` / ``work_order_nacked`` events attributing the failure to
  a per-robot ``dispatch_failed``, printed a plausible audit reconstruction, and
  **exited 0**.

Two rules here.  The behavioural tests drive each real builder with a session
that will not open and require an actionable refusal.  The structural rule is
derived from the builders themselves rather than hand-listed, so a sixth peer
start is graded on arrival: every function that starts a peer and guards it for
``None`` must also refuse on ``alive``.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FLEET_DIR = _REPO_ROOT / "examples" / "fleet"

# The peer-starting example scripts.  ``capabilities.py`` starts no peer.
_PEER_SCRIPTS = (
    "02_cross_zone_transport.py",
    "03_failover_and_degraded_ops.py",
    "04_emergency_evacuation.py",
    "05_work_order_dispatch.py",
    "dashboard.py",
)

# The live-fleet builders, with the peer each one starts first.  ``peer`` is read
# off the loaded module so a renamed constant updates the expectation with it.
_BUILDERS = (
    ("02_cross_zone_transport.py", "_build_live_zones", lambda m: next(iter(m.ZONES))),
    ("03_failover_and_degraded_ops.py", "_build_live_fleet", lambda m: next(iter(m.ROBOT_EMBODIMENT))),
    ("04_emergency_evacuation.py", "_build_live_world", lambda m: m.FLEET_PEER_ID),
    ("05_work_order_dispatch.py", "_build_live_transport", lambda m: "fleet-sites"),
)


class _StubSim:
    """World construction always succeeds; the mesh is what is under test.

    Only the calls the builders make are answered, so an unexpected one fails
    loudly rather than being absorbed by a catch-all.
    """

    tool_name_str = "stub-sim"

    def __init__(self) -> None:
        self.mesh: Any = None
        self.peer_id: str | None = None
        self.destroyed = 0

    def _ok(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    create_world = add_robot = add_object = add_camera = start_policy = stop_policy = _ok

    def destroy(self) -> None:
        self.destroyed += 1


def _load(filename: str) -> Any:
    """Import a fleet example as a module, as the example tests do."""
    name = f"_fleet_example_{filename.split('_')[0]}"
    spec = importlib.util.spec_from_file_location(name, _FLEET_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fleet_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Mesh enabled, audit confined, world construction stubbed out.

    The suite sets ``STRANDS_MESH=false`` process-wide so tests never touch a
    real mesh, and that kill switch makes ``init_mesh`` return ``None`` before it
    reaches a session -- the deliberate-opt-out branch, not the one under test.
    """
    monkeypatch.delenv("STRANDS_MESH", raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
    monkeypatch.syspath_prepend(str(_FLEET_DIR))
    monkeypatch.setattr("strands_robots.simulation.Simulation", _StubSim)


def _refusal_from(module: Any, builder: str) -> RuntimeError:
    """Run one builder with no session available; return the refusal it raised.

    Every refusal these builders raise is a ``RuntimeError`` -- the
    ``mesh.alive`` path and the deliberate ``STRANDS_MESH=0`` opt-out alike -- so
    anything else is a leak to report here rather than an exception to hand back
    for the caller to classify. Reporting it here is also what keeps the clause
    that catches a leak of any kind ending in a lexical ``raise``, matching the
    tree's other two leak-catching test helpers.
    """
    with patch("strands_robots.mesh.core.get_session", return_value=None):
        try:
            getattr(module, builder)()
        except RuntimeError as exc:
            return exc
        except BaseException as exc:  # noqa: BLE001 - the point is to catch a leak
            raise AssertionError(
                f"{builder} raised {type(exc).__name__} ({exc}) instead of refusing to build a "
                f"fleet whose peers are not on the mesh"
            ) from exc
    raise AssertionError(f"{builder} returned a fleet whose peers are not on the mesh")


class TestEachLiveBuilderRefusesAMeshThatDidNotStart:
    """The refusal happens where the cause is known, not at the next wait."""

    @pytest.mark.parametrize(("script", "builder", "expected_peer"), _BUILDERS, ids=[b[1] for b in _BUILDERS])
    def test_the_builder_refuses_and_names_the_remedy(
        self, script: str, builder: str, expected_peer: Any, fleet_env: None
    ) -> None:
        module = _load(script)
        peer = expected_peer(module)

        message = str(_refusal_from(module, builder))

        assert "mesh.alive is False" in message, f"the refusal does not name the observable it read: {message}"
        assert peer in message, f"the refusal does not name the peer that failed ({peer!r}): {message}"
        assert "strands-robots[mesh]" in message, f"the refusal does not name the remedy: {message}"
        assert "--dry-run" in message, f"the refusal does not offer the simulator-free alternative: {message}"

    @pytest.mark.parametrize(("script", "builder", "expected_peer"), _BUILDERS, ids=[b[1] for b in _BUILDERS])
    def test_the_builder_leaves_no_world_behind(
        self, script: str, builder: str, expected_peer: Any, fleet_env: None
    ) -> None:
        """Refusing mid-construction still tears the sim down.

        The refusal is asserted first so a builder that fails for some other
        reason reports that, rather than reporting an undestroyed world.
        """
        module = _load(script)
        built: list[_StubSim] = []
        original = _StubSim.__init__

        def record(self: _StubSim) -> None:
            original(self)
            built.append(self)

        with patch.object(_StubSim, "__init__", record):
            exc = _refusal_from(module, builder)

        assert "mesh.alive is False" in str(exc), (
            f"{builder} did not refuse on the alive observable, so this proves nothing about its "
            f"cleanup path: {type(exc).__name__}: {exc}"
        )
        assert built, "premise: the builder constructed no world, so there is nothing to release"
        assert all(sim.destroyed for sim in built), (
            f"{builder} refused but left {sum(not s.destroyed for s in built)} world(s) undestroyed"
        )


class TestTheDeliberateOptOutKeepsItsOwnAdvice:
    """Mesh switched off on purpose and a mesh that would not start differ."""

    @pytest.mark.parametrize(("script", "builder", "expected_peer"), _BUILDERS, ids=[b[1] for b in _BUILDERS])
    def test_a_disabled_mesh_still_reports_the_switch(
        self, script: str, builder: str, expected_peer: Any, fleet_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH", "false")
        module = _load(script)

        exc = _refusal_from(module, builder)

        message = str(exc)
        assert "STRANDS_MESH=0" in message, f"the opt-out refusal no longer names the switch: {message}"
        assert "strands-robots[mesh]" not in message, (
            f"an install remedy cannot fix a mesh that was switched off on purpose: {message}"
        )


def _peer_starting_functions(tree: ast.AST) -> list[ast.FunctionDef]:
    """Functions that call ``init_mesh``."""
    found: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                func = call.func
                if (func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")) == "init_mesh":
                    found.append(node)
                    break
    return found


def _guards_none(fn: ast.FunctionDef) -> bool:
    """The function compares something against ``None``."""
    return any(
        isinstance(node, ast.Compare) and any(isinstance(c, ast.Constant) and c.value is None for c in node.comparators)
        for node in ast.walk(fn)
    )


def _refuses_on_alive(fn: ast.FunctionDef) -> bool:
    """The function has an ``if`` that reads ``.alive`` and raises."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        reads_alive = any(isinstance(t, ast.Attribute) and t.attr == "alive" for t in ast.walk(node.test))
        raises = any(isinstance(b, ast.Raise) for b in ast.walk(node))
        if reads_alive and raises:
            return True
    return False


def _unguarded_peer_starts(sources: dict[str, str]) -> list[str]:
    """``"<script>:<function>"`` for every peer start that only checks ``None``."""
    offenders = []
    for name, source in sources.items():
        for fn in _peer_starting_functions(ast.parse(source)):
            if _guards_none(fn) and not _refuses_on_alive(fn):
                offenders.append(f"{name}:{fn.name}")
    return offenders


def _shipped_sources() -> dict[str, str]:
    return {name: (_FLEET_DIR / name).read_text(encoding="utf-8") for name in _PEER_SCRIPTS}


class TestEveryPeerStartRefusesOnAlive:
    """Derived from the scripts, so a sixth peer start is graded on arrival."""

    def test_no_peer_start_checks_only_none(self) -> None:
        offenders = _unguarded_peer_starts(_shipped_sources())

        assert offenders == [], (
            "these start a mesh peer and only check `is None`, so a peer whose mesh did not "
            f"start passes the guard and the next wait expires instead: {offenders}"
        )

    def test_the_scan_found_the_peer_starts(self) -> None:
        """A clean result must mean the rule was applied, not that nothing was read."""
        counted = sum(len(_peer_starting_functions(ast.parse(src))) for src in _shipped_sources().values())

        assert counted >= len(_BUILDERS) + 1, f"the scan reached only {counted} peer starts across {_PEER_SCRIPTS}"

    def test_the_scan_reports_a_none_only_guard(self) -> None:
        planted = """
def _build():
    mesh = init_mesh(sim, peer_id="a")
    if mesh is None:
        raise RuntimeError("mesh is disabled (STRANDS_MESH=0); rerun with --dry-run")
    return mesh
"""
        assert _unguarded_peer_starts({"planted.py": planted}) == ["planted.py:_build"]

    def test_the_scan_accepts_the_dashboard_shape(self) -> None:
        """The shape the dashboard already ships is what the rule asks for."""
        clean = """
def main():
    mesh = init_mesh(owner, peer_id="d")
    if mesh is None:
        raise SystemExit("mesh is disabled (STRANDS_MESH=0)")
    if not mesh.alive:
        raise SystemExit('mesh did not start (pip install "strands-robots[mesh]")')
    return mesh
"""
        assert _unguarded_peer_starts({"clean.py": clean}) == []
