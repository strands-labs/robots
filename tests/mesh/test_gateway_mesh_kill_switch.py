"""``STRANDS_MESH=false`` must also kill the robot-less gateway peer.

README's Configuration table documents ``STRANDS_MESH`` as: "``false``/``0``/``no``
is a hard kill switch that also overrides an explicit ``mesh=True``". That switch
was resolved in exactly one place --
:func:`~strands_robots.mesh.core.init_mesh`, by an inline ``os.getenv`` read --
and ``robot_mesh._gateway_mesh`` is the one :class:`~strands_robots.mesh.core.Mesh`
in the tree that does not go through ``init_mesh``. It builds and starts its
``Mesh`` directly, so the switch never reached it.

The consequence, reproduced on the pre-fix tree with ``STRANDS_MESH=false`` set::

    RESULT gw= Mesh(peer_id='gateway-...', type='gateway', alive) elapsed=3.51
    Zenoh mesh session opened (listener on tcp/127.0.0.1:7447)
    mesh threads: ['mesh-hand-...', 'mesh-health-...', 'mesh-heartbeat-...',
                   'mesh-imu-...', 'mesh-lidar-...', 'mesh-map-info-...',
                   'mesh-odom-...', 'mesh-pose-...', 'mesh-state-...']

An operator who asked for no mesh got a real Zenoh session, this process
advertised to the fleet as a live ``gateway-*`` peer, and nine publishing
threads. Because ``_GATEWAY`` caches for the process lifetime (only
:func:`~strands_robots.tools.robot_mesh._stop_gateway_mesh`, at ``atexit``,
clears it), they survive until the interpreter dies. A kill switch that leaves
nine threads publishing is not a kill switch.

The same escape reached CI. ``tests/conftest.py`` sets ``STRANDS_MESH=false`` for
the whole suite for precisely this reason -- "so the ``Robot()`` /
``Simulation()`` factory does not spin up real Zenoh sessions and background
heartbeat threads when ``eclipse-zenoh`` is installed" -- but the gateway escaped
that guard too. ``test_deep_mesh.py::TestRobotMeshTool::test_peers_action_no_mesh``
calls the real tool with no session fixture, so it started a live gateway; its
``_health_loop`` then published into whichever later test had patched the
module-level ``put``, which is how
``test_mesh_rpc.py::test_publish_step_when_not_running_does_nothing`` failed with
a ``strands/gateway-<host>-<hex>/health`` payload it never provoked.

Pinned here: the switch is resolved by one shared authority, the gateway consults
it before constructing anything, the refusal names the variable, and -- the half
that keeps the fix from being over-broad -- an enabled mesh still brings the
gateway up exactly as before.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh.core import _MESH_KILL_SWITCH_VALUES, init_mesh, mesh_disabled_by_env

# ``from strands_robots.tools import robot_mesh`` yields the tool rather than the
# module, so the module-level cache and lock under test are unreachable that way.
rmt = importlib.import_module("strands_robots.tools.robot_mesh")

#: Spellings that must NOT trip the switch. ``off`` and ``foo`` are deliberate:
#: the documented domain is exactly ``false``/``0``/``no``, so an out-of-domain
#: value leaves the mesh enabled. Pinned to record the boundary, not to bless it
#: -- widening the domain is a separate change that would land here as a failure.
_NON_KILL_VALUES = ("true", "1", "yes", "", "off", "foo")


class _FakeMesh:
    """A Mesh stand-in that records bring-up without opening a session.

    Deliberately not a ``MagicMock``: ``_gateway_mesh`` reads ``.alive`` to
    decide whether bring-up succeeded, and a mock answers that truthfully-by-
    fabrication, so a mock would make the enabled-path control pass even if the
    production code never called ``start()``.
    """

    instances: list[_FakeMesh] = []

    def __init__(self, robot: Any, peer_id: str = "", peer_type: str = "robot") -> None:
        self.robot = robot
        self.peer_id = peer_id
        self.peer_type = peer_type
        self.started = False
        self.stopped = False
        _FakeMesh.instances.append(self)

    @property
    def alive(self) -> bool:
        return self.started and not self.stopped

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture(autouse=True)
def _isolate_gateway_cache(monkeypatch):
    """Leave no gateway behind -- the defect under test, applied to this module.

    Stops and clears anything cached both before and after each test, so a test
    here cannot become the leak it is pinning against.
    """
    _FakeMesh.instances = []

    def _drain() -> None:
        leaked = rmt._GATEWAY.pop("mesh", None)
        if leaked is not None:
            try:
                leaked.stop()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass

    _drain()
    # Bring-up sleeps one heartbeat period on success; no test here needs to wait.
    monkeypatch.setattr(rmt, "_gateway_discovery_wait_s", lambda: 0.0)
    yield
    _drain()


def _mesh_thread_names() -> set[str]:
    """Names of the live ``mesh-*`` threads ``Mesh.start`` would have spawned."""
    return {t.name for t in threading.enumerate() if t.name.startswith("mesh-")}


class TestTheKillSwitchDomain:
    """One predicate answers "may I start a mesh?" for every call site."""

    @pytest.mark.parametrize("raw", _MESH_KILL_SWITCH_VALUES)
    def test_every_documented_spelling_disables(self, monkeypatch, raw) -> None:
        # Parametrized over the shared tuple rather than a restated list, so a
        # spelling added to the domain is graded here without an edit.
        monkeypatch.setenv("STRANDS_MESH", raw)

        assert mesh_disabled_by_env() is True

    @pytest.mark.parametrize("raw", ["  false  ", "FALSE", "No", "\tNO\n"])
    def test_case_and_surrounding_space_are_ignored(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("STRANDS_MESH", raw)

        assert mesh_disabled_by_env() is True

    @pytest.mark.parametrize("raw", _NON_KILL_VALUES)
    def test_a_value_outside_the_domain_does_not_disable(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("STRANDS_MESH", raw)

        assert mesh_disabled_by_env() is False

    def test_an_unset_variable_does_not_disable(self, monkeypatch) -> None:
        monkeypatch.delenv("STRANDS_MESH", raising=False)

        # The switch only ever forces mesh OFF; absence is not a decision.
        assert mesh_disabled_by_env() is False


class TestTheGatewayRefusesADisabledMesh:
    """``_gateway_mesh`` consults the switch before it builds anything."""

    @pytest.mark.parametrize("raw", _MESH_KILL_SWITCH_VALUES)
    def test_it_returns_none(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("STRANDS_MESH", raw)

        assert rmt._gateway_mesh() is None

    def test_no_mesh_is_constructed_at_all(self, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_MESH", "false")
        monkeypatch.setattr(mesh_core, "Mesh", _FakeMesh)

        rmt._gateway_mesh()

        # Not merely "no session opened": the refusal precedes construction, so
        # there is no peer_id claimed and nothing to have to tear down.
        assert _FakeMesh.instances == []

    def test_nothing_is_cached_for_a_later_call_to_reuse(self, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_MESH", "false")

        rmt._gateway_mesh()

        # The cache is what made this leak outlive its test: it is only cleared
        # by the atexit teardown.
        assert "mesh" not in rmt._GATEWAY

    def test_it_starts_no_publishing_threads(self, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_MESH", "false")
        before = _mesh_thread_names()

        rmt._gateway_mesh()

        # Nine of them pre-fix: heartbeat, state, and the seven sensor loops.
        # These are the threads that published into unrelated tests.
        assert _mesh_thread_names() - before == set()

    def test_the_refusal_does_not_wait_on_the_bring_up_lock(self, monkeypatch) -> None:
        """The check is outside ``_GATEWAY_LOCK``, so a disabled mesh cannot queue.

        Pre-fix the switch was not consulted at all, so this call took the lock
        and blocked behind whatever held it -- on the real path, a bring-up
        holding it across its discovery sleep.
        """
        monkeypatch.setenv("STRANDS_MESH", "false")
        done = threading.Event()

        def _call() -> None:
            rmt._gateway_mesh()
            done.set()

        with rmt._GATEWAY_LOCK:
            worker = threading.Thread(target=_call, daemon=True)
            worker.start()
            answered = done.wait(timeout=5.0)

        worker.join(timeout=5.0)
        assert answered, "_gateway_mesh blocked on _GATEWAY_LOCK while the mesh was disabled"


class TestAnEnabledMeshStillBringsTheGatewayUp:
    """The scope boundary: this refuses a disabled mesh, it does not retire the gateway."""

    @pytest.mark.parametrize("raw", ["true", "1", "yes"])
    def test_an_explicit_opt_in_still_starts_and_caches(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("STRANDS_MESH", raw)
        monkeypatch.setattr(mesh_core, "Mesh", _FakeMesh)

        gw = rmt._gateway_mesh()

        assert isinstance(gw, _FakeMesh)
        assert gw.started is True
        assert gw.peer_type == "gateway"
        assert rmt._GATEWAY["mesh"] is gw

    def test_an_unset_variable_still_starts_the_gateway(self, monkeypatch) -> None:
        # The gateway serves robot-less coordinator processes (#10) and is not
        # opt-in: absence of the variable must not be read as a refusal.
        monkeypatch.delenv("STRANDS_MESH", raising=False)
        monkeypatch.setattr(mesh_core, "Mesh", _FakeMesh)

        gw = rmt._gateway_mesh()

        assert isinstance(gw, _FakeMesh)
        assert gw.started is True

    def test_a_live_cached_gateway_is_still_reused(self, monkeypatch) -> None:
        monkeypatch.delenv("STRANDS_MESH", raising=False)
        monkeypatch.setattr(mesh_core, "Mesh", _FakeMesh)

        first = rmt._gateway_mesh()
        second = rmt._gateway_mesh()

        assert first is second
        assert len(_FakeMesh.instances) == 1


class TestInitMeshIsUnchanged:
    """The extracted predicate preserves the behaviour it was lifted out of."""

    @pytest.mark.parametrize("raw", _MESH_KILL_SWITCH_VALUES)
    def test_the_switch_still_overrides_an_explicit_opt_in(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("STRANDS_MESH", raw)

        assert init_mesh(MagicMock(), peer_id="arm-1", mesh=True) is None

    def test_an_explicit_opt_out_still_wins_with_the_switch_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("STRANDS_MESH", raising=False)

        assert init_mesh(MagicMock(), peer_id="arm-1", mesh=False) is None

    def test_the_variable_still_never_forces_mesh_on(self, monkeypatch) -> None:
        # ``STRANDS_MESH=true`` with an explicit ``mesh=False`` stays off; the
        # opt-in path lives in the Robot factory, not here.
        monkeypatch.setenv("STRANDS_MESH", "true")

        assert init_mesh(MagicMock(), peer_id="arm-1", mesh=False) is None

    def test_an_enabled_mesh_is_still_constructed_and_started(self, monkeypatch) -> None:
        monkeypatch.delenv("STRANDS_MESH", raising=False)
        monkeypatch.setattr(mesh_core, "Mesh", _FakeMesh)

        instance = init_mesh(MagicMock(), peer_id="arm-1", mesh=True)

        assert isinstance(instance, _FakeMesh)
        assert instance.started is True


class TestTheSwitchHasOneAuthority:
    """A second inline spelling is how the gateway escaped in the first place."""

    def test_the_kill_values_are_spelled_once_in_the_package(self) -> None:
        from pathlib import Path

        package = Path(mesh_core.__file__).parent.parent
        literal = '("false", "0", "no")'
        hits = [
            f"{path.relative_to(package.parent)}:{n}"
            for path in sorted(package.rglob("*.py"))
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if literal in line
        ]

        assert len(hits) == 1, f"the kill-switch domain is spelled in {len(hits)} places: {hits}"

    def test_init_mesh_resolves_the_switch_through_the_predicate(self) -> None:
        import inspect

        body = inspect.getsource(init_mesh)

        assert "mesh_disabled_by_env()" in body
        # Non-vacuity: the inline read this replaced is really gone, so the two
        # call sites cannot drift apart again.
        assert 'getenv("STRANDS_MESH"' not in body


class TestTheToolNamesTheVariable:
    """A refusal an operator can act on, not a remedy the switch would also refuse."""

    def test_a_disabled_mesh_is_reported_as_a_disabled_mesh(self, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_MESH", "false")

        out = rmt.robot_mesh(action="inbox", target="peer-b", tool_context=MagicMock())

        assert out["status"] == "error"
        text = out["content"][0]["text"]
        # The generic answer advises constructing a Robot(), which this switch
        # would refuse too -- so the operator has to be told which knob it is.
        assert "STRANDS_MESH" in text
        assert "kill switch" in text

    def test_the_generic_message_is_kept_when_the_mesh_is_enabled(self, monkeypatch) -> None:
        monkeypatch.delenv("STRANDS_MESH", raising=False)
        # Mesh enabled, but zenoh unavailable: the pre-existing advice is right.
        monkeypatch.setattr(rmt, "_gateway_mesh", lambda: None)

        out = rmt.robot_mesh(action="inbox", target="peer-b", tool_context=MagicMock())

        assert out["status"] == "error"
        assert "no local mesh found" in out["content"][0]["text"]
