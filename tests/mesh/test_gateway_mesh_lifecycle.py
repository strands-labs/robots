"""The gateway mesh must survive its own configuration and close itself.

The robot-less gateway is the mesh entry point for a process that owns no robot
-- a dashboard, a scheduler, a logger. Two of its properties are pinned here
because neither is visible from the mesh actions it enables.

Its presence wait was read as ``float(os.environ.get(..., "3"))`` directly inside
the :func:`time.sleep` argument, which gave one operator knob two failures that
look nothing like a misconfigured wait. A non-numeric value raised
:class:`ValueError`, which the best-effort handler around gateway bring-up
absorbed, so a typo surfaced as "gateway mesh unavailable" and every action fell
back to ``no local mesh found`` -- naming neither the variable nor the typo.
``inf`` was worse than raising: :func:`time.sleep` accepts it and blocks forever
while holding ``_GATEWAY_LOCK``, so the call never returns and no later call can
take the lock either. This is the same shape as the step-telemetry rate, and the
last mesh knob that was still read inline; a wait that is merely wrong costs
first-call peer completeness, so an unusable value falls back to the default
rather than refusing to start the gateway at all.

The gateway is also the one :class:`~strands_robots.mesh.core.Mesh` in the tree
with no owner to stop it -- every other one is closed by the ``Robot`` or
``Simulation`` that built it. Cached at module scope for the process lifetime, it
held an open session plus its heartbeat and state threads, and stayed advertised
to the fleet as a live peer, until the interpreter died.
"""

from __future__ import annotations

import atexit
import importlib
import logging
import math

import pytest

# The ``@tool`` decorator binds a DecoratedFunctionTool named ``robot_mesh`` into
# ``strands_robots.tools``, shadowing the submodule of the same name, so
# ``from strands_robots.tools import robot_mesh`` yields the tool rather than the
# module. These tests need the module object to reach its cache and resolvers.
robot_mesh = importlib.import_module("strands_robots.tools.robot_mesh")

_ENV = robot_mesh._GATEWAY_WAIT_ENV
_DEFAULT = robot_mesh._GATEWAY_DISCOVERY_WAIT_S


class TestTheDiscoveryWaitIsAlwaysASpanSleepCanHonor:
    """Whatever the variable holds, the resolved wait is one sleep can take."""

    @pytest.mark.parametrize("raw", ["abc", "3s", "", "   ", "inf", "-inf", "1e999", "nan", "-1", "-0.5"])
    def test_an_unusable_value_falls_back_to_the_default(self, monkeypatch, raw):
        monkeypatch.setenv(_ENV, raw)

        assert robot_mesh._gateway_discovery_wait_s() == _DEFAULT

    @pytest.mark.parametrize("raw", ["abc", "3s", "inf", "1e999", "nan", "-1"])
    def test_the_resolved_wait_is_finite_and_non_negative(self, monkeypatch, raw):
        """The two properties :func:`time.sleep` actually needs.

        ``inf`` blocks forever while the gateway lock is held and a negative
        value raises, so neither may reach the sleep no matter which branch
        produced the answer.
        """
        monkeypatch.setenv(_ENV, raw)
        wait = robot_mesh._gateway_discovery_wait_s()

        assert math.isfinite(wait)
        assert wait >= 0.0

    @pytest.mark.parametrize("raw", ["abc", "inf", "-1"])
    def test_the_fallback_names_the_variable(self, monkeypatch, caplog, raw):
        """Otherwise the only symptom is an unrelated "no local mesh found"."""
        monkeypatch.setenv(_ENV, raw)

        with caplog.at_level(logging.WARNING):
            robot_mesh._gateway_discovery_wait_s()

        assert any(_ENV in record.getMessage() for record in caplog.records), caplog.text

    def test_the_knob_is_not_read_inline_any_more(self):
        """Structural: the resolver exists only if nothing bypasses it.

        Extends the rule the step-telemetry rate already carries to the knob that
        was still divided -- here slept on -- straight from the environment.
        """
        import inspect

        source = inspect.getsource(robot_mesh._gateway_mesh)

        assert _ENV not in source, f"_gateway_mesh reads {_ENV} directly instead of via the resolver"
        assert "_gateway_discovery_wait_s()" in source


class TestAUsableWaitIsStillObeyed:
    """The control: the knob still works, including the value meaning "do not wait".

    Falling back for what cannot be honored must not become falling back for
    everything -- an operator who sets a usable wait is entitled to get it.
    """

    def test_an_unset_variable_uses_the_documented_default(self, monkeypatch):
        monkeypatch.delenv(_ENV, raising=False)

        assert robot_mesh._gateway_discovery_wait_s() == _DEFAULT
        assert _DEFAULT == 3.0

    @pytest.mark.parametrize("raw", ["0", "0.0", "0.25", "1", "10"])
    def test_a_usable_value_is_returned_unchanged(self, monkeypatch, raw):
        """``0`` is honored: not waiting is a choice, not an unusable value."""
        monkeypatch.setenv(_ENV, raw)

        assert robot_mesh._gateway_discovery_wait_s() == pytest.approx(float(raw))


class _RecordingMesh:
    """Stands in for a started gateway; records that it was asked to stop."""

    def __init__(self) -> None:
        self.stopped = 0

    def stop(self) -> None:
        self.stopped += 1


class TestTheGatewayIsClosedAtExit:
    """Nothing else owns the gateway, so the exit hook is its whole teardown."""

    def test_the_teardown_is_registered_with_atexit(self):
        """A hook that exists but is never registered would leak just as much."""
        assert callable(robot_mesh._stop_gateway_mesh)
        count = getattr(atexit, "_ncallbacks", None)
        assert count is None or count() > 0

    def test_a_started_gateway_is_stopped(self, monkeypatch):
        gateway = _RecordingMesh()
        monkeypatch.setitem(robot_mesh._GATEWAY, "mesh", gateway)

        robot_mesh._stop_gateway_mesh()

        assert gateway.stopped == 1

    def test_the_cache_is_cleared_so_a_stopped_gateway_is_not_reused(self, monkeypatch):
        """A stopped mesh left in the cache would be handed out as if alive."""
        monkeypatch.setitem(robot_mesh._GATEWAY, "mesh", _RecordingMesh())

        robot_mesh._stop_gateway_mesh()

        assert "mesh" not in robot_mesh._GATEWAY

    def test_stopping_twice_stops_once(self, monkeypatch):
        """atexit may run alongside an explicit teardown; the second is a no-op."""
        gateway = _RecordingMesh()
        monkeypatch.setitem(robot_mesh._GATEWAY, "mesh", gateway)

        robot_mesh._stop_gateway_mesh()
        robot_mesh._stop_gateway_mesh()

        assert gateway.stopped == 1

    def test_no_gateway_is_a_no_op(self, monkeypatch):
        """The common case: a process that never needed a gateway exits quietly."""
        monkeypatch.delitem(robot_mesh._GATEWAY, "mesh", raising=False)

        robot_mesh._stop_gateway_mesh()

        assert "mesh" not in robot_mesh._GATEWAY
