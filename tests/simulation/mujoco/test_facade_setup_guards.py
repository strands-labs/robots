"""Facade guards for a partially set-up MuJoCo Simulation.

``test_no_world_guard_contract`` pins the behaviour when *no world exists at
all*. This module pins the adjacent states an agent hits while bringing a sim
up incrementally, where a world does exist but the setup is not finished:

* ``physics_timestep()`` returns ``None`` before ``create_world`` (so
  :class:`PolicyRunner` can detect that substepping at the control rate is not
  yet possible) and the world's real timestep once a world is live.
* ``send_action(robot_name=None)`` on a live world that has *no robots yet*
  returns a clear "No robots in the world." error instead of dereferencing an
  empty registry - a distinct signal from the no-world guard so an agent that
  created a world but forgot to add a robot can self-correct.
* ``_cheap_robot_count`` degrades to ``0`` (with a warning) when the model
  registry lookup fails, so a probe of available sim models never turns a
  transient registry error into a hard crash.
"""

import logging

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation(tool_name="facade_setup_guards", mesh=False)
    yield s
    s.cleanup()


class TestPhysicsTimestepGuard:
    def test_returns_none_before_world_created(self, sim):
        """No world -> None, so the control loop knows it cannot substep yet."""
        assert sim.physics_timestep() is None

    def test_returns_world_timestep_once_live(self, sim):
        """A live world -> the world's integration timestep as a float."""
        sim.create_world()
        ts = sim.physics_timestep()
        assert isinstance(ts, float)
        assert ts > 0.0


class TestSendActionOnRobotlessWorld:
    def test_defaulted_robot_on_empty_world_errors_clearly(self, sim):
        """A world with no robots + robot_name=None -> actionable error, no crash."""
        sim.create_world()
        result = sim.send_action({"1": 0.5})
        assert result["status"] == "error"
        assert result["content"][0]["text"] == "No robots in the world."

    def test_adding_a_robot_makes_the_defaulted_action_resolve(self, sim):
        """Once a robot exists, the same defaulted call is accepted (contrast)."""
        sim.create_world()
        sim.add_robot("so101")
        result = sim.send_action({"1": 0.5})
        assert result["status"] == "success"


class TestCheapRobotCountFallback:
    def test_registry_failure_degrades_to_zero_with_warning(self, sim, monkeypatch, caplog):
        """A raising model-registry lookup -> 0 (not a crash) + a warning."""

        def _boom():
            raise OSError("registry unavailable")

        monkeypatch.setattr(
            "strands_robots.simulation.mujoco.simulation.count_sim_robots",
            _boom,
        )
        with caplog.at_level(logging.WARNING):
            assert sim._cheap_robot_count() == 0
        assert any("count sim robots" in r.message.lower() for r in caplog.records)

    def test_healthy_registry_reports_a_positive_count(self, sim):
        """The happy path returns the real number of available sim models."""
        assert sim._cheap_robot_count() > 0
