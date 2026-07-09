"""Integration tests for NewtonSimEngine (gated on Newton + GPU availability).

These exercise the real Newton physics engine: building a model from an MJCF
asset, stepping it, observing/actuating joints, and headless rendering. They
are skipped when Newton/Warp are not installed or no compute device is usable.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


@pytest.fixture
def engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    sim = NewtonSimEngine(solver="mujoco")
    sim.create_world()
    yield sim
    sim.destroy()


@pytest.fixture
def engine_with_so100(engine):
    engine.add_robot("so100")
    return engine


class TestConstruction:
    def test_unknown_solver_raises(self):
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        with pytest.raises(ValueError, match="Unknown Newton solver"):
            NewtonSimEngine(solver="not_a_solver")

    def test_describe_reports_backend_and_solvers(self, engine):
        info = engine.describe()
        assert info["backend"] == "newton"
        assert info["solver"] == "mujoco"
        assert "featherstone" in info["available_solvers"]


class TestRobotManagement:
    def test_add_robot_resolves_joints(self, engine_with_so100):
        joints = engine_with_so100.robot_joint_names("so100")
        # Matches the MuJoCo backend's short joint names for parity.
        assert joints == ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

    def test_list_robots(self, engine_with_so100):
        assert engine_with_so100.list_robots() == ["so100"]

    def test_duplicate_robot_rejected(self, engine_with_so100):
        result = engine_with_so100.add_robot("so100")
        assert result["status"] == "error"

    def test_unknown_robot_reports_error(self, engine):
        result = engine.add_robot("definitely_not_a_robot_xyz")
        assert result["status"] == "error"
        assert "list_robots" in result["content"][0]["text"]

    def test_remove_robot(self, engine_with_so100):
        assert engine_with_so100.remove_robot("so100")["status"] == "success"
        assert engine_with_so100.list_robots() == []


class TestObservationAction:
    def test_observation_keys_match_joints(self, engine_with_so100):
        obs = engine_with_so100.get_observation("so100")
        assert set(obs) == set(engine_with_so100.robot_joint_names("so100"))
        assert all(isinstance(v, float) for v in obs.values())

    def test_send_action_moves_joint(self, engine_with_so100):
        before = engine_with_so100.get_observation("so100")["Rotation"]
        engine_with_so100.send_action({"Rotation": 0.6}, robot_name="so100", n_substeps=200)
        after = engine_with_so100.get_observation("so100")["Rotation"]
        assert after != pytest.approx(before, abs=1e-4)

    def test_send_action_unresolved_keys_surfaced(self, engine_with_so100):
        result = engine_with_so100.send_action({"NotAJoint": 1.0}, robot_name="so100")
        assert result["status"] == "error"
        payload = result["content"][-1]["json"]
        assert payload["unresolved_keys"] == ["NotAJoint"]

    def test_send_action_nonscalar_value_is_actionable_error(self, engine_with_so100):
        """A non-scalar dict value returns a clean error, not an uncaught TypeError.

        Backend parity with the MuJoCo path: the shared ``_coerce_action``
        validates every mapping value coerces to a scalar float before the
        per-actuator apply loop, so a vector-valued key (e.g. a policy emitting
        ``base_velocity: [vx, vy, omega]``) is rejected atomically instead of
        crashing the caller mid-rollout.
        """
        result = engine_with_so100.send_action({"Rotation": [0.1, 0.2, 0.3]}, robot_name="so100")
        assert result["status"] == "error"
        assert "scalar" in result["content"][0]["text"]

    def test_physics_timestep_positive(self, engine_with_so100):
        assert engine_with_so100.physics_timestep() > 0


class TestRendering:
    def test_render_returns_png_image_block(self, engine_with_so100):
        result = engine_with_so100.render(width=160, height=120)
        assert result["status"] == "success"
        image_block = next(b["image"] for b in result["content"] if "image" in b)
        assert image_block["format"] == "png"
        assert image_block["source"]["bytes"]

    def test_render_unknown_camera_errors(self, engine_with_so100):
        result = engine_with_so100.render(camera_name="nonexistent")
        assert result["status"] == "error"


class TestObjects:
    def test_add_and_remove_box(self, engine_with_so100):
        assert engine_with_so100.add_object("cube", shape="box", position=[0.3, 0, 0.05])["status"] == "success"
        assert "cube" in engine_with_so100.describe()["objects"]
        assert engine_with_so100.remove_object("cube")["status"] == "success"

    def test_unsupported_shape_rejected(self, engine_with_so100):
        assert engine_with_so100.add_object("blob", shape="torus")["status"] == "error"


class TestPolicyRollout:
    def test_run_policy_mock(self, engine_with_so100):
        result = engine_with_so100.run_policy(
            robot_name="so100", policy_provider="mock", instruction="wave", n_steps=10, control_frequency=20.0
        )
        assert result["status"] == "success"

    def test_run_policy_writes_video(self, engine_with_so100, tmp_path):
        out = tmp_path / "rollout.mp4"
        result = engine_with_so100.run_policy(
            robot_name="so100",
            policy_provider="mock",
            instruction="sweep",
            n_steps=8,
            control_frequency=20.0,
            video={"path": str(out), "fps": 20, "width": 160, "height": 120},
        )
        assert result["status"] == "success"
        assert out.exists() and out.stat().st_size > 0


class TestSolverParity:
    def test_featherstone_solver_steps(self):
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        sim = NewtonSimEngine(solver="featherstone")
        try:
            sim.create_world()
            sim.add_robot("so100")
            assert sim.step(5)["status"] == "success"
            obs = sim.get_observation("so100")
            assert len(obs) == 6
            assert all(np.isfinite(v) for v in obs.values())
        finally:
            sim.destroy()
