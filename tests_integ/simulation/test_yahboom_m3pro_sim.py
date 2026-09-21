"""End-to-end: the Yahboom M3 Pro resolves from its GitHub asset source and simulates.

Network + MuJoCo integration, the same path ``test_lekiwi_sim.py`` covers for
the other GitHub-sourced mobile manipulator: registry lookup -> asset
auto-download (dimwael/yahboom_m3pro_description) -> MuJoCo compile -> step ->
mock policy rollout over all 9 actuators -> render from both declared cameras.

The model-shape claims the network-free registry test cannot make are made
here: joint names, actuator count, the ``home`` keyframe, the gripper's closed
end, and the camera intrinsics the MJCF declares.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

# 3 kinematic base velocities + 5 arm servos + 1 gripper servo.
_EXPECTED_ACTUATORS = 9
_EXPECTED_JOINTS = (
    "base_x",
    "base_y",
    "base_yaw",
    "arm1",
    "arm2",
    "arm3",
    "arm4",
    "arm5",
    "rlink1",
    "llink1",
)
# Measured RGB calibration the MJCF cameras declare (640x480).
_FX, _FY, _CX, _CY = 477.57, 477.56, 319.38, 238.64


@pytest.fixture
def m3pro_sim(tmp_path, monkeypatch):
    """Resolve + build the M3 Pro in sim, downloading assets into an isolated cache."""
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
    import strands_robots as sr
    from strands_robots.registry import reload

    reload()
    sim = sr.Robot("yahboom_m3pro", mode="sim", keyframe="home", mesh=False)
    yield sim
    sim.destroy()


def test_m3pro_builds_and_steps(m3pro_sim) -> None:
    """The model compiles with 9 actuators / 10 joints and steps stably (no NaN)."""
    name = m3pro_sim.list_robots()[0]
    assert name == "yahboom_m3pro"

    assert tuple(m3pro_sim.robot_joint_names(name)) == _EXPECTED_JOINTS
    assert m3pro_sim.mj_model.nu == _EXPECTED_ACTUATORS

    m3pro_sim.step(n_steps=300)
    state = m3pro_sim.get_observation(robot_name=name, skip_images=True)
    assert all(np.isfinite(v) for v in state.values())
    # ``keyframe="home"`` is the vendor grasp-init pose: arm3/arm4 folded at -pi/2.
    assert state["arm3"] == pytest.approx(-np.pi / 2, abs=0.02)
    assert state["arm4"] == pytest.approx(-np.pi / 2, abs=0.02)


def test_m3pro_mock_policy_rollout_and_render(m3pro_sim) -> None:
    """A mock policy drives all 9 actuators and both namespaced cameras render."""
    name = m3pro_sim.list_robots()[0]

    result = m3pro_sim.run_policy(
        robot_name=name,
        policy_provider="mock",
        instruction="drive forward and move the arm",
        duration=2.0,
        control_frequency=30.0,
    )
    assert result["status"] == "success", result

    for camera in ("yahboom_m3pro/front", "yahboom_m3pro/wrist"):
        render = m3pro_sim.render(camera_name=camera, width=320, height=240)
        assert render["status"] == "success", render


def test_m3pro_cameras_declare_the_measured_intrinsics(m3pro_sim) -> None:
    """``get_camera_params`` reads the physical-camera attributes, not a fovy fallback."""
    params = m3pro_sim.get_camera_params(camera_name="yahboom_m3pro/front")
    k = np.asarray(params.K, dtype=float)
    assert k[0, 0] == pytest.approx(_FX, abs=0.05)
    assert k[1, 1] == pytest.approx(_FY, abs=0.05)
    assert k[0, 2] == pytest.approx(_CX, abs=0.05)
    assert k[1, 2] == pytest.approx(_CY, abs=0.05)


def test_m3pro_gripper_closes_at_the_low_end_of_ctrlrange(m3pro_sim) -> None:
    """The registry's ``closed="low"`` matches the model: low ctrl reduces jaw separation."""
    import mujoco

    model, data = m3pro_sim.mj_model, m3pro_sim.mj_data
    act = model.actuator("yahboom_m3pro/gripper").id
    low, high = model.actuator_ctrlrange[act]

    def separation_at(ctrl: float) -> float:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.ctrl[act] = ctrl
        mujoco.mj_step(model, data, nstep=600)
        return float(np.linalg.norm(data.body("yahboom_m3pro/rlink2").xpos - data.body("yahboom_m3pro/llink2").xpos))

    closed, opened = separation_at(low), separation_at(high)
    assert closed < opened, (closed, opened)
    # The full servo travel is reachable (the jaw pair is contact-excluded).
    assert data.qpos[model.joint("yahboom_m3pro/rlink1").qposadr[0]] == pytest.approx(high, abs=0.02)
