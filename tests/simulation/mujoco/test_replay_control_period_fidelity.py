"""Regression: replay steps a FULL control period per recorded frame.

A recorded ``LeRobotDataset`` frame corresponds to one control step taken at the
dataset's fps. During recording (``run_policy`` / ``eval_policy``) the engine
advances physics for the full control period per action - derived from the
control frequency and physics timestep (``PolicyRunner._control_substeps``) -
so a position-servo robot actually tracks each commanded target.

``PolicyRunner.replay`` previously fell through to ``send_action``'s default
``n_substeps=1``: a single ~2 ms physics step per recorded frame regardless of
the recording's control period. On an SO-101 (position servos, ~2 ms physics
dt, 30 Hz control -> 17 substeps/frame) the arm got ~1/17 of the integration
time per target, so it could not track the recorded trajectory: replay produced
a heavily under-integrated, attenuated motion that did not reproduce the
recording, while still reporting ``Frames: N/N`` and ``status="success"`` - a
silent record -> replay fidelity gap.

These tests pin (1) the mechanism - every replayed frame steps the full control
period - and (2) the behaviour - a self-recorded episode replays back to the
same final pose it was recorded at.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402


@pytest.fixture
def so101_sim():
    s = Simulation()
    s.create_world(ground_plane=True)
    s.add_robot("so101")
    yield s
    s.cleanup()


def _arm_qpos(sim: Simulation) -> np.ndarray:
    """Current SO-101 arm joint positions (so101/1..6) straight from mjData."""
    m = sim.mj_model
    d = sim.mj_data
    assert m is not None and d is not None
    adrs = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"so101/{j}")]) for j in range(1, 7)]
    return d.qpos[adrs].copy()


def _record_episode(sim: Simulation, repo: str, root: str, fps: int = 30) -> None:
    """Record a short mock (sinusoidal) rollout, action-only (no cameras)."""
    assert sim.start_recording(repo_id=repo, task="rt", fps=fps, root=root, cameras=[])["status"] == "success"
    assert (
        sim.run_policy(
            robot_name="so101",
            policy_provider="mock",
            n_steps=90,
            control_frequency=fps,
            action_horizon=8,
            fast_mode=True,
        )["status"]
        == "success"
    )
    assert sim.stop_recording()["status"] == "success"


def test_replay_steps_full_control_period(so101_sim):
    """Every replayed frame steps a full control period, not a single dt.

    Pre-fix each ``send_action`` call used the default ``n_substeps=1``; post-fix
    it uses ``_control_substeps(fps)`` (17 for 30 Hz control at a 2 ms physics
    dt), the same integration the recording used.
    """
    pytest.importorskip("lerobot")
    root = tempfile.mkdtemp(prefix="so101_replay_period_")
    repo = "local/so101_replay_period"
    _record_episode(so101_sim, repo, root, fps=30)

    expected = PolicyRunner(so101_sim)._control_substeps(30)
    assert expected > 1, "precondition: a 30 Hz control period must span >1 physics dt"

    used: list[int] = []
    original = so101_sim.send_action

    def spy(action, robot_name=None, n_substeps=1):
        used.append(n_substeps)
        return original(action, robot_name=robot_name, n_substeps=n_substeps)

    so101_sim.send_action = spy  # type: ignore[method-assign]
    try:
        rep = PolicyRunner(so101_sim).replay(repo, robot_name="so101", root=root, speed=1000.0)
    finally:
        so101_sim.send_action = original  # type: ignore[method-assign]

    assert rep["status"] == "success"
    assert used, "replay applied no actions"
    # Pre-fix: every element is 1. Post-fix: every element is the full period.
    assert set(used) == {expected}


def test_replay_reproduces_recorded_trajectory(so101_sim):
    """A self-recorded episode replays back to the pose it was recorded at.

    Physics is deterministic, so replaying the exact recorded action sequence
    from the same reset state must reproduce the recorded final pose - but only
    if replay integrates the same full control period per frame. Pre-fix the
    under-integrated replay diverged by ~0.95 rad; post-fix it matches to
    float32 round-trip precision.
    """
    pytest.importorskip("lerobot")
    root = tempfile.mkdtemp(prefix="so101_replay_traj_")
    repo = "local/so101_replay_traj"

    _record_episode(so101_sim, repo, root, fps=30)
    recorded_final = _arm_qpos(so101_sim)  # arm pose at the end of the recording

    so101_sim.reset()
    after_reset = _arm_qpos(so101_sim)
    assert np.linalg.norm(after_reset) < 1e-6, "so101 should reset to the zero rest pose"

    rep = PolicyRunner(so101_sim).replay(repo, robot_name="so101", root=root, speed=1000.0)
    assert rep["status"] == "success"
    replay_final = _arm_qpos(so101_sim)

    gap = float(np.linalg.norm(recorded_final - replay_final))
    # The recorded motion sweeps ~1 rad on several joints; the under-stepped
    # (pre-fix) replay diverged by ~0.95 rad. A faithful full-control-period
    # replay reproduces it to float32 round-trip precision (<1e-3).
    assert gap < 1e-2, f"replay did not reproduce the recording: {gap:.4f} rad gap"
