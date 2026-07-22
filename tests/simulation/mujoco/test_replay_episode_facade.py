"""Contract tests for the public ``SimEngine.replay_episode`` facade.

``replay_episode`` is one of the three headline sim entrypoints
(``run_policy`` / ``eval_policy`` / ``replay_episode``) and the only public,
documented way to play back a recorded ``LeRobotDataset`` episode. No backend
overrides it, so the base implementation in
:meth:`strands_robots.simulation.base.SimEngine.replay_episode` is the
production path for every backend; its sole job is to delegate to
:meth:`strands_robots.simulation.policy_runner.PolicyRunner.replay` with the
caller's arguments untouched.

The existing replay suite exercises ``PolicyRunner.replay`` directly and never
routes through the public facade, so a delegation regression (a dropped or
renamed keyword between the facade and the runner - e.g. ``episode`` silently
defaulting to 0, or ``action_key_map`` never forwarded) would ship without a
failing test. These tests pin (1) the delegation wiring - every documented
argument reaches the runner unchanged and the runner's result is returned
verbatim - and (2) the end-to-end behaviour - a self-recorded episode replays
back through the public facade to the pose it was recorded at.
"""

from __future__ import annotations

import pathlib
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


def test_replay_episode_forwards_all_documented_kwargs(so101_sim, monkeypatch):
    """The facade hands every documented argument to ``PolicyRunner.replay``.

    The facade's contract is pure delegation: whatever the caller passes must
    reach the runner unchanged, and whatever the runner returns must be handed
    back verbatim (no wrapping / status rewrite). Spying on the runner method
    pins that wiring without needing a real dataset, so a dropped or renamed
    keyword between the two layers fails here rather than silently replaying
    the wrong episode / speed / action mapping.
    """
    captured: dict = {}
    sentinel = {"status": "success", "sentinel": object()}

    def spy(self, repo_id, robot_name=None, *, episode=0, root=None, speed=1.0, action_key_map=None):
        captured.update(
            repo_id=repo_id,
            robot_name=robot_name,
            episode=episode,
            root=root,
            speed=speed,
            action_key_map=action_key_map,
        )
        return sentinel

    monkeypatch.setattr(PolicyRunner, "replay", spy)

    result = so101_sim.replay_episode(
        "local/some_dataset",
        robot_name="so101",
        episode=3,
        root="/tmp/some_root",
        speed=2.5,
        action_key_map=["so101/1", "so101/2"],
    )

    # Result is passed through untouched - the facade adds no wrapping.
    assert result is sentinel
    # Every documented argument reached the runner unchanged.
    assert captured == {
        "repo_id": "local/some_dataset",
        "robot_name": "so101",
        "episode": 3,
        "root": "/tmp/some_root",
        "speed": 2.5,
        "action_key_map": ["so101/1", "so101/2"],
    }


def test_replay_episode_defaults_match_runner(so101_sim, monkeypatch):
    """Calling the facade with only ``repo_id`` applies the documented defaults.

    ``episode=0``, ``root=None``, ``speed=1.0`` and ``action_key_map=None`` are
    the documented defaults; pin that the facade forwards those exact defaults
    (not, say, a truthy ``speed`` sentinel) so a default drift is caught.
    """
    captured: dict = {}

    def spy(self, repo_id, robot_name=None, *, episode=0, root=None, speed=1.0, action_key_map=None):
        captured.update(
            repo_id=repo_id,
            robot_name=robot_name,
            episode=episode,
            root=root,
            speed=speed,
            action_key_map=action_key_map,
        )
        return {"status": "success"}

    monkeypatch.setattr(PolicyRunner, "replay", spy)

    so101_sim.replay_episode("local/only_repo")

    assert captured == {
        "repo_id": "local/only_repo",
        "robot_name": None,
        "episode": 0,
        "root": None,
        "speed": 1.0,
        "action_key_map": None,
    }


def test_replay_episode_reproduces_recorded_trajectory(so101_sim):
    """A self-recorded episode replays through the public facade to its pose.

    End-to-end proof that the public entrypoint (not just the internal runner)
    actually plays back a recording: record a short deterministic rollout, reset
    to the rest pose, then replay via ``sim.replay_episode`` and assert the arm
    lands back on the recorded final pose to float32 round-trip precision.
    """
    pytest.importorskip("lerobot")
    root = tempfile.mkdtemp(prefix="so101_facade_replay_")
    # Unique repo id + overwrite so a leftover dataset from a prior run in the
    # same cache cannot make start_recording refuse (record -> replay must be
    # self-contained and re-runnable).
    repo = f"local/{pathlib.Path(root).name}"

    assert (
        so101_sim.start_recording(repo_id=repo, task="rt", fps=30, root=root, cameras=[], overwrite=True)["status"]
        == "success"
    )
    assert (
        so101_sim.run_policy(
            robot_name="so101",
            policy_provider="mock",
            n_steps=90,
            control_frequency=30,
            action_horizon=8,
            fast_mode=True,
        )["status"]
        == "success"
    )
    assert so101_sim.stop_recording()["status"] == "success"
    recorded_final = _arm_qpos(so101_sim)

    so101_sim.reset()
    assert np.linalg.norm(_arm_qpos(so101_sim)) < 1e-6, "so101 should reset to the zero rest pose"

    result = so101_sim.replay_episode(repo, robot_name="so101", root=root, speed=1000.0)
    assert result["status"] == "success"

    gap = float(np.linalg.norm(recorded_final - _arm_qpos(so101_sim)))
    assert gap < 1e-2, f"facade replay did not reproduce the recording: {gap:.4f} rad gap"
