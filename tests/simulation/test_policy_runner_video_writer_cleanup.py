"""The rollout video writer must be finalized on every ``PolicyRunner.run`` exit.

``PolicyRunner.run`` opens an MP4 writer before the step loop when a caller
requests ``video={...}``. If the policy raises mid-rollout the run must not
leak that writer: an unclosed imageio/ffmpeg writer holds a subprocess pipe and
an open file descriptor, and on a training host that records thousands of
rollouts a leaked writer per crashed episode exhausts descriptors and leaves
truncated, unplayable MP4 files. The contract: whether the rollout completes,
is cooperatively stopped, or raises, the writer is always closed and the run
returns a structured ``{"status": "error", ...}`` (never a bare exception) so an
agent driving ``run_policy`` can react instead of crashing.
"""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

import imageio

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.policy_runner import PolicyRunner, VideoConfig
from tests.simulation.test_policy_runner import FakeSim


class _RenderableSim(FakeSim):
    """``FakeSim`` whose ``render`` returns a status-tagged frame.

    The base ``FakeSim.render`` omits the ``"status"`` key, which the video
    writer's up-front camera probe treats as an unrenderable camera. Adding the
    key lets the writer open so the failure-path close can be exercised without
    a real MuJoCo backend.
    """

    def render(self, camera_name="default", width=None, height=None):
        return {
            "status": "success",
            "image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8),
        }


class _SpyWriter:
    """Stand-in for the imageio writer that records whether ``close`` ran."""

    def __init__(self) -> None:
        self.closed = False
        self.frames = 0

    def append_data(self, img) -> None:
        self.frames += 1

    def close(self) -> None:
        self.closed = True


class _RaisingPolicy(MockPolicy):
    """A policy whose inference call raises on the first query."""

    async def get_actions(self, observation, instruction="", **kwargs):
        raise RuntimeError("boom mid-rollout")


def test_run_closes_video_writer_when_policy_raises(monkeypatch, tmp_path) -> None:
    spy = _SpyWriter()
    monkeypatch.setattr(imageio, "get_writer", lambda *a, **k: spy)

    sim = _RenderableSim()
    policy = _RaisingPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    result = PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=0.2,
        control_frequency=20.0,
        fast_mode=True,
        video=VideoConfig.from_dict({"path": str(tmp_path / "rollout.mp4"), "fps": 20, "camera": "default"}),
    )

    # A crashed rollout returns a structured error, never a raised exception.
    assert result["status"] == "error"
    assert "Policy failed" in result["content"][0]["text"]
    # The writer opened (camera probe succeeded) and was finalized on the way out.
    assert spy.closed, "video writer was leaked when the rollout raised"
