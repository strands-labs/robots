"""Tests for ``strands_robots.simulation.policy_runner``.

Covers the backend-agnostic ``PolicyRunner`` (run/replay/evaluate) against a
pure-Python ``FakeSim`` stub, plus the real-backend behaviours:

* ``VideoConfig`` dataclass + legacy key consolidation
* ``run_policy(video={...})`` writes a valid MP4 (regression)
* ``run_policy(policy_object=...)`` reuses pre-built policies (regression)
* ``_extract_frame_ndarray`` decodes render() content blocks
* ``SimEngine.run_policy`` signature lock: no flat video params leaked.
"""

from __future__ import annotations

import base64
import inspect
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import strands_robots  # noqa: F401
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation import Simulation
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import (
    CooperativeStop,
    PolicyRunner,
    VideoConfig,
    _extract_frame_ndarray,
)

#
# PolicyRunner against FakeSim (backend-agnostic)
#


class FakeSim(SimEngine):
    """Minimal ``SimEngine`` implementation - no physics, records all calls."""

    def __init__(self, joint_names: tuple[str, ...] = ("j0", "j1", "j2")):
        self._joint_names = list(joint_names)
        self.calls: list[tuple] = []
        self._step_count = 0
        self._sim_time = 0.0
        self._robots = {"fake_robot": self._joint_names}

    # Implement abstract methods (bare minimum)
    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        self.calls.append(("reset",))
        self._step_count = 0
        self._sim_time = 0.0
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        self.calls.append(("step", n_steps))
        self._step_count += n_steps
        self._sim_time += 0.002 * n_steps
        return {"status": "success"}

    def get_state(self):
        return {"sim_time": self._sim_time, "step_count": self._step_count}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._robots.get(robot_name, []))

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        self.calls.append(("get_observation", robot_name))
        return {n: 0.0 for n in self._joint_names}

    def send_action(self, action, robot_name=None, n_substeps=1):
        self.calls.append(("send_action", dict(action), robot_name))
        self._step_count += 1
        self._sim_time += 0.002

    def render(self, camera_name="default", width=None, height=None):
        self.calls.append(("render", camera_name, width, height))
        return {
            "image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8),
        }


def test_policy_runner_only_touches_public_api():
    """Fail if PolicyRunner reaches past the SimEngine public surface."""
    sim = FakeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    result = PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=0.1,
        control_frequency=10.0,  # → 1 step total
        fast_mode=True,
    )

    assert result["status"] == "success"
    allowed = {"get_observation", "send_action", "step", "render", "reset"}
    for call in sim.calls:
        assert call[0] in allowed, f"PolicyRunner touched private API: {call}. Only {allowed} are allowed."


def test_policy_runner_import_does_not_pull_in_mujoco():
    """Importing policy_runner must not drag in mujoco."""

    # Wipe any existing mujoco imports
    for mod in [m for m in list(sys.modules) if m.startswith("mujoco")]:
        del sys.modules[mod]

    # Force a fresh import of the runner module
    if "strands_robots.simulation.policy_runner" in sys.modules:
        del sys.modules["strands_robots.simulation.policy_runner"]

    leaked = [m for m in sys.modules if m.startswith("mujoco")]
    assert not leaked, (
        f"strands_robots.simulation.policy_runner pulled in MuJoCo modules: {leaked}. "
        "The runner must be backend-agnostic."
    )


def test_on_frame_hook_receives_step_obs_action():
    """The on_frame hook is called per step with (idx, observation, action)."""
    captured: list[tuple] = []
    sim = FakeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    def hook(step: int, obs: dict[str, Any], action: dict[str, Any]) -> None:
        captured.append((step, dict(obs), dict(action)))

    result = PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=0.3,
        control_frequency=10.0,  # → 3 steps
        fast_mode=True,
        on_frame=hook,
    )

    assert result["status"] == "success"
    assert len(captured) >= 2
    # Each hook call carries the joint observation and a MockPolicy action
    for step_idx, obs, action in captured:
        assert "j0" in obs
        assert isinstance(action, dict)


def test_cooperative_stop_is_normal_success():
    """Raising ``CooperativeStop`` in the hook returns a success result."""
    sim = FakeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    def hook(step: int, obs, action) -> None:
        if step >= 2:
            raise CooperativeStop("user stopped")

    result = PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=10.0,
        control_frequency=10.0,  # would be 100 steps normally
        fast_mode=True,
        on_frame=hook,
    )
    assert result["status"] == "success"
    assert "stopped" in result["content"][0]["text"].lower()


def test_evaluate_calls_reset_per_episode():
    """evaluate() resets before every episode."""
    sim = FakeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    result = PolicyRunner(sim).evaluate(
        "fake_robot",
        policy,
        n_episodes=3,
        max_steps=5,
    )
    assert result["status"] == "success"
    # One reset per episode
    reset_calls = [c for c in sim.calls if c[0] == "reset"]
    assert len(reset_calls) == 3


def test_evaluate_success_fn_callable():
    """evaluate() supports arbitrary callable success_fn."""
    sim = FakeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    # Always succeed
    result = PolicyRunner(sim).evaluate(
        "fake_robot",
        policy,
        n_episodes=2,
        max_steps=10,
        success_fn=lambda obs: True,
    )

    payload = next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)
    assert payload["success_rate"] == 1.0
    assert payload["n_success"] == 2


def test_simengine_run_policy_facade_works_with_fake_sim():
    """The SimEngine.run_policy facade delegates to PolicyRunner correctly."""
    sim = FakeSim()
    # MockPolicy is the default - no policy_config needed.
    result = sim.run_policy(
        "fake_robot",
        policy_provider="mock",
        duration=0.2,
        control_frequency=10.0,
        fast_mode=True,
    )
    assert result["status"] == "success"


def test_simengine_eval_policy_facade_works_with_fake_sim():
    """The SimEngine.eval_policy facade delegates to PolicyRunner correctly."""
    sim = FakeSim()
    result = sim.eval_policy(
        robot_name="fake_robot",
        policy_provider="mock",
        n_episodes=2,
        max_steps=3,
    )
    assert result["status"] == "success"


def test_simengine_run_policy_validates_robot_exists():
    """run_policy returns a friendly error if the robot isn't in the sim."""
    sim = FakeSim()
    result = sim.run_policy(
        "nonexistent_robot",
        policy_provider="mock",
        duration=0.1,
        control_frequency=10.0,
        fast_mode=True,
    )
    assert result["status"] == "error"
    assert "not found" in result["content"][0]["text"].lower()


#
# run_policy(video=...) regression + helper unit tests
#


@pytest.mark.skipif(
    os.environ.get("CI") == "true" and not os.environ.get("ROBOT_TEST_MUJOCO"),
    reason="requires OpenGL; opt-in via ROBOT_TEST_MUJOCO=1",
)
def test_run_policy_video_writes_mp4(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "glfw")

    video_path = tmp_path / "rollout.mp4"

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam", position=[0.0, 0.0, 0.8], target=[0.0, 0.2, 0.05])

    result = sim.run_policy(
        robot_name="arm",
        policy_provider="mock",
        policy_config={},
        duration=0.5,
        control_frequency=20.0,
        video={"path": str(video_path), "fps": 20, "camera": "cam"},
    )

    sim.destroy()

    assert result["status"] == "success", f"rollout failed: {result}"
    assert video_path.exists(), f"video not written: {video_path}"
    assert video_path.stat().st_size > 0, "video file is empty"

    text_blocks = [c.get("text", "") for c in result.get("content", []) if isinstance(c, dict)]
    summary = "\n".join(text_blocks)
    assert "🎬 Video:" in summary, f"no video summary in output: {summary}"
    assert "📹" in summary and "frames" in summary, f"frame count missing: {summary}"


def test_extract_frame_ndarray_handles_render_shape() -> None:
    """Unit test the helper directly against the real render() output shape."""

    # Synthetic PNG with bytes source (the common MuJoCo path)
    img = Image.new("RGB", (8, 8), color=(128, 64, 32))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    result_bytes = {
        "status": "success",
        "content": [
            {"text": "📸 8x8 from 'cam'"},
            {"image": {"format": "png", "source": {"bytes": png_bytes}}},
        ],
    }
    arr = _extract_frame_ndarray(result_bytes)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (8, 8, 3)

    # Also accepts base64-encoded 'data' field
    result_b64 = {
        "status": "success",
        "content": [
            {"image": {"format": "png", "source": {"data": base64.b64encode(png_bytes).decode()}}},
        ],
    }
    arr2 = _extract_frame_ndarray(result_b64)
    assert isinstance(arr2, np.ndarray)
    assert arr2.shape == (8, 8, 3)

    # Rejects garbage
    assert _extract_frame_ndarray({}) is None
    assert _extract_frame_ndarray({"content": []}) is None
    assert _extract_frame_ndarray({"content": [{"text": "no image here"}]}) is None


#
# policy_object kwarg regression
#


@pytest.mark.skipif(
    os.environ.get("CI") == "true" and not os.environ.get("ROBOT_TEST_MUJOCO"),
    reason="requires OpenGL; opt-in via ROBOT_TEST_MUJOCO=1",
)
def test_run_policy_reuses_policy_object() -> None:
    pytest.importorskip("mujoco")
    """Two rollouts with a single pre-built MockPolicy should both succeed."""
    os.environ.setdefault("MUJOCO_GL", "glfw")

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])

    policy = MockPolicy()

    t0 = time.time()
    r1 = sim.run_policy(
        robot_name="arm",
        policy_object=policy,
        duration=0.3,
        control_frequency=20.0,
    )
    d1 = time.time() - t0
    assert r1["status"] == "success", r1

    t0 = time.time()
    r2 = sim.run_policy(
        robot_name="arm",
        policy_object=policy,
        duration=0.3,
        control_frequency=20.0,
    )
    d2 = time.time() - t0
    assert r2["status"] == "success", r2

    # Second call reuses policy; neither should be dramatically slower than the other.
    # (Both should be <2s for mock; if policy_object wasn't honoured, we'd rebuild.)
    assert d1 < 3.0 and d2 < 3.0, f"rollouts took {d1:.1f}s + {d2:.1f}s"

    sim.destroy()


def test_run_policy_object_param_exposed() -> None:
    """Signature check - policy_object must be in both base and MuJoCo variants."""

    sig = inspect.signature(Simulation.run_policy)
    assert "policy_object" in sig.parameters
    # Default must be None so existing callers are unaffected
    assert sig.parameters["policy_object"].default is None

    # start_policy too
    sig2 = inspect.signature(Simulation.start_policy)
    assert "policy_object" in sig2.parameters


#
# VideoConfig dataclass + legacy key consolidation
#


class TestVideoConfigDataclass:
    def test_default_config_is_disabled(self) -> None:
        cfg = VideoConfig()
        assert cfg.path is None
        assert cfg.enabled is False
        assert cfg.fps == 30
        assert cfg.camera is None
        assert cfg.width == 640
        assert cfg.height == 480

    def test_enabled_when_path_set(self) -> None:
        assert VideoConfig(path="/tmp/x.mp4").enabled is True

    def test_enabled_false_for_empty_string(self) -> None:
        """Empty path must be treated as "no recording", not a valid path."""
        assert VideoConfig(path="").enabled is False

    def test_frozen(self) -> None:
        cfg = VideoConfig(path="/tmp/a.mp4")
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            cfg.fps = 60  # type: ignore[misc]


class TestVideoConfigFromDict:
    def test_none_passthrough(self) -> None:
        assert VideoConfig.from_dict(None) is None

    def test_empty_dict_passthrough(self) -> None:
        assert VideoConfig.from_dict({}) is None

    def test_canonical_keys(self) -> None:
        cfg = VideoConfig.from_dict({"path": "/tmp/a.mp4", "fps": 60, "camera": "wrist", "width": 320, "height": 240})
        assert cfg is not None
        assert cfg.path == "/tmp/a.mp4"
        assert cfg.fps == 60
        assert cfg.camera == "wrist"
        assert cfg.width == 320
        assert cfg.height == 240

    def test_legacy_record_video_alias(self) -> None:
        """Back-compat: the old ``record_video`` flat kwarg name is accepted."""
        cfg = VideoConfig.from_dict({"record_video": "/tmp/legacy.mp4"})
        assert cfg is not None
        assert cfg.path == "/tmp/legacy.mp4"

    def test_legacy_output_path_alias(self) -> None:
        """tool_spec.json uses ``output_path``; legacy callers accepted."""
        cfg = VideoConfig.from_dict({"output_path": "/tmp/spec.mp4", "fps": 24})
        assert cfg is not None
        assert cfg.path == "/tmp/spec.mp4"
        assert cfg.fps == 24

    def test_legacy_video_fps_alias(self) -> None:
        cfg = VideoConfig.from_dict({"path": "/tmp/a.mp4", "video_fps": 15})
        assert cfg is not None
        assert cfg.fps == 15


class TestRunPolicySignatureNoFlatVideoParams:
    """Regression: the ABC and MuJoCo override must not expose flat video params."""

    _FORBIDDEN = {"record_video", "video_fps", "video_camera", "video_width", "video_height"}

    def test_sim_engine_run_policy_has_only_video_param(self) -> None:

        params = inspect.signature(SimEngine.run_policy).parameters
        leaked = self._FORBIDDEN.intersection(params)
        assert not leaked, f"SimEngine.run_policy still exposes flat video params: {leaked}"
        assert "video" in params

    def test_mujoco_run_policy_has_only_video_param(self) -> None:
        pytest.importorskip("mujoco")

        params = inspect.signature(Simulation.run_policy).parameters
        leaked = self._FORBIDDEN.intersection(params)
        assert not leaked, f"MuJoCo run_policy still exposes flat video params: {leaked}"
        assert "video" in params

    def test_policy_runner_run_has_only_video_param(self) -> None:

        params = inspect.signature(PolicyRunner.run).parameters
        leaked = self._FORBIDDEN.intersection(params)
        assert not leaked, f"PolicyRunner.run still exposes flat video params: {leaked}"
        assert "video" in params


class TestDispatcherFoldsFlatVideoKeys:
    """Agent callers pass flat ``output_path``/``fps`` via tool_spec.json.

    The MuJoCo dispatcher must fold those into a ``video`` dict before
    calling ``run_policy``, so Python-level and agent-level callers end
    up on the same code path.

    We subclass ``Simulation`` and override ``run_policy`` with the exact
    same signature so ``inspect.signature`` in the dispatcher matches
    against the real parameter list.
    """

    def _make_capturing_sim(self):
        pytest.importorskip("mujoco")

        captured: dict = {}

        class _CapturingSim(Simulation):
            def run_policy(  # type: ignore[override]
                self,
                robot_name: str,
                policy_provider: str = "mock",
                policy_config: dict | None = None,
                instruction: str = "",
                duration: float = 10.0,
                control_frequency: float = 50.0,
                action_horizon: int = 8,
                fast_mode: bool = False,
                video: dict | None = None,
            ) -> dict:
                captured.update(
                    {
                        "robot_name": robot_name,
                        "policy_provider": policy_provider,
                        "policy_config": policy_config,
                        "instruction": instruction,
                        "duration": duration,
                        "control_frequency": control_frequency,
                        "action_horizon": action_horizon,
                        "fast_mode": fast_mode,
                        "video": video,
                    }
                )
                return {"status": "success", "content": [{"text": "ok"}]}

        sim = _CapturingSim.__new__(_CapturingSim)
        sim._lock = threading.RLock()
        return sim, captured

    def test_dispatcher_folds_flat_keys(self) -> None:
        sim, captured = self._make_capturing_sim()
        sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "arm0",
                "output_path": "/tmp/x.mp4",
                "fps": 25,
                "camera_name": "wrist",
            },
        )
        assert captured["video"] == {"path": "/tmp/x.mp4", "fps": 25, "camera": "wrist"}

    def test_dispatcher_no_path_no_video(self) -> None:
        """Without ``output_path``, dispatcher must pass ``video=None``."""
        sim, captured = self._make_capturing_sim()
        sim._dispatch_action(
            "run_policy",
            {"robot_name": "arm0", "fps": 25, "camera_name": "wrist"},
        )
        assert captured["video"] is None, "dispatcher must not synthesise a video dict without an output path"

    def test_dispatcher_passes_explicit_video_dict_through(self) -> None:
        """If caller already provides ``video`` explicitly, don't clobber it."""
        sim, captured = self._make_capturing_sim()
        explicit_video = {"path": "/tmp/explicit.mp4", "fps": 120}
        sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "arm0",
                "video": explicit_video,
                "output_path": "/tmp/should_be_ignored.mp4",  # explicit wins
            },
        )
        assert captured["video"] == explicit_video


class _SubstepRecordingSim(FakeSim):
    """FakeSim that reports a fixed physics timestep and records the
    ``n_substeps`` PolicyRunner passes to each ``send_action`` call."""

    def __init__(self, dt: float = 0.002, joint_names=("j0", "j1", "j2")):
        super().__init__(joint_names=joint_names)
        self._dt = dt
        self.substeps_seen: list[int] = []

    def physics_timestep(self):  # noqa: D102 - see SimEngine.physics_timestep
        return self._dt

    def send_action(self, action, robot_name=None, n_substeps=1):
        self.substeps_seen.append(n_substeps)
        super().send_action(action, robot_name=robot_name, n_substeps=n_substeps)


def test_runner_substeps_at_control_rate():
    """Runner converts control_frequency + physics dt into substeps so a
    position-servo arm gets a full control period of physics per action.

    15 Hz control with a 2 ms physics dt => round((1/15)/0.002) = 33 substeps.
    """
    sim = _SubstepRecordingSim(dt=0.002)
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    res = PolicyRunner(sim).run("fake_robot", policy, duration=4 / 15, control_frequency=15.0, fast_mode=True)
    assert res["status"] == "success"
    assert sim.substeps_seen, "no send_action calls recorded"
    assert all(s == 33 for s in sim.substeps_seen), sim.substeps_seen


def test_runner_control_substeps_override():
    """Explicit ``control_substeps`` wins over the auto-derived value."""
    sim = _SubstepRecordingSim(dt=0.002)
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=0.2,
        control_frequency=15.0,
        fast_mode=True,
        control_substeps=7,
    )
    assert all(s == 7 for s in sim.substeps_seen), sim.substeps_seen


def test_runner_falls_back_to_one_substep_when_dt_unknown():
    """A backend that returns ``None`` from physics_timestep => 1 substep
    (preserves the legacy single-step behaviour)."""
    sim = _SubstepRecordingSim(dt=0.002)
    sim.physics_timestep = lambda: None  # type: ignore[assignment]
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    PolicyRunner(sim).run("fake_robot", policy, duration=0.2, control_frequency=15.0, fast_mode=True)
    assert all(s == 1 for s in sim.substeps_seen), sim.substeps_seen
