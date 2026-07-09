"""Regression: ``evaluate_benchmark`` records one rollout MP4 per episode (video=).

``eval_policy`` (the success_fn eval path) could already record a per-episode
rollout video, but ``evaluate_benchmark`` (the spec/benchmark eval path) could
not - the spec route hard-rejected ``video`` so a benchmark eval could only be
read as an aggregate ``success_rate`` and never watched to see WHY episodes
fail. This pins the per-episode ``_ep{i}.mp4`` recording, the up-front camera
validation, the ``video_paths`` result field, and the no-video no-op.

Frames are captured synchronously on the eval thread (render is read-only over
``mjData``), so recording does not perturb the bit-stable spec-path rollout.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import (  # noqa: E402
    BenchmarkProtocol,
    StepInfo,
    register_benchmark,
    unregister_benchmark,
)
from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No OpenGL context available (EGL/OSMesa required for offscreen rendering)",
)

ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
    <camera name="side" pos="0.8 -0.8 0.4" xyaxes="0.707 0.707 0 -0.2 0.2 0.96"/>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="30"/>
  </actuator>
</mujoco>
"""


class _NoopSpec(BenchmarkProtocol):
    """Minimal always-running spec: fixed max_steps, never succeeds/fails."""

    max_steps = 6

    @property
    def supported_robots(self) -> list[str]:
        return ["arm1"]

    @property
    def default_robot(self) -> str:
        return "arm1"

    def on_episode_start(self, sim: Any, rng: random.Random) -> None:
        return None

    def on_step(self, sim: Any, obs: dict[str, Any], action: dict[str, Any]) -> StepInfo:
        return StepInfo(reward=0.0)

    def is_success(self, sim: Any) -> bool:
        return False

    def is_failure(self, sim: Any) -> bool:
        return False


@pytest.fixture
def sim_and_bench(tmp_path):
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    sim = Simulation(tool_name="bench_video", mesh=False)
    name = "noop_video_bench"
    register_benchmark(name, _NoopSpec())
    try:
        sim.create_world()
        r = sim.add_robot(name="arm1", urdf_path=str(xml_path))
        assert r["status"] == "success", r
        yield sim, name
    finally:
        try:
            unregister_benchmark(name)
        except Exception:
            # Best-effort teardown: the benchmark may already be gone if the
            # test body failed before/after registration. Never mask the real
            # test failure with a cleanup error.
            pass
        sim.cleanup(policy_stop_timeout=0.5)


def _result_json(result: dict) -> dict:
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in result: {result}")


class TestEvaluateBenchmarkVideo:
    @requires_gl
    def test_records_one_mp4_per_episode(self, sim_and_bench, tmp_path):
        """A 2-episode benchmark eval with video= writes bench_ep0.mp4 +
        bench_ep1.mp4, each with real frames, and lists them in video_paths."""
        import imageio.v3 as iio

        sim, bench = sim_and_bench
        base = tmp_path / "bench.mp4"
        result = sim.evaluate_benchmark(
            benchmark_name=bench,
            robot_name="arm1",
            policy_provider="mock",
            n_episodes=2,
            control_frequency=20.0,
            video={"path": str(base), "camera": "arm1/side", "fps": 10, "width": 160, "height": 120},
        )
        assert result["status"] == "success", result
        payload = _result_json(result)
        video_paths = payload["video_paths"]
        assert len(video_paths) == 2, video_paths

        ep0 = str(tmp_path / "bench_ep0.mp4")
        ep1 = str(tmp_path / "bench_ep1.mp4")
        assert video_paths == [ep0, ep1], video_paths
        # The un-suffixed base path must never be written - only per-episode files.
        assert not base.exists(), "base path should not be written; per-episode files only"
        for vp in video_paths:
            frames = iio.imread(vp, plugin="pyav")
            assert len(frames) > 0, f"{vp} has no frames"
            assert frames.shape[1:3] == (120, 160), frames.shape

    @requires_gl
    def test_bad_camera_fails_fast(self, sim_and_bench, tmp_path):
        """A wrong camera name is caught up-front (before any episode records),
        not after N episodes of silent 0-frame MP4s."""
        sim, bench = sim_and_bench
        base = tmp_path / "bad.mp4"
        result = sim.evaluate_benchmark(
            benchmark_name=bench,
            robot_name="arm1",
            policy_provider="mock",
            n_episodes=3,
            video={"path": str(base), "camera": "does_not_exist", "fps": 10},
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"].lower()
        assert "not renderable" in text or "not found" in text, text
        assert not (tmp_path / "bad_ep0.mp4").exists()

    def test_no_video_is_noop(self, sim_and_bench):
        """Without video=, evaluate_benchmark records nothing (opt-in) and
        returns an empty video_paths list - no GL needed."""
        sim, bench = sim_and_bench
        result = sim.evaluate_benchmark(
            benchmark_name=bench,
            robot_name="arm1",
            policy_provider="mock",
            n_episodes=2,
        )
        assert result["status"] == "success", result
        assert _result_json(result)["video_paths"] == []
