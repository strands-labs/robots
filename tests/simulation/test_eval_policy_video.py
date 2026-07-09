"""Regression: ``eval_policy`` records one rollout MP4 per episode (video=).

``run_policy`` could already record rollout video, but ``eval_policy`` - the
multi-episode success-measuring path - could not, so an eval could only be read
as an aggregate ``success_rate`` and never watched to see WHY episodes failed.
This pins the per-episode ``_ep{i}.mp4`` recording, the up-front camera
validation, and the benchmark-path rejection.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import BenchmarkProtocol, StepInfo  # noqa: E402
from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402

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


@pytest.fixture
def sim_with_arm(tmp_path):
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    sim = Simulation(tool_name="eval_video", mesh=False)
    try:
        sim.create_world()
        r = sim.add_robot(name="arm1", urdf_path=str(xml_path))
        assert r["status"] == "success", r
        yield sim
    finally:
        sim.cleanup(policy_stop_timeout=0.5)


def _result_json(result: dict) -> dict:
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in result: {result}")


class TestEvalPolicyVideo:
    @requires_gl
    def test_records_one_mp4_per_episode(self, sim_with_arm, tmp_path):
        """A 2-episode eval with video= writes eval_ep0.mp4 + eval_ep1.mp4,
        each with real frames, and lists them in the result json."""
        import imageio.v3 as iio

        base = tmp_path / "eval.mp4"
        result = sim_with_arm.eval_policy(
            robot_name="arm1",
            policy_provider="mock",
            n_episodes=2,
            max_steps=10,
            control_frequency=20.0,
            video={"path": str(base), "camera": "arm1/side", "fps": 10, "width": 160, "height": 120},
        )
        assert result["status"] == "success", result
        video_paths = _result_json(result)["video_paths"]
        assert len(video_paths) == 2, video_paths

        ep0 = str(tmp_path / "eval_ep0.mp4")
        ep1 = str(tmp_path / "eval_ep1.mp4")
        assert video_paths == [ep0, ep1], video_paths
        # The un-suffixed base path must never be written - only per-episode files.
        assert not base.exists(), "base path should not be written; per-episode files only"
        for vp in video_paths:
            frames = iio.imread(vp, plugin="pyav")
            assert len(frames) > 0, f"{vp} has no frames"
            assert frames.shape[1:3] == (120, 160), frames.shape

    def test_bad_camera_fails_fast(self, sim_with_arm, tmp_path):
        """A wrong camera name is caught up-front (before any episode runs),
        not after N episodes of silent 0-frame MP4s."""
        base = tmp_path / "bad.mp4"
        result = sim_with_arm.eval_policy(
            robot_name="arm1",
            policy_provider="mock",
            n_episodes=3,
            max_steps=5,
            video={"path": str(base), "camera": "side", "fps": 10},
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"].lower()
        assert "not renderable" in text or "not found" in text, text
        # No per-episode stub file should have been written.
        assert not (tmp_path / "bad_ep0.mp4").exists()

    @requires_gl
    def test_video_recorded_on_benchmark_spec_path(self, sim_with_arm, tmp_path):
        """Recording works on BOTH eval routes: the spec/benchmark path now
        records a per-episode MP4 too (parity with eval_policy), captured
        synchronously so the bit-stable rollout is unperturbed. See
        test_evaluate_benchmark_video.py for the full facade coverage."""

        class _NoopSpec(BenchmarkProtocol):
            max_steps = 4

            @property
            def supported_robots(self) -> list[str]:
                return ["arm1"]

            @property
            def default_robot(self) -> str:
                return "arm1"

            def on_step(self, sim, obs, action):
                return StepInfo(reward=0.0)

            def is_success(self, sim):
                return False

        result = PolicyRunner(sim_with_arm).evaluate(
            "arm1",
            __import__("strands_robots.policies", fromlist=["create_policy"]).create_policy("mock"),
            spec=_NoopSpec(),
            n_episodes=1,
            control_frequency=20.0,
            video={"path": str(tmp_path / "x.mp4"), "camera": "arm1/side", "fps": 10},
        )
        assert result["status"] == "success", result
        assert _result_json(result)["video_paths"] == [str(tmp_path / "x_ep0.mp4")]
