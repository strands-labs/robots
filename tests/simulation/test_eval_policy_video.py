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


class TestEvalPolicyVideoZeroFramesCaptured:
    """A camera that passes the up-front probe but decodes no in-loop frames
    (e.g. it is unplugged mid-eval) must not silently list empty per-episode
    MP4s as if they held a rollout.

    ``run_policy`` already flags a 0-frame rollout, but the multi-episode paths
    - ``eval_policy`` (per-episode ``_ep{i}.mp4``) and the ``spec``/benchmark
    route (:meth:`PolicyRunner.evaluate` with a spec) - independently decide
    whether to append each episode's writer path to the returned ``video_paths``.
    This pins that both routes stay ``status=success`` (a dead camera does not
    kill a scored eval), OMIT the empty episode from ``video_paths`` so the
    aggregate honestly reflects which episodes captured, and log a per-episode
    warning naming the offending path.
    """

    @staticmethod
    def _blind_render(*_a, **_k):
        # Probe (``_RolloutVideoWriter.open``) requires status=success, so this
        # opens the writer; but it carries no image block, so every in-loop
        # ``capture()`` decodes to None and no frame is ever appended.
        return {"status": "success", "content": [{"text": "no image block"}]}

    @requires_gl
    def test_eval_policy_omits_empty_episode_videos(self, sim_with_arm, tmp_path, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(sim_with_arm, "render", self._blind_render)
        base = tmp_path / "eval.mp4"
        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.policy_runner"):
            result = sim_with_arm.eval_policy(
                robot_name="arm1",
                policy_provider="mock",
                n_episodes=2,
                max_steps=6,
                control_frequency=20.0,
                video={"path": str(base), "camera": "arm1/side", "fps": 10, "width": 160, "height": 120},
            )
        # A dead camera does not fail the scored eval.
        assert result["status"] == "success", result
        # Both episodes captured 0 frames -> neither is listed.
        assert _result_json(result)["video_paths"] == []
        # Every episode that requested a video but wrote nothing is flagged.
        zero_frame_warnings = [r for r in caplog.records if "wrote 0 frames" in r.getMessage()]
        assert len(zero_frame_warnings) == 2, [r.getMessage() for r in caplog.records]
        assert "eval_policy episode" in zero_frame_warnings[0].getMessage()

    @requires_gl
    def test_benchmark_spec_path_omits_empty_episode_videos(self, sim_with_arm, tmp_path, monkeypatch, caplog):
        import logging

        from strands_robots.policies import create_policy

        class _NoopSpec(BenchmarkProtocol):
            max_steps = 5

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

        monkeypatch.setattr(sim_with_arm, "render", self._blind_render)
        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.policy_runner"):
            result = PolicyRunner(sim_with_arm).evaluate(
                "arm1",
                create_policy("mock"),
                spec=_NoopSpec(),
                n_episodes=2,
                control_frequency=20.0,
                video={"path": str(tmp_path / "bench.mp4"), "camera": "arm1/side", "fps": 10},
            )
        assert result["status"] == "success", result
        assert _result_json(result)["video_paths"] == []
        zero_frame_warnings = [r for r in caplog.records if "wrote 0 frames" in r.getMessage()]
        assert len(zero_frame_warnings) == 2, [r.getMessage() for r in caplog.records]
        assert "evaluate_benchmark episode" in zero_frame_warnings[0].getMessage()
