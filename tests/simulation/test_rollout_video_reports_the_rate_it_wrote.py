"""A rollout reports the rate its MP4 plays at, not the rate that was asked for.

``video={"fps": N}`` is a *request*. A rollout renders at most one frame per
applied control step, so ``_RolloutVideoWriter.open`` caps the writer at
``control_frequency`` when ``fps`` exceeds it (pinned in
``test_rollout_video_realtime_fps.py``) - the file on disk then plays at the
capped rate, not at ``N``.

The result has to say so. Quoting the requested ``fps`` for a file written at
another rate makes every length a caller derives from the report wrong by
exactly the capping factor: at ``control_frequency=10`` with the default
``fps=30``, a 2.0-second MP4 is reported as ``20 frames, 30fps`` and computes to
0.67 seconds. So the reported rate is pinned against the rate the decoder reads
back off the file, and the machine-readable payload - the surface that exists so
an agent need not parse the prose - carries it as ``video_fps``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("imageio")
pytest.importorskip("mujoco")

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


@pytest.fixture
def sim_with_arm(tmp_path):
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    sim = Simulation(tool_name="video_fps_report", mesh=False)
    try:
        sim.create_world()
        assert sim.add_robot(name="arm1", urdf_path=str(xml_path))["status"] == "success"
        yield sim
    finally:
        sim.cleanup(policy_stop_timeout=0.5)


def _result_json(result: dict) -> dict:
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in result: {result}")


def _file_fps(path) -> float:
    """The rate the MP4 on disk actually plays at, read back with a decoder."""
    import imageio.v3 as iio

    return float(iio.immeta(str(path), plugin="pyav")["fps"])


def _record(sim, path, *, fps: int, control_frequency: float, n_steps: int) -> dict:
    result = sim.run_policy(
        robot_name="arm1",
        policy_provider="mock",
        n_steps=n_steps,
        control_frequency=control_frequency,
        fast_mode=True,
        video={"path": str(path), "camera": "arm1/side", "fps": fps, "width": 160, "height": 120},
    )
    assert result["status"] == "success", result
    return result


class TestTheReportedRateIsTheRateOnDisk:
    """The human-readable report: what it quotes must be what plays."""

    @requires_gl
    def test_a_capped_rollout_reports_the_capped_rate(self, sim_with_arm, tmp_path):
        """fps above control_frequency: the text quotes what was written (15),
        not what was requested (30)."""
        path = tmp_path / "capped.mp4"
        text = _record(sim_with_arm, path, fps=30, control_frequency=15.0, n_steps=30)["content"][0]["text"]

        assert _file_fps(path) == 15.0  # the cap took effect
        assert "15fps" in text
        assert "30fps" not in text

    @requires_gl
    def test_an_uncapped_rollout_still_reports_the_requested_rate(self, sim_with_arm, tmp_path):
        """Control case: at fps <= control_frequency the capture cadence
        down-samples, nothing is capped, and the requested rate IS the written
        rate - the fix must not shift this reading."""
        path = tmp_path / "uncapped.mp4"
        text = _record(sim_with_arm, path, fps=10, control_frequency=20.0, n_steps=40)["content"][0]["text"]

        assert _file_fps(path) == 10.0
        assert "10fps" in text


class TestThePayloadCarriesTheRate:
    """The machine-readable surface exists so an agent need not parse the prose,
    so the rate has to be in it - under a key that is always present."""

    @requires_gl
    def test_the_payload_rate_is_the_rate_on_disk(self, sim_with_arm, tmp_path):
        path = tmp_path / "capped.mp4"
        payload = _result_json(_record(sim_with_arm, path, fps=30, control_frequency=15.0, n_steps=30))

        assert payload["video_fps"] == _file_fps(path) == 15.0

    @requires_gl
    def test_the_reported_frames_and_rate_give_the_files_real_length(self, sim_with_arm, tmp_path):
        """video_frames / video_fps is the MP4's own duration - the derivation a
        caller makes off the payload, which the requested rate got wrong by the
        capping factor (2.0s computing to 0.67s at fps=30, cf=10)."""
        path = tmp_path / "length.mp4"
        payload = _result_json(_record(sim_with_arm, path, fps=30, control_frequency=10.0, n_steps=20))

        import imageio.v3 as iio

        on_disk = float(iio.immeta(str(path), plugin="pyav")["duration"])
        assert payload["video_frames"] / payload["video_fps"] == pytest.approx(on_disk, abs=0.05)

    @requires_gl
    def test_a_rollout_with_no_recording_reports_no_rate(self, sim_with_arm):
        """The key is stable, so a caller can read it unconditionally: no MP4
        means no rate, reported as None rather than a rate nothing wrote."""
        payload = _result_json(
            sim_with_arm.run_policy(
                robot_name="arm1", policy_provider="mock", n_steps=5, control_frequency=20.0, fast_mode=True
            )
        )
        assert payload["video_fps"] is None
        assert payload["video_path"] is None
        assert payload["video_frames"] == 0
