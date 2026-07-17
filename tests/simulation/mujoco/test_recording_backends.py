"""T12: Video recording backends.

* start_recording (LeRobotDataset) requires the lerobot extra; when it's
  not installed, the error message must point to start_cameras_recording
  for plain MP4 and to the [lerobot] extra for dataset recording.
* start_cameras_recording works under [sim-mujoco] alone (imageio-ffmpeg)
  and does not need lerobot.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No GL context available (headless CI without EGL/OSMesa)",
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="rec_backend_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


class TestStartRecordingErrorWithoutLerobot:
    """start_recording must degrade to an actionable structured error - never a
    raw ImportError traceback - when the [lerobot] extra is unavailable.

    Both failure modes are exercised deterministically (independent of whether
    lerobot happens to be installed in the test environment) because the
    contract must hold in every environment:

    * the extra is cleanly absent (``has_lerobot_dataset()`` returns False), and
    * a partial/broken install where importing ``DatasetRecorder`` itself
      raises ImportError (e.g. lerobot-from-source API drift).

    Previously this was guarded by ``skipif(has_lerobot)``, so with lerobot
    installed the whole error contract went untested and the ImportError
    fallback was never covered.
    """

    def test_extra_absent_points_to_start_cameras_recording(self, sim, monkeypatch):
        import strands_robots.dataset_recorder as dr

        monkeypatch.setattr(dr, "has_lerobot_dataset", lambda: False)
        result = sim.start_recording(repo_id="local/test_rec")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "start_cameras_recording" in text
        assert "lerobot" in text.lower()

    def test_broken_import_falls_through_to_structured_error(self, sim, monkeypatch):
        import strands_robots.dataset_recorder as dr

        # Simulate a partial/broken lerobot install: the symbol is gone, so the
        # in-function ``from ... import DatasetRecorder`` raises ImportError,
        # which must be swallowed into the same actionable error rather than
        # propagating a traceback out of the tool call.
        monkeypatch.delattr(dr, "DatasetRecorder", raising=False)
        result = sim.start_recording(repo_id="local/test_rec")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "start_cameras_recording" in text
        assert "lerobot" in text.lower()


@requires_gl
class TestCamerasRecordingWithoutLerobot:
    """start_cameras_recording must work under [sim-mujoco] alone."""

    def test_start_stop_writes_mp4(self, sim, tmp_path):
        # Ensure at least one camera exists.
        r = sim.add_camera(name="cam1", position=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.0])
        assert r["status"] == "success"

        out = tmp_path / "mp4out"
        r = sim.start_cameras_recording(
            cameras=["cam1"],
            output_dir=str(out),
            fps=10,
            width=160,
            height=120,
            name="t12_smoke",
        )
        assert r["status"] == "success", r

        # Capture a few frames via stepping the sim.
        for _ in range(10):
            sim.step(n_steps=1)
            # tiny sleep to let the background capture thread tick
            import time

            time.sleep(0.05)

        r = sim.stop_cameras_recording()
        assert r["status"] == "success", r

        # At least one .mp4 must have landed in output_dir.
        assert out.exists()
        files = [f for f in os.listdir(out) if f.endswith(".mp4")]
        assert files, f"no mp4 files in {out}"
