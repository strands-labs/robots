"""#191 — Synchronous-mode camera recording tests.

The synchronous recorder (:meth:`Simulation.start_cameras_recording_synchronous`)
returns ``(on_frame, finalize)`` callables instead of spawning a daemon
thread. Tests here exercise the full lifecycle without requiring a real
GL context — ``self.render(...)`` is monkey-patched on the Simulation
instance to return a synthetic PNG-encoded result (the same shape
``RenderingMixin.render`` produces). This lets the test run on
CI/dev-box environments that lack X11/EGL while still hitting the buffer
bookkeeping, the imageio MP4 flush, the idempotency contract, and the
mutual-exclusion guard with the daemon-thread variant.

The companion plumbing tests
(``tests/simulation/test_policy_runner_benchmark.py::TestOnFrameHookForSpec``)
cover ``policy_runner._evaluate_with_spec``'s ``on_frame`` invocation;
this file covers the recorder side of the contract.

Why a dedicated module: the existing
``tests/simulation/mujoco/test_rendering.py`` is gated on a working GL
context for the daemon-thread MP4 round-trip tests, so it skips wholesale
in headless environments. The synchronous-mode recorder is GL-free in its
bookkeeping path (the fake render result is what matters); pulling it
into a separate module keeps it discoverable without entangling with the
GL gate.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed - pip install strands-robots[sim-mujoco]")
imageio = pytest.importorskip("imageio", reason="imageio not installed - pip install imageio imageio-ffmpeg")
# Validate ``imageio.v2`` is importable since
# ``RenderingMixin._flush_cameras_recording_state`` uses it; ``importorskip``
# on bare ``imageio`` doesn't catch a v2 split-package install. Skips
# the whole module if v2 is missing.
pytest.importorskip("imageio.v2", reason="imageio.v2 not available - upgrade imageio (>=2.5)")

from strands_robots.simulation import Simulation  # noqa: E402


def _png_bytes(arr: np.ndarray) -> bytes:
    """Encode a (H, W, 3) uint8 ndarray as PNG bytes.

    Mirrors the wire format ``RenderingMixin.render`` returns inside its
    content blocks (``content[1]["image"]["source"]["bytes"]``). Used by
    the fake ``render`` to feed the synchronous recorder without needing
    a real GL context.
    """
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _make_render_result(width: int = 32, height: int = 24, fill: int = 128) -> dict[str, Any]:
    """Construct a fake render() return dict with a real PNG payload.

    The recorder calls ``_extract_frame_ndarray`` on the result, which
    walks ``content[*]["image"]["source"]["bytes"]`` and PIL-decodes
    the PNG. So the test fixture has to produce real PNG bytes — a
    bare ``b"png"`` placeholder won't decode and the recorder would
    silently increment ``state["errors"][cam]`` instead of buffering
    a frame.
    """
    arr = np.full((height, width, 3), fill, dtype=np.uint8)
    # Add a marker pixel so we can detect frames at finalize time.
    arr[0, 0] = [255, 0, 0]
    return {
        "status": "success",
        "content": [
            {"text": f"📸 {width}x{height}"},
            {"image": {"format": "png", "source": {"bytes": _png_bytes(arr)}}},
        ],
    }


def _make_sim_with_fake_render() -> Simulation:
    """Construct a real Simulation with a fake ``render`` method.

    Builds a real world + robot so ``self._world._model`` / ``self._world._data``
    are populated (the synchronous recorder validates the world is loaded
    before returning callables). ``render`` is replaced post-construction
    with a counter-tracking stub so the test asserts how many times each
    camera was rendered.
    """
    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    # Add cameras so _active_camera_list resolves without hitting the
    # "no cameras" error path. The actual camera positions don't matter
    # since we replace ``render`` below.
    sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.add_camera("cam_b", position=[0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])

    render_calls: list[dict[str, Any]] = []

    def _fake_render(camera_name: str, width: int | None = None, height: int | None = None, **_kw):
        render_calls.append({"camera": camera_name, "width": width, "height": height})
        return _make_render_result(width=width or 32, height=height or 24)

    # Stash on the instance, not the class, so other tests aren't affected.
    sim.render = _fake_render  # type: ignore[assignment,method-assign]
    sim._test_render_calls = render_calls  # type: ignore[attr-defined]
    return sim


class TestStartCamerasRecordingSynchronous:
    """``start_cameras_recording_synchronous`` returns ``(on_frame, finalize)``
    closures that capture frames synchronously on the calling thread.

    Pinned so the daemon-thread regressions surfaced in #191
    (2-3% capture rate + greenish gradient artifacts under multi-threaded
    eval) cannot recur silently — any drift in the synchronous-mode
    bookkeeping shows up here.
    """

    def test_returns_on_frame_and_finalize_callables(self):
        """The success result carries the closures in the JSON content block."""
        sim = _make_sim_with_fake_render()
        try:
            result = sim.start_cameras_recording_synchronous(
                cameras=["cam_a", "cam_b"], output_dir=None, fps=15, name="t1"
            )
            assert result["status"] == "success", result

            json_block = next(c["json"] for c in result["content"] if "json" in c)
            assert callable(json_block["on_frame"]), "on_frame must be a callable"
            assert callable(json_block["finalize"]), "finalize must be a callable"
            assert json_block["name"] == "t1"
            assert "output_dir" in json_block
        finally:
            sim.destroy()

    def test_on_frame_renders_each_camera_per_call(self):
        """Each ``on_frame`` invocation triggers one ``self.render(...)`` per
        camera, populating per-camera buffers."""
        sim = _make_sim_with_fake_render()
        try:
            result = sim.start_cameras_recording_synchronous(cameras=["cam_a", "cam_b"], fps=15, name="t2")
            on_frame = next(c["json"] for c in result["content"] if "json" in c)["on_frame"]

            for step in range(5):
                on_frame(step, {}, {})

            # 5 calls × 2 cameras = 10 renders.
            assert len(sim._test_render_calls) == 10
            # Equal split per camera.
            cams_seen = [c["camera"] for c in sim._test_render_calls]
            assert cams_seen.count("cam_a") == 5
            assert cams_seen.count("cam_b") == 5

            # Buffers are populated.
            state = sim._cams_rec_state
            assert len(state["buffers"]["cam_a"]) == 5
            assert len(state["buffers"]["cam_b"]) == 5
        finally:
            sim.destroy()

    def test_finalize_writes_one_mp4_per_camera_with_correct_frame_count(self, tmp_path: Path):
        """``finalize()`` flushes the per-camera buffers to MP4 files in the
        output dir. Frame counts in the JSON artifacts match the number of
        ``on_frame`` calls that captured each camera."""
        sim = _make_sim_with_fake_render()
        try:
            result = sim.start_cameras_recording_synchronous(
                cameras=["cam_a", "cam_b"], output_dir=str(tmp_path), fps=15, name="t3"
            )
            json_block = next(c["json"] for c in result["content"] if "json" in c)
            on_frame = json_block["on_frame"]
            finalize = json_block["finalize"]

            for step in range(7):
                on_frame(step, {}, {})

            final = finalize()
            assert final["status"] == "success", final

            artifacts = next(c["json"] for c in final["content"] if "json" in c)["artifacts"]
            artifact_by_cam = {a["camera"]: a for a in artifacts}
            assert artifact_by_cam["cam_a"]["frames"] == 7
            assert artifact_by_cam["cam_b"]["frames"] == 7
            assert artifact_by_cam["cam_a"]["errors"] == 0
            assert artifact_by_cam["cam_b"]["errors"] == 0

            # MP4 files actually exist on disk.
            mp4_files = sorted(tmp_path.glob("*.mp4"))
            assert len(mp4_files) == 2, f"expected 2 mp4 files, got {[f.name for f in mp4_files]}"
            for f in mp4_files:
                assert f.stat().st_size > 0, f"empty mp4: {f}"
        finally:
            sim.destroy()

    def test_finalize_is_idempotent(self):
        """Calling ``finalize()`` twice is safe — second call returns
        ``"Was not recording cameras."`` instead of re-flushing or
        crashing. Matches ``stop_cameras_recording``'s contract.
        """
        sim = _make_sim_with_fake_render()
        try:
            result = sim.start_cameras_recording_synchronous(cameras=["cam_a"], fps=15, name="t4")
            json_block = next(c["json"] for c in result["content"] if "json" in c)
            on_frame = json_block["on_frame"]
            finalize = json_block["finalize"]
            on_frame(0, {}, {})

            first = finalize()
            assert first["status"] == "success"
            assert "Was not recording" not in first["content"][0]["text"]

            second = finalize()
            assert second["status"] == "success"
            assert "Was not recording" in second["content"][0]["text"]
        finally:
            sim.destroy()

    def test_stop_cameras_recording_also_finalizes_sync_mode(self, tmp_path: Path):
        """``stop_cameras_recording`` works equivalently to ``finalize()``
        for synchronous-mode recordings — useful for callers that don't
        keep the closure handle around (e.g. tool-spec dispatch over
        an agent contract that can't return Python callables).
        """
        sim = _make_sim_with_fake_render()
        try:
            result = sim.start_cameras_recording_synchronous(
                cameras=["cam_a"], output_dir=str(tmp_path), fps=15, name="t5"
            )
            on_frame = next(c["json"] for c in result["content"] if "json" in c)["on_frame"]
            for step in range(3):
                on_frame(step, {}, {})

            stop = sim.stop_cameras_recording()
            assert stop["status"] == "success", stop
            artifacts = next(c["json"] for c in stop["content"] if "json" in c)["artifacts"]
            assert artifacts[0]["frames"] == 3
            assert artifacts[0]["camera"] == "cam_a"

            # State is cleared.
            assert sim._cams_rec_state is None
        finally:
            sim.destroy()

    def test_on_frame_swallows_render_errors_into_state_errors(self):
        """A single failing render call should not crash the eval rollout.
        It increments ``state["errors"][cam]`` instead — the same policy
        the daemon-thread recorder uses.

        Pinned so a refactor that promotes per-frame failures to
        exceptions can't silently break long-running evals.
        """
        sim = _make_sim_with_fake_render()
        try:
            # Replace render with a function that raises on cam_b only.
            calls: list[str] = []

            def _flaky_render(camera_name: str, width: int | None = None, height: int | None = None, **_kw):
                calls.append(camera_name)
                if camera_name == "cam_b":
                    raise RuntimeError("camera offline")
                return _make_render_result(width=width or 32, height=height or 24)

            sim.render = _flaky_render  # type: ignore[assignment,method-assign]

            result = sim.start_cameras_recording_synchronous(cameras=["cam_a", "cam_b"], fps=15, name="t6")
            on_frame = next(c["json"] for c in result["content"] if "json" in c)["on_frame"]

            for step in range(4):
                on_frame(step, {}, {})

            state = sim._cams_rec_state
            assert len(state["buffers"]["cam_a"]) == 4
            assert len(state["buffers"]["cam_b"]) == 0
            assert state["errors"]["cam_a"] == 0
            assert state["errors"]["cam_b"] == 4
        finally:
            sim.destroy()

    def test_max_frames_caps_buffer_growth(self):
        """``max_frames_per_camera`` bounds in-memory buffer growth.
        Frames captured beyond the cap are silently dropped (no error
        increment, just a no-op append).

        Pin so a long-running eval can't OOM the recorder.
        """
        sim = _make_sim_with_fake_render()
        try:
            result = sim.start_cameras_recording_synchronous(
                cameras=["cam_a"], fps=15, name="t7", max_frames_per_camera=3
            )
            on_frame = next(c["json"] for c in result["content"] if "json" in c)["on_frame"]

            for step in range(10):
                on_frame(step, {}, {})

            state = sim._cams_rec_state
            assert len(state["buffers"]["cam_a"]) == 3, "buffer must cap at max_frames_per_camera"
            # Errors stay zero — the cap is a deliberate skip, not a render failure.
            assert state["errors"]["cam_a"] == 0
        finally:
            sim.destroy()

    def test_already_recording_returns_error(self):
        """Concurrent recordings (sync vs daemon, or sync vs sync) are
        rejected. Caller must explicitly stop the active recording first.
        """
        sim = _make_sim_with_fake_render()
        try:
            r1 = sim.start_cameras_recording_synchronous(cameras=["cam_a"], fps=15, name="first")
            assert r1["status"] == "success"

            r2 = sim.start_cameras_recording_synchronous(cameras=["cam_a"], fps=15, name="second")
            assert r2["status"] == "error"
            assert "Already recording 'first'" in r2["content"][0]["text"]
        finally:
            sim.destroy()

    def test_no_world_returns_error(self):
        """Calling before ``create_world`` is a structured error, not a
        traceback. Matches the daemon-thread variant's behaviour.
        """
        sim = Simulation()
        try:
            r = sim.start_cameras_recording_synchronous(cameras=["cam_a"], fps=15, name="t8")
            assert r["status"] == "error"
            assert "No world" in r["content"][0]["text"]
        finally:
            sim.destroy()

    def test_unresolved_camera_name_returns_error(self):
        """User-provided camera names that don't resolve are rejected up
        front. Matches the daemon-thread variant.
        """
        sim = _make_sim_with_fake_render()
        try:
            r = sim.start_cameras_recording_synchronous(cameras=["does_not_exist"], fps=15, name="t9")
            assert r["status"] == "error"
            assert "not found" in r["content"][0]["text"]
        finally:
            sim.destroy()

    def test_state_mode_marker(self):
        """The recorder state dict carries a ``mode`` field so introspection
        / status tooling can distinguish synchronous vs daemon-thread
        recordings.
        """
        sim = _make_sim_with_fake_render()
        try:
            sim.start_cameras_recording_synchronous(cameras=["cam_a"], fps=15, name="t10")
            assert sim._cams_rec_state["mode"] == "synchronous"
            # Thread is None in sync mode (no daemon spawned).
            assert sim._cams_rec_state["thread"] is None
        finally:
            sim.destroy()
