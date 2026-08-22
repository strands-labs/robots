"""A render the renderer REFUSED is not a camera that is merely "cold".

The recorder thread warms each camera's GL context before the timing loop, and
until now it had one verdict for every camera that never produced usable
output: still cold after 30 attempts, first frames may show gradient artifact.
That reads as a context that has not settled yet.

A render can instead come back as a structured ERROR RESULT - the
``status="error"`` envelope :meth:`render` returns when ``_get_renderer`` yields
no renderer at all. Two things follow, and both were invisible:

* No number of attempts fixes a missing GL context, so "cold" is not a slower
  version of the same story - every captured frame will be empty.
* Because the refusal is a *returned dict* and not a raised exception, the
  warmup loop's ``except`` branch never sees it, so the reason was dropped at
  every log level and the recording finished as ``status="success"`` with zero
  frames, an empty MP4, and nothing in the artifact saying why.

These tests drive the real daemon recorder. ``start_cameras_recording`` waits on
``state["ready"]``, which ``_loop`` sets only after warmup finishes, so the call
returning IS the synchronisation point: no sleeps and no thread patching, and
every assertion below reads state the warmup has already written.
"""

from __future__ import annotations

import io
import logging

import pytest

pytest.importorskip("mujoco")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_LOGGER = "strands_robots.simulation.mujoco.rendering"


@pytest.fixture
def sim():
    s = MuJoCoSimEngine(tool_name="warmup_refusal_test", mesh=False)
    s.create_world()
    s.add_camera(name="cam_a", position=[0.6, -0.5, 0.4], target=[0.0, 0.0, 0.1])
    s.add_camera(name="cam_b", position=[-0.6, 0.5, 0.4], target=[0.0, 0.0, 0.1])
    yield s
    try:
        s.stop_cameras_recording()
    finally:
        s.cleanup()


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


#: A frame with real per-column variance: ``arr.std(axis=0).mean()`` well over
#: the warmup's 5.0 threshold, so the camera reports warm on the first attempt.
_WARM_PNG = _png(np.concatenate([np.full((12, 32, 3), 255, np.uint8), np.zeros((12, 32, 3), np.uint8)]))
#: A uniform frame: per-column std-dev is exactly 0, which is what "cold" means
#: here - a real frame that carries no geometry yet.
_COLD_PNG = _png(np.full((24, 32, 3), 128, np.uint8))


def _frame(png: bytes) -> dict:
    return {
        "status": "success",
        "content": [{"image": {"format": "png", "source": {"bytes": png, "media_type": "image/png"}}}],
    }


def _refusal(text: str = "Rendering unavailable (no OpenGL context). Install EGL or OSMesa") -> dict:
    """What :meth:`render` returns when ``_get_renderer`` yields no renderer."""
    return {"status": "error", "content": [{"text": text}]}


def _record(sim, tmp_path, per_camera: dict[str, dict], cameras: list[str], caplog):
    """Run one real recording whose renders are dictated by ``per_camera``.

    Returns ``(warnings, stop_result)``. ``caplog`` is cleared after
    ``start_cameras_recording`` returns so the unrelated "not ready" notice
    cannot be mistaken for a warmup verdict.
    """

    def stub_render(camera_name, width=None, height=None, **kwargs):
        return per_camera[camera_name]

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        sim.render = stub_render  # type: ignore[method-assign]
        started = sim.start_cameras_recording(cameras=cameras, output_dir=str(tmp_path), fps=10, width=32, height=24)
        assert started["status"] == "success", started
        warnings = [r.getMessage() for r in caplog.records if r.name == _LOGGER]
        stopped = sim.stop_cameras_recording()
    return warnings, stopped


def _artifacts(stopped: dict) -> dict[str, dict]:
    payload = next(b["json"] for b in stopped["content"] if "json" in b)
    return {a["camera"]: a for a in payload["artifacts"]}


class TestARefusalIsNotColdness:
    """The warmup verdict has to name the failure the operator actually has."""

    def test_a_refused_render_is_reported_as_refused_not_as_a_cold_camera(self, sim, tmp_path, caplog):
        """The headline: 30 attempts against a missing GL context are not warmup.

        Reporting this as "cold" tells the operator to wait for something that
        will never settle, and its suggested consequence - a gradient artifact
        on the first frames - understates a recording in which every frame is
        missing.
        """
        warnings, _ = _record(sim, tmp_path, {"cam_a": _refusal()}, ["cam_a"], caplog)

        assert any("REFUSED" in w for w in warnings), (
            f"a refused render was not reported as a refusal; warmup said: {warnings}"
        )
        assert not any("still cold after" in w for w in warnings), (
            f"a refused render was reported as a camera that is merely cold: {warnings}"
        )

    def test_the_refusal_reason_reaches_the_operator(self, sim, tmp_path, caplog):
        """The renderer already said why; the warmup loop has to carry it."""
        warnings, _ = _record(sim, tmp_path, {"cam_a": _refusal()}, ["cam_a"], caplog)

        assert any("no OpenGL context" in w for w in warnings), f"the reason the renderer gave was dropped: {warnings}"

    def test_the_warning_says_the_frames_will_be_empty_not_merely_degraded(self, sim, tmp_path, caplog):
        warnings, _ = _record(sim, tmp_path, {"cam_a": _refusal()}, ["cam_a"], caplog)

        refused = [w for w in warnings if "REFUSED" in w]
        assert refused, warnings
        assert "empty" in refused[0]
        assert "gradient artifact" not in refused[0]

    def test_the_artifact_names_why_the_camera_has_no_frames(self, sim, tmp_path, caplog):
        """``frames: 0`` beside ``status: success`` is not a diagnosis.

        Without the reason the caller has to guess between an empty scene, a
        window too short to catch a frame, and a machine that cannot render at
        all - and only the last one is worth acting on.
        """
        _, stopped = _record(sim, tmp_path, {"cam_a": _refusal()}, ["cam_a"], caplog)

        artifact = _artifacts(stopped)["cam_a"]
        assert artifact["frames"] == 0, artifact
        assert "no OpenGL context" in artifact.get("render_refused", ""), artifact
        text = next(b["text"] for b in stopped["content"] if "text" in b)
        assert "rendering refused" in text

    def test_a_mixed_batch_separates_the_refused_from_the_cold(self, sim, tmp_path, caplog):
        """Two cameras, two different failures, two different verdicts.

        A single "cold" line covering both would hide the one that cannot be
        waited out behind the one that can.
        """
        warnings, stopped = _record(
            sim,
            tmp_path,
            {"cam_a": _refusal(), "cam_b": _frame(_COLD_PNG)},
            ["cam_a", "cam_b"],
            caplog,
        )

        refused = [w for w in warnings if "REFUSED" in w]
        cold = [w for w in warnings if "still cold after" in w]
        assert len(refused) == 1, warnings
        assert len(cold) == 1, warnings
        assert "cam_a" in refused[0] and "cam_b" not in refused[0]
        assert "cam_b" in cold[0] and "cam_a" not in cold[0]
        # Only the refused camera carries a reason; the cold one has none to give.
        arts = _artifacts(stopped)
        assert "render_refused" in arts["cam_a"]
        assert "render_refused" not in arts["cam_b"]

    def test_an_error_result_with_no_readable_text_still_records_a_placeholder(self, sim, tmp_path, caplog):
        """A refusal whose envelope carries no text must not lose the fact of it.

        The reason is read out of a caller-shaped dict, so the read can fail on
        a missing key, an empty content list or a non-subscriptable block. That
        must not cost the recorder thread, and it must not silently downgrade
        the camera back to "cold".
        """
        warnings, stopped = _record(sim, tmp_path, {"cam_a": {"status": "error", "content": []}}, ["cam_a"], caplog)

        assert any("REFUSED" in w for w in warnings), warnings
        assert not any("still cold after" in w for w in warnings), warnings
        assert _artifacts(stopped)["cam_a"]["render_refused"]


class TestTheColdPathIsUnchanged:
    """Cold is still a real state, and it still reads the way it always did."""

    def test_a_genuinely_cold_camera_is_still_reported_as_cold(self, sim, tmp_path, caplog):
        """A frame that arrives but carries no geometry is the original case.

        This is what fails if the two paths are ever collapsed into one: a
        camera that really is warming up must not be told its renders were
        refused, because waiting is exactly the right thing for it.
        """
        warnings, stopped = _record(sim, tmp_path, {"cam_a": _frame(_COLD_PNG)}, ["cam_a"], caplog)

        assert any("still cold after" in w for w in warnings), warnings
        assert any("gradient artifact" in w for w in warnings), warnings
        assert not any("REFUSED" in w for w in warnings), warnings
        assert "render_refused" not in _artifacts(stopped)["cam_a"]

    def test_a_camera_that_warms_reports_neither(self, sim, tmp_path, caplog):
        warnings, stopped = _record(sim, tmp_path, {"cam_a": _frame(_WARM_PNG)}, ["cam_a"], caplog)

        assert not any("still cold after" in w for w in warnings), warnings
        assert not any("REFUSED" in w for w in warnings), warnings
        assert "render_refused" not in _artifacts(stopped)["cam_a"]

    def test_the_synchronous_path_flushes_without_a_refusals_key(self, sim, tmp_path):
        """Only the daemon recorder warms up, so only it can record a refusal.

        ``start_cameras_recording_synchronous`` renders on the caller's thread
        and has no warmup loop, so its state never carries the key at all - the
        shared flush has to read it as absent rather than assume it is there.
        """
        sim.render = lambda camera_name, width=None, height=None, **kw: _frame(_WARM_PNG)  # type: ignore[method-assign]
        started = sim.start_cameras_recording_synchronous(
            cameras=["cam_a"], output_dir=str(tmp_path), fps=10, width=32, height=24
        )
        assert started["status"] == "success", started
        handles = next(b["json"] for b in started["content"] if "json" in b)
        handles["on_frame"](0, {}, {})
        stopped = handles["finalize"]()

        assert stopped["status"] == "success", stopped
        assert "refusals" not in (sim._cams_rec_state or {})
        assert "render_refused" not in _artifacts(stopped)["cam_a"]
