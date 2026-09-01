"""A camera-recorder stop that cannot join its thread says so, and encodes nothing.

``stop_cameras_recording`` asks the daemon recorder loop to exit, waits
:data:`~strands_robots.simulation.mujoco.rendering._CAMS_REC_JOIN_TIMEOUT_S` for
it, and then has to decide what to report. ``Thread.join`` returns ``None``
whether or not the thread finished, so the liveness read after it is the only
thing that distinguishes a stopped recorder from one still inside ``render`` -
and that one reading decides three separate answers: the envelope's verdict,
whether the buffers may be encoded, and whether the recording stays registered.

The window is ordinary rather than exotic: ``render`` on a wedged GL context, an
EGL device that stops answering, or a frame large enough that one encode outlasts
the budget. What made it worth pinning is that the *consequences* of getting the
verdict wrong all read as success - an MP4 encoded from a frame list that is
still growing, a status verb answering ``[idle]`` about a live thread, and a
second recorder started on the same cameras because the guard that should have
refused it reads a flag the first loop already outlived.

The refusal it settles on promises the buffered frames are recoverable through a
second stop, which makes every read *between* the two calls part of the same
defect: the state decays -- the slow ``render`` returns, the loop exits -- into
one that no liveness read can see but that still holds those frames, so a guard
keyed on liveness would let a `start` drop them under ``status="success"``.

Each cell drives a real daemon thread and wedges it inside ``render``, mirroring
``test_daemon_camera_recording.py``'s fake-render harness so no GL context is
needed. The join budget is monkeypatched down so a cell costs a fraction of a
second rather than the production wait.
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed - pip install strands-robots[sim-mujoco]")
imageio = pytest.importorskip("imageio", reason="imageio not installed - pip install imageio imageio-ffmpeg")

from strands_robots.simulation import Simulation  # noqa: E402
from strands_robots.simulation.mujoco import rendering  # noqa: E402

# Short enough that a cell's failed join is not felt, long enough that the
# capture loop reaches the wedge on a loaded box.
_TEST_JOIN_BUDGET_S = 0.5


def _render_result(width: int, height: int) -> dict[str, Any]:
    """A ``render()`` envelope carrying a real PNG the recorder can decode.

    A gradient rather than a flat fill: the recorder's warmup discards frames
    whose per-column standard deviation reads as a cold GL gradient.
    """
    from PIL import Image

    row = np.linspace(0, 255, width, dtype=np.uint8)
    arr = np.repeat(row[None, :], height, axis=0)
    arr = np.stack([arr, arr[::-1], arr], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return {
        "status": "success",
        "content": [
            {"text": f"{width}x{height}"},
            {"image": {"format": "png", "source": {"bytes": buf.getvalue()}}},
        ],
    }


class _WedgeableRecorder:
    """A sim whose recorder thread can be parked inside ``render`` on demand.

    ``arm()`` makes the next ``render`` block until ``release()``, so a test can
    let real frames buffer first and only then produce the state the join budget
    cannot clear. Without the two-phase shape the wedge lands during the warmup
    loop instead, and the buffers a stop would have flushed are empty - which is
    a different situation with the same symptom.
    """

    def __init__(self) -> None:
        self.sim = Simulation()
        self.sim.create_world()
        self.sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
        self.sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
        self._armed = threading.Event()
        self._released = threading.Event()
        self.wedged = threading.Event()
        self.sim.render = self._render  # type: ignore[assignment,method-assign]

    def _render(self, camera_name: str, width: int | None = None, height: int | None = None, **_kw: Any) -> dict:
        if self._armed.is_set():
            self.wedged.set()
            self._released.wait(timeout=30)
        return _render_result(width or 32, height or 24)

    def buffered(self, camera: str = "cam_a") -> int:
        state = self.sim._cams_rec_state
        return 0 if state is None else len(state["buffers"][camera])

    def wedge(self) -> threading.Thread:
        """Buffer a few real frames, then park the loop inside ``render``."""
        deadline = time.monotonic() + 30.0
        while self.buffered() < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert self.buffered() >= 3, "premise: the recorder buffers frames before it is wedged"
        thread = self.sim._cams_rec_state["thread"]
        self._armed.set()
        assert self.wedged.wait(timeout=30), "premise: the recorder thread reaches the blocking render"
        return thread

    def release(self) -> None:
        self._armed.clear()
        self._released.set()

    def settle(self) -> int:
        """Let the wedged ``render`` return so the loop exits, and count what it left.

        The state a failed stop decays into on the ordinary recovery path: the
        budget is sized for a slow render, i.e. one that does return, just late.
        ``running`` is already clear, so the loop appends the frame that render
        was holding and then leaves -- and the registration outlives it, because
        only a flush deregisters.
        """
        thread = self.sim._cams_rec_state["thread"]
        self.release()
        thread.join(timeout=30)
        assert not thread.is_alive(), "premise: the released render lets the loop exit"
        buffered = self.buffered()
        assert buffered >= 3, "premise: the settled state still holds the captured frames"
        return buffered


@pytest.fixture
def recorder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A started daemon recording whose thread is released at teardown."""
    monkeypatch.setattr(rendering, "_CAMS_REC_JOIN_TIMEOUT_S", _TEST_JOIN_BUDGET_S)
    rec = _WedgeableRecorder()
    started = rec.sim.start_cameras_recording(cameras=["cam_a"], output_dir=str(tmp_path), fps=30, name="wedge")
    assert started["status"] == "success", started
    try:
        yield rec
    finally:
        rec.release()
        thread = (rec.sim._cams_rec_state or {}).get("thread")
        if thread is not None:
            thread.join(timeout=10)


def _json_block(envelope: dict) -> dict:
    """The single JSON payload of a tool envelope."""
    blocks = [c["json"] for c in envelope["content"] if "json" in c]
    assert len(blocks) == 1, envelope
    return blocks[0]


class TestAStopWhoseJoinExpires:
    """The recorder thread outlives the budget: one reading, four consequences."""

    def test_the_stop_refuses_instead_of_reporting_a_stop_that_did_not_happen(self, recorder) -> None:
        """A caller cannot read success while the loop is still capturing."""
        thread = recorder.wedge()

        result = recorder.sim.stop_cameras_recording()

        assert thread.is_alive(), "premise: the wedged thread outlives the join budget"
        assert result["status"] == "error", result
        payload = _json_block(result)
        assert payload["stopped"] is False
        assert payload["recording"] == "wedge"
        # The frames it did capture are reported, so the refusal says what is
        # pending rather than only that something went wrong.
        assert payload["buffered_frames"]["cam_a"] >= 3
        text = result["content"][0]["text"]
        assert f"{_TEST_JOIN_BUDGET_S:.1f}s" in text, text
        assert "cam_a" in text

    def test_no_mp4_is_encoded_from_a_buffer_the_loop_is_still_appending_to(self, recorder, tmp_path) -> None:
        """The flush reads each buffer twice; a live appender makes the two disagree."""
        recorder.wedge()

        recorder.sim.stop_cameras_recording()

        assert list(tmp_path.glob("*.mp4")) == []

    def test_the_recording_stays_registered_so_a_later_call_re_joins_and_flushes_it(self, recorder, tmp_path) -> None:
        """The unflushed buffers are recoverable: the second stop writes the MP4."""
        recorder.wedge()
        first = recorder.sim.stop_cameras_recording()
        assert first["status"] == "error"
        assert recorder.sim._cams_rec_state is not None, "the handle is the only route back to the loop"

        recorder.release()
        second = recorder.sim.stop_cameras_recording()

        assert second["status"] == "success", second
        artifact = _json_block(second)["artifacts"][0]
        assert artifact["frames"] >= 3
        mp4 = Path(artifact["path"])
        assert mp4.exists()
        # Round-trip: what was written is a readable clip, not a sealed stub.
        with imageio.v2.get_reader(str(mp4)) as reader:
            assert sum(1 for _ in reader) == artifact["frames"]
        assert recorder.sim._cams_rec_state is None, "a real stop deregisters the recording"

    def test_the_status_verb_reports_the_loop_that_has_not_exited(self, recorder) -> None:
        """``running`` and ``thread_alive`` differ exactly across this window."""
        recorder.wedge()
        recorder.sim.stop_cameras_recording()

        status = recorder.sim.get_cameras_recording_status()

        assert status["status"] == "success"
        text = status["content"][0]["text"]
        assert text.startswith("[stopping]"), text
        assert "[idle]" not in text
        payload = _json_block(status)
        assert payload["running"] is False
        assert payload["thread_alive"] is True

    def test_a_second_recorder_cannot_start_beside_the_live_one(self, recorder, tmp_path) -> None:
        """Two threads on one camera set is what the start guard exists to refuse."""
        recorder.wedge()
        recorder.sim.stop_cameras_recording()

        again = recorder.sim.start_cameras_recording(cameras=["cam_a"], output_dir=str(tmp_path), fps=30, name="second")

        assert again["status"] == "error", again
        assert "Already recording 'wedge'" in again["content"][0]["text"]

    def test_the_synchronous_variant_is_refused_for_the_same_reason(self, recorder, tmp_path) -> None:
        """Both start verbs share one registration, so both read the same liveness."""
        recorder.wedge()
        recorder.sim.stop_cameras_recording()

        again = recorder.sim.start_cameras_recording_synchronous(
            cameras=["cam_a"], output_dir=str(tmp_path), fps=30, name="second"
        )

        assert again["status"] == "error", again
        assert "Already recording 'wedge'" in again["content"][0]["text"]


class TestASettledRecordingIsNotDroppable:
    """The failed stop decays: thread gone, buffers still registered and unencoded.

    This is where the recovery contract is kept or broken. The refusal tells the
    caller the frames are recoverable via a second stop, so nothing between the
    two calls may discard them -- and a guard that reads liveness stops seeing
    this state the moment the slow ``render`` returns, which is precisely when
    the caller is most likely to retry. Reading the registration instead makes
    the same caller sequence answer the same way whether or not the wedge
    cleared in between.
    """

    def test_a_start_cannot_discard_the_frames_the_failed_stop_promised_were_recoverable(
        self, recorder, tmp_path
    ) -> None:
        """The frames survive the refused start, so the second stop can still encode them."""
        recorder.wedge()
        assert recorder.sim.stop_cameras_recording()["status"] == "error"
        buffered = recorder.settle()

        again = recorder.sim.start_cameras_recording(cameras=["cam_a"], output_dir=str(tmp_path), fps=30, name="second")

        assert again["status"] == "error", again
        text = again["content"][0]["text"]
        assert "stop_cameras_recording() first" in text, text
        assert str(buffered) in text, text
        assert _json_block(again)["phase"] == "unflushed"
        # The refusal is only worth anything if the buffer it protected is intact.
        assert recorder.sim._cams_rec_state["name"] == "wedge"
        assert recorder.buffered() == buffered

    def test_the_synchronous_start_reads_the_same_registration(self, recorder, tmp_path) -> None:
        """Both start verbs replace one attribute, so one guard answers for both."""
        recorder.wedge()
        assert recorder.sim.stop_cameras_recording()["status"] == "error"
        buffered = recorder.settle()

        again = recorder.sim.start_cameras_recording_synchronous(
            cameras=["cam_a"], output_dir=str(tmp_path), fps=30, name="second"
        )

        assert again["status"] == "error", again
        assert "stop_cameras_recording() first" in again["content"][0]["text"]
        assert recorder.sim._cams_rec_state["name"] == "wedge"
        assert recorder.buffered() == buffered

    def test_the_status_verb_does_not_call_an_unencoded_buffer_idle(self, recorder) -> None:
        """``[idle]`` promises nothing is left to encode; here something is."""
        recorder.wedge()
        recorder.sim.stop_cameras_recording()
        buffered = recorder.settle()

        status = recorder.sim.get_cameras_recording_status()

        text = status["content"][0]["text"]
        assert text.startswith("[unflushed]"), text
        assert "[idle]" not in text
        payload = _json_block(status)
        assert payload["phase"] == "unflushed"
        # Neither boolean is set here, which is why the phase had to be carried:
        # a caller reading only these two cannot tell this from a stale handle.
        assert payload["running"] is False
        assert payload["thread_alive"] is False
        assert payload["frames"]["cam_a"] == buffered

    def test_the_second_stop_still_encodes_the_settled_buffer(self, recorder) -> None:
        """The advertised remedy: a stop on a settled recording joins at once and flushes."""
        recorder.wedge()
        assert recorder.sim.stop_cameras_recording()["status"] == "error"
        buffered = recorder.settle()

        second = recorder.sim.stop_cameras_recording()

        assert second["status"] == "success", second
        artifact = _json_block(second)["artifacts"][0]
        assert artifact["frames"] == buffered
        with imageio.v2.get_reader(str(artifact["path"])) as reader:
            assert sum(1 for _ in reader) == buffered
        assert recorder.sim._cams_rec_state is None


class TestARecorderThatExitsIsUnaffected:
    """Controls: the ordinary paths keep their existing answers."""

    def test_a_joined_recorder_reports_success_and_writes_its_mp4(self, recorder) -> None:
        """The whole point is a verdict, not a refusal - an exiting loop still flushes."""
        deadline = time.monotonic() + 30.0
        while recorder.buffered() < 2 and time.monotonic() < deadline:
            time.sleep(0.02)

        result = recorder.sim.stop_cameras_recording()

        assert result["status"] == "success", result
        artifact = _json_block(result)["artifacts"][0]
        assert artifact["frames"] >= 2
        assert Path(artifact["path"]).exists()
        assert recorder.sim._cams_rec_state is None

    def test_nothing_registered_is_still_the_idempotent_no_op(self, recorder) -> None:
        """A second stop after a real one is a success, not an error."""
        recorder.sim.stop_cameras_recording()

        again = recorder.sim.stop_cameras_recording()

        assert again["status"] == "success"
        assert "Was not recording cameras." in again["content"][0]["text"]

    def test_status_is_idle_once_the_recording_is_deregistered(self, recorder) -> None:
        """``[stopping]`` is a third phase, not a replacement for ``[idle]``."""
        recorder.sim.stop_cameras_recording()

        status = recorder.sim.get_cameras_recording_status()

        assert "[idle]" in status["content"][0]["text"]
