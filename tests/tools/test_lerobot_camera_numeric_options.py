"""The ``lerobot_camera`` tool must refuse a numeric option it cannot honor.

Every numeric option this agent tool exposes is forwarded straight into a camera
configuration, an MP4 container header or a capture loop's bound, and none of them
was checked. The consequences differed per option and none of them was reportable:

* ``capture_duration <= 0`` made the recording loop's bound
  ``int(fps * capture_duration)`` zero, so the loop body never ran. The tool then
  returned ``status="success"`` and a summary whose ``Saved:`` line named a
  258-byte MP4 that no decoder will open - a recording reported as complete that
  contains no video.
* ``capture_duration=nan`` leaked ``cannot convert float NaN to integer`` - an
  ``int()`` internal naming neither the tool, the action nor the parameter - and
  still left that stub file behind.
* ``capture_duration=True`` silently recorded one second, since ``True`` is an
  ``int`` worth 1.
* A ``capture_duration`` that is positive and finite but shorter than one frame
  period - every span below ``0.0333`` at the default ``fps=30`` - made that same
  bound zero, so refusing only the non-positive spans left the reported-complete
  empty recording reachable through the other factor.
* ``width=640.0`` reached ``cv2.VideoWriter`` and died in a raw OpenCV overload
  resolution dump, even though the sibling plain-MP4 recorders honor an integral
  float for exactly this parameter.
* ``fps`` in ``{0, -10, 2.7, nan, inf, True}`` was refused only by the camera
  driver, which compares the requested rate against the rate the attached device
  reports - so the complaint named an ``actual_fps`` and arrived only after the
  device had been opened and reconfigured, making a request that is impossible on
  every camera look like a property of this one.

The tool already validated its ``save_path`` at the boundary, so the numeric
options were the one option class forwarded unchecked.

These tests pin the domain, the per-action scoping (an option an action never
reads must not be refused), the coercion that lets an integral float through, and
parity with the shared helpers so this tool cannot drift from the recorders.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

# The module object itself is needed - for the ``cv2``/``os`` handles the tool
# module imports and for the drift guard's ``vars()`` scan - so every name in it,
# the tool included, is reached through this one alias.
import strands_robots.tools.lerobot_camera as cam_mod
from strands_robots.utils import positive_finite_number_error, positive_whole_number_error

# Actions that open a camera configured with the caller's geometry. Every one of
# them must therefore have its geometry validated.
CAMERA_ACTIONS = ("capture", "capture_batch", "record", "preview", "test", "configure")

# One value per rejection reason, for each domain.
BAD_WHOLE_NUMBERS = (0, -10, 2.7, float("nan"), float("inf"), True, "10", None)
BAD_SPANS = (0.0, -3.0, float("nan"), float("inf"), True, "1", None, [1.0])


class _Recorder:
    """A camera stand-in that records the geometry it was configured with."""

    def __init__(self) -> None:
        self.opened: list[tuple[Any, Any, Any]] = []
        self.width = 8
        self.height = 6
        self.fps = 30
        self.color_mode = type("_M", (), {"value": "RGB"})()
        self.rotation: Any = None

    def connect(self, warmup: bool = True) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def read(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Substitute the camera factory and record every geometry it is handed."""
    cam = _Recorder()

    def _create(camera_type: str, camera_id: Any, width: Any, height: Any, fps: Any, *rest: Any) -> _Recorder:
        cam.opened.append((width, height, fps))
        return cam

    monkeypatch.setattr(cam_mod, "_create_camera", _create)
    return cam


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool with agent-shaped keyword values.

    The tool annotates these options ``int`` / ``float``, but an agent supplies
    JSON, and the values under test here are deliberately outside those
    annotations - a string rate, a ``None`` span, an integral-float geometry.
    Funnelling every call through one ``**kwargs: Any`` helper states that once,
    rather than scattering a suppression over every call site.
    """
    return cam_mod.lerobot_camera(**kwargs)


#: ``(fps, capture_duration)`` pairs where each value is individually usable but
#: ``int(fps * capture_duration)`` is zero, so the loop body never runs.
EMPTY_RECORDINGS = ((30, 0.02), (30, 1.0 / 60), (10, 0.09), (60, 0.016), (1, 0.9))

#: The same product at its boundary: the shortest span each rate can honor, which
#: must still record. These are what keep the pairing from over-refusing.
ONE_FRAME_RECORDINGS = ((10, 0.1), (2, 0.5), (30, 1.0 / 30), (1, 1.0))


class TestARecordingThatCannotHappenIsRefused:
    """The headline: a span that captures no frame reported a complete recording.

    The span's own sign is only half of it. The loop bound is the product
    ``int(fps * capture_duration)``, so which side of the line a span falls on is
    not a property of the span - a positive, finite span below one frame period
    produces the identical empty recording, and the pair has to be judged together.
    """

    @pytest.mark.parametrize("duration", BAD_SPANS)
    def test_a_capture_duration_that_records_no_frame_is_refused(
        self, recorder: _Recorder, tmp_path: Any, duration: Any
    ) -> None:
        result = _call(
            action="record",
            camera_id=0,
            save_path=str(tmp_path),
            fps=10,
            capture_duration=duration,
        )

        assert result["status"] == "error"
        assert "capture_duration" in _text(result)
        # The refusal precedes the device, so no camera is opened and - the point
        # of the fix - no file is left behind for the summary to call "Saved".
        assert recorder.opened == []
        assert list(tmp_path.iterdir()) == []

    def test_the_refusal_names_the_action_and_the_option(self, recorder: _Recorder, tmp_path: Any) -> None:
        text = _text(_call(action="record", camera_id=0, save_path=str(tmp_path), fps=10, capture_duration=0.0))

        assert text.startswith("record:")
        assert "capture_duration" in text
        assert all(ord(c) < 128 for c in text), text

    def test_a_usable_span_still_records(
        self, recorder: _Recorder, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = type("_W", (), {"write": lambda self, f: None, "release": lambda self: None})()
        monkeypatch.setattr(cam_mod.cv2, "VideoWriter", lambda *a, **k: writer)
        monkeypatch.setattr(cam_mod.cv2, "VideoWriter_fourcc", lambda *a, **k: 0, raising=False)
        monkeypatch.setattr(cam_mod.os.path, "getsize", lambda p: 1234)

        result = _call(action="record", camera_id=0, save_path=str(tmp_path), fps=2, capture_duration=0.5)

        assert result["status"] == "success"
        assert "Frames: 1" in _text(result)

    @pytest.mark.parametrize(("fps", "duration"), EMPTY_RECORDINGS)
    def test_a_span_shorter_than_one_frame_period_is_refused(
        self, recorder: _Recorder, tmp_path: Any, fps: int, duration: float
    ) -> None:
        """Each value is in its own domain, so only the product can refuse this."""
        assert positive_whole_number_error(fps, "fps", "record") is None
        assert positive_finite_number_error(duration, "capture_duration", "record") is None

        result = _call(action="record", camera_id=0, save_path=str(tmp_path), fps=fps, capture_duration=duration)

        assert result["status"] == "error"
        # The refusal precedes the device, so - the point of the fix - no MP4 stub
        # is left for the summary to call "Saved".
        assert recorder.opened == []
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(("fps", "duration"), ONE_FRAME_RECORDINGS)
    def test_the_shortest_span_a_rate_can_honor_still_records(
        self, recorder: _Recorder, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, fps: int, duration: float
    ) -> None:
        """A frame count of exactly one is honored, not rounded away."""
        writer = type("_W", (), {"write": lambda self, f: None, "release": lambda self: None})()
        monkeypatch.setattr(cam_mod.cv2, "VideoWriter", lambda *a, **k: writer)
        monkeypatch.setattr(cam_mod.cv2, "VideoWriter_fourcc", lambda *a, **k: 0, raising=False)
        monkeypatch.setattr(cam_mod.os.path, "getsize", lambda p: 1234)

        result = _call(action="record", camera_id=0, save_path=str(tmp_path), fps=fps, capture_duration=duration)

        assert result["status"] == "success"
        assert "Frames: 1" in _text(result)

    def test_the_refusal_names_both_factors_and_the_span_that_would_work(
        self, recorder: _Recorder, tmp_path: Any
    ) -> None:
        text = _text(_call(action="record", camera_id=0, save_path=str(tmp_path), fps=30, capture_duration=0.02))

        assert text.startswith("record:")
        assert "capture_duration=0.02" in text
        assert "fps=30" in text
        # A refusal an agent can act on has to quote the span that would work.
        assert "0.0333333" in text
        assert all(ord(c) < 128 for c in text), text

    def test_a_preview_shorter_than_one_frame_period_is_not_refused(
        self, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``preview_duration`` is a deadline, not a factor of a frame count.

        The preview loop is bounded by ``time.monotonic() - start < duration``, so
        its first iteration always runs and a short preview displays a frame rather
        than none. Pairing it with ``fps`` the way ``capture_duration`` is paired
        would refuse a preview that works, so the scoping is pinned here.
        """
        monkeypatch.setattr(cam_mod.cv2, "cvtColor", lambda frame, code: frame, raising=False)
        for name in ("putText", "imshow", "destroyAllWindows"):
            monkeypatch.setattr(cam_mod.cv2, name, lambda *a, **k: None, raising=False)
        monkeypatch.setattr(cam_mod.cv2, "waitKey", lambda delay: 0, raising=False)

        result = _call(action="preview", camera_id=0, fps=30, preview_duration=0.001)

        assert result["status"] == "success", _text(result)
        assert recorder.opened == [(640, 480, 30)]


class TestGeometryIsValidatedOnEveryCameraAction:
    """A frame size or rate no camera can be configured with is refused up front."""

    @pytest.mark.parametrize("action", CAMERA_ACTIONS)
    @pytest.mark.parametrize("param", ("width", "height", "fps"))
    def test_a_geometry_no_camera_can_honor_is_refused(
        self, recorder: _Recorder, tmp_path: Any, action: str, param: str
    ) -> None:
        result = _call(action=action, camera_id=0, camera_ids=[0], save_path=str(tmp_path), **{param: 0})

        assert result["status"] == "error"
        assert param in _text(result)
        assert recorder.opened == []

    @pytest.mark.parametrize("value", BAD_WHOLE_NUMBERS)
    def test_every_unusable_rate_is_refused(self, recorder: _Recorder, tmp_path: Any, value: Any) -> None:
        result = _call(action="record", camera_id=0, save_path=str(tmp_path), fps=value, capture_duration=1.0)

        assert result["status"] == "error"
        assert "fps" in _text(result)
        assert recorder.opened == []

    def test_an_integral_float_geometry_is_honored_as_an_int(
        self, recorder: _Recorder, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``640.0`` is usable, and reaches the camera and the container as ``640``."""
        writer = type("_W", (), {"write": lambda self, f: None, "release": lambda self: None})()
        monkeypatch.setattr(cam_mod.cv2, "VideoWriter", lambda *a, **k: writer)
        monkeypatch.setattr(cam_mod.cv2, "VideoWriter_fourcc", lambda *a, **k: 0, raising=False)
        monkeypatch.setattr(cam_mod.os.path, "getsize", lambda p: 1234)

        result = _call(
            action="record",
            camera_id=0,
            save_path=str(tmp_path),
            width=640.0,
            height=480.0,
            fps=2.0,
            capture_duration=0.5,
        )

        assert result["status"] == "success"
        assert recorder.opened == [(640, 480, 2)]
        assert [type(v) for v in recorder.opened[0]] == [int, int, int]


class TestOnlyTheOptionsAnActionReadsAreValidated:
    """Refusing a value an action never consumes would be a false rejection."""

    @pytest.mark.parametrize("action", ("discover", "list"))
    def test_an_action_that_opens_no_configured_camera_ignores_geometry(self, action: str) -> None:
        result = _call(action=action, camera_type="opencv", width="nonsense", fps=-1)

        assert result["status"] != "error" or "width" not in _text(result)

    def test_a_synchronous_recording_does_not_refuse_a_budget_it_never_reads(
        self, recorder: _Recorder, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The synchronous read takes no timeout, so ``record`` must accept any value here.

        This case reads as it always did, but its reason has changed: the budget
        used to be inert on ``record`` at every value because the handler was
        never passed it. It is now inert only because this call leaves
        ``async_mode`` off, which is the same reason it is inert on every other
        action - and the asynchronous recording refuses the value.
        """
        writer = type("_W", (), {"write": lambda self, f: None, "release": lambda self: None})()
        monkeypatch.setattr(cam_mod.cv2, "VideoWriter", lambda *a, **k: writer)
        monkeypatch.setattr(cam_mod.cv2, "VideoWriter_fourcc", lambda *a, **k: 0, raising=False)
        monkeypatch.setattr(cam_mod.os.path, "getsize", lambda p: 1234)

        result = _call(
            action="record",
            camera_id=0,
            save_path=str(tmp_path),
            fps=2,
            capture_duration=0.5,
            timeout_ms=-1.0,
        )

        assert result["status"] == "success"

    def test_a_timeout_is_effective_only_on_the_asynchronous_read(self, recorder: _Recorder, tmp_path: Any) -> None:
        common = {"action": "capture", "camera_id": 0, "save_path": str(tmp_path), "timeout_ms": -1.0}

        assert _call(async_mode=False, **common)["status"] == "success"

        refused = _call(async_mode=True, **common)
        assert refused["status"] == "error"
        assert "timeout_ms" in _text(refused)


class TestTheDomainCannotDriftFromTheSharedHelpers:
    """One domain per kind, shared with the recorders that write the same MP4."""

    @pytest.mark.parametrize("value", BAD_WHOLE_NUMBERS + BAD_SPANS + (1, 30, 0.5, 2.5, np.int64(64)))
    def test_the_tool_agrees_with_the_shared_domain_for_a_rate(
        self, recorder: _Recorder, tmp_path: Any, value: Any
    ) -> None:
        tool_refuses = (
            _call(action="record", camera_id=0, save_path=str(tmp_path), fps=value, capture_duration=1.0)["status"]
            == "error"
        )

        assert tool_refuses == (positive_whole_number_error(value, "fps", "record") is not None)

    @pytest.mark.parametrize("value", BAD_SPANS + (0.5, 2.5, 30, np.float32(1.5)))
    def test_the_tool_agrees_with_the_shared_domain_for_a_span(
        self, recorder: _Recorder, tmp_path: Any, value: Any
    ) -> None:
        tool_refuses = (
            _call(action="record", camera_id=0, save_path=str(tmp_path), fps=10, capture_duration=value)["status"]
            == "error"
        )

        assert tool_refuses == (positive_finite_number_error(value, "capture_duration", "record") is not None)

    def test_the_span_parity_is_over_the_value_domain_alone(self) -> None:
        """The row above is a claim about one value, not about the whole request.

        Membership in the shared span domain is necessary but not sufficient: the
        loop bound is a product, so a span in the domain is still refused when the
        rate makes it capture no frame. Every value the parity row probes is many
        frames at its ``fps=10``, which is what keeps the two claims compatible -
        pinned here so a later widening of that row cannot quietly assert that the
        shared domain is the whole guard.
        """
        assert all(10 * float(value) >= 1.0 for value in (0.5, 2.5, 30, np.float32(1.5)))
        assert cam_mod._numeric_option_error(
            "record",
            width=640,
            height=480,
            fps=10,
            capture_duration=0.05,
            preview_duration=10.0,
            timeout_ms=1000,
            async_mode=False,
        )

    def test_every_camera_opening_handler_has_a_table_row(self) -> None:
        """A new action configured with the caller's geometry must be validated too."""
        handlers = {
            name
            for name, obj in vars(cam_mod).items()
            if inspect.isfunction(obj)
            and "width" in inspect.signature(obj).parameters
            and name not in ("_create_camera", "_numeric_option_error")
        }

        assert handlers == {
            "_capture_single_image",
            "_capture_batch_images",
            "_record_video_sequence",
            "_preview_camera_live",
            "_test_camera_performance",
            "_configure_camera_settings",
        }
        assert set(cam_mod._ACTION_NUMERIC_OPTIONS) == set(CAMERA_ACTIONS)
        assert len(handlers) == len(cam_mod._ACTION_NUMERIC_OPTIONS)
