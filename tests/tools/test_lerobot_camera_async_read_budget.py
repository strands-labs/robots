"""The caller's asynchronous read budget must reach the camera driver.

``lerobot_camera`` documents ``timeout_ms`` as the budget for its asynchronous
reads, and every handler that selects that read path hands the caller's value to
``async_read`` - except ``_record_video_sequence``, which the dispatch never
passed the value at all and which read with a fixed ``1000`` instead. The
parameter was the only one of its kind missing from that handler's signature,
which is also why ``record`` was left out of the guard's ``timeout_ms`` rows: an
option no handler consumes must not be refused, so the drop propagated into the
validation table as intent.

Measured on a real UVC camera at 640x480, ``action="record"`` with ``timeout_ms``
in ``{default, 1, -5, 9000}`` produced byte-identical output every time - 12
frames, 5971 bytes - so the option had no effect at any value, and ``-5`` was not
refused either. The sibling ``action="capture"`` honored ``timeout_ms=1`` on the
same device ("Timed out waiting for frame ... after 1 ms") and refused ``-5``.

The consequence is a budget the caller cannot set. A camera that needs longer
than a second for a frame - a high resolution, a contended USB bus, a sensor
still warming - aborts the recording at 1000 ms and reports a timeout against a
budget nobody chose, while raising ``timeout_ms`` to cover it changes nothing. A
caller who instead needs a tighter bound gets a loop that blocks for a second per
frame.

These tests pin that the budget reaches the driver on every asynchronous action,
that the recording completes once the budget covers the camera, that ``record``
now validates the option it reads, and that a handler cannot select the
asynchronous read again without taking a budget.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

import numpy as np
import pytest

# The module object itself is needed - for the ``cv2``/``os`` handles the tool
# module imports and for the signature guard's ``vars()`` scan - so every name in
# it, the tool included, is reached through this one alias.
import strands_robots.tools.lerobot_camera as cam_mod
from strands_robots.utils import positive_finite_number_error

# Actions whose handler selects the asynchronous read, and therefore consumes the
# caller's budget.
ASYNC_ACTIONS = ("capture", "capture_batch", "record", "preview", "test")

# One value per rejection reason of the shared span domain.
BAD_BUDGETS = (0.0, -3.0, float("nan"), float("inf"), True, "1", None, [1.0])


class _BudgetCamera:
    """A camera stand-in that records every read budget it is handed.

    ``needs_ms`` models a camera slower than the budget: a read whose budget does
    not cover it raises the driver's own ``TimeoutError``, naming the budget, the
    way :meth:`lerobot.cameras.opencv.OpenCVCamera.async_read` does.
    """

    def __init__(self, needs_ms: float = 0.0) -> None:
        self.budgets: list[Any] = []
        self.sync_reads = 0
        self.needs_ms = needs_ms
        self.width = 8
        self.height = 6
        self.fps = 30
        self.color_mode = type("_M", (), {"value": "RGB"})()
        self.rotation: Any = None

    def connect(self, warmup: bool = True) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def _frame(self) -> np.ndarray:
        # A measurable span keeps the tool's own rate arithmetic off zero.
        time.sleep(0.0005)
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def read(self) -> np.ndarray:
        self.sync_reads += 1
        return self._frame()

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        self.budgets.append(timeout_ms)
        if self.needs_ms > float(timeout_ms):
            raise TimeoutError(f"Timed out waiting for frame from camera after {timeout_ms} ms.")
        return self._frame()


@pytest.fixture
def camera(monkeypatch: pytest.MonkeyPatch) -> _BudgetCamera:
    """Substitute the camera factory and neutralise every sink a handler writes to."""
    cam = _BudgetCamera()

    monkeypatch.setattr(cam_mod, "_create_camera", lambda *a, **k: cam)
    writer = type("_W", (), {"write": lambda self, f: None, "release": lambda self: None})()
    monkeypatch.setattr(cam_mod.cv2, "VideoWriter", lambda *a, **k: writer)
    monkeypatch.setattr(cam_mod.cv2, "VideoWriter_fourcc", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(cam_mod.cv2, "imwrite", lambda *a, **k: True)
    monkeypatch.setattr(cam_mod.cv2, "imshow", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(cam_mod.cv2, "waitKey", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(cam_mod.cv2, "destroyAllWindows", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(cam_mod.os.path, "getsize", lambda p: 1234)
    return cam


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool with agent-shaped keyword values.

    The tool annotates ``timeout_ms`` as a ``float``, but an agent supplies JSON
    and several values under test here are deliberately outside that annotation -
    a string budget, a ``None`` budget. Funnelling every call through one
    ``**kwargs: Any`` helper states that once rather than scattering a
    suppression over every call site.
    """
    return cam_mod.lerobot_camera(**kwargs)


def _action_kwargs(action: str, tmp_path: Any) -> dict[str, Any]:
    """The smallest set of options that drives ``action`` through one read."""
    common: dict[str, Any] = {
        "action": action,
        "camera_type": "opencv",
        "save_path": str(tmp_path),
        "width": 8,
        "height": 6,
        "fps": 2,
        "async_mode": True,
    }
    if action == "capture_batch":
        common["camera_ids"] = [0]
    else:
        common["camera_id"] = 0
    if action == "record":
        common["capture_duration"] = 0.5
    if action == "preview":
        common["preview_duration"] = 0.01
    return common


class TestTheBudgetReachesTheDriver:
    """The option is documented as the asynchronous read's budget on every action."""

    @pytest.mark.parametrize("action", ASYNC_ACTIONS)
    def test_every_asynchronous_action_hands_the_callers_budget_to_the_read(
        self, camera: _BudgetCamera, tmp_path: Any, action: str
    ) -> None:
        result = _call(timeout_ms=137.0, **_action_kwargs(action, tmp_path))

        assert result["status"] == "success"
        assert camera.budgets, f"{action} performed no asynchronous read"
        assert set(camera.budgets) == {137.0}

    def test_the_default_budget_still_reaches_the_read_unchanged(self, camera: _BudgetCamera, tmp_path: Any) -> None:
        """A caller who sets nothing keeps the budget the recording always used."""
        result = _call(**_action_kwargs("record", tmp_path))

        assert result["status"] == "success"
        assert set(camera.budgets) == {1000.0}

    def test_a_synchronous_recording_is_handed_no_budget(self, camera: _BudgetCamera, tmp_path: Any) -> None:
        kwargs = _action_kwargs("record", tmp_path) | {"async_mode": False}

        result = _call(timeout_ms=137.0, **kwargs)

        assert result["status"] == "success"
        assert camera.budgets == []
        assert camera.sync_reads > 0


class TestABudgetTheCameraExceedsIsTheCallersToRaise:
    """The failure must name the budget the caller set, and raising it must help."""

    def test_a_camera_slower_than_the_budget_reports_the_budget_the_caller_set(
        self, camera: _BudgetCamera, tmp_path: Any
    ) -> None:
        camera.needs_ms = 1500.0

        result = _call(timeout_ms=250.0, **_action_kwargs("record", tmp_path))

        assert result["status"] == "error"
        assert "250.0 ms" in _text(result)

    def test_raising_the_budget_over_the_camera_completes_the_same_recording(
        self, camera: _BudgetCamera, tmp_path: Any
    ) -> None:
        """The remedy the option exists for: a camera needing 1.5 s per frame."""
        camera.needs_ms = 1500.0

        result = _call(timeout_ms=3000.0, **_action_kwargs("record", tmp_path))

        assert result["status"] == "success"
        assert set(camera.budgets) == {3000.0}


class TestRecordValidatesTheBudgetItReads:
    """An option a handler consumes has to be usable before the camera opens."""

    @pytest.mark.parametrize("value", BAD_BUDGETS)
    def test_an_unusable_budget_is_refused_before_the_camera_opens(
        self, camera: _BudgetCamera, tmp_path: Any, value: Any
    ) -> None:
        result = _call(timeout_ms=value, **_action_kwargs("record", tmp_path))

        assert result["status"] == "error"
        assert _text(result).startswith("record:")
        assert "timeout_ms" in _text(result)
        assert camera.budgets == []

    @pytest.mark.parametrize("value", BAD_BUDGETS + (1.0, 250.0, 1000, np.float32(1.5)))
    def test_the_budget_domain_matches_the_shared_helper(
        self, camera: _BudgetCamera, tmp_path: Any, value: Any
    ) -> None:
        refused = _call(timeout_ms=value, **_action_kwargs("record", tmp_path))["status"] == "error"

        assert refused == (positive_finite_number_error(value, "timeout_ms", "record") is not None)


class TestTheBudgetCannotBeDroppedAgain:
    """Structural pins: selecting the asynchronous read implies taking a budget."""

    def test_every_handler_that_selects_the_asynchronous_read_takes_a_budget(self) -> None:
        def params(obj: Any) -> Any:
            return inspect.signature(obj).parameters

        handlers = [obj for obj in vars(cam_mod).values() if inspect.isfunction(obj)]
        selects = {obj.__name__ for obj in handlers if "async_mode" in params(obj)}
        takes = {obj.__name__ for obj in handlers if "timeout_ms" in params(obj)}

        assert selects == takes, f"selects the asynchronous read without a budget: {selects - takes}"

    def test_every_asynchronous_action_carries_the_guard_row(self) -> None:
        validated = {action for action, options in cam_mod._ACTION_NUMERIC_OPTIONS.items() if "timeout_ms" in options}

        assert validated == set(ASYNC_ACTIONS)
