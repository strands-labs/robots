"""The earthrover verbs: one accessor table, one handle judgement, thin pass-through.

Every cell runs against a recording driver double. The double matters for the
same reason the driver's own test double does: ``rover_lamp`` claims the lamp
rides a zero twist and ``rover_move`` claims a timed hold ends in a stop -
claims about what was *sent*, which only a double that records can grade.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, cast

import pytest

from strands_robots.tools.earthrover import (
    _VERBS,
    MAX_MOVE_DURATION_S,
    rover_camera,
    rover_lamp,
    rover_move,
    rover_speak,
    rover_state,
    rover_stop,
)

_ALL_VERBS: dict[str, Any] = {
    "rover_move": rover_move,
    "rover_stop": rover_stop,
    "rover_lamp": rover_lamp,
    "rover_state": rover_state,
    "rover_camera": rover_camera,
    "rover_speak": rover_speak,
}

_OK = {"status": "success", "content": [{"json": {"commanded": {"linear": 0.0, "angular": 0.0}}}]}


class _Driver:
    """Records every call; answers from per-method envelopes."""

    def __init__(self, **answers: Any) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._answers = answers

    def _answer(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((name, args, kwargs))
        return self._answers.get(name, dict(_OK))

    def move(self, *a: Any, **k: Any) -> Any:
        return self._answer("move", *a, **k)

    def stop_task(self, *a: Any, **k: Any) -> Any:
        return self._answer("stop_task", *a, **k)

    def send_action(self, *a: Any, **k: Any) -> Any:
        return self._answer("send_action", *a, **k)

    def read_state(self, *a: Any, **k: Any) -> Any:
        return self._answer("read_state", *a, **k)

    def capture_frame(self, *a: Any, **k: Any) -> Any:
        return self._answer("capture_frame", *a, **k)

    def speak(self, *a: Any, **k: Any) -> Any:
        return self._answer("speak", *a, **k)


_STATE = {
    "battery": 87,
    "signal_level": 3,
    "orientation": 128,
    "lamp": 0,
    "speed": 0,
    "gps_signal": 31,
    "latitude": 41.08,
    "longitude": 29.01,
}


def _args(verb: str, driver: Any) -> dict[str, Any]:
    """The minimal valid call for each verb, so one table grades them all."""
    extras: dict[str, dict[str, Any]] = {"rover_speak": {"text": "hi"}}
    return {"driver": driver, **extras.get(verb, {})}


class TestTheHandleJudgementCoversEveryVerb:
    """The four handle facts, graded across the whole table."""

    @pytest.mark.parametrize("verb", sorted(_VERBS))
    def test_none_is_refused_naming_the_parameter(self, verb: str) -> None:
        result = _ALL_VERBS[verb](**_args(verb, None))
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith(f"{verb}: `driver` is required")

    @pytest.mark.parametrize("verb", sorted(_VERBS))
    def test_a_wrong_handle_is_refused_naming_its_type(self, verb: str) -> None:
        result = _ALL_VERBS[verb](**_args(verb, "earthrover"))
        assert result["status"] == "error"
        assert "'str' does not expose" in result["content"][0]["text"]

    @pytest.mark.parametrize("verb", sorted(_VERBS))
    def test_a_data_shaped_fake_is_refused_not_called(self, verb: str) -> None:
        fake = type("Cache", (), {_VERBS[verb][0]: "not-callable"})()
        assert _ALL_VERBS[verb](**_args(verb, fake))["status"] == "error"

    @pytest.mark.parametrize("verb", sorted(_VERBS))
    def test_the_table_and_the_lazy_surface_agree(self, verb: str) -> None:
        import strands_robots.tools as tools

        assert getattr(tools, verb).tool_name == verb  # registered and importable


class TestTheWriteVerbsRecordWhatTheySent:
    def test_move_passes_the_twist_through(self) -> None:
        driver = _Driver()
        assert rover_move(driver=driver, linear=0.4, angular=-0.2)["status"] == "success"
        assert driver.calls == [("move", (0.4, -0.2), {})]

    def test_a_timed_move_ends_in_a_stop_and_reports_both(self) -> None:
        driver = _Driver()
        result = rover_move(driver=driver, linear=0.3, duration_s=0.01)
        assert [name for name, _, _ in driver.calls] == ["move", "stop_task"]
        payload = result["content"][0]["json"]
        assert payload["stopped"] is True and payload["held_s"] == 0.01

    def test_a_failed_trailing_stop_is_an_error_not_a_success(self) -> None:
        driver = _Driver(stop_task={"status": "error", "content": [{"text": "gone"}]})
        result = rover_move(driver=driver, linear=0.3, duration_s=0.01)
        assert result["status"] == "error"
        assert "still be rolling" in result["content"][0]["text"]

    def test_a_refused_move_skips_the_hold_and_the_stop(self) -> None:
        driver = _Driver(move={"status": "error", "content": [{"text": "not connected"}]})
        assert rover_move(driver=driver, linear=0.3, duration_s=5.0)["status"] == "error"
        assert [name for name, _, _ in driver.calls] == ["move"]  # no sleep, no stop

    @pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf"), MAX_MOVE_DURATION_S + 1], ids=repr)
    def test_an_unholdable_duration_is_refused_before_the_wire(self, duration: float) -> None:
        driver = _Driver()
        assert rover_move(driver=driver, linear=0.3, duration_s=duration)["status"] == "error"
        assert driver.calls == []

    def test_stop_is_the_drivers_stop_task_verbatim(self) -> None:
        driver = _Driver()
        assert rover_stop(driver=driver) == _OK
        assert driver.calls == [("stop_task", (), {})]

    @pytest.mark.parametrize("on", [True, False], ids=["on", "off"])
    def test_the_lamp_rides_a_zero_twist(self, on: bool) -> None:
        driver = _Driver()
        assert rover_lamp(driver=driver, on=on)["status"] == "success"
        assert driver.calls == [("send_action", ({"linear": 0.0, "angular": 0.0, "lamp": on},), {})]

    def test_a_non_boolean_lamp_is_refused(self) -> None:
        driver = _Driver()
        assert rover_lamp(driver=driver, on=cast(bool, "bright"))["status"] == "error"
        assert driver.calls == []

    def test_speak_passes_the_text_and_the_refusal_back(self) -> None:
        driver = _Driver(speak={"status": "error", "content": [{"text": "speak: text must be"}]})
        assert rover_speak(driver=driver, text="")["status"] == "error"
        assert driver.calls == [("speak", ("",), {})]


class TestTheReadVerbsAnswerHonestly:
    def test_state_summarises_and_carries_the_full_snapshot(self) -> None:
        driver = _Driver(read_state=dict(_STATE))
        result = rover_state(driver=driver)
        summary = result["content"][0]["text"]
        assert "battery 87%" in summary and "41.08" in summary
        assert result["content"][1]["json"] == _STATE

    def test_a_missing_gps_fix_is_said_not_fabricated(self) -> None:
        driver = _Driver(read_state={**_STATE, "gps_signal": 0})
        assert "GPS no fix" in rover_state(driver=driver)["content"][0]["text"]

    def test_an_empty_cache_is_a_refusal_with_the_remedy(self) -> None:
        result = rover_state(driver=_Driver(read_state={}))
        assert result["status"] == "error"
        assert "earth-rovers-sdk" in result["content"][0]["text"]

    def test_a_frame_becomes_a_visible_image_block(self) -> None:
        import base64

        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4
        driver = _Driver(
            capture_frame={
                "status": "success",
                "content": [{"json": {"camera": "front", "format": "png", "b64": base64.b64encode(raw).decode()}}],
            }
        )
        result = rover_camera(driver=driver)
        image = result["content"][1]["image"]
        assert image["format"] == "png" and image["source"]["bytes"] == raw

    def test_a_camera_refusal_passes_through_verbatim(self) -> None:
        refusal = {"status": "error", "content": [{"text": "video session is not up"}]}
        driver = _Driver(capture_frame=refusal)
        assert rover_camera(driver=driver, camera="rear") == refusal
        assert driver.calls == [("capture_frame", ("rear",), {})]


class TestTheVerbsImportNoDriver:
    """The handle contract is the ``_VERBS`` accessor names, not an import.

    The module types ``driver`` as ``Any`` and names ``EarthRoverDriver``
    literally rather than through a ``:class:`` role, so nothing here promises
    the reader a resolvable path into ``strands_robots.drivers``. Pinning that
    keeps the claim load-bearing in both directions: an import added for a type
    annotation the tool schema never sees would couple this surface - and every
    agent that loads a rover verb - to a driver module and to the extras that
    driver needs, and it would put the docstring back into the shape whose
    pointer only resolves while that module happens to sit in the tree.
    """

    def test_a_clean_interpreter_loads_the_verbs_and_pulls_no_driver(self) -> None:
        code = (
            "import sys, strands_robots.tools.earthrover as m; "
            "assert callable(m.rover_move), 'the verbs did not load'; "
            "leaked = sorted(n for n in sys.modules if n.startswith('strands_robots.drivers')); "
            "assert not leaked, f'importing a rover verb pulled {leaked}'"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
