"""The EarthRover native driver: the twist wire, the refusals, the parting stop.

Everything here runs with no rover and no SDK process attached. One double
stands in for ``requests`` and it is faithful in the one respect the tests
depend on: it **records** every request rather than discarding it. A double
that dropped the POST could not tell "the driver clamped the twist" from "the
driver sent nothing" - and the parting zero twist in ``cleanup()`` is exactly
one recorded POST, which is the property that stops the wheels.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from strands_robots.drivers import get_native_driver_class
from strands_robots.drivers.base import HardwareDriver, missing_driver_members
from strands_robots.drivers.earthrover import (
    CAMERA_VIEWS,
    DEFAULT_SDK_URL,
    DRIVE_CHANNELS,
    EarthRoverDriver,
    base_url_error,
    detect_image_format,
)

_DATA = {
    "battery": 87,
    "signal_level": 3,
    "orientation": 128,
    "latitude": 41.0,
    "longitude": 29.0,
    "speed": 0,
    "lamp": 0,
}


# --------------------------------------------------------------------------- #
# The requests double.                                                        #
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeSession:
    """Records every request; answers from a mutable route table."""

    def __init__(self) -> None:
        self.gets: list[tuple[str, float]] = []
        self.posts: list[tuple[str, Any, float]] = []
        self.closed = False
        self.routes: dict[str, Any] = {"/data": _FakeResponse(200, dict(_DATA))}
        self.post_response: Any = _FakeResponse(200, {})

    def _resolve(self, url: str) -> _FakeResponse:
        for suffix, answer in self.routes.items():
            if url.endswith(suffix):
                if isinstance(answer, Exception):
                    raise answer
                return answer
        return _FakeResponse(404, None, "no such route")

    def get(self, url: str, timeout: float = 0.0, **_: Any) -> _FakeResponse:
        self.gets.append((url, timeout))
        return self._resolve(url)

    def post(self, url: str, json: Any = None, timeout: float = 0.0, **_: Any) -> _FakeResponse:
        self.posts.append((url, json, timeout))
        if isinstance(self.post_response, Exception):
            raise self.post_response
        return self.post_response

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    """Install a fake ``requests`` whose ``Session()`` is one shared recorder."""
    fake_session = _FakeSession()
    fake = types.ModuleType("requests")
    fake.Session = lambda: fake_session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake_session


def _live_driver(session: _FakeSession, **kwargs: Any) -> EarthRoverDriver:
    driver = EarthRoverDriver(**kwargs)
    assert driver.connect_eagerly() is None
    return driver


# --------------------------------------------------------------------------- #
# Construction.                                                               #
# --------------------------------------------------------------------------- #


class TestConstructionRefusesTheWrongShape:
    @pytest.mark.parametrize(
        ("kwargs", "needle"),
        [
            ({"port": "/dev/ttyUSB0"}, "filesystem path"),
            ({"port": "   "}, "base URL"),
            ({"timeout_s": 0.0}, "timeout_s"),
            ({"timeout_s": float("nan")}, "timeout_s"),
            ({"timeout_s": True}, "timeout_s"),
            ({"turn_sign": 0.5}, "turn_sign"),
            ({"turn_sign": 0.0}, "turn_sign"),
        ],
        ids=[
            "serial-path",
            "blank-url",
            "zero-timeout",
            "nan-timeout",
            "bool-timeout",
            "half-turn-sign",
            "zero-turn-sign",
        ],
    )
    def test_a_wrong_argument_is_refused_by_name(self, kwargs: dict[str, Any], needle: str) -> None:
        with pytest.raises(ValueError, match=needle):
            EarthRoverDriver(**kwargs)

    @pytest.mark.parametrize(
        ("port", "base"),
        [
            (None, DEFAULT_SDK_URL),
            ("http://10.0.0.9:8001/", "http://10.0.0.9:8001"),
            ("10.0.0.9:8001", "http://10.0.0.9:8001"),
            ("https://rover.local:8001", "https://rover.local:8001"),
        ],
        ids=["default", "trailing-slash", "bare-host-port", "https"],
    )
    def test_the_base_url_is_normalised(self, port: str | None, base: str) -> None:
        assert EarthRoverDriver(port=port)._base == base

    def test_the_url_shape_guard_is_reusable(self) -> None:
        assert base_url_error("http://x:1", "port", "t") is None
        assert base_url_error("/tmp/sock", "port", "t") is not None


class TestTheSeamCanBuildIt:
    def test_the_driver_satisfies_the_whole_surface(self) -> None:
        assert missing_driver_members(EarthRoverDriver) == ()
        assert isinstance(EarthRoverDriver(), HardwareDriver)

    def test_the_shipped_registration_names_this_class(self) -> None:
        assert get_native_driver_class("earthrover") is EarthRoverDriver

    def test_the_factory_extras_are_tolerated(self) -> None:
        driver = EarthRoverDriver(cameras={"front": {}}, data_config="x", unused_extra=1)
        assert driver.tool_name == "earthrover"
        assert driver.tool_type == "robot"


# --------------------------------------------------------------------------- #
# Lifecycle.                                                                  #
# --------------------------------------------------------------------------- #


class TestConnectIsProvenNotAssumed:
    def test_success_caches_the_snapshot(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        assert driver.is_connected
        assert driver.read_state()["battery"] == _DATA["battery"]

    def test_connect_is_idempotent(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        assert driver.connect_eagerly() is None
        assert len(session.gets) >= 1

    def test_an_unreachable_sdk_is_a_named_reason(self, session: _FakeSession) -> None:
        session.routes["/data"] = OSError("connection refused")
        driver = EarthRoverDriver()
        reason = driver.connect_eagerly()
        assert reason is not None and "/data" in reason
        assert not driver.is_connected
        assert session.closed  # the half-open session is released, not kept

    def test_an_sdk_without_a_rover_is_a_named_reason(self, session: _FakeSession) -> None:
        session.routes["/data"] = _FakeResponse(200, None)
        reason = EarthRoverDriver().connect_eagerly()
        assert reason is not None and "not connected" in reason

    def test_a_missing_requests_is_a_named_reason_not_an_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "requests", None)
        reason = EarthRoverDriver().connect_eagerly()
        assert reason is not None and "requests" in reason

    def test_the_driver_stays_usable_off_hardware(self, session: _FakeSession) -> None:
        session.routes["/data"] = OSError("down")
        driver = EarthRoverDriver()
        driver.connect_eagerly()
        assert driver.read_state() == {}
        assert driver.get_observation() == {}
        refusal = driver.send_action({"linear": 0.5})
        assert refusal["status"] == "error"
        assert "not connected" in refusal["content"][0]["text"]


class TestCleanupIsAStopFirst:
    def test_cleanup_sends_the_parting_zero_twist_then_closes(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        driver.cleanup()
        url, body, _ = session.posts[-1]
        assert url.endswith("/control")
        assert body == {"command": {"linear": 0.0, "angular": 0.0}}
        assert session.closed
        assert not driver.is_connected  # teardown is not a state a flag can outlive

    def test_a_dead_link_does_not_block_the_close(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        session.post_response = OSError("link dropped")
        driver.cleanup()
        assert session.closed
        assert not driver.is_connected

    def test_cleanup_is_idempotent(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        driver.cleanup()
        driver.cleanup()
        assert not driver.is_connected

    def test_stop_commands_a_zero_twist_and_stays_connected(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        asyncio.run(driver.stop())
        _, body, _ = session.posts[-1]
        assert body == {"command": {"linear": 0.0, "angular": 0.0}}
        assert driver.is_connected


# --------------------------------------------------------------------------- #
# The write path.                                                             #
# --------------------------------------------------------------------------- #


class TestSendActionReachesTheWire:
    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            ({"linear": 0.5, "angular": -0.25}, {"linear": 0.5, "angular": -0.25}),
            ({"linear": 2.0}, {"linear": 1.0, "angular": 0.0}),
            ({"angular": -3.0}, {"linear": 0.0, "angular": -1.0}),
            ({}, {"linear": 0.0, "angular": 0.0}),
            ({"lamp": True}, {"linear": 0.0, "angular": 0.0, "lamp": 1}),
            ({"lamp": 0}, {"linear": 0.0, "angular": 0.0, "lamp": 0}),
        ],
        ids=["plain", "clamp-linear", "clamp-angular", "empty-is-stop", "lamp-on", "lamp-off"],
    )
    def test_the_posted_command_is_the_clamped_twist(
        self, session: _FakeSession, action: dict[str, Any], expected: dict[str, float]
    ) -> None:
        driver = _live_driver(session)
        result = driver.send_action(action)
        assert result["status"] == "success"
        url, body, _ = session.posts[-1]
        assert url == f"{DEFAULT_SDK_URL}/control"
        assert body == {"command": expected}
        assert result["content"][0]["json"]["commanded"] == expected

    def test_turn_sign_flips_the_commanded_angular(self, session: _FakeSession) -> None:
        driver = _live_driver(session, turn_sign=-1.0)
        driver.send_action({"angular": 0.5})
        _, body, _ = session.posts[-1]
        assert body["command"]["angular"] == -0.5

    def test_move_is_sugar_over_send_action(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        assert driver.move(0.3, 0.1)["status"] == "success"
        _, body, _ = session.posts[-1]
        assert body["command"] == {"linear": 0.3, "angular": 0.1}


class TestSendActionRefusesRatherThanGuesses:
    @pytest.mark.parametrize(
        ("action", "needle"),
        [
            ({"linaer": 0.5}, "unknown drive channel"),
            ({"linear": float("nan")}, "linear"),
            ({"angular": float("inf")}, "angular"),
            ({"linear": "fast"}, "linear"),
        ],
        ids=["typo-channel", "nan", "inf", "string"],
    )
    def test_a_bad_action_is_refused_before_the_wire(
        self, session: _FakeSession, action: dict[str, Any], needle: str
    ) -> None:
        driver = _live_driver(session)
        posts_before = len(session.posts)
        refusal = driver.send_action(action)
        assert refusal["status"] == "error"
        assert needle in refusal["content"][0]["text"]
        assert len(session.posts) == posts_before  # nothing reached the SDK

    def test_the_channel_refusal_names_the_valid_set(self, session: _FakeSession) -> None:
        refusal = _live_driver(session).send_action({"warp": 9})
        assert str(list(DRIVE_CHANNELS)) in refusal["content"][0]["text"]

    @pytest.mark.parametrize(
        ("post_response", "needle"),
        [
            (_FakeResponse(500, None, "boom"), "HTTP 500"),
            (OSError("gone"), "did not reach"),
        ],
        ids=["http-500", "dead-link"],
    )
    def test_a_failed_send_is_reported_not_swallowed(
        self, session: _FakeSession, post_response: Any, needle: str
    ) -> None:
        driver = _live_driver(session)
        session.post_response = post_response
        refusal = driver.send_action({"linear": 0.5})
        assert refusal["status"] == "error"
        assert needle in refusal["content"][0]["text"]


class TestSpeakIsGuardedTheSameWay:
    def test_text_reaches_the_speaker_endpoint(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        assert driver.speak("on my way")["status"] == "success"
        url, body, _ = session.posts[-1]
        assert url.endswith("/speak") and body == {"text": "on my way"}

    @pytest.mark.parametrize("text", ["", "   ", 7], ids=["empty", "blank", "number"])
    def test_unspeakable_text_is_refused(self, session: _FakeSession, text: Any) -> None:
        assert _live_driver(session).speak(text)["status"] == "error"


# --------------------------------------------------------------------------- #
# Task paths.                                                                 #
# --------------------------------------------------------------------------- #


class TestTaskPathsAnswerHonestly:
    def test_stop_task_is_a_zero_twist_envelope(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        result = driver.stop_task()
        assert result["status"] == "success"
        assert result["content"][0]["json"]["commanded"] == {"linear": 0.0, "angular": 0.0}

    def test_a_failed_stop_is_not_reported_as_stopped(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        session.post_response = OSError("gone")
        assert driver.stop_task()["status"] == "error"

    def test_policy_paths_refuse_with_a_route(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        assert "send_action" in driver.start_task("go")["content"][0]["text"]
        assert "send_action" in driver.run_policy(object())["content"][0]["text"]  # type: ignore[arg-type]

    def test_task_status_reports_the_last_command(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        assert driver.get_task_status()["content"][0]["json"]["last_command"] is None
        driver.send_action({"linear": 0.4})
        assert driver.get_task_status()["content"][0]["json"]["last_command"]["linear"] == 0.4


# --------------------------------------------------------------------------- #
# The read path.                                                              #
# --------------------------------------------------------------------------- #


class TestTheReadSurface:
    def test_read_state_polls_fresh_telemetry(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        session.routes["/data"] = _FakeResponse(200, {**_DATA, "battery": 42})
        assert driver.read_state()["battery"] == 42

    def test_a_failed_poll_falls_back_to_the_cache(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        session.routes["/data"] = OSError("blip")
        assert driver.read_state()["battery"] == _DATA["battery"]

    def test_a_wheeled_base_reports_no_joints(self, session: _FakeSession) -> None:
        assert _live_driver(session).get_observation() == {}

    def test_status_reports_the_connection_and_the_last_command(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        driver.send_action({"linear": 0.2})
        payload = asyncio.run(driver.get_status())["content"][0]["json"]
        assert payload["connected"] is True
        assert payload["sdk_url"] == DEFAULT_SDK_URL
        assert payload["battery_pct"] == _DATA["battery"]
        assert payload["last_command"]["linear"] == 0.2

    def test_status_answers_for_a_robot_that_never_connected(self) -> None:
        payload = asyncio.run(EarthRoverDriver().get_status())["content"][0]["json"]
        assert payload["connected"] is False
        assert payload["last_command"] is None


class TestCameraFrames:
    _JPEG = b"\xff\xd8\xff" + b"\x00" * 8
    _PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    _WEBP = b"RIFF\x00\x00\x00\x00WEBP"

    @pytest.mark.parametrize(
        ("raw", "fmt"),
        [(_JPEG, "jpeg"), (_PNG, "png"), (_WEBP, "webp"), (b"????", "jpeg")],
        ids=["jpeg", "png", "webp", "unknown-defaults-jpeg"],
    )
    def test_the_format_is_read_off_the_magic_bytes(self, raw: bytes, fmt: str) -> None:
        assert detect_image_format(raw) == fmt

    def test_a_frame_comes_back_with_its_format(self, session: _FakeSession) -> None:
        import base64

        driver = _live_driver(session)
        session.routes["/v2/front"] = _FakeResponse(200, {"front_frame": base64.b64encode(self._PNG).decode()})
        payload = driver.capture_frame("front")["content"][0]["json"]
        assert payload["camera"] == "front" and payload["format"] == "png"

    @pytest.mark.parametrize(
        ("camera", "route", "needle"),
        [
            ("side", None, "camera must be one of"),
            ("front", _FakeResponse(200, {}), "video session is not up"),
            ("front", _FakeResponse(503, None, "starting"), "HTTP 503"),
            ("front", _FakeResponse(200, {"front_frame": "%%%not-base64%%%"}), "not base64"),
        ],
        ids=["unknown-view", "no-frame", "http-503", "bad-b64"],
    )
    def test_an_unusable_frame_is_refused_by_name(
        self, session: _FakeSession, camera: str, route: Any, needle: str
    ) -> None:
        driver = _live_driver(session)
        if route is not None:
            session.routes["/v2/front"] = route
        refusal = driver.capture_frame(camera)
        assert refusal["status"] == "error"
        assert needle in refusal["content"][0]["text"]

    def test_every_declared_view_is_a_route(self) -> None:
        assert CAMERA_VIEWS == ("front", "rear")


# --------------------------------------------------------------------------- #
# The agent surface.                                                          #
# --------------------------------------------------------------------------- #


class TestTheAgentSurface:
    def _events(self, driver: EarthRoverDriver, action: str) -> list[Any]:
        async def collect() -> list[Any]:
            return [
                event
                async for event in driver.stream(
                    {"toolUseId": "t-1", "name": "earthrover", "input": {"action": action}}, {}
                )
            ]

        return asyncio.run(collect())

    def test_sensors_yields_one_result_with_the_snapshot(self, session: _FakeSession) -> None:
        events = self._events(_live_driver(session), "sensors")
        assert len(events) == 1
        assert events[0]["toolUseId"] == "t-1"
        assert events[0]["content"][0]["json"]["battery"] == _DATA["battery"]

    def test_stop_reaches_the_wire(self, session: _FakeSession) -> None:
        driver = _live_driver(session)
        events = self._events(driver, "stop")
        assert events[0]["status"] == "success"
        assert session.posts[-1][1] == {"command": {"linear": 0.0, "angular": 0.0}}

    def test_the_spec_offers_exactly_the_read_only_trio(self, session: _FakeSession) -> None:
        spec = _live_driver(session).tool_spec
        assert spec["name"] == "earthrover"
        assert spec["inputSchema"]["json"]["properties"]["action"]["enum"] == ["sensors", "status", "stop"]
