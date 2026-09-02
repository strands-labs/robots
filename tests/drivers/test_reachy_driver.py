"""The Reachy Mini native driver: seam contract, daemon probe, sensors, envelope.

Everything here runs with no Reachy attached and no daemon listening. Two
doubles stand in, and each is faithful in the one respect the tests depend on:

* The daemon double replaces
  :func:`strands_robots.device_connect.reachy_transport.api` and returns the
  shape that function really returns - a decoded body, or ``{"error": ...}`` for
  every failure including an unreachable host. That is why the driver has no
  ``try`` around its REST calls, so a double that raised instead would be
  testing a code path that cannot happen.
* The link double is installed through
  :meth:`~strands_robots.drivers.reachy.ReachyDriver._build_link`, which exists
  as an overridable method for exactly this. It **records** the commands it is
  given rather than discarding them: a double that dropped its argument could
  not tell "the driver converted degrees to radians" from "the driver sent
  nothing", which is the whole subject of the wire-format tests.

The link double is driven on the driver's real background asyncio loop, so the
loop-and-thread plumbing in :meth:`ReachyDriver._start_link` is exercised rather
than mocked away.
"""

from __future__ import annotations

import asyncio
import math
import sys
from typing import Any

import pytest

import strands_robots.drivers.reachy as reachy_mod
from strands_robots.drivers import get_native_driver_class, resolve_driver
from strands_robots.drivers.base import HardwareDriver, missing_driver_members
from strands_robots.drivers.reachy import ReachyDriver
from strands_robots.tools.reachy import HEAD_BODY_YAW_DELTA_LIMIT_DEG, MOTION_ENVELOPE_DEG

# A status body shaped like the daemon's: the variant flag the driver reads, plus
# fields it passes over. ``wireless_version=False`` is a Lite, which is the
# variant that needs no Zenoh transport and so is the simplest to bring up.
_LITE_STATUS: dict[str, Any] = {"wireless_version": False, "motors": "on", "control_freq": 100.0}
_WIRELESS_STATUS: dict[str, Any] = {"wireless_version": True, "motors": "on"}


class _RecordingLink:
    """A ``HardwareLink`` that records commands and replays sensor payloads.

    Attributes:
        commands: Every command dict handed to :meth:`send_cmd`, in order.
        started: Whether :meth:`start` ran.
        stopped: Whether :meth:`stop` ran.
    """

    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []
        self.started = False
        self.stopped = False
        self._on_joints: Any = None
        self._on_imu: Any = None
        #: How many times the driver asked for a link. The double answers with
        #: itself every time, so identity alone cannot tell a re-build from a
        #: no-op - this counter is what grades the idempotence claim.
        self.build_calls = 0

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        """Record the callbacks the driver installed."""
        self._on_joints, self._on_imu = on_joints, on_imu
        self.started = True

    async def stop(self) -> None:
        """Record that teardown ran."""
        self.stopped = True

    async def send_cmd(self, cmd: dict[str, Any]) -> None:
        """Record one command verbatim."""
        self.commands.append(cmd)

    def deliver_joints(self, payload: dict[str, Any]) -> None:
        """Push a joints payload through the driver's own callback."""
        assert self._on_joints is not None, "link was never started"
        self._on_joints(payload)

    def deliver_imu(self, payload: dict[str, Any]) -> None:
        """Push an IMU payload through the driver's own callback."""
        assert self._on_imu is not None, "link was never started"
        self._on_imu(payload)


class _DaemonDouble:
    """Records REST calls and answers them from a table.

    Attributes:
        calls: ``(host, port, path, method)`` for every call, in order.
    """

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, int, str, str]] = []

    def __call__(
        self,
        host: str,
        port: int,
        path: str,
        method: str = "GET",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Answer one REST call, recording it first."""
        del data
        self.calls.append((host, port, path, method))
        return dict(self._responses.get(path, {}))


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: dict[str, Any] | None = None,
    stop_result: dict[str, Any] | None = None,
    link: _RecordingLink | None = None,
    tool_name: str = "reachy_mini",
    **driver_kwargs: Any,
) -> tuple[ReachyDriver, _DaemonDouble, _RecordingLink]:
    """Build a driver wired to a daemon double and a recording link.

    Args:
        monkeypatch: pytest's patcher.
        status: Body for ``/api/daemon/status``; defaults to a Lite.
        stop_result: Body for ``/api/move/stop``; defaults to success.
        link: Link double to install; a fresh one is made when omitted.
        tool_name: Driver's tool name and mesh peer id.
        **driver_kwargs: Forwarded to :class:`ReachyDriver`.

    Returns:
        ``(driver, daemon, link)``. The driver is *not* connected yet, so a test
        can assert on the pre-connect state.
    """
    daemon = _DaemonDouble(
        {
            reachy_mod._PATH_STATUS: _LITE_STATUS if status is None else status,
            reachy_mod._PATH_STOP: {"ok": True} if stop_result is None else stop_result,
        }
    )
    monkeypatch.setattr("strands_robots.device_connect.reachy_transport.api", daemon)
    installed = _RecordingLink() if link is None else link

    def _build(self: ReachyDriver, *, is_lite: bool) -> Any:
        installed.build_calls += 1
        return installed

    monkeypatch.setattr(ReachyDriver, "_build_link", _build)
    driver = ReachyDriver(tool_name=tool_name, **driver_kwargs)
    return driver, daemon, installed


def _connected(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> tuple[ReachyDriver, _DaemonDouble, _RecordingLink]:
    """Build a driver and connect it, asserting the connect succeeded.

    Args:
        monkeypatch: pytest's patcher.
        **kwargs: Forwarded to :func:`_install`.

    Returns:
        ``(driver, daemon, link)`` with the link running on the driver's loop.
    """
    driver, daemon, link = _install(monkeypatch, **kwargs)
    assert driver.connect_eagerly() is None
    return driver, daemon, link


def _text(envelope: dict[str, Any]) -> str:
    """Join every text block in an envelope, for substring assertions."""
    return " ".join(block.get("text", "") for block in envelope.get("content", []))


class TestTheDriverSatisfiesTheSeam:
    """The driver is reachable through the #353 seam, by every spelling."""

    def test_the_class_has_every_driver_member(self) -> None:
        assert missing_driver_members(ReachyDriver) == ()

    def test_an_instance_is_a_hardware_driver(self) -> None:
        assert isinstance(ReachyDriver(port="reachy-a.local"), HardwareDriver)

    def test_the_driver_is_registered_for_the_robot_on_import(self) -> None:
        assert get_native_driver_class("reachy_mini") is ReachyDriver

    @pytest.mark.parametrize("alias", ["reachy", "reachymini", "reachy-mini", "pollen_reachy_mini"])
    def test_every_registry_alias_reaches_the_same_driver(self, alias: str) -> None:
        assert get_native_driver_class(alias) is ReachyDriver

    def test_the_registry_makes_the_native_driver_the_default(self) -> None:
        # Without this the robot would resolve to lerobot, which has no robot
        # class for a Mini - the reason the native driver exists.
        assert resolve_driver("reachy_mini") == "strands"

    def test_the_tool_surface_names_the_peer(self) -> None:
        driver = ReachyDriver(tool_name="tiny-a")
        assert driver.tool_name == "tiny-a"
        assert driver.tool_type == "robot"
        assert driver.tool_spec["name"] == "tiny-a"

    def test_the_tool_spec_declares_only_verbs_the_driver_implements(self) -> None:
        enum = ReachyDriver().tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
        assert sorted(enum) == ["sensors", "status", "stop"]

    def test_the_constructor_takes_the_three_factory_keywords(self) -> None:
        # The factory builds every native driver this way; see
        # strands_robots.drivers.base's constructor contract.
        driver = ReachyDriver(tool_name="reachy_mini", cameras={"head": {}}, data_config="cfg")
        assert driver.tool_name == "reachy_mini"

    def test_the_constructor_tolerates_extras_the_factory_forwards(self) -> None:
        assert ReachyDriver(unknown_future_kwarg=1).tool_name == "reachy_mini"


class TestTheHostAndPortComeFromOnePolymorphicArgument:
    """``port=`` names a host, with or without a ``:port`` suffix."""

    @pytest.mark.parametrize(
        ("port", "expected_host", "expected_port"),
        [
            (None, "localhost", 8000),
            ("reachy-a.local", "reachy-a.local", 8000),
            ("reachy-a.local:8000", "reachy-a.local", 8000),
            ("reachy-b.local:9100", "reachy-b.local", 9100),
            ("192.168.1.5:8001", "192.168.1.5", 8001),
        ],
    )
    def test_a_host_and_an_optional_suffix_both_parse(
        self, port: str | None, expected_host: str, expected_port: int
    ) -> None:
        driver = ReachyDriver(port=port)
        assert (driver._host, driver._api_port) == (expected_host, expected_port)

    def test_an_explicit_suffix_beats_the_api_port_keyword(self) -> None:
        assert ReachyDriver(port="host:9000", api_port=8000)._api_port == 9000

    @pytest.mark.parametrize("bad", ["host:notaport", "host:80x"])
    def test_a_suffix_that_is_not_a_number_is_refused_by_name(self, bad: str) -> None:
        with pytest.raises(ValueError, match="is not a number"):
            ReachyDriver(port=bad)

    @pytest.mark.parametrize("bad", [0, -1, 70000])
    def test_a_port_outside_the_tcp_range_is_refused(self, bad: int) -> None:
        with pytest.raises(ValueError, match="api_port"):
            ReachyDriver(api_port=bad)

    def test_the_zenoh_prefix_defaults_to_the_peer_name(self) -> None:
        # Two Minis left at the default must not share a Zenoh key space.
        assert ReachyDriver(tool_name="tiny-b")._zenoh_prefix == "tiny-b"


class TestTheDaemonProbeDecidesTheConnection:
    """``connect_eagerly`` probes REST, then starts the variant's link."""

    def test_an_unreachable_daemon_is_named_with_its_address(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _install(
            monkeypatch,
            status={"error": "Connection refused"},
            port="reachy-a.local:8000",
        )
        reason = driver.connect_eagerly()
        assert reason is not None
        assert "daemon unreachable" in reason
        assert "reachy-a.local:8000" in reason
        assert not link.started

    def test_an_unreachable_daemon_leaves_the_driver_usable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A peer for a Mini that is switched off must still be constructible.
        driver, _, _ = _install(monkeypatch, status={"error": "timed out"})
        driver.connect_eagerly()
        status = asyncio.run(driver.get_status())["content"][0]["json"]
        assert status["connected"] is False
        assert "daemon unreachable" in status["connect_error"]

    def test_the_probe_asks_the_documented_status_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, daemon, _ = _connected(monkeypatch, port="reachy-a.local:8000")
        assert daemon.calls[0] == ("reachy-a.local", 8000, "/api/daemon/status", "GET")

    def test_a_lite_connects_and_reports_its_variant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _connected(monkeypatch, status=_LITE_STATUS)
        assert link.started
        status = asyncio.run(driver.get_status())["content"][0]["json"]
        assert status == {**status, "connected": True, "variant": "lite", "connect_error": None}

    def test_a_wireless_reports_its_variant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch, status=_WIRELESS_STATUS, transport=object())
        status = asyncio.run(driver.get_status())["content"][0]["json"]
        assert status["variant"] == "wireless"

    def test_a_second_connect_does_not_build_a_second_link(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Rebuilding would drop the only reference to the running link and
        # leave its reader subscribed. Graded by the build count, not by
        # identity: the double answers with itself, so identity would hold even
        # if the driver rebuilt on every call.
        driver, _, link = _connected(monkeypatch)
        assert link.build_calls == 1
        assert driver.connect_eagerly() is None
        assert link.build_calls == 1
        assert driver._link is link

    def test_a_wireless_without_a_transport_is_refused_by_name(self) -> None:
        # Exercises the real _build_link rather than the double, because the
        # refusal *is* the thing under test.
        driver = ReachyDriver(port="reachy-a.local", transport=None)
        link = driver._build_link(is_lite=False)
        assert isinstance(link, str)
        assert "Zenoh" in link and "transport=" in link

    def test_a_lite_gets_the_websocket_link(self) -> None:
        from strands_robots.device_connect.reachy_transport import WebSocketLink

        assert isinstance(ReachyDriver(port="h:8000")._build_link(is_lite=True), WebSocketLink)

    def test_a_wireless_with_a_transport_gets_the_zenoh_link(self) -> None:
        from strands_robots.device_connect.reachy_transport import ZenohLink

        driver = ReachyDriver(port="h", transport=object())
        assert isinstance(driver._build_link(is_lite=False), ZenohLink)

    def test_a_link_that_fails_to_start_is_a_named_connect_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Broken(_RecordingLink):
            async def start(self, on_joints: Any, on_imu: Any) -> None:
                raise RuntimeError("handshake rejected")

        driver, _, _ = _install(monkeypatch, link=_Broken(), port="reachy-a.local:8000")
        reason = driver.connect_eagerly()
        assert reason is not None
        assert "failed to start" in reason and "handshake rejected" in reason
        assert driver._connected is False

    def test_cleanup_stops_the_link_and_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _connected(monkeypatch)
        driver.cleanup()
        assert link.stopped
        assert driver._connected is False
        driver.cleanup()  # must not raise


class TestSensorCachesPopulateFromTheLink:
    """The mesh reads ``_imu``/``_pose``/``_battery`` off the driver itself."""

    def test_an_imu_payload_populates_the_imu_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _connected(monkeypatch)
        link.deliver_imu(
            {
                "accelerometer": [0.0, 0.0, 9.81],
                "gyroscope": [0.1, 0.2, 0.3],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "temperature": 31.5,
            }
        )
        imu = driver._imu
        assert imu is not None
        assert imu["accelerometer"] == [0.0, 0.0, 9.81]
        assert imu["gyroscope"] == [0.1, 0.2, 0.3]
        assert imu["temperature"] == 31.5

    def test_the_head_pose_is_the_imu_quaternion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The Mini's IMU is in the head, so its quaternion is the head's
        # orientation - a measurement rather than derived kinematics.
        driver, _, link = _connected(monkeypatch)
        link.deliver_imu({"quaternion": [0.707, 0.0, 0.707, 0.0]})
        pose = driver._pose
        assert pose is not None
        assert pose["quat"] == [0.707, 0.0, 0.707, 0.0]
        assert pose["frame"] == "head"
        assert pose["source"] == "imu"

    def test_an_imu_without_a_quaternion_leaves_the_pose_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Publishing a pose with no orientation would put an unmeasured number
        # on the mesh.
        driver, _, link = _connected(monkeypatch)
        link.deliver_imu({"accelerometer": [0.0, 0.0, 9.81]})
        assert driver._imu is not None
        assert driver._pose is None

    def test_joint_positions_are_reported_in_degrees(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _connected(monkeypatch)
        link.deliver_joints(
            {
                "head_joint_positions": [0.0, math.pi / 2],
                "antennas_joint_positions": [math.pi, -math.pi / 4],
            }
        )
        joints = driver._joints
        assert joints is not None
        assert joints["head_leg_deg"] == pytest.approx([0.0, 90.0])
        assert joints["antennas_deg"] == pytest.approx([180.0, -45.0])

    def test_a_battery_in_the_status_payload_populates_the_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch, status={**_LITE_STATUS, "battery_level": 82.5})
        assert driver._battery == {**(driver._battery or {}), "pct": 82.5, "source": "battery_level"}

    def test_a_status_payload_without_a_battery_leaves_the_cache_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The mesh treats every sensor attribute as optional, so no battery
        # field means no battery topic - not a broken driver.
        driver, _, _ = _connected(monkeypatch, status=_LITE_STATUS)
        assert driver._battery is None
        assert asyncio.run(driver.get_status())["content"][0]["json"]["battery_pct"] is None

    def test_a_boolean_battery_field_is_not_read_as_a_percentage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `battery: true` means "a battery is fitted", not "1 percent".
        driver, _, _ = _connected(monkeypatch, status={**_LITE_STATUS, "battery": True})
        assert driver._battery is None

    def test_a_malformed_payload_is_logged_rather_than_raised_on_the_link_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver, _, link = _connected(monkeypatch)
        link.deliver_joints({"head_joint_positions": ["not a number"]})
        link.deliver_imu({"quaternion": 5})
        assert driver._joints is None
        assert driver._pose is None

    def test_the_sensors_verb_returns_every_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _connected(monkeypatch, status={**_LITE_STATUS, "battery_level": 50.0})
        link.deliver_imu({"quaternion": [1.0, 0.0, 0.0, 0.0]})
        link.deliver_joints({"head_joint_positions": [0.0]})
        payload = _run_tool(driver, "sensors")["content"][0]["json"]
        assert sorted(payload) == ["battery", "imu", "joints", "pose"]
        assert payload["battery"]["pct"] == 50.0
        assert payload["pose"]["frame"] == "head"

    def test_a_snapshot_does_not_hand_out_the_live_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _connected(monkeypatch)
        link.deliver_imu({"quaternion": [1.0, 0.0, 0.0, 0.0]})
        snapshot = driver._snapshot("_imu")
        assert snapshot is not None
        snapshot["quaternion"] = "clobbered"
        assert (driver._imu or {})["quaternion"] == [1.0, 0.0, 0.0, 0.0]


class TestTheEnvelopeRefusesWhatTheNeckCannotDo:
    """Out-of-envelope motion is refused with the limit named, never clamped."""

    @pytest.mark.parametrize(
        ("axis", "value"),
        [
            ("head_pitch", 40.5),
            ("head_pitch", -40.5),
            ("head_roll", 41.0),
            ("head_yaw", 181.0),
            ("body_yaw", 161.0),
            ("body_yaw", -161.0),
        ],
    )
    def test_an_axis_past_its_travel_is_refused_by_name(
        self, monkeypatch: pytest.MonkeyPatch, axis: str, value: float
    ) -> None:
        driver, _, link = _connected(monkeypatch)
        result = driver.send_action({axis: value})
        assert result["status"] == "error"
        assert axis in _text(result)
        assert f"{MOTION_ENVELOPE_DEG[axis]:g}" in _text(result)
        assert link.commands == [], "a refused action must not reach the wire"

    @pytest.mark.parametrize("axis", sorted(MOTION_ENVELOPE_DEG))
    def test_every_axis_is_accepted_exactly_at_its_limit(self, monkeypatch: pytest.MonkeyPatch, axis: str) -> None:
        # The bound is inclusive; without this the refusal tests would pass
        # against a driver that refuses everything.
        driver, _, link = _connected(monkeypatch)
        assert driver.send_action({axis: MOTION_ENVELOPE_DEG[axis]})["status"] == "success"
        assert link.commands != []

    def test_a_head_and_body_yaw_pair_past_the_coupling_limit_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Both values are individually legal; only the pair is not. Nothing that
        # checks one axis at a time can see this.
        driver, _, link = _connected(monkeypatch)
        result = driver.send_action({"head_yaw": 60.0, "body_yaw": -60.0})
        assert result["status"] == "error"
        assert "120" in _text(result)
        assert f"{HEAD_BODY_YAW_DELTA_LIMIT_DEG:g}" in _text(result)
        assert link.commands == []

    def test_each_half_of_a_refused_pair_is_legal_against_a_robot_it_does_not_twist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The discriminator: the pair is refused for being a pair, not for its values.

        One driver each, so neither half is judged against the other. Without
        this the coupling tests would pass against a driver that refuses 60
        degrees of yaw outright.
        """
        head_only, _, _ = _connected(monkeypatch)
        body_only, _, _ = _connected(monkeypatch)
        assert head_only.send_action({"head_yaw": 60.0})["status"] == "success"
        assert body_only.send_action({"body_yaw": -60.0})["status"] == "success"

    def test_the_two_halves_in_sequence_are_still_the_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Splitting a refused pair across two calls does not make it legal.

        The head pose the first call commands is the one the daemon holds while
        the second is carried out, so the twist is the same 120 degrees whether
        the two values arrive together or apart. Owned by
        ``tests/test_reachy_a_lone_body_yaw_is_bounded_by_the_head_target.py``.
        """
        driver, _, link = _connected(monkeypatch)
        assert driver.send_action({"head_yaw": 60.0})["status"] == "success"
        link.commands.clear()

        result = driver.send_action({"body_yaw": -60.0})

        assert result["status"] == "error"
        assert f"{HEAD_BODY_YAW_DELTA_LIMIT_DEG:g}" in _text(result)
        assert link.commands == []

    def test_the_coupling_limit_is_inclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch)
        result = driver.send_action({"head_yaw": HEAD_BODY_YAW_DELTA_LIMIT_DEG, "body_yaw": 0.0})
        assert result["status"] == "success"

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "40", None])
    def test_a_value_that_is_not_a_finite_number_is_refused(self, monkeypatch: pytest.MonkeyPatch, value: Any) -> None:
        driver, _, link = _connected(monkeypatch)
        result = driver.send_action({"head_pitch": value})
        assert result["status"] == "error"
        assert "finite number" in _text(result)
        assert link.commands == []

    @pytest.mark.parametrize("key", ["head_x", "head_y", "head_z", "antenna_left", "antenna_right"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_a_pass_through_value_that_is_not_finite_is_still_refused(
        self, monkeypatch: pytest.MonkeyPatch, key: str, value: float
    ) -> None:
        # These keys carry no travel bound, so the shared envelope ignores them
        # entirely - the driver's own finiteness pass is the only thing between
        # a nan and the wire. Without it a non-finite offset reaches
        # ``rpy_to_pose`` and puts a matrix of nans on the link, with the call
        # reported as a success.
        driver, _, link = _connected(monkeypatch)
        from strands_robots.tools.reachy import envelope_error

        assert envelope_error({key: value}, "send_action") is None, (
            f"{key} is bounded after all; this test no longer grades the driver's own pass"
        )
        result = driver.send_action({key: value})
        assert result["status"] == "error"
        assert "finite number" in _text(result)
        assert key in _text(result)
        assert link.commands == []

    def test_the_driver_and_the_shared_envelope_cannot_disagree(self) -> None:
        # The limits are imported, not restated, so the reachy_* tools and this
        # driver bound the same robot the same way.
        import inspect

        source = inspect.getsource(reachy_mod)
        assert "from strands_robots.tools.reachy import envelope_error" in source
        for limit in ("40.0", "160.0", "65.0"):
            assert limit not in source, f"{limit} is restated in the driver instead of imported"


class TestActionsReachTheWireInTheDaemonsUnits:
    """Degrees and millimetres in; radians and a 4x4 pose out."""

    def test_a_body_yaw_is_converted_to_radians(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _connected(monkeypatch)
        assert driver.send_action({"body_yaw": 90.0})["status"] == "success"
        assert link.commands == [{"body_yaw": pytest.approx(math.pi / 2)}]

    def test_antennas_are_converted_to_radians_as_a_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _connected(monkeypatch)
        assert driver.send_action({"antenna_left": 60.0, "antenna_right": -60.0})["status"] == "success"
        assert link.commands == [
            {"antennas_joint_positions": [pytest.approx(math.radians(60)), pytest.approx(math.radians(-60))]}
        ]

    def test_one_antenna_still_sends_both_because_the_daemon_takes_a_pair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver, _, link = _connected(monkeypatch)
        driver.send_action({"antenna_left": 30.0})
        assert link.commands[0]["antennas_joint_positions"][1] == pytest.approx(0.0)

    def test_a_head_axis_becomes_a_four_by_four_pose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.device_connect.reachy_transport import rpy_to_pose

        driver, _, link = _connected(monkeypatch)
        assert driver.send_action({"head_pitch": 20.0, "head_z": 15.0})["status"] == "success"
        assert link.commands == [{"head_pose": rpy_to_pose(20.0, 0.0, 0.0, 0.0, 0.0, 15.0)}]

    def test_each_group_is_one_command_and_only_named_groups_are_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Commanding the antennas must not also re-send a head pose, which
        # would move the head as a side effect of moving an antenna.
        driver, _, link = _connected(monkeypatch)
        driver.send_action({"antenna_left": 10.0})
        assert [sorted(c) for c in link.commands] == [["antennas_joint_positions"]]

    def test_a_whole_body_action_sends_head_then_body_then_antennas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _connected(monkeypatch)
        driver.send_action({"head_yaw": 10.0, "body_yaw": 10.0, "antenna_left": 5.0, "antenna_right": 5.0})
        assert [sorted(c) for c in link.commands] == [
            ["head_pose"],
            ["body_yaw"],
            ["antennas_joint_positions"],
        ]

    def test_an_action_naming_nothing_this_driver_sends_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Reporting success for an action that moved nothing is the failure
        # mode this refusal exists to prevent.
        driver, _, link = _connected(monkeypatch)
        result = driver.send_action({"shoulder_pan": 10.0})
        assert result["status"] == "error"
        assert "nothing to send" in _text(result)
        assert link.commands == []

    def test_a_write_before_connect_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _install(monkeypatch)
        result = driver.send_action({"body_yaw": 10.0})
        assert result["status"] == "error"
        assert "not connected" in _text(result)
        assert link.commands == []

    def test_an_action_for_another_robot_is_refused_rather_than_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, link = _connected(monkeypatch, tool_name="tiny-a")
        result = driver.send_action({"body_yaw": 10.0}, robot_name="tiny-b")
        assert result["status"] == "error"
        assert "tiny-a" in _text(result) and "tiny-b" in _text(result)
        assert link.commands == []

    def test_this_drivers_own_name_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch, tool_name="tiny-a")
        assert driver.send_action({"body_yaw": 10.0}, robot_name="tiny-a")["status"] == "success"

    def test_a_link_that_refuses_a_command_is_reported_not_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Refusing(_RecordingLink):
            async def send_cmd(self, cmd: dict[str, Any]) -> None:
                raise RuntimeError("socket closed")

        driver, _, _ = _connected(monkeypatch, link=_Refusing())
        result = driver.send_action({"body_yaw": 10.0})
        assert result["status"] == "error"
        assert "socket closed" in _text(result)


class TestTheStopPathReachesTheDaemon:
    """A Mini has a real stop: a recorded move can be halted mid-play."""

    def test_stop_posts_the_documented_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, daemon, _ = _connected(monkeypatch, port="reachy-a.local:8000")
        asyncio.run(driver.stop())
        assert ("reachy-a.local", 8000, "/api/move/stop", "POST") in daemon.calls

    def test_stop_records_that_motion_was_halted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch)
        asyncio.run(driver.stop())
        assert asyncio.run(driver.get_status())["content"][0]["json"]["motion_stopped"] is True

    def test_a_daemon_that_refuses_the_stop_does_not_report_it_as_halted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Reporting a halt that did not happen is an affirmative lie on a
        # safety path.
        driver, _, _ = _connected(monkeypatch, stop_result={"error": "busy"})
        asyncio.run(driver.stop())
        assert asyncio.run(driver.get_status())["content"][0]["json"]["motion_stopped"] is False

    def test_the_stop_verb_leaves_sensors_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A halted Mini is still observable, which is what an operator wants.
        driver, _, link = _connected(monkeypatch)
        _run_tool(driver, "stop")
        assert not link.stopped
        link.deliver_imu({"quaternion": [1.0, 0.0, 0.0, 0.0]})
        assert driver._pose is not None

    def test_stop_task_halts_a_recorded_move(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, daemon, _ = _connected(monkeypatch)
        assert driver.stop_task()["status"] == "success"
        assert any(call[2] == "/api/move/stop" for call in daemon.calls)

    def test_stop_task_reports_a_daemon_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch, stop_result={"error": "busy"})
        result = driver.stop_task()
        assert result["status"] == "error"
        assert "busy" in _text(result)


class TestThePolicyPathRefusesRatherThanPretending:
    """The Mini has no action space, so a rollout is refused, not deferred."""

    @pytest.mark.parametrize("verb", ["start_task", "run_policy"])
    def test_a_policy_verb_names_the_recorded_move_path_instead(
        self, monkeypatch: pytest.MonkeyPatch, verb: str
    ) -> None:
        driver, _, _ = _connected(monkeypatch)
        # The driver refuses before it reads the policy, so a bare stand-in is
        # enough; annotated Any because the parameter is typed Policy.
        policy: Any = object()
        result = driver.start_task("wave hello") if verb == "start_task" else driver.run_policy(policy)
        assert result["status"] == "error"
        assert "no policy action space" in _text(result)
        assert "recorded move" in _text(result)

    def test_the_task_status_question_is_still_answerable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch)
        result = driver.get_task_status()
        assert result["status"] == "success"
        assert result["content"][0]["json"]["running"] is False


class TestTwoMinisAreTwoPeers:
    """Two Minis on one mesh differ by peer id and share no state."""

    def test_two_drivers_carry_distinct_peer_ids_and_key_spaces(self) -> None:
        a, b = ReachyDriver(tool_name="tiny-a"), ReachyDriver(tool_name="tiny-b")
        assert a.tool_name != b.tool_name
        assert a._zenoh_prefix != b._zenoh_prefix
        assert a.tool_spec["name"] != b.tool_spec["name"]

    def test_two_drivers_address_different_daemons(self) -> None:
        a = ReachyDriver(tool_name="tiny-a", port="reachy-a.local:8000")
        b = ReachyDriver(tool_name="tiny-b", port="reachy-b.local:8000")
        assert (a._host, a._api_port) != (b._host, b._api_port)

    def test_one_minis_sensor_reading_does_not_appear_on_the_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The caches are per instance; a class attribute would fail this.
        a, _, link_a = _connected(monkeypatch, tool_name="tiny-a")
        b, _, _ = _connected(monkeypatch, tool_name="tiny-b", link=_RecordingLink())
        link_a.deliver_imu({"quaternion": [1.0, 0.0, 0.0, 0.0]})
        assert a._pose is not None
        assert b._pose is None

    def test_commanding_one_mini_does_not_command_the_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        a, _, link_a = _connected(monkeypatch, tool_name="tiny-a")
        b, _, link_b = _connected(monkeypatch, tool_name="tiny-b", link=_RecordingLink())
        a.send_action({"body_yaw": 30.0})
        assert len(link_a.commands) == 1
        assert link_b.commands == []
        b.cleanup()
        a.cleanup()


class TestTheLerobotPathIsUnaffected:
    """Declaring a native driver must not change any other robot."""

    def test_asking_for_lerobot_explicitly_still_goes_to_lerobot(self) -> None:
        assert resolve_driver("reachy_mini", "lerobot") == "lerobot"

    def test_the_other_reachy_is_untouched(self) -> None:
        # reachy2 is a different robot with a real lerobot type; it must not
        # inherit the Mini's driver.
        assert resolve_driver("reachy2") == "lerobot"
        assert get_native_driver_class("reachy2") is None

    def test_the_g1_registration_survives_a_second_shipped_driver(self) -> None:
        # The registration loop must not let one driver's arrival cost another's.
        from strands_robots.drivers.g1 import G1Driver

        assert get_native_driver_class("unitree_g1") is G1Driver

    def test_every_shipped_driver_satisfies_the_seam(self) -> None:
        # Derived from the shipped table, so a driver added later is held to the
        # same contract without editing this test.
        import importlib

        from strands_robots.drivers import _SHIPPED_DRIVERS, shipped_robot_names

        assert _SHIPPED_DRIVERS, "the shipped-driver table is empty"
        for module_path, class_name, robot_names in _SHIPPED_DRIVERS:
            module = importlib.import_module(module_path)
            driver_cls = getattr(module, class_name)
            assert missing_driver_members(driver_cls) == (), f"{class_name} is missing driver members"
            # Resolved through the same helper the registration loop uses, so an
            # entry that names its family on its own module is graded on the
            # names actually registered rather than on the attribute name.
            assert shipped_robot_names(module, robot_names), f"{class_name} is registered for no robot"


def _run_tool(driver: ReachyDriver, action: str) -> dict[str, Any]:
    """Drive one agent tool call to completion and return the single result.

    Args:
        driver: The driver to invoke.
        action: The ``action`` verb to request.

    Returns:
        The one envelope the driver yields.
    """

    async def _drive() -> dict[str, Any]:
        results = [
            event
            async for event in driver.stream(
                {"name": driver.tool_name, "toolUseId": "t1", "input": {"action": action}}, {}
            )
        ]
        assert len(results) == 1, f"expected exactly one tool result, got {len(results)}"
        return results[0]

    return asyncio.run(_drive())


class TestAnUnimportableTransportIsRefusedByNameRatherThanCrashing:
    """An unimportable transport must be reported, not raised.

    Every daemon touch here goes through
    :mod:`strands_robots.device_connect.reachy_transport`, a stdlib-only leaf
    whose parent package resolves its third-party imports lazily. Nothing an
    extra installs decides whether that import succeeds, so a failure reaching
    these surfaces is a broken install of a module the core distribution ships -
    a shadowing module, a partial wheel - which the ``ImportError`` describes and
    no ``pip install`` line repairs.

    What must hold either way is that it does not raise: these surfaces document
    a returned reason, and an escaping ``ModuleNotFoundError`` becomes a
    traceback through the agent tool surface instead.

    The failure is simulated rather than installed, the way Python itself signals
    a blocked module: ``sys.modules[name] = None`` makes
    ``importlib.import_module`` raise ``ImportError``.
    """

    @staticmethod
    def _block_transport(monkeypatch: pytest.MonkeyPatch) -> None:
        """Make the transport module unimportable for the current test."""
        monkeypatch.setitem(sys.modules, reachy_mod._TRANSPORT_MODULE, None)

    def test_the_reason_names_the_module_and_the_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reason nobody can act on is barely better than a traceback.

        What a caller can act on here is the module that failed and the error it
        failed with. An install command is not part of that: no ``pip install``
        supplies this module, so naming one would be a diagnosis this branch
        cannot establish.
        """
        self._block_transport(monkeypatch)
        reason = reachy_mod._resolve_transport()
        assert isinstance(reason, str)
        assert reachy_mod._TRANSPORT_MODULE in reason
        assert "halted" in reason, f"the underlying ImportError is not reported: {reason}"
        assert "pip install" not in reason, f"prescribes an install it cannot establish: {reason}"

    def test_connect_eagerly_returns_the_reason_instead_of_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``connect_eagerly`` documents a returned reason, so it must return one."""
        self._block_transport(monkeypatch)
        driver = ReachyDriver(tool_name="reachy_mini", port="reachy-a.local")
        reason = driver.connect_eagerly()
        assert reason is not None
        assert reachy_mod._TRANSPORT_MODULE in reason
        assert "pip install" not in reason
        # Left disconnected but usable, and the reason cached for a later gate.
        assert driver._connected is False
        assert driver._connect_error == reason

    def test_connect_eagerly_does_not_blame_the_network(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing package is not an unreachable daemon.

        The probe wraps any error it gets in ``daemon unreachable (host:port)``.
        Reporting a missing extra through that wrapper would send an operator to
        check a network this call never touched, so the transport is resolved
        before the probe runs.
        """
        self._block_transport(monkeypatch)
        driver = ReachyDriver(tool_name="reachy_mini", port="reachy-a.local")
        reason = driver.connect_eagerly()
        assert reason is not None
        assert "daemon unreachable" not in reason

    def test_send_action_refuses_rather_than_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The connected gate should refuse first; the write path must not raise anyway.

        ``send_action`` is reached through the agent tool surface, where an
        exception escapes as a traceback rather than an error envelope. The
        driver is forced past its connected gate here so the wire path itself is
        graded, not the gate in front of it.
        """
        self._block_transport(monkeypatch)
        driver = ReachyDriver(tool_name="reachy_mini", port="reachy-a.local")
        driver._connected = True
        envelope = driver.send_action({"head_pitch": 5.0})
        assert envelope["status"] == "error"
        assert reachy_mod._TRANSPORT_MODULE in _text(envelope)
        assert "pip install" not in _text(envelope)

    def test_stop_does_not_claim_a_halt_it_could_not_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stop that never reached the daemon is not a stop.

        Same contract as a daemon that refuses the stop: ``_stopped`` stays
        ``False`` so nothing downstream reports a halted robot.
        """
        self._block_transport(monkeypatch)
        driver = ReachyDriver(tool_name="reachy_mini", port="reachy-a.local")
        asyncio.run(driver.stop())
        assert driver._stopped is False

    def test_the_module_still_imports_and_registers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The driver stays constructible, which is why the refusal has to be named.

        Nothing at module load touches the transport, so the class imports, the
        registry keeps its entry and ``Robot(..., mode="real")`` builds. That is
        precisely why the failure has to arrive as a reason at the first daemon
        touch rather than as a ``ModuleNotFoundError``.
        """
        self._block_transport(monkeypatch)
        assert get_native_driver_class("reachy_mini") is ReachyDriver
        driver = ReachyDriver(tool_name="reachy_mini", port="reachy-a.local")
        assert isinstance(driver, HardwareDriver)
