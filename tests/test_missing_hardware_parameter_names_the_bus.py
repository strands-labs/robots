"""A required hardware parameter the caller did not supply is refused by name.

Eight of lerobot's serial robot types declare ``port`` with no default, so
forgetting it is the commonest caller mistake on the hardware path. Letting the
resolved dataclass raise reported it as ``SOFollowerRobotConfig.__init__()
missing 1 required positional argument: 'port'`` -- a lerobot internal, naming
neither the parameter the caller supplies nor the devices this host has, while
the port list needed to answer it had already been enumerated one frame up by
``mode="auto"`` and discarded.

The mirror-image mistake was already reported properly: a kwarg the dataclass
does not declare (a typo like ``prot=``) is refused with the accepted field
names. These tests pin the other half, and pin that the rule deciding "which
serial device is a robot's motor bus" has one owner -- the hardware layer and
``_auto_detect_mode`` both consult it, and a second copy of the vendor-id table
would let them disagree about what a robot is.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
from typing import Any

import pytest

import strands_robots

_PACKAGE_DIR = pathlib.Path(strands_robots.__file__).parent
_OWNER = _PACKAGE_DIR / "_serial_discovery.py"

# The vendor ids, restated here rather than imported: this is the value the
# one-owner guard pins, and a guard that reads it from the module it is checking
# cannot fail when that module is absent.
_SERVO_BUS_VIDS = (0x1A86, 0x0403)  # WCH CH34x, FTDI

# A WCH CH34x servo bridge as an SO-101 enumerates: a generic description that
# no keyword matches, so only the vendor id recognises it.
_ARM_VID, _ARM_PID = 0x1A86, 0x55D3


class _FakePort:
    """A stand-in for pyserial's ``ListPortInfo``."""

    def __init__(
        self,
        device: str,
        description: str = "n/a",
        vid: int | None = None,
        pid: int | None = None,
        serial_number: str | None = None,
        manufacturer: str | None = None,
        location: str | None = None,
    ) -> None:
        self.device = device
        self.name = device.rsplit("/", 1)[-1]
        self.description = description
        self.manufacturer = manufacturer
        self.vid = vid
        self.pid = pid
        self.serial_number = serial_number
        self.location = location


def _arm(device: str, serial_number: str | None, location: str | None = None) -> _FakePort:
    return _FakePort(device, "USB Single Serial", _ARM_VID, _ARM_PID, serial_number, location=location)


def _uart(device: str) -> _FakePort:
    """An on-board UART: no vid, no pid, no serial number."""
    return _FakePort(device)


@pytest.fixture
def bus(monkeypatch: pytest.MonkeyPatch):
    """Return a callable that makes ``comports()`` report the given devices."""
    serial_tools = pytest.importorskip("serial.tools.list_ports")

    def _set(*ports: Any) -> None:
        monkeypatch.setattr(serial_tools, "comports", lambda: list(ports))

    return _set


def _config_for(robot_type: str, tool_name: str, **kwargs: Any) -> Any:
    """Build a config through the real hardware path, returning it or raising."""
    pytest.importorskip("lerobot")
    from strands_robots.hardware_robot import Robot as HwRobot

    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = tool_name
    return hw._create_minimal_config(robot_type, None, **kwargs)


def _refusal_for(robot_type: str, tool_name: str, **kwargs: Any) -> str:
    """Return the refusal text, failing loudly if the call is not refused."""
    try:
        built = _config_for(robot_type, tool_name, **kwargs)
    except ValueError as refused:
        return str(refused)
    raise AssertionError(f"{robot_type} built {type(built).__name__} instead of refusing a missing required parameter")


class TestAMissingRequiredParameterIsNamed:
    """The refusal names the parameter the caller supplies, not a lerobot internal."""

    def test_the_refusal_names_the_missing_parameter(self, bus) -> None:
        bus(_arm("/dev/ttyACM0", "5AB0181806"))
        text = _refusal_for("so101_follower", "so101")

        assert "port" in text, text
        assert "missing required parameter" in text, text

    def test_the_refusal_does_not_leak_the_dataclass_initialiser(self, bus) -> None:
        """``SOFollowerRobotConfig.__init__()`` is lerobot's business, not the caller's."""
        bus(_arm("/dev/ttyACM0", "5AB0181806"))
        text = _refusal_for("so101_follower", "so101")

        assert "__init__()" not in text, text
        assert "positional argument" not in text, text

    def test_the_refusal_offers_the_call_that_supplies_it(self, bus) -> None:
        bus(_arm("/dev/ttyACM0", "5AB0181806"))
        text = _refusal_for("so101_follower", "so101")

        assert "Robot('so101', mode='real', port=...)" in text, text

    def test_every_missing_parameter_is_named_not_just_the_first(self, bus) -> None:
        """``hope_jr_hand`` requires both ``port`` and ``side``."""
        bus(_arm("/dev/ttyACM0", "5AB0181806"))
        text = _refusal_for("hope_jr_hand", "hope_jr_hand")

        assert "port" in text and "side" in text, text
        assert "port=..., side=..." in text, text

    def test_the_refusal_still_names_the_config_class_and_the_assembled_values(self, bus) -> None:
        """The wrapper's existing contract: name the class and what was assembled."""
        bus(_arm("/dev/ttyACM0", "5AB0181806"))
        text = _refusal_for("so101_follower", "so101")

        assert "Failed to construct SOFollowerRobotConfig for robot type 'so101_follower'" in text, text
        assert "Config: {" in text, text


class TestThePortRefusalNamesTheBus:
    """A missing port is answered with the devices this host actually has."""

    def test_a_servo_bus_candidate_is_named_with_its_stable_id(self, bus) -> None:
        bus(_arm("/dev/ttyACM0", "5AB0181806"), _arm("/dev/ttyACM3", "5AB0158428"))
        text = _refusal_for("so101_follower", "so101")

        assert "/dev/ttyACM0 (usb id 5AB0181806)" in text, text
        assert "/dev/ttyACM3 (usb id 5AB0158428)" in text, text

    def test_devices_that_are_not_a_servo_bus_are_named_as_such(self, bus) -> None:
        """Two on-board UARTs are present but neither is a robot -- say so."""
        bus(_uart("/dev/ttyS0"), _uart("/dev/ttyAMA10"))
        text = _refusal_for("so101_follower", "so101")

        assert "looks like a servo bus" in text, text
        assert "/dev/ttyS0" in text and "/dev/ttyAMA10" in text, text

    def test_an_empty_bus_is_reported_as_empty(self, bus) -> None:
        bus()
        text = _refusal_for("so101_follower", "so101")

        assert "No serial devices are present on this host." in text, text

    def test_the_refusal_says_a_port_is_not_an_identity(self, bus) -> None:
        """The reason to name the usb id: the path is what moves across a replug."""
        bus(_arm("/dev/ttyACM0", "5AB0181806"))
        text = _refusal_for("so101_follower", "so101")

        assert "position on the bus, not an identity" in text, text

    def test_an_unusable_bus_still_produces_the_parameter_refusal(self, monkeypatch) -> None:
        """Enumeration is best-effort: a hub glitch must not replace the refusal."""
        serial_tools = pytest.importorskip("serial.tools.list_ports")

        def boom() -> list[Any]:
            raise RuntimeError("libusb hub glitch")

        monkeypatch.setattr(serial_tools, "comports", boom)
        text = _refusal_for("so101_follower", "so101")

        assert "missing required parameter" in text and "port" in text, text
        assert "No serial devices are present on this host." in text, text


class TestOnlyAPortIsAnsweredByASerialScan:
    """A missing network address is not answered with this host's serial devices."""

    def test_a_missing_remote_ip_names_the_parameter_but_not_the_serial_bus(self, bus) -> None:
        bus(_arm("/dev/ttyACM0", "5AB0181806"))
        text = _refusal_for("lekiwi_client", "lekiwi")

        assert "remote_ip" in text, text
        assert "/dev/ttyACM0" not in text, text
        assert "servo bus" not in text, text


class TestTheServoBusRecognitionRule:
    """The rule is pure and testable without a bus."""

    def test_a_generic_description_is_recognised_by_vendor_id(self) -> None:
        """An SO-101 enumerates as ``USB Single Serial``, which no keyword matches."""
        from strands_robots._serial_discovery import matches_servo_bus

        assert matches_servo_bus(_arm("/dev/ttyACM0", "5AB0181806")) is True

    def test_an_onboard_uart_is_not_a_servo_bus(self) -> None:
        from strands_robots._serial_discovery import matches_servo_bus

        assert matches_servo_bus(_uart("/dev/ttyS0")) is False

    def test_a_named_servo_bus_is_recognised_without_a_vendor_id(self) -> None:
        from strands_robots._serial_discovery import matches_servo_bus

        assert matches_servo_bus(_FakePort("/dev/ttyUSB0", "Feetech STS3215 bus")) is True

    def test_an_excluded_device_is_refused_however_it_matched(self) -> None:
        """A matching vendor id does not make a Bluetooth bridge a robot."""
        from strands_robots._serial_discovery import matches_servo_bus

        assert matches_servo_bus(_FakePort("/dev/cu.Bluetooth", "Bluetooth-Incoming-Port", _ARM_VID, _ARM_PID)) is False

    def test_a_partial_stand_in_is_answered_rather_than_raising(self) -> None:
        """The rule reads its inputs with defaults, so a bare object is answered."""
        from strands_robots._serial_discovery import matches_servo_bus

        assert matches_servo_bus(object()) is False


class TestTheStableIdentity:
    """What survives a replug, and what stands in when nothing does."""

    def test_a_serial_number_is_the_identity(self) -> None:
        from strands_robots._serial_discovery import stable_device_id

        assert stable_device_id(_arm("/dev/ttyACM0", "5AB0181806")) == "5AB0181806"

    def test_the_same_hardware_keeps_its_identity_when_the_port_moves(self) -> None:
        """The point of the whole exercise: the path changes, the id does not."""
        from strands_robots._serial_discovery import stable_device_id

        before = _arm("/dev/ttyACM0", "5AB0181806")
        after = _arm("/dev/ttyACM3", "5AB0181806")

        assert before.device != after.device
        assert stable_device_id(before) == stable_device_id(after)

    def test_a_device_without_a_serial_number_falls_back_to_vid_pid(self) -> None:
        from strands_robots._serial_discovery import stable_device_id

        no_serial = _FakePort("/dev/ttyUSB0", "USB Serial", 0x0403, 0x6001)

        assert stable_device_id(no_serial) == "0403:6001"

    def test_the_fallback_carries_the_bus_location_when_there_is_one(self) -> None:
        """Two identical serial-less devices are still told apart by position."""
        from strands_robots._serial_discovery import stable_device_id

        left = _FakePort("/dev/ttyUSB0", "USB Serial", 0x0403, 0x6001, location="1-2:1.0")
        right = _FakePort("/dev/ttyUSB1", "USB Serial", 0x0403, 0x6001, location="1-3:1.0")

        assert stable_device_id(left) == "0403:6001:1-2:1.0"
        assert stable_device_id(left) != stable_device_id(right)

    def test_a_device_with_no_usb_identity_at_all_has_none(self) -> None:
        from strands_robots._serial_discovery import stable_device_id

        assert stable_device_id(_uart("/dev/ttyS0")) is None

    def test_a_candidate_with_no_identity_is_still_named_by_port(self, bus) -> None:
        """A serial-less servo bus is reported without inventing an id for it."""
        from strands_robots._serial_discovery import describe_serial_candidates, scan_serial_devices

        bus(_FakePort("/dev/ttyUSB0", "Feetech bus"))
        described = describe_serial_candidates(scan_serial_devices())

        assert "/dev/ttyUSB0" in described
        assert "usb id" not in described


class TestTheRuleHasOneOwner:
    """A second copy of the vendor-id table would let the two callers disagree."""

    def test_only_the_owner_module_carries_the_vendor_id_table(self) -> None:
        carriers = sorted(
            path.relative_to(_PACKAGE_DIR).as_posix()
            for path in _PACKAGE_DIR.rglob("*.py")
            if path != _OWNER
            and any(f"{vid:#06x}" in path.read_text(encoding="utf-8").lower() for vid in _SERVO_BUS_VIDS)
        )

        assert carriers == [], (
            f"the servo-bus vendor ids are named outside {_OWNER.name} by {carriers}; "
            "route those call sites through matches_servo_bus() instead"
        )

    def test_the_mode_probe_consults_the_owner(self) -> None:
        """``_auto_detect_mode`` must not re-derive what a robot's bus looks like."""
        source = (_PACKAGE_DIR / "robot.py").read_text(encoding="utf-8")

        assert "scan_serial_devices" in source, "robot.py no longer consults the owner"
        assert "comports" not in source, "robot.py enumerates the bus itself instead of asking the owner"

    def test_the_hardware_refusal_consults_the_owner(self) -> None:
        source = (_PACKAGE_DIR / "hardware_robot.py").read_text(encoding="utf-8")

        assert "scan_serial_devices" in source
        assert "describe_serial_candidates" in source

    def test_the_owner_is_reachable_without_pyserial(self, monkeypatch) -> None:
        """pyserial is not a declared dependency, so its absence reports no devices."""
        import builtins

        real_import = builtins.__import__

        def refuse_serial(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("serial"):
                raise ImportError("no pyserial")
            return real_import(name, *args, **kwargs)

        from strands_robots._serial_discovery import scan_serial_devices

        monkeypatch.setattr(builtins, "__import__", refuse_serial)

        assert scan_serial_devices() == []


class TestNothingElseChanges:
    """Which calls are refused is unchanged -- only what a refused call is told."""

    def test_a_supplied_port_still_builds(self, bus) -> None:
        bus(_arm("/dev/ttyACM0", "5AB0181806"))
        config = _config_for("so101_follower", "so101", port="/dev/ttyACM0")

        assert config.port == "/dev/ttyACM0"

    def test_a_config_that_refuses_its_own_values_keeps_the_generic_wrapper(self, monkeypatch, bus) -> None:
        """A ``__post_init__`` refusal is not a missing parameter -- it still wraps."""
        pytest.importorskip("lerobot")
        from lerobot.robots.config import RobotConfig

        @dataclasses.dataclass
        class ExplodingConfig:
            id: str = ""
            cameras: dict[str, Any] = dataclasses.field(default_factory=dict)

            def __post_init__(self) -> None:
                raise ValueError("post-init rejected the assembled config")

        monkeypatch.setattr(RobotConfig, "get_choice_class", lambda robot_type: ExplodingConfig)
        bus(_arm("/dev/ttyACM0", "5AB0181806"))

        with pytest.raises(ValueError, match="post-init rejected the assembled config"):
            _config_for("so101_follower", "so101")

    @pytest.mark.parametrize(
        ("label", "devices", "expected"),
        [
            ("a servo bus is present", (_arm("/dev/ttyACM0", "5AB0181806"),), "real"),
            ("only on-board uarts", (_uart("/dev/ttyS0"), _uart("/dev/ttyAMA10")), "sim"),
            ("nothing at all", (), "sim"),
        ],
    )
    def test_the_mode_probe_reaches_the_same_verdict(self, bus, label, devices, expected) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.robot import _auto_detect_mode

        bus(*devices)

        assert _auto_detect_mode("so101") == expected, label

    def test_an_unusable_bus_still_falls_back_to_sim(self, monkeypatch) -> None:
        serial_tools = pytest.importorskip("serial.tools.list_ports")
        from strands_robots.robot import _auto_detect_mode

        def boom() -> list[Any]:
            raise RuntimeError("libusb hub glitch")

        monkeypatch.setattr(serial_tools, "comports", boom)

        assert _auto_detect_mode("so101") == "sim"


class TestTheScanIsNotVacuous:
    """A rule that recognises nothing would satisfy every assertion above."""

    def test_the_owner_declares_the_vendor_ids_this_file_pins(self) -> None:
        """The restated literal above must be the set the owner really declares."""
        from strands_robots._serial_discovery import SERVO_BUS_VIDS

        assert set(_SERVO_BUS_VIDS) == set(SERVO_BUS_VIDS)
        assert 0x1A86 in SERVO_BUS_VIDS, "the WCH CH34x bridge every SO-10x carries"

    def test_the_one_owner_scan_reaches_the_package(self) -> None:
        scanned = [path for path in _PACKAGE_DIR.rglob("*.py")]

        assert len(scanned) > 100, f"only {len(scanned)} modules scanned"
        assert _OWNER in scanned

    def test_the_owner_module_parses_as_the_source_the_guards_read(self) -> None:
        """The structural guards read source text; confirm it is the real module."""
        tree = ast.parse(_OWNER.read_text(encoding="utf-8"))
        names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

        assert {"matches_servo_bus", "stable_device_id", "scan_serial_devices"} <= names
