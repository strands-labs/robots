"""Behavior tests for hardware Robot lerobot-config construction.

Covers ``strands_robots.hardware_robot.Robot._create_minimal_config`` and the
``_initialize_robot`` dispatch - the seam that turns a strands ``Robot(...)``
call into a concrete lerobot ``RobotConfig``/driver. The contracts pinned here:

    - a registered robot_type resolves to its lerobot config dataclass, with
      ``id`` defaulting to the strands tool name and known kwargs forwarded;
    - a camera dict becomes an ``OpenCVCameraConfig`` (with lerobot defaults),
      and an unsupported camera type is rejected;
    - a typo'd / unknown kwarg is rejected with a ``ValueError`` rather than
      silently dropped (Review Learnings #86);
    - a cross-robot allowlist kwarg absent from the resolved dataclass is
      dropped, not raised (the ``Robot('so101', kp=[...])`` polymorphism
      carve-out), while a dataclass field outside the allowlist is still
      forwarded (new-lerobot-field future-proofing);
    - ``_initialize_robot`` passes a prebuilt driver / config straight through
      and rejects an unsupported object type.

The resolved config classes come from the installed lerobot's draccus
``ChoiceRegistry``, so the module ``importorskip``s ``lerobot`` and self-skips
where it is not installed. No serial/USB hardware is touched: only config
dataclasses are constructed, and the ``make_robot_from_config`` factory is
stubbed for the driver-construction branches.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

pytest.importorskip("lerobot")

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState


def _make_robot() -> HwRobot:
    """A Robot wired with just the attributes ``_create_minimal_config`` /
    ``_initialize_robot`` need, plus the handful the destructor's cleanup path
    reads (so teardown is silent) - never touching hardware init."""
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "test_arm"
    hw._shutdown_event = threading.Event()
    hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_arm_executor")
    hw.mesh = None
    return hw


class TestCreateMinimalConfig:
    def test_registered_type_resolves_config_and_forwards_known_kwargs(self):
        hw = _make_robot()
        cfg = hw._create_minimal_config("so101_follower", None, port="/dev/ttyACM0")
        # Resolved to the lerobot SO-follower config dataclass.
        assert type(cfg).__name__ == "SOFollowerRobotConfig"
        # ``id`` defaults to the strands tool name (namespaces calibration).
        assert cfg.id == "test_arm"
        # A declared, allowlisted kwarg is forwarded verbatim.
        assert cfg.port == "/dev/ttyACM0"

    def test_camera_dict_becomes_opencv_config_with_defaults(self):
        hw = _make_robot()
        cfg = hw._create_minimal_config(
            "so101_follower",
            {"base": {"type": "opencv", "index_or_path": 0}},
            port="/dev/ttyACM0",
        )
        cam = cfg.cameras["base"]
        assert type(cam).__name__ == "OpenCVCameraConfig"
        assert cam.index_or_path == 0
        # Unspecified camera keys fall back to the documented defaults.
        assert cam.fps == 30
        assert cam.width == 640
        assert cam.height == 480

    def test_unsupported_camera_type_is_rejected(self):
        hw = _make_robot()
        with pytest.raises(ValueError, match="Unsupported camera type: realsense"):
            hw._create_minimal_config(
                "so101_follower",
                {"c": {"type": "realsense", "index_or_path": 0}},
                port="/dev/ttyACM0",
            )

    def test_unknown_kwarg_is_rejected_not_silently_dropped(self):
        hw = _make_robot()
        # ``prot`` is a typo for ``port`` - neither in the allowlist nor a
        # field of the dataclass, so it must surface loudly (Learnings #86).
        with pytest.raises(ValueError, match=r"Unknown kwarg\(s\).*prot"):
            hw._create_minimal_config("so101_follower", None, prot="/dev/ttyACM0")

    def test_unsupported_robot_type_lists_known_choices(self):
        hw = _make_robot()
        with pytest.raises(ValueError) as excinfo:
            hw._create_minimal_config("not_a_real_robot", None)
        msg = str(excinfo.value)
        assert "Unsupported robot type: 'not_a_real_robot'" in msg
        # The error is actionable: it enumerates the known lerobot types.
        assert "so101_follower" in msg

    def test_cross_robot_allowlist_kwarg_absent_from_dataclass_is_dropped(self):
        hw = _make_robot()
        # ``kp`` is in the cross-robot allowlist but not a field of the
        # SO-follower config - the documented polymorphism carve-out: dropped,
        # never raised (so a heterogeneous-fleet call does not fail).
        cfg = hw._create_minimal_config("so101_follower", None, port="/dev/ttyACM0", kp=[1.0, 2.0])
        assert not hasattr(cfg, "kp")
        assert cfg.port == "/dev/ttyACM0"

    def test_dataclass_field_outside_allowlist_is_forwarded(self):
        hw = _make_robot()
        # ``side`` is a real field of the hope-jr hand config but not in the
        # cross-robot allowlist - it must still forward (new-lerobot-field
        # future-proofing), no strands release required.
        cfg = hw._create_minimal_config("hope_jr_hand", None, port="/dev/ttyACM0", side="left")
        assert cfg.side == "left"

    def test_explicit_id_overrides_tool_name_default(self):
        hw = _make_robot()
        cfg = hw._create_minimal_config("so101_follower", None, port="/dev/ttyACM0", id="left_arm")
        assert cfg.id == "left_arm"


class TestInitializeRobotDispatch:
    def test_prebuilt_driver_instance_passes_through(self):
        from lerobot.robots.robot import Robot as LeRobotRobot

        # A concrete lerobot Robot subclass. The abstract-method set is cleared
        # because this test only asserts the isinstance passthrough (identity),
        # not any driver behavior - no hardware method is ever called.
        class _FakeDriver(LeRobotRobot):
            pass

        _FakeDriver.__abstractmethods__ = frozenset()

        hw = _make_robot()
        driver = _FakeDriver.__new__(_FakeDriver)
        assert hw._initialize_robot(driver, None) is driver

    def test_robot_config_goes_through_lerobot_factory(self, monkeypatch):
        import lerobot.robots.utils as lru
        from lerobot.robots.config import RobotConfig

        hw = _make_robot()
        cfg = hw._create_minimal_config("so101_follower", None, port="/dev/ttyACM0")
        assert isinstance(cfg, RobotConfig)

        sentinel = object()
        seen: dict[str, Any] = {}

        def _fake_make(config):
            seen["config"] = config
            return sentinel

        monkeypatch.setattr(lru, "make_robot_from_config", _fake_make)
        assert hw._initialize_robot(cfg, None) is sentinel
        assert seen["config"] is cfg

    def test_type_string_builds_config_then_factory(self, monkeypatch):
        import lerobot.robots.utils as lru
        from lerobot.robots.config import RobotConfig

        hw = _make_robot()
        sentinel = object()
        seen: dict[str, Any] = {}

        def _fake_make(config):
            seen["config"] = config
            return sentinel

        monkeypatch.setattr(lru, "make_robot_from_config", _fake_make)
        result = hw._initialize_robot("so101_follower", None, port="/dev/ttyACM0")
        assert result is sentinel
        # The string was resolved into a concrete lerobot RobotConfig first.
        assert isinstance(seen["config"], RobotConfig)
        assert seen["config"].port == "/dev/ttyACM0"

    def test_unsupported_object_type_is_rejected(self):
        hw = _make_robot()
        with pytest.raises(ValueError, match="Unsupported robot type"):
            hw._initialize_robot(12345, None)
