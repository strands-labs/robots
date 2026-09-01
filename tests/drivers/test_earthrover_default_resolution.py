"""``Robot("earthrover", mode="real")`` stays lerobot-backed; the native driver is asked for.

The EarthRover is the first robot to have *both* a working lerobot robot class
and a native strands driver, and the two do not carry the same surface. So
"which one does the bare call get?" is a decision rather than a detail, and this
module grades the decision instead of restating it.

lerobot 0.6.1 ships ``lerobot/robots/earthrover_mini_plus/``, reached over the
same ``earth-rovers-sdk`` HTTP service the native driver speaks
(``RobotConfig.get_choice_class("earthrover_mini_plus")`` declares ``sdk_url``,
which :mod:`tests.test_robot_factory` already pins). Because a lerobot class
exists, the bare call resolves to
:class:`strands_robots.hardware_robot.Robot`, and three things ride on that
wrapper which the native driver does not carry:

* ``attach_teleop`` / ``teleoperate`` - :class:`~strands_robots.teleop_mixin.TeleopMixin`
  is mixed into the hardware wrapper and not into any native driver. README.md
  and ``docs/hardware/teleoperation.md`` both document these two reads on
  exactly this construction, so a flip makes four documented lines raise
  ``AttributeError``. ``tests/test_docs_robot_attribute_reads_resolve.py``
  catches that half.
* the *action vocabulary*. lerobot's ``action_features`` is
  ``{linear_velocity, angular_velocity}``; the native driver's
  :data:`~strands_robots.drivers.earthrover.DRIVE_CHANNELS` is
  ``{linear, angular, lamp}`` and it *refuses* an unknown channel rather than
  ignoring it. A flip therefore does not merely change which class answers - it
  changes which words a caller, a teleoperator and a recorded dataset must use,
  and the refusal arrives at the first frame. This is the half no docs grader
  can see, so it is pinned below.
* dataset recording. lerobot's ``observation_features`` declares the front/rear
  frames plus speed, battery, orientation and GPS; the native driver's
  ``get_observation()`` is ``{}`` by design (a wheeled base has no joints).

None of that makes the native driver worse - it adds the lamp, the second
camera, ``speak`` and an agent tool surface lerobot has no seam for. It makes
the *default* a question that is answered by ``driver="strands"`` at the call
site until the native driver carries the surface the documented lines read.
That keyword is the documented way to reach a driver for a robot whose registry
entry declares nothing, so the driver is fully reachable with the registry left
alone.

A future change that declares ``hardware.driver="strands"`` on this entry should
fail here, and the failure should say what else has to land with it.
"""

from __future__ import annotations

import pytest

from strands_robots.drivers import get_native_driver_class, resolve_driver
from strands_robots.drivers.earthrover import DRIVE_CHANNELS, EarthRoverDriver
from strands_robots.registry import get_robot, resolve_name

#: What lerobot's ``EarthRoverMiniPlus.action_features`` declares, and therefore
#: what ``keyboard_rover`` is documented as driving zero-config.
_LEROBOT_ACTION_KEYS = ("linear_velocity", "angular_velocity")

#: The two reads README.md and docs/hardware/teleoperation.md perform on
#: ``Robot("earthrover_mini_plus", mode="real")``.
_DOCUMENTED_TELEOP_READS = ("attach_teleop", "teleoperate")


class TestTheBareCallStaysLerobotBacked:
    """The registry declares no driver, so the documented reads keep resolving."""

    def test_the_registry_entry_declares_no_driver(self) -> None:
        """An absent ``hardware.driver`` is what defers to the default."""
        entry = get_robot(resolve_name("earthrover"))
        # Asserted rather than coalesced to ``{}``: the check below is an absence,
        # so a missing entry would satisfy it vacuously and report a default this
        # module never read.
        assert entry is not None, "premise: 'earthrover' no longer resolves to a registry entry"
        hardware = entry["hardware"]
        assert "driver" not in hardware, (
            "earthrover must not declare hardware.driver: the native driver does not carry "
            f"{list(_DOCUMENTED_TELEOP_READS)} (documented in README.md and "
            "docs/hardware/teleoperation.md) and refuses lerobot's action vocabulary "
            f"{list(_LEROBOT_ACTION_KEYS)} - see this module's docstring"
        )

    def test_the_default_resolution_is_lerobot(self) -> None:
        """The surface the documented teleop lines are written against."""
        assert resolve_driver(resolve_name("earthrover")) == "lerobot"

    def test_the_hardware_wrapper_carries_the_documented_teleop_reads(self) -> None:
        """Premise guard: the reads exist on what the default resolves to."""
        from strands_robots.hardware_robot import Robot as HardwareRobot

        for name in _DOCUMENTED_TELEOP_READS:
            assert hasattr(HardwareRobot, name), name

    def test_the_native_driver_does_not_carry_them(self) -> None:
        """The other half of the premise - why the flip would break the docs."""
        for name in _DOCUMENTED_TELEOP_READS:
            assert not hasattr(EarthRoverDriver, name), (
                f"EarthRoverDriver grew {name}; if the native driver now carries the "
                "documented teleop surface, revisit the default and the action "
                "vocabulary cell below together"
            )


class TestTheNativeDriverIsReachableAnyway:
    """Leaving the registry alone does not stall the driver."""

    def test_the_driver_is_registered_for_the_robot(self) -> None:
        assert get_native_driver_class(resolve_name("earthrover")) is EarthRoverDriver

    def test_an_explicit_choice_reaches_it(self) -> None:
        """``driver="strands"`` beats an absent registry declaration."""
        assert resolve_driver(resolve_name("earthrover"), "strands") == "strands"


class TestTheVocabulariesDoNotAgree:
    """The half no docs grader can see: a flip changes which words work.

    Driven against a connected driver with a recording stand-in for the wire, so
    the refusal is the driver's own judgement of the frame rather than a
    not-connected gate answering first.
    """

    @staticmethod
    def _connected() -> tuple[EarthRoverDriver, list[dict[str, object]]]:
        """A driver that reports connected, with every POST recorded, not sent."""
        posted: list[dict[str, object]] = []

        class _Response:
            status_code = 200
            text = "{}"

        class _Session:
            def post(self, url: str, **kwargs: object) -> _Response:
                posted.append({"url": url, **kwargs})
                return _Response()

        driver = EarthRoverDriver()
        driver._session = _Session()  # type: ignore[assignment]
        driver._connected = True
        return driver, posted

    def test_the_native_channels_are_not_lerobots(self) -> None:
        """Stated as a set relation so a rename on either side is caught."""
        assert not set(_LEROBOT_ACTION_KEYS) & set(DRIVE_CHANNELS), (
            f"native DRIVE_CHANNELS {list(DRIVE_CHANNELS)} now overlaps lerobot's "
            f"action_features {list(_LEROBOT_ACTION_KEYS)}; the vocabulary argument "
            "against flipping the default may no longer hold"
        )

    def test_a_native_twist_is_accepted(self) -> None:
        """Premise guard: the refusal below is about the keys, not the wire."""
        driver, posted = self._connected()
        assert driver.send_action({"linear": 0.5, "angular": 0.0})["status"] == "success"
        assert len(posted) == 1

    @pytest.mark.parametrize("key", _LEROBOT_ACTION_KEYS)
    def test_a_lerobot_action_key_is_refused_before_the_wire(self, key: str) -> None:
        """What a flipped default would do to a documented ``keyboard_rover`` frame.

        ``docs/hardware/teleoperation.md`` records this pairing as
        ``identity`` - zero-config - and it is, against lerobot's robot. Against
        the native driver the same frame is refused, and nothing is posted, so a
        flip would strand the documented recipe at its first tick rather than
        drive the rover with a partial twist.
        """
        driver, posted = self._connected()
        result = driver.send_action({key: 0.5})
        assert result["status"] == "error", result
        reason = result["content"][0]["text"]
        assert key in reason and "unknown drive channel" in reason, reason
        assert posted == [], "a refused frame must not reach the wire"

    def test_the_whole_documented_frame_is_refused(self) -> None:
        """Both keys at once - the shape ``keyboard_rover`` actually emits."""
        driver, posted = self._connected()
        result = driver.send_action(dict.fromkeys(_LEROBOT_ACTION_KEYS, 0.5))
        assert result["status"] == "error", result
        assert posted == []
