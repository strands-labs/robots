"""The 2F-85 driver's command path, graded against a scripted gripper.

Every case here runs over a real TCP socket to a fake that enforces the
gripper's own preconditions (see :class:`FakeGripper`): a position lands only
when the gripper is activated and the frame set ``rGTO``. So these are
behaviour pins, not envelope pins - a driver that built a perfect frame but
skipped activation would produce the same envelopes and move nothing, and the
assertions read what reached the wire.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import pytest

from strands_robots import Robot
from strands_robots.drivers.robotiq import STROKE_MM, RobotiqDriver
from strands_robots.drivers.robotiq.protocol import ObjectStatus
from tests.drivers.robotiq.conftest import FakeGripper

if TYPE_CHECKING:
    from strands_robots.policies import Policy

Connected = Callable[..., tuple[RobotiqDriver, FakeGripper]]


def test_connecting_activates_the_gripper_before_reporting_success(connected: Connected) -> None:
    """An unactivated 2F-85 ignores positions, so connecting must activate it.

    The gripper is the authority: it only records a commanded position once
    ``gSTA`` reads ACTIVE. If ``connect_eagerly`` returned on an open socket
    alone, the first ``send_action`` would be dropped and this would see an
    empty ``commanded``.
    """
    driver, fake = connected()

    assert driver.is_connected
    assert fake.activation == 3, "gripper should have finished its calibration stroke"
    # The manual's sequence: clear rACT, then set it. Both frames must be sent,
    # because a gripper holding a fault will not activate without the reset.
    assert [write["activate"] for write in fake.writes[:2]] == [0, 1]

    assert driver.send_action({"gripper": 1.0})["status"] == "success"
    assert fake.commanded == [255], "the commanded position did not reach the gripper"


def test_an_already_activated_gripper_is_not_reset(connected: Connected) -> None:
    """Re-activating costs a calibration stroke, so a live gripper is left alone."""
    driver, fake = connected(starts_activated=True)

    assert driver.is_connected
    assert fake.writes == [], "an activated gripper needs no activation frames"


def test_activation_that_never_completes_is_reported_not_hidden(gripper: Callable[..., FakeGripper]) -> None:
    """A gripper stuck mid-calibration yields a reason naming the state."""
    fake = gripper(never_activates=True)
    driver = RobotiqDriver(port="127.0.0.1", tcp_port=fake.port, timeout=1.0, activation_timeout=0.3)

    reason = driver.connect_eagerly()

    assert reason is not None
    assert "did not finish activating" in reason
    assert "ACTIVATING" in reason, f"the reason should name the state it was stuck in: {reason}"
    assert not driver.is_connected


@pytest.mark.parametrize(
    ("action", "expected_counts", "expected_mm"),
    [
        # The direction pin. rPR runs backwards to aperture, so a sign error
        # here closes a gripper that was asked to open - and it would still
        # look like a working driver.
        ({"gripper": 0.0}, 0, STROKE_MM),
        ({"gripper": 1.0}, 255, 0.0),
        ({"gripper": 0.5}, 128, pytest.approx(42.3, abs=0.3)),
        # The same command in the spelling a policy action dict uses.
        ({"gripper.pos": 0.0}, 0, STROKE_MM),
        ({"gripper.pos": 1.0}, 255, 0.0),
        # Millimetres of fingertip aperture, both spellings.
        ({"position": STROKE_MM}, 0, STROKE_MM),
        ({"position": 0.0}, 255, 0.0),
        ({"aperture_mm": 40.0}, 135, pytest.approx(40.0, abs=0.4)),
        # Out of stroke means "as far as it goes", not a refusal.
        ({"aperture_mm": 500.0}, 0, STROKE_MM),
        ({"gripper": -3.0}, 0, STROKE_MM),
        ({"gripper": 7.0}, 255, 0.0),
    ],
)
def test_an_aperture_command_lands_the_position_it_names(
    connected: Connected,
    action: dict[str, float],
    expected_counts: int,
    expected_mm: float,
) -> None:
    """Each accepted spelling reaches the gripper as the position it means."""
    driver, fake = connected()

    envelope = driver.send_action(action)

    assert envelope["status"] == "success", envelope
    reported = envelope["content"][0]["json"]
    assert reported["position"] == expected_counts
    assert reported["aperture_mm"] == expected_mm
    assert fake.commanded == [expected_counts], "the gripper received a different position"
    assert fake.position == expected_counts


def test_the_two_command_spellings_agree_on_the_wire(connected: Connected) -> None:
    """A closed fraction and the millimetres it equals produce one position.

    Pinned because the two paths convert through different functions, and a
    caller switching spelling must not silently change where the fingers go.
    """
    driver, fake = connected()

    driver.send_action({"gripper": 1.0})
    driver.send_action({"aperture_mm": 0.0})

    assert fake.commanded[0] == fake.commanded[1] == 255


def test_speed_and_force_scale_from_a_fraction(connected: Connected) -> None:
    """Per-command speed and force reach the gripper as counts."""
    driver, fake = connected()

    driver.send_action({"gripper": 1.0, "speed": 0.0, "force": 0.5})

    assert fake.speed == 0
    assert fake.force == 128


def test_a_grasp_is_distinguished_from_an_empty_close(connected: Connected) -> None:
    """``gOBJ`` says whether the fingers stopped on an object or on the target.

    Reading "stopped" without reading which kind reports every empty close as a
    successful pick, which is the difference between a grasp check that works
    and one that always says yes.
    """
    holding_driver, _holding = connected(object_status=ObjectStatus.CONTACT_CLOSING)
    empty_driver, _empty = connected(object_status=ObjectStatus.AT_REQUEST)

    holding = holding_driver.read_status()["content"][0]["json"]
    empty = empty_driver.read_status()["content"][0]["json"]

    assert holding["object"] == "CONTACT_CLOSING"
    assert holding["holding"] is True
    assert empty["object"] == "AT_REQUEST"
    assert empty["holding"] is False


def test_the_observation_reports_the_one_joint_a_gripper_has(connected: Connected) -> None:
    """``get_observation`` is what puts the gripper on the mesh state topic.

    The driver owns its bus, so it answers the joint read itself rather than
    through an inner device - see
    :func:`strands_robots.bus_access.joint_read_source`.
    """
    from strands_robots.bus_access import joint_read_source

    driver, _fake = connected()
    driver.send_action({"gripper": 1.0})

    assert joint_read_source(driver) is driver, "the mesh must resolve this driver as its own joint source"
    assert driver.get_observation() == {"gripper.pos": 1.0}

    driver.send_action({"gripper": 0.0})
    assert driver.get_observation() == {"gripper.pos": 0.0}


def test_a_disconnected_gripper_reports_no_joints_rather_than_failing() -> None:
    """ "No joints" is not an error; the mesh publishes no section for it."""
    driver = RobotiqDriver(port="127.0.0.1", tcp_port=1)

    assert driver.get_observation() == {}


@pytest.mark.parametrize(
    ("action", "robot_name", "expected"),
    [
        ({"gripper": 0.0, "position": 10.0}, None, "two spellings of the same command"),
        ({"wrist_flex": 0.2}, None, "none of ['wrist_flex'] names an aperture"),
        ({}, None, "nothing to command"),
        ({"gripper": float("nan")}, None, "gripper"),
        ({"gripper": "shut"}, None, "gripper"),
        ({"gripper": 0.0}, "so101", "fronts 'robotiq_2f85' only"),
    ],
)
def test_a_command_it_cannot_honour_is_refused_in_an_envelope(
    connected: Connected,
    action: dict[str, object],
    robot_name: str | None,
    expected: str,
) -> None:
    """A bad action is an error envelope, never an exception past the boundary.

    The mesh command path and the agent tool dispatch both read the envelope, so
    a raise here would escape as a traceback where a reason was expected.
    """
    driver, fake = connected()

    envelope = driver.send_action(action, robot_name=robot_name)

    assert envelope["status"] == "error", envelope
    assert expected in envelope["content"][0]["text"]
    assert fake.commanded == [], "a refused command must not reach the gripper"


def test_a_modbus_exception_reply_becomes_a_reason(gripper: Callable[..., FakeGripper]) -> None:
    """A gripper refusing the write is reported with its exception code."""
    fake = gripper(exception_code=0x02, starts_activated=True)
    driver = RobotiqDriver(port="127.0.0.1", tcp_port=fake.port, timeout=1.0)

    reason = driver.connect_eagerly()

    assert reason is not None
    assert "0x02" in reason, reason


def test_stopping_halts_the_fingers_without_dropping_activation(connected: Connected) -> None:
    """Clearing ``rGTO`` stops motion; clearing ``rACT`` would cost a re-stroke."""
    driver, fake = connected()

    envelope = driver.stop_task()

    assert envelope["status"] == "success"
    assert fake.writes[-1]["go_to"] == 0, "rGTO must be cleared to halt"
    assert fake.writes[-1]["activate"] == 1, "rACT must stay set to keep the gripper activated"
    assert fake.activation == 3


@pytest.mark.parametrize("verb", ["start_task", "run_policy"])
def test_a_gripper_refuses_a_policy_rollout_and_names_the_path_that_works(verb: str) -> None:
    """A 1-DOF end effector has no rollout, so the refusal points at send_action.

    A refusal that only said "unsupported" would leave a caller guessing; naming
    the working path is what makes it actionable.
    """
    driver = RobotiqDriver()

    envelope = (
        driver.start_task("pick up the cube") if verb == "start_task" else driver.run_policy(cast("Policy", object()))
    )

    assert envelope["status"] == "error"
    text = envelope["content"][0]["text"]
    assert verb in text
    assert "send_action" in text


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"timeout": 0}, "timeout"),
        ({"timeout": float("nan")}, "timeout"),
        ({"activation_timeout": -1.0}, "activation_timeout"),
        ({"stroke_mm": 0.0}, "stroke_mm"),
        ({"tcp_port": 0}, "tcp_port"),
        ({"tcp_port": "502"}, "tcp_port"),
        ({"unit_id": 300}, "unit_id"),
        ({"unit_id": True}, "unit_id"),
        ({"speed": 1.5}, "speed"),
        ({"force": float("inf")}, "force"),
    ],
)
def test_a_knob_the_transport_cannot_use_is_refused_at_construction(kwargs: dict[str, object], expected: str) -> None:
    """Each of these reaches a consumer that cannot report what it was handed.

    Raised from ``__init__`` rather than returned from ``connect_eagerly``,
    which is declared ``-> str | None``: a value the socket cannot use is not a
    connection the driver can degrade to reporting.
    """
    with pytest.raises(ValueError, match=expected):
        RobotiqDriver(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["robotiq_2f85", "robotiq_2f85_v4", "robotiq"])
def test_real_mode_reaches_this_driver_without_naming_it(name: str, gripper: Callable[..., FakeGripper]) -> None:
    """``mode="real"`` alone resolves here, which is what the registry buys.

    The entries declare ``hardware.driver = "strands"`` because lerobot has no
    robot type for a gripper, so the default resolution must land here rather
    than on a lerobot driver that cannot build it. Without that declaration a
    caller would need ``driver="strands"`` and the documented call would fail.
    """
    fake = gripper(starts_activated=True)
    built = Robot(name, mode="real", port="127.0.0.1", tcp_port=fake.port, timeout=1.0)

    assert isinstance(built, RobotiqDriver)
    assert built.connect_eagerly() is None
    assert built.send_action({"gripper": 1.0})["status"] == "success"
    assert fake.commanded == [255]
    built.cleanup()


# ---------- regression: intra-category duplicate keys ----------


class TestSendActionRefusesIntraCategoryDuplicateKeys:
    """Pin the guard widened to ``len(normalised) + len(aperture) > 1``.

    Two keys inside *one* category (e.g. ``{"gripper": 0.0, "gripper.pos": 1.0}``)
    are contradictory.  Before the fix, the guard checked only for cross-category
    conflicts and the first tuple element won silently -- a close command could be
    dropped while the driver reported success.
    """

    @pytest.mark.parametrize(
        "action",
        [
            pytest.param({"gripper": 0.0, "gripper.pos": 1.0}, id="two-normalised"),
            pytest.param({"position": 85.0, "aperture_mm": 0.0}, id="two-aperture"),
        ],
    )
    def test_intra_category_duplicate_keys_are_refused(self, connected: Connected, action: dict[str, float]) -> None:
        driver, _fake = connected()
        result = driver.send_action(action)
        assert result["status"] == "error", (
            f"Intra-category duplicate keys {sorted(action)} must be refused; got {result}"
        )
        assert "two spellings" in result["content"][0]["text"].lower()
