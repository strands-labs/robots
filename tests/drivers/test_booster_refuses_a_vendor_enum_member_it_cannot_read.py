# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A T1 verb refuses a vendor enum member the installed SDK lacks.

``booster_robotics_sdk_python`` is a vendor wheel pinned to the robot's
firmware, not a dependency this project resolves, so the *installed* build's
vocabulary is an input. MODULE ``strands_robots.drivers.booster`` freezes two
claims about that vocabulary - ``ROBOT_MODES`` and the keys of
``CMD_TYPE_STATE_FIELD`` - and both are handed straight to an SDK enum. Read
with a bare ``getattr`` a build that lacks one raises :exc:`AttributeError` out
of the verb, past the ``{"status": "error", ...}`` envelope every driver verb is
contracted to return: the agent sees a traceback instead of a reason, and the
reason it needed ("this SDK build declares no ``RobotMode.kSoccer``") is exactly
the one nothing else can tell it.

Why no cell here could previously see that: the SDK double in
MODULE ``tests.drivers.test_booster_driver`` declares the full vendor vocabulary,
so its ``RobotMode`` cannot disagree with ``ROBOT_MODES`` and the disagreement
has no shape to occur in. The doubles below take their vocabulary as a
*parameter*, which is what makes the two claims falsifiable.

Three kinds of cell, kept apart so they are not read for each other:

* **Refusal cells** drive a verb against a build missing the member. These fail
  on the pre-fix tree with :exc:`AttributeError`.
* **Control cells** drive the same verb against a build that has it, so the
  refusal is shown to discriminate rather than to refuse everything. These pass
  either way.
* **A closure cell** holds the whole package to zero two-argument ``getattr``
  calls on an SDK module, so the class is closed rather than the two sites
  patched.

The vendor-truth cell that grades ``ROBOT_MODES`` against a real wheel lives
with its siblings in ``TestTheVendorNumbersAreTheVendors`` -
MODULE ``tests.drivers.test_booster_driver`` - because that is where the
``JointIndex`` and ``FallDownStateType`` vocabularies are already graded.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types
from typing import Any

import pytest

import strands_robots
from strands_robots.drivers.booster import (
    BOOSTER_JOINT_INDEX,
    CMD_TYPE_STATE_FIELD,
    ROBOT_MODES,
    BoosterDriver,
    declared_members,
    resolve_vendor_member,
)

_WIDTH = len(BOOSTER_JOINT_INDEX)

#: The full ``RobotMode`` vocabulary of SDK 1.6.1, values included: every mode
#: the driver offers plus the ``kUnknown`` only ``GetMode`` reports.
VENDOR_ROBOT_MODES: dict[str, int] = {
    "kUnknown": -1,
    "kDamping": 0,
    "kPrepare": 1,
    "kWalking": 2,
    "kCustom": 3,
    "kSoccer": 4,
}

#: The full ``LowCmdType`` vocabulary, at the vendor's own values.
VENDOR_CMD_TYPES: dict[str, int] = {"PARALLEL": 0, "SERIAL": 1}


# --------------------------------------------------------------------------- #
# An SDK double whose vocabulary is a parameter.                              #
# --------------------------------------------------------------------------- #


def _enum(name: str, members: dict[str, int]) -> Any:
    """Build a stand-in for one pybind11 enum type declaring ``members``.

    Carries the ``name``/``value`` descriptors a real pybind11 enum type carries
    so :func:`declared_members` is exercised against the same noise it filters
    on hardware. Typed ``Any``: the members are built at runtime, which is the
    whole point, so a static type could not name them.
    """
    return type(name, (), {**members, "name": property(lambda self: ""), "value": property(lambda self: 0)})


class _MotorCmd:
    """One ``MotorCmd`` slot, recording what was written into it."""

    def __init__(self) -> None:
        self.q = self.dq = self.tau = self.kp = self.kd = 0.0
        self.mode = 0


class _LowCmd:
    """A ``LowCmd`` frame the publisher double can read back."""

    def __init__(self) -> None:
        self.cmd_type: Any = None
        self._motors: list[_MotorCmd] = []

    def resize_motor_cmd(self, count: int) -> None:
        self._motors = [_MotorCmd() for _ in range(count)]

    def motor_cmd_at(self, slot: int) -> _MotorCmd:
        return self._motors[slot]


class _Publisher:
    """``B1LowCmdPublisher``, recording every accepted frame."""

    def __init__(self) -> None:
        self.written: list[_LowCmd] = []

    def Write(self, cmd: _LowCmd) -> bool:  # noqa: N802 - the vendor's spelling
        self.written.append(cmd)
        return True


class _LocoClient:
    """``B1LocoClient``, recording the mode it was asked for."""

    def __init__(self) -> None:
        self.modes: list[Any] = []

    def ChangeMode(self, mode: Any) -> None:  # noqa: N802 - the vendor's spelling
        self.modes.append(mode)


def _install_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    robot_modes: dict[str, int] | None = None,
    cmd_types: dict[str, int] | None = None,
) -> Any:
    """Install an SDK double declaring exactly the vocabulary asked for."""
    sdk: Any = types.ModuleType("booster_robotics_sdk_python")
    sdk.RobotMode = _enum("RobotMode", VENDOR_ROBOT_MODES if robot_modes is None else robot_modes)
    sdk.LowCmdType = _enum("LowCmdType", VENDOR_CMD_TYPES if cmd_types is None else cmd_types)
    sdk.LowCmd = _LowCmd
    monkeypatch.setitem(sys.modules, "booster_robotics_sdk_python", sdk)
    return sdk


def _mode_driver() -> tuple[BoosterDriver, _LocoClient]:
    """A driver holding a loco client, which is all ``change_mode`` needs."""
    driver = BoosterDriver()
    client = _LocoClient()
    driver._client = client
    return driver, client


def _write_driver(cmd_type: str) -> tuple[BoosterDriver, _Publisher]:
    """A driver past every ``send_action`` gate, so only the enum is left."""
    driver = BoosterDriver(cmd_type=cmd_type)
    publisher = _Publisher()
    driver._client = _LocoClient()
    driver._publisher = publisher
    driver._connected = True
    driver._upper_body_enabled = True
    driver._fall_state = "IS_READY"
    driver._last_state = {"joints": [0.0] * _WIDTH}
    return driver, publisher


def _reason(result: dict[str, Any]) -> str:
    """The refusal text of an error envelope."""
    assert result["status"] == "error", result
    return str(result["content"][0]["text"])


# --------------------------------------------------------------------------- #
# The refusal, and that it discriminates.                                     #
# --------------------------------------------------------------------------- #


class TestAModeTheBuildLacksIsRefusedRatherThanRaised:
    """``change_mode`` answers an envelope for every mode it offers."""

    @pytest.mark.parametrize("missing", ROBOT_MODES)
    def test_a_dropped_mode_is_a_reason_not_a_traceback(self, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
        """Whichever of the five a build drops, the verb still returns."""
        _install_sdk(monkeypatch, robot_modes={n: v for n, v in VENDOR_ROBOT_MODES.items() if n != missing})
        driver, client = _mode_driver()

        result = driver.change_mode(missing)

        assert client.modes == [], "a mode that could not be resolved must not reach the robot"
        reason = _reason(result)
        assert f"RobotMode.{missing}" in reason, reason
        assert "booster_robotics_sdk_python" in reason, reason

    def test_the_reason_names_the_vocabulary_the_build_does_have(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An operator has to be able to pick a mode from the refusal alone."""
        _install_sdk(monkeypatch, robot_modes={n: v for n, v in VENDOR_ROBOT_MODES.items() if n != "kSoccer"})
        driver, _ = _mode_driver()

        reason = _reason(driver.change_mode("kSoccer"))

        for present in ("kDamping", "kPrepare", "kWalking", "kCustom"):
            assert present in reason, f"{present} is installed and the refusal does not name it: {reason}"
        assert "SDK version mismatch" in reason, reason

    @pytest.mark.parametrize("mode", ROBOT_MODES)
    def test_a_build_that_declares_the_mode_still_reaches_the_robot(
        self, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        """The control: the guard refuses a missing member, not every member."""
        sdk = _install_sdk(monkeypatch)
        driver, client = _mode_driver()

        result = driver.change_mode(mode)

        assert result["status"] == "success", result
        assert result["content"][0]["json"] == {"mode": mode}
        assert client.modes == [getattr(sdk.RobotMode, mode)], "the vendor member itself must be handed over"

    def test_a_mode_outside_the_offered_set_is_refused_before_the_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``kUnknown`` is declared by the vendor and not offered by the driver.

        ``read_mode`` passes the vendor's name through verbatim, so a T1 can
        *report* ``kUnknown``; asking to be put into it is meaningless. The two
        refusals are distinct and the declared-set one comes first.
        """
        _install_sdk(monkeypatch)
        driver, client = _mode_driver()

        reason = _reason(driver.change_mode("kUnknown"))

        assert "mode must be one of" in reason, reason
        assert "booster_robotics_sdk_python" not in reason, "the SDK was consulted for a mode never offered"
        assert client.modes == []


class TestACmdTypeTheBuildLacksIsRefusedBeforeAFrameIsPublished:
    """``send_action``'s convention literal is read off the enum too."""

    @pytest.mark.parametrize("cmd_type", sorted(CMD_TYPE_STATE_FIELD))
    def test_a_dropped_convention_publishes_nothing_and_says_why(
        self, monkeypatch: pytest.MonkeyPatch, cmd_type: str
    ) -> None:
        """A frame is not published on a guess about the wire convention."""
        member = cmd_type.upper()
        _install_sdk(monkeypatch, cmd_types={n: v for n, v in VENDOR_CMD_TYPES.items() if n != member})
        driver, publisher = _write_driver(cmd_type)

        result = driver.send_action({"left_shoulder_pitch": 0.1})

        assert publisher.written == [], "a frame whose cmd_type is unresolved must not reach the wire"
        reason = _reason(result)
        assert f"LowCmdType.{member}" in reason, reason
        assert "send_action" in reason, reason

    @pytest.mark.parametrize("cmd_type", sorted(CMD_TYPE_STATE_FIELD))
    def test_a_build_that_declares_the_convention_still_publishes(
        self, monkeypatch: pytest.MonkeyPatch, cmd_type: str
    ) -> None:
        """The control, and it also pins which member lands on the frame."""
        sdk = _install_sdk(monkeypatch)
        driver, publisher = _write_driver(cmd_type)

        result = driver.send_action({"left_shoulder_pitch": 0.1})

        assert result["status"] == "success", result
        assert len(publisher.written) == 1
        assert publisher.written[0].cmd_type == getattr(sdk.LowCmdType, cmd_type.upper())


# --------------------------------------------------------------------------- #
# The resolver's own contract.                                                #
# --------------------------------------------------------------------------- #


class TestTheResolverReportsTheBuildRatherThanTheClaim:
    """:func:`resolve_vendor_member` and the vocabulary it reads."""

    def test_a_member_is_returned_and_a_reason_is_a_string(self) -> None:
        """The caller's ``isinstance(result, str)`` check needs both halves."""
        enum = _enum("RobotMode", VENDOR_ROBOT_MODES)

        found = resolve_vendor_member(enum, "kCustom", enum_name="RobotMode", verb="change_mode")
        absent = resolve_vendor_member(enum, "kFlying", enum_name="RobotMode", verb="change_mode")

        assert found is enum.kCustom
        assert not isinstance(found, str), "a member that is a str would be read as a refusal"
        assert isinstance(absent, str)

    def test_a_member_whose_value_is_falsy_is_still_found(self) -> None:
        """``kDamping`` is ``0``; a truthiness test on the member loses it."""
        enum = _enum("RobotMode", VENDOR_ROBOT_MODES)

        assert resolve_vendor_member(enum, "kDamping", enum_name="RobotMode", verb="change_mode") == 0

    def test_the_reported_vocabulary_is_the_members_and_not_the_bindings(self) -> None:
        """``name``/``value`` are per-member descriptors, not modes."""
        enum = _enum("RobotMode", VENDOR_ROBOT_MODES)

        assert declared_members(enum) == tuple(sorted(VENDOR_ROBOT_MODES))

    def test_an_enum_declaring_nothing_still_produces_a_readable_reason(self) -> None:
        """A binding that failed to populate must not render an empty list."""
        reason = resolve_vendor_member(_enum("RobotMode", {}), "kCustom", enum_name="RobotMode", verb="change_mode")

        assert isinstance(reason, str)
        assert "(no members)" in reason


# --------------------------------------------------------------------------- #
# The class is closed, not the two sites patched.                             #
# --------------------------------------------------------------------------- #


class TestNoVendorLookupIsLeftUnguarded:
    """No module reads an SDK attribute it has not established is there.

    A two-argument ``getattr`` on an SDK module is the shape this file is about:
    it has no default, so an absent attribute raises out of whatever verb is
    holding it. Graded over the whole package rather than over the two sites
    fixed here, so a third driver cannot reintroduce it silently.
    """

    @staticmethod
    def _unguarded_lookups(tree: ast.Module) -> list[str]:
        """Two-argument ``getattr`` calls whose base is an imported SDK alias."""
        aliases = {
            alias.asname or alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if "sdk" in alias.name.lower() or "sdk" in (alias.asname or "").lower()
        }
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) != 2:
                continue
            if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
                continue
            base: ast.expr = node.args[0]
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in aliases:
                found.append(f"line {node.lineno}: {ast.unparse(node)}")
        return found

    def test_the_package_reads_no_sdk_attribute_without_a_default(self) -> None:
        offenders = {}
        root = pathlib.Path(strands_robots.__file__).parent
        for path in sorted(root.rglob("*.py")):
            lookups = self._unguarded_lookups(ast.parse(path.read_text(encoding="utf-8")))
            if lookups:
                offenders[str(path.relative_to(root))] = lookups

        assert not offenders, (
            f"a vendor SDK attribute is read with no default, so an SDK build that lacks it raises "
            f"out of the verb instead of refusing: {offenders}. Resolve it through "
            "resolve_vendor_member (or an equivalent that returns a reason)."
        )

    def test_the_rule_sees_the_shape_it_is_named_for(self) -> None:
        """The exemplar: the pre-fix spelling of both sites fixed here."""
        source = (
            "import booster_robotics_sdk_python as sdk\n"
            "cmd.cmd_type = getattr(sdk.LowCmdType, self._cmd_type.upper())\n"
            "client.ChangeMode(getattr(sdk.RobotMode, mode))\n"
            "safe = getattr(sdk.RobotMode, mode, None)\n"
            "unrelated = getattr(self, mode)\n"
        )

        found = self._unguarded_lookups(ast.parse(source))

        assert len(found) == 2, found
        assert "LowCmdType" in found[0] and "RobotMode" in found[1]
