# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin: ``write_goal_positions`` refuses non-numeric and bool targets.

``float(value)`` on a non-numeric value (``None``, a list, a string that is
not a number) raises ``TypeError`` rather than ``ValueError``, escaping the
``send_action`` except tuple and violating the driver's never-raises
contract. A ``bool`` is an ``int`` subclass and ``float(True)`` silently
commands 1.0, so ``{"gripper": true}`` energizes a motor at 1 percent
without a refusal -- the shared numeric domains in
:mod:`strands_robots.utils` reject ``bool`` for exactly this reason.

The guard validates the value before ``float()`` is called, so none of these
reach the bus at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.drivers.feetech.bus import FeetechBus
from strands_robots.drivers.feetech.driver import FeetechDriver
from tests.drivers.conftest import FakeServoPort


def _open_bus() -> FeetechBus:
    """A bus already holding a fake port, as if connected."""
    bus = FeetechBus(port="/dev/fake")  # type: ignore[arg-type]
    bus._conn = FakeServoPort(dict.fromkeys((1, 2, 3, 4, 5, 6), 2048))
    return bus


def _wired(**kwargs: Any) -> FeetechDriver:
    """A driver whose bus already holds a fake port, as if connected."""
    driver = FeetechDriver(tool_name="so101", port="/dev/fake", **kwargs)
    driver.bus._conn = FakeServoPort(dict.fromkeys((1, 2, 3, 4, 5, 6), 2048))
    return driver


class TestNonNumericTargetsAreRefused:
    """A value that ``float()`` would raise ``TypeError`` on is caught
    as ``ValueError`` before it reaches the bus."""

    @pytest.mark.parametrize(
        "value",
        [
            None,
            [0.0],
            (45.0,),
            "90",
            {"deg": 45},
            object(),
        ],
        ids=["None", "list", "tuple", "string", "dict", "object"],
    )
    def test_non_numeric_value_raises_value_error(self, value: object) -> None:
        bus = _open_bus()
        with pytest.raises(ValueError, match="must be a finite number"):
            bus.write_goal_positions({"shoulder_pan": value})  # type: ignore[dict-item]


class TestBoolTargetsAreRefused:
    """A ``bool`` is an ``int`` subclass; ``float(True)`` is 1.0. The
    guard refuses it so ``{"gripper": true}`` does not silently command
    1.0 percent."""

    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_bool_raises_value_error(self, value: object) -> None:
        bus = _open_bus()
        with pytest.raises(ValueError, match="must be a finite number"):
            bus.write_goal_positions({"gripper": value})  # type: ignore[dict-item]


class TestAcceptedValuesAreUnchanged:
    """Positive controls: the values the driver was already accepting
    must still be accepted after the guard is added."""

    @pytest.mark.parametrize(
        "value",
        [
            45.0,
            45,
            np.float32(45.0),
            np.float64(45.0),
            np.int64(45),
            np.int32(45),
        ],
        ids=["float", "int", "np.float32", "np.float64", "np.int64", "np.int32"],
    )
    def test_real_scalars_are_accepted(self, value: object) -> None:
        bus = _open_bus()
        # Should not raise -- the value is real and finite.
        bus.write_goal_positions({"shoulder_pan": value})  # type: ignore[dict-item]


class TestDriverSendActionNeverRaises:
    """The driver's ``send_action`` catches the ``ValueError`` the bus
    raises, so a non-numeric target returns an error envelope rather
    than escaping."""

    def test_none_target_returns_error_envelope(self) -> None:
        driver = _wired()
        result = driver.send_action({"gripper": None})
        assert result["status"] == "error"
        assert "must be a finite number" in result["content"][0]["text"]

    def test_list_target_returns_error_envelope(self) -> None:
        driver = _wired()
        result = driver.send_action({"gripper": [0.0]})
        assert result["status"] == "error"
        assert "must be a finite number" in result["content"][0]["text"]

    def test_bool_target_returns_error_envelope(self) -> None:
        driver = _wired()
        result = driver.send_action({"gripper": True})
        assert result["status"] == "error"
        assert "must be a finite number" in result["content"][0]["text"]
