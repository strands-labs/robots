# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bound ``serial_tool`` documents must be the bound ``serial_tool`` enforces.

``serial_tool`` is an agent tool, so the ``Args:`` entry for each register field
is not commentary: ``docstring_parser`` lifts it into the tool's input schema and
that schema is the whole of what the model driving the tool knows about the
field's domain. A bound restated in prose beside a bound decided in code is two
copies of one fact, and nothing compared them.

Measured on ``de9762a``, one of the three had already drifted::

    field     documented        enforced
    motor_id  [1, 254]          [1, 254]
    position  [0, 4095]         [0, 4095]
    velocity  [0, 65535]        [0, 32767]     <-- 50% of the advertised domain

The enforced ceiling is the correct one and was tightened deliberately:
``Goal_Velocity`` is sign-magnitude, so 65535 sets bit 15 and the servo reads it
as full speed in the *opposite* direction -- a different command, not a faster
one. ``tests/tools/test_serial_tool_numeric_domain.py`` records that change and
asserts 32767. What went unchanged was the docstring, so the tool went on
offering the model 32768 values it refuses, each with a well-worded refusal for a
call the schema said was in range.

The second half is the drift that had not happened yet. The full scale was
spelled twice -- once as the ``position`` ceiling and once as the divisor the
reported angle turns a count into degrees with -- while the ceiling's own reason
string claimed they are "the same full scale". That claim was asserted and never
enforced, so a correction to one would have been invisible to the other. Issue
#2812 reports that the family the registry declares spans two resolutions
(``sts3215`` at 4096 counts, ``scs0009`` at 1024) and asks for the ceiling and
the divisor to be single-sourced whichever way the model question is settled.
This module pins that, so the correction is one edit rather than two that can
disagree.

Which full scale is right was left open by that change and is settled here: it is
the STS/SMS one, because the two-byte order this tool encodes a position into is
itself an STS/SMS property. The vendor SDK reverses that order on a per-model
protocol number, so an SCS-series servo reads the same two bytes as the
byte-swapped value -- ``position=1023``, full scale on an ``scs0009``, reaches it
as 65283. Covering that series is therefore a second word order and a second full
scale, not a scale option on this action, and the number and the order now come
from :mod:`strands_robots.drivers.feetech.protocol`, which decides both once.
These cells grade that the bound is honest, that it is not restated here, and
that the schema says which series it belongs to.
"""

from __future__ import annotations

import ast
import inspect
import re
from typing import Any

import pytest
import serial

import strands_robots.tools.serial_tool as serial_mod
from strands_robots.drivers.feetech.protocol import MAX_GOAL_POSITION

#: The bound a field's schema entry declares, as the model reads it. ``motor_id``
#: carries a second interval for the reply-expecting actions, so the *first*
#: interval is the domain of the parameter itself.
_INTERVAL = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")

#: Every register field the tool bounds, derived from the module rather than
#: listed here, so a field added without a documented bound is graded on arrival.
_REGISTER_FIELDS = tuple(serial_mod._REGISTER_FIELDS)


class _FakeSerial:
    """Stand-in for ``serial.Serial`` recording the bytes that reach the wire."""

    def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
        self.writes: list[bytes] = []
        self.in_waiting = 0

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def read(self, n: int = 1) -> bytes:
        return b""

    def close(self) -> None:
        return None


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[_FakeSerial]:
    created: list[_FakeSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> _FakeSerial:
        instance = _FakeSerial(port, baudrate, timeout)
        created.append(instance)
        return instance

    monkeypatch.setattr(serial, "Serial", _ctor)
    return created


def _call(**kwargs: Any) -> dict[str, Any]:
    return serial_mod.serial_tool(port="/dev/fake0", **kwargs)


def _text(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _documented_bound(field: str) -> tuple[int, int]:
    """The interval this field's entry declares in the generated tool schema."""
    properties = serial_mod.serial_tool.tool_spec["inputSchema"]["json"]["properties"]
    description = " ".join(properties[field]["description"].split())
    match = _INTERVAL.search(description)
    assert match is not None, f"{field}: schema entry declares no [lo, hi] interval: {description!r}"
    return int(match.group(1)), int(match.group(2))


class TestTheDocumentedBoundIsTheEnforcedBound:
    """A domain the schema advertises and the validator refuses is not a domain."""

    @pytest.mark.parametrize("field", _REGISTER_FIELDS)
    def test_the_schema_declares_the_bound_the_validator_applies(self, field: str) -> None:
        floor, ceiling, _reason = serial_mod._REGISTER_FIELDS[field]

        assert _documented_bound(field) == (floor, ceiling), (
            f"{field}: the tool schema tells the model {_documented_bound(field)} "
            f"while the validator enforces {(floor, ceiling)}"
        )

    @pytest.mark.parametrize(
        "action,field",
        [("feetech_position", "position"), ("feetech_velocity", "velocity")],
    )
    def test_the_documented_ceiling_is_accepted_and_one_past_it_is_refused(
        self, action: str, field: str, opened: list[_FakeSerial]
    ) -> None:
        ceiling = _documented_bound(field)[1]

        accepted = _call(action=action, motor_id=1, **{field: ceiling})
        assert accepted["status"] == "success", f"{field}={ceiling} is documented as in range: {_text(accepted)}"

        refused = _call(action=action, motor_id=1, **{field: ceiling + 1})
        assert refused["status"] == "error"
        assert str(ceiling) in _text(refused)


class TestTheSchemaNamesTheSeriesTheBoundBelongsTo:
    """A model reading the schema is told which servos the bound is true of.

    ``[0, 4095]`` and the degrees the tool reports back are true of the STS/SMS
    series and of nothing else on this bus: an ``scs0009`` addresses a quarter of
    that range and reads the two bytes in the opposite order, so the same call
    commands a different position. A schema entry that names the manufacturer and
    not the series tells the model the action covers servos it cannot address,
    and the refusal it would rather have -- the one this suite exists to keep
    honest -- never comes, because the value is in range for the series the tool
    does speak.
    """

    @pytest.mark.parametrize("field", ["position", "velocity"])
    def test_the_schema_entry_names_the_series(self, field: str) -> None:
        properties = serial_mod.serial_tool.tool_spec["inputSchema"]["json"]["properties"]
        description = " ".join(properties[field]["description"].split())

        assert "STS/SMS" in description, (
            f"{field}: the schema tells the model {description!r}, which names the "
            "manufacturer rather than the series the bound and the encoding belong to"
        )

    def test_the_position_reason_says_which_series_the_full_scale_is(self) -> None:
        # The reason is what a refusal quotes back, so it is the one place a
        # caller who was refused learns why the ceiling is where it is.
        _floor, _ceiling, why = serial_mod._REGISTER_FIELDS["position"]

        assert "STS/SMS" in why and "SCS" in why, (
            f"the position ceiling's reason reads {why!r}; it states a 12-bit scale "
            "without saying which series is 12-bit, and the family has a 10-bit half"
        )

    def test_a_refusal_names_the_series_alongside_the_ceiling(self, opened: list[_FakeSerial]) -> None:
        ceiling = serial_mod._REGISTER_FIELDS["position"][1]

        refused = _call(action="feetech_position", motor_id=1, position=ceiling + 1)

        assert refused["status"] == "error"
        assert "STS/SMS" in _text(refused), _text(refused)


class TestTheFullScaleIsSingleSourced:
    """The ceiling and the report divisor are one number, not two that agree."""

    def test_the_full_scale_is_not_spelled_in_this_module_at_all(self) -> None:
        # Taken from the enforced ceiling rather than from the constant that now
        # holds it, so this cell states the property and not the fix's spelling.
        # It is a property of the servo series, decided by the codec that frames
        # the register, so this module reads it and never restates it: the
        # ceiling, the reported angle's divisor and the byte order a correction
        # would have to move together all read one name.
        full_scale = serial_mod._REGISTER_FIELDS["position"][1]
        tree = ast.parse(inspect.getsource(serial_mod))
        literals = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value.__class__ is int and node.value == full_scale
        ]

        assert literals == [], (
            f"the position full scale {full_scale} is spelled as {len(literals)} "
            f"integer literals, on lines {[node.lineno for node in literals]}; it is "
            "an STS/SMS-series property and MAX_GOAL_POSITION is where it is decided"
        )

    def test_the_ceiling_is_the_codec_full_scale(self) -> None:
        assert serial_mod._REGISTER_FIELDS["position"][1] == MAX_GOAL_POSITION

    def test_the_reported_angle_divides_by_the_enforced_ceiling(self, opened: list[_FakeSerial]) -> None:
        _floor, ceiling, _reason = serial_mod._REGISTER_FIELDS["position"]

        result = _call(action="feetech_position", motor_id=1, position=ceiling)

        assert result["status"] == "success"
        # Full scale is a full turn. Were the divisor to keep a scale the ceiling
        # no longer uses, the count the servo treats as its limit would report as
        # a fraction of one -- #2812 measures that as 89.9 deg on an scs0009.
        assert "360.0 deg" in _text(result), (
            f"position={ceiling} is the enforced ceiling, so it is full scale, but the tool reported: {_text(result)}"
        )

    def test_half_the_enforced_ceiling_reports_half_a_turn(self, opened: list[_FakeSerial]) -> None:
        _floor, ceiling, _reason = serial_mod._REGISTER_FIELDS["position"]

        result = _call(action="feetech_position", motor_id=1, position=ceiling // 2)

        assert result["status"] == "success"
        assert f"{ceiling // 2 / ceiling * 360:.1f} deg" in _text(result)
