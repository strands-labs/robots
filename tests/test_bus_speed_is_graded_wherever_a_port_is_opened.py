# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The speed a serial bus is opened at is graded on every surface that takes one.

:mod:`~strands_robots.tools.serial_tool` holds its ``baudrate`` to
:func:`~strands_robots.utils.positive_count_error` and says why in its own module
docstring: the option is "coerced rather than checked by pyserial (``2.7`` becomes
2 baud)". The four constructors that reach the same ``serial.Serial`` took the same
quantity and held it to nothing - two converted it with ``int()`` and two stored it
raw - so a speed that is not a count was *applied* rather than reported, and the
tool refused a value the driver beside it accepted.

What the ungraded value did, measured rather than argued (:class:`TestPyserialCoerces`
pins the third-party half against a real pty):

* ``baud_rate=0`` opened the port **successfully**, at a speed no servo answers.
  Every read then times out, which is indistinguishable from an unplugged arm.
* ``baud_rate=True`` opened it at 1 baud and ``baud_rate=2.7`` at 2 - a different
  speed from the one asked for, while ``get_status`` reported the converted number
  as the configured one.
* ``baud_rate=-9600`` / ``None`` / ``[1_000_000]`` reached pyserial's own
  ``ValueError: Not a valid baudrate``, and ``nan`` / ``inf`` an ``int()``
  conversion error - each raised at connect time by a third library, naming
  neither the driver nor the parameter.
* ``1_000_000.0`` and ``'1000000'`` were converted to the speed the caller meant.
  They are refused now, which is the one narrowing here: the tool surface already
  refuses both, and that divergence between a tool and the driver it drives is
  what this domain removes.

:class:`TestEverySurfaceThatOpensAPortIsRostered` keeps the roster honest by
reading the syntax tree for ``serial.Serial`` calls, so a fifth surface cannot
open a port at an ungraded speed without failing here.
:mod:`~strands_robots.teleoperator` is deliberately absent: it forwards
``baud_rate`` into a lerobot config dataclass and never opens the port itself, so
the domain that applies is lerobot's.

Everything here is offline. The only port opened is a pty this file creates.
"""

from __future__ import annotations

import ast
import os
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

import strands_robots
from strands_robots.drivers.feetech.bus import FeetechBus
from strands_robots.drivers.feetech.driver import FeetechDriver
from strands_robots.tools.pose_tool import MotorController
from strands_robots.tools.serial_tool import serial_tool
from tests._package_ast import parse_file, walk_tree

_PORT = "/dev/ttyTEST-never-opened"

# Every surface that accepts a bus speed from a caller: how one is built, the
# attribute the accepted speed lands in, and the parameter a refusal must name.
_SURFACES: tuple[tuple[str, Callable[[Any], Any], str, str], ...] = (
    ("FeetechDriver", lambda v: FeetechDriver(tool_name="so101", baud_rate=v), "_baud_rate", "baud_rate"),
    ("FeetechBus", lambda v: FeetechBus(port=_PORT, baud_rate=v), "baud_rate", "baud_rate"),
    ("MotorController", lambda v: MotorController(_PORT, v), "baudrate", "baudrate"),
)
_SURFACE_IDS = [name for name, _, _, _ in _SURFACES]

# Speeds that are not a count, grouped by what each one did instead of failing.
# Kept as (value, what it did) pairs rather than a set: ``0 == False`` and
# ``1_000_000 == 1_000_000.0``, so a mapping keyed on the value would silently
# collapse the exact distinctions this domain is about.
_UNUSABLE: tuple[tuple[Any, str], ...] = (
    (0, "opened the port successfully, at a speed no servo answers"),
    (True, "opened it at 1 baud - a bool is an int, so a bare `< 1` test admits it"),
    (2.7, "opened it at 2 baud, the value serial_tool's docstring names"),
    (-9600, "reached pyserial's own 'Not a valid baudrate', naming no parameter"),
    (float("nan"), "escaped as 'cannot convert float NaN to integer'"),
    (float("inf"), "escaped as an OverflowError from int()"),
    (None, "reached pyserial's own 'Not a valid baudrate: None'"),
    ([1_000_000], "reached pyserial's own 'Not a valid baudrate: [1000000]'"),
    (1_000_000.0, "was converted to the speed meant; the tool surface refuses it"),
    ("1000000", "was converted to the speed meant; the tool surface refuses it"),
)

# Speeds a servo bus really runs at, plus the extremes of the accepted domain.
# The rule is a floor on the type, not a whitelist of standard rates: a caller
# with a servo configured to something unusual is not second-guessed.
_USABLE: tuple[int, ...] = (1, 9600, 57600, 115200, 500_000, 1_000_000, 4_000_000)


def _refusal(param: str, build: Callable[[Any], Any], value: Any) -> str | None:
    """The refusal text a surface produced, or ``None`` when it accepted.

    A ``ValueError`` whose message does not name ``param`` is not a refusal by
    this domain - it is the pre-fix behaviour, where the conversion itself failed
    and reported neither the surface nor the option. Grading those as refusals
    would let this file pass against the code it was written to catch.
    """
    try:
        build(value)
    except ValueError as error:
        if param in str(error):
            return str(error)
        pytest.fail(f"{param}={value!r} raised a ValueError naming neither surface nor option: {error}")
    except Exception as error:
        # Any other escape is the defect under test - pre-fix, `inf` left a
        # constructor as OverflowError - so the class is reported rather than
        # asserted. Exception, not BaseException: an interrupt or a pytest
        # outcome is not something a surface did with the value.
        pytest.fail(f"{param}={value!r} escaped as {type(error).__name__}: {error}")
    return None


class TestAnUnusableSpeedIsRefusedEverywhere:
    """No surface may apply a speed that is not a positive integer count."""

    @pytest.mark.parametrize(("name", "build", "attr", "param"), _SURFACES, ids=_SURFACE_IDS)
    def test_it_is_refused_by_name(self, name: str, build: Callable[[Any], Any], attr: str, param: str) -> None:
        del attr
        for value, did in _UNUSABLE:
            refusal = _refusal(param, build, value)
            assert refusal is not None, f"{name} accepted {param}={value!r}, which {did}"
            assert param in refusal and "positive integer" in refusal, f"{value!r}: {refusal}"

    def test_the_refusal_names_the_surface_that_received_it(self) -> None:
        """A message naming only the option sends the caller to the wrong layer."""
        for name, build, _attr, param in _SURFACES:
            refusal = _refusal(param, build, 0)
            assert refusal is not None
            assert name in refusal, f"{name}'s refusal does not name it: {refusal}"


class TestAUsableSpeedIsAcceptedUnchanged:
    """The over-reach control: nothing that worked stops working, or is rewritten."""

    @pytest.mark.parametrize(("name", "build", "attr", "param"), _SURFACES, ids=_SURFACE_IDS)
    def test_it_is_stored_exactly_as_supplied(
        self, name: str, build: Callable[[Any], Any], attr: str, param: str
    ) -> None:
        del param
        for value in _USABLE:
            stored = getattr(build(value), attr)
            assert stored == value and type(stored) is int, f"{name} stored {stored!r} for a supplied {value!r}"

    def test_the_documented_default_is_within_the_domain(self) -> None:
        """A domain that refuses its own default is the one real risk here."""
        for name, build, attr, _param in _SURFACES:
            del name
            assert getattr(build(1_000_000), attr) == 1_000_000


class TestTheToolAndTheDriverAgree:
    """The divergence this domain removes: one speed, one answer."""

    def test_the_tool_refuses_what_the_drivers_refuse(self) -> None:
        for value, _did in _UNUSABLE:
            result = serial_tool(action="send", port=_PORT, data="x", baudrate=value)
            assert result["status"] == "error", f"{value!r}"
            assert "baudrate must be a positive integer" in result["content"][0]["text"], f"{value!r}"

    def test_a_usable_speed_gets_past_the_tools_option_check(self) -> None:
        """So the cell above measures the option and not the missing port.

        The approval prompt that may follow names the speed it will open at,
        so the check is on the option refusal itself, not the word.
        """
        result = serial_tool(action="send", port=_PORT, data="x", baudrate=1_000_000)
        assert "baudrate must be a positive integer" not in result["content"][0]["text"]


class TestPyserialCoerces:
    """The premise every docstring here rests on, measured against pyserial and a pty.

    Two halves. ``baudrate`` is coerced with ``int()`` in pyserial's setter,
    before any port is touched, so the conversion is measured on an unopened
    ``Serial`` - on every OS. Whether the converted number is then *applied*
    rather than refused is measured against a real pty, at a speed a pty on
    every OS accepts: macOS sets a non-standard rate (1, 2, 1_000_000 baud) with
    the ``IOSSIOSPEED`` ioctl, which a pty answers with ``ENOTTY``, so the
    values from the first half cannot be the ones opened.

    If a future pyserial starts refusing these itself, this fails and the reason
    the domain gives for existing needs rewriting.
    """

    def test_a_speed_that_is_not_a_count_is_coerced_not_refused(self) -> None:
        serial = pytest.importorskip("serial")
        coercions: list[tuple[Any, int]] = [(0, 0), (True, 1), (2.7, 2), ("1000000", 1_000_000)]
        for value, coerced in coercions:
            conn = serial.Serial(None, value)
            assert conn.is_open is False, "a port here would put the case back on the platform's ioctl"
            assert conn.baudrate == coerced, f"{value!r}"

    def test_the_coerced_speed_is_applied_to_the_port(self) -> None:
        serial = pytest.importorskip("serial")
        applied: list[Any] = [9600.7, "9600"]
        for value in applied:
            master, follower = os.openpty()
            try:
                conn = serial.Serial(os.ttyname(follower), value, timeout=0.1)
                try:
                    assert conn.baudrate == 9600, f"{value!r}"
                finally:
                    conn.close()
            finally:
                os.close(master)
                os.close(follower)


class TestEverySurfaceThatOpensAPortIsRostered:
    """A fifth surface cannot open a port at a speed nothing graded."""

    #: Modules the scan below is allowed to find. ``serial_tool`` grades its
    #: ``baudrate`` as a tool option rather than at a constructor, so it is named
    #: here instead of in ``_SURFACES``.
    _EXPECTED = frozenset(
        {
            "strands_robots/tools/serial_tool.py",
            "strands_robots/tools/pose_tool.py",
            "strands_robots/drivers/feetech/bus.py",
        }
    )

    @staticmethod
    def _modules_opening_a_port() -> set[str]:
        root = pathlib.Path(strands_robots.__file__).parent
        found: set[str] = set()
        for path in sorted(root.rglob("*.py")):
            tree = parse_file(path)
            for node in walk_tree(tree):
                if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("serial.Serial"):
                    found.add(f"strands_robots/{path.relative_to(root).as_posix()}")
        return found

    def test_the_scan_finds_the_surfaces_the_roster_accounts_for(self) -> None:
        assert self._modules_opening_a_port() == self._EXPECTED, (
            "a module opens a serial port that this file does not account for; "
            "give its speed a domain and add it to _SURFACES"
        )

    def test_every_rostered_constructor_is_exercised(self) -> None:
        """The roster cannot fall behind the surfaces it claims to grade."""
        assert {name for name, _, _, _ in _SURFACES} == {
            "FeetechDriver",
            "FeetechBus",
            "MotorController",
        }
