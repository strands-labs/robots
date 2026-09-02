"""Both Reachy consumers hold the same axis to the same travel envelope.

:mod:`strands_robots.tools.reachy` exists for exactly one reason, which its own
package docstring states: it "holds only what the *two* Reachy consumers must
agree on and neither owns: the motion envelope". Only one of those two consumed
it. :meth:`~strands_robots.drivers.reachy.ReachyDriver.send_action` ran
:func:`~strands_robots.tools.reachy.envelope_error` over its action dict; the
Device Connect driver's three movement RPCs ran
:func:`~strands_robots.utils.finite_number_error` and stopped there, so a value
one driver refused the other put on the wire, for the same physical robot:

* ``look(pitch=200)`` reported ``success`` and sent ``head_pose`` built from 200
  degrees of pitch on an axis whose travel is +/-40.
* ``body(yaw=400)`` reported ``success`` and sent ``{"body_yaw": 6.98}`` - 400
  degrees in radians - on an axis whose travel is +/-160.
* ``send_action({"head_pitch": 200})`` refused both, naming the limit.

The exclusion was argued rather than overlooked. ``_motion_domain_error`` said
the reachable workspace "is the daemon's to enforce -- it depends on hardware
this library does not model", and that was true when it was written: the reason
landed on 2026-08-07 and ``MOTION_ENVELOPE_DEG`` landed on 2026-08-26, nineteen
days later, in a package that imports no transport and no driver and is
importable with no Reachy attached. The reason is a claim about what the library
can model, and a later change gave the library the model.

Why nothing caught it: the two surfaces spell the same axis differently.
``look`` takes ``pitch``/``roll``/``yaw`` where the envelope keys
``head_pitch``/``head_roll``/``head_yaw``, and ``envelope_error`` ignores a key
it has no limit for - so handing it the RPC's own keyword dict bounds nothing
and reports no error. :data:`_ENVELOPE_AXIS_BY_PARAM` is that mapping, and
``TestTheEnvelopeIsNotReImplemented`` grades it against the live limits so an
axis added to the envelope cannot be silently unmapped.

Scope. Per-axis travel is the half that transfers. The envelope's head-body yaw
coupling limit bounds ``head_yaw - body_yaw`` and no single RPC here carries both
values - ``look`` carries the head yaw and ``body`` the body yaw.
``TestTheCouplingLimitIsNotReachableHere`` pins that as a property of the mapping
rather than leaving it implied, and grades the one-member case rather than
assuming it: the native driver reaches the limit on a lone ``body_yaw`` too, by
checking it against the head yaw it last commanded, and this surface keeps no
such record, so its ``body`` RPC stays per-axis only. Both halves of that are
asserted, so the difference between the two consumers is a measured property and
not a paragraph. The millimetre offsets and the antenna angles carry no envelope
entry and stay finiteness-only, which ``TestWhatTheEnvelopeDoesNotBound`` holds
from both sides so this change cannot grow into a bound the envelope never
declared.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import pathlib
import textwrap
import threading
from typing import Any

import pytest

from strands_robots.drivers.reachy import ReachyDriver
from strands_robots.tools.reachy import (
    HEAD_BODY_YAW_DELTA_LIMIT_DEG,
    MOTION_ENVELOPE_DEG,
    envelope_error,
)
from tests.test_reachy_mini_driver import _force_real_device_connect_edge
from tests.test_reachy_motion_domain import USABLE_MOTION_VALUES


def _outside(limit: float) -> float:
    """A travel request comfortably outside ``limit``, in degrees."""
    return limit * 5.0 + 10.0


def _inside(limit: float) -> float:
    """A travel request comfortably inside ``limit``, in degrees."""
    return limit * 0.5


@pytest.fixture
def rmd() -> Any:
    """The reachy_mini_driver module bound to the real device_connect_edge."""
    _force_real_device_connect_edge()
    import strands_robots.device_connect.reachy_mini_driver as module

    return module


class _RecordingLink:
    """A hardware link that records what a driver hands it instead of dialing."""

    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send_cmd(self, cmd: dict[str, Any]) -> None:
        self.commands.append(cmd)


def _device_connect(rmd: Any) -> tuple[Any, _RecordingLink]:
    """A Device Connect driver with a recording link, built without dialing."""
    driver = rmd.ReachyMiniDriver.__new__(rmd.ReachyMiniDriver)
    driver._host = "bot.local"
    driver._prefix = "reachy_mini"
    driver._api_port = 8000
    driver._latest_joints = None
    driver._latest_imu = None
    link = _RecordingLink()
    driver._hw = link
    return driver, link


def _native() -> tuple[ReachyDriver, list[dict[str, Any]]]:
    """A native driver reporting connected, recording what it would send.

    Carries the cache lock and the head yaw target as well as the connection
    flag: ``send_action`` reads the target when an action names no head pose, so
    a stand-in without it could not reach the coupling check at all.
    """
    driver = ReachyDriver.__new__(ReachyDriver)
    driver._tool_name = "reachy_mini"
    driver._connected = True
    driver._cache_lock = threading.Lock()
    driver._head_yaw_target = None
    sent: list[dict[str, Any]] = []

    def _send(command: dict[str, Any]) -> str | None:
        sent.append(command)
        return None

    driver._send_cmd = _send  # type: ignore[method-assign]
    return driver, sent


def _call(driver: Any, rpc_name: str, **kwargs: Any) -> Any:
    """Invoke a movement RPC as an authorized caller would."""
    return asyncio.run(getattr(driver, rpc_name)(**kwargs))


def _text(envelope: dict[str, Any]) -> str:
    """Every text block of a driver envelope, joined."""
    return " ".join(block["text"] for block in envelope.get("content", []) if "text" in block)


def _bounded_cases(rmd: Any) -> list[tuple[str, str, str, float]]:
    """Every ``(rpc, param, axis, limit)`` the mapping bounds, from the tree.

    Derived from the driver's own map and the envelope's own limits, so an axis
    that gains a bound - or a parameter that gains a mapping - is graded here
    without this file being edited.
    """
    return [
        (rpc, param, axis, MOTION_ENVELOPE_DEG[axis])
        for rpc, axes in sorted(rmd._ENVELOPE_AXIS_BY_PARAM.items())
        for param, axis in sorted(axes.items())
    ]


def _case_ids(rmd: Any) -> list[str]:
    return [f"{rpc}-{param}" for rpc, param, _axis, _limit in _bounded_cases(rmd)]


class TestBothDriversRefuseTheSameTravel:
    """The defect: one consumer refused a request the other carried to the wire."""

    def test_the_device_connect_rpc_refuses_travel_outside_the_axis(self, rmd: Any) -> None:
        for rpc, param, axis, limit in _bounded_cases(rmd):
            driver, link = _device_connect(rmd)
            result = _call(driver, rpc, **{param: _outside(limit)})
            assert result["status"] == "error", f"{rpc}({param}={_outside(limit)}) -> {result}"
            assert link.commands == [], f"{rpc}({param}) reached the wire: {link.commands}"
            assert axis in result["reason"], result["reason"]

    def test_the_native_driver_refuses_the_same_travel(self, rmd: Any) -> None:
        """The consumer that already consulted the envelope, as the control."""
        for _rpc, _param, axis, limit in _bounded_cases(rmd):
            driver, sent = _native()
            result = driver.send_action({axis: _outside(limit)})
            assert result["status"] == "error", f"send_action({axis}) -> {result}"
            assert sent == []
            assert axis in _text(result)

    def test_the_two_drivers_agree_on_every_bounded_axis(self, rmd: Any) -> None:
        """The property the shared package exists for, over both verdicts."""
        for rpc, param, axis, limit in _bounded_cases(rmd):
            for value in (_inside(limit), _outside(limit)):
                dc_driver, _link = _device_connect(rmd)
                dc_status = _call(dc_driver, rpc, **{param: value})["status"]
                native_driver, _sent = _native()
                native_status = native_driver.send_action({axis: value})["status"]
                assert dc_status == native_status, f"{axis}={value}: {dc_status} vs {native_status}"


class TestTheRefusalNamesTheAxisAndTheLimit:
    """A refusal a caller can act on: which axis, and what its travel is."""

    def test_the_reason_names_the_rpc_the_axis_and_the_bound(self, rmd: Any) -> None:
        driver, _link = _device_connect(rmd)
        result = _call(driver, "body", yaw=400.0)
        assert result["reason"] == "body: body_yaw 400 deg is outside the envelope +/-160 deg"

    def test_an_unusable_value_is_still_named_by_the_callers_own_parameter(self, rmd: Any) -> None:
        """Finiteness runs first, so ``nan`` is reported as ``pitch``, not ``head_pitch``.

        The envelope keys the axis; the caller typed the parameter. Asking
        finiteness first keeps the message in the caller's own vocabulary for the
        one input class that cannot be compared against a travel bound at all.
        """
        driver, link = _device_connect(rmd)
        result = _call(driver, "look", pitch=float("nan"))
        assert result["reason"] == "look: pitch must be a finite number, got nan."
        assert "head_pitch" not in result["reason"]
        assert link.commands == []


class TestTheEnvelopeIsNotReImplemented:
    """One owner for the limits, and a mapping graded against it."""

    def test_the_driver_consults_the_shared_envelope(self, rmd: Any) -> None:
        """Graded on the CALL, not the source text.

        This helper's docstring names ``envelope_error`` to explain the bound, so
        a substring check would be satisfied by the prose alone - it passes on a
        body that dropped the call.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(rmd._motion_domain_error)))
        called = {
            node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "envelope_error" in called, f"the helper calls {sorted(called)}"
        assert "finite_number_error" in called, f"the helper calls {sorted(called)}"

    def test_every_mapped_axis_is_one_the_envelope_bounds(self, rmd: Any) -> None:
        """A mapping entry naming an axis with no limit would bound nothing."""
        for rpc, axes in rmd._ENVELOPE_AXIS_BY_PARAM.items():
            for param, axis in axes.items():
                assert axis in MOTION_ENVELOPE_DEG, f"{rpc}.{param} -> {axis} has no envelope limit"

    def test_the_driver_declares_no_travel_limit_of_its_own(self, rmd: Any) -> None:
        """The numbers live in the envelope, so this module restates none of them.

        Read as numeric literals rather than as text, so a limit that appears in
        a docstring explaining the bound is not mistaken for a second copy of it.
        """
        tree = ast.parse(pathlib.Path(str(rmd.__file__)).read_text(encoding="utf-8"))
        literals = {
            float(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        }
        restated = sorted(literals & set(MOTION_ENVELOPE_DEG.values()))
        assert not restated, f"travel limits restated as literals here: {restated}"

    def test_the_mapping_is_not_empty(self, rmd: Any) -> None:
        """Non-vacuity: an empty map would make every cell above pass trivially."""
        cases = _bounded_cases(rmd)
        assert len(cases) >= 4, f"expected the head axes and body yaw, mapped {cases}"


class TestWhatTheEnvelopeDoesNotBound:
    """The boundary held from both sides: no bound the envelope never declared."""

    def test_a_usable_in_envelope_request_still_reaches_the_robot(self, rmd: Any) -> None:
        for rpc, param, _axis, limit in _bounded_cases(rmd):
            driver, link = _device_connect(rmd)
            result = _call(driver, rpc, **{param: _inside(limit)})
            assert result["status"] == "success", f"{rpc}({param}={_inside(limit)}) -> {result}"
            assert len(link.commands) == 1

    def test_a_millimetre_offset_is_not_bounded_by_the_envelope(self, rmd: Any) -> None:
        """``look``'s offsets are millimetres; the envelope declares no limit."""
        for param in ("x", "y", "z"):
            assert param not in rmd._ENVELOPE_AXIS_BY_PARAM["look"]
            driver, link = _device_connect(rmd)
            result = _call(driver, "look", **{param: 9999.0})
            assert result["status"] == "success", f"look({param}=9999) -> {result}"
            assert len(link.commands) == 1

    def test_an_antenna_angle_is_not_bounded_by_the_envelope(self, rmd: Any) -> None:
        assert "antennas" not in rmd._ENVELOPE_AXIS_BY_PARAM
        driver, link = _device_connect(rmd)
        result = _call(driver, "antennas", left=500.0, right=-500.0)
        assert result["status"] == "success", result
        assert len(link.commands) == 1

    def test_an_unmapped_rpc_keeps_the_finiteness_domain(self, rmd: Any) -> None:
        """The change is additive for a surface the mapping does not name."""
        driver, link = _device_connect(rmd)
        assert _call(driver, "antennas", left=float("inf"))["status"] == "error"
        assert link.commands == []


class TestTheCouplingLimitIsNotReachableHere:
    """Scope, pinned rather than implied: no RPC here carries both values.

    Holds before and after the change. It records why the envelope's second
    limit is not part of it, so a reader does not take the omission for an
    oversight.

    "No RPC maps both" is not on its own a reason the limit cannot apply, so the
    one-member case is graded rather than assumed. What decides it is whether the
    surface knows the counterpart: ``send_action`` records the head pose it last
    sent and so applies the coupling to a lone ``body_yaw`` too, while this one
    keeps no such record and stays per-axis. The rows below hold both halves.
    """

    def test_no_single_rpc_carries_both_members_of_the_yaw_pair(self, rmd: Any) -> None:
        for rpc, axes in rmd._ENVELOPE_AXIS_BY_PARAM.items():
            mapped = set(axes.values())
            assert not {"head_yaw", "body_yaw"} <= mapped, f"{rpc} maps both, so the coupling applies"

    def test_the_coupling_limit_exists_and_the_native_driver_can_reach_it(self) -> None:
        """The half that does have a surface taking both values at once."""
        driver, sent = _native()
        result = driver.send_action(
            {"head_yaw": HEAD_BODY_YAW_DELTA_LIMIT_DEG + 20.0, "body_yaw": -HEAD_BODY_YAW_DELTA_LIMIT_DEG}
        )
        assert result["status"] == "error"
        assert sent == []
        assert "coupling" in _text(result)

    def test_one_member_of_the_pair_alone_clears_the_check_without_a_counterpart(self) -> None:
        """The single-member case, graded rather than left to be assumed.

        The row above pins that no Device Connect RPC carries both members. That
        is a property of the *pair*, not of that surface, so an action reaching
        the envelope with one member reaches the check only if the counterpart is
        supplied alongside it - which is the edited expectation #3094 asked for.
        A lone ``head_yaw`` stays cleared on its own terms: the daemon serves it
        by turning the body under the head, so there is no second value to bound.
        """
        far = HEAD_BODY_YAW_DELTA_LIMIT_DEG + 20.0
        assert envelope_error({"head_yaw": far}, "reachy_look") is None
        assert envelope_error({"body_yaw": far}, "reachy_body_turn") is None
        assert envelope_error({"body_yaw": far}, "reachy_body_turn", head_yaw_target=0.0) is not None
        # The same head value paired with a counterpart is refused, so what
        # decides the verdict is the action's key set and not the angle.
        assert envelope_error({"head_yaw": far, "body_yaw": 0.0}, "send_action") is not None

    def test_a_lone_member_reaches_the_wire_until_the_driver_knows_the_counterpart(self) -> None:
        """End to end: the envelope is consulted by a driver, not obeyed by one.

        The same lone ``body_yaw``, twice: through a driver that has commanded no
        head pose and so cannot know the twist, and through one that has.
        """
        far = HEAD_BODY_YAW_DELTA_LIMIT_DEG + 20.0

        unaware, sent = _native()
        assert unaware.send_action({"body_yaw": far})["status"] == "success"
        assert len(sent) == 1

        aware, sent = _native()
        aware.send_action({"head_pitch": 0.0, "head_yaw": 0.0})
        sent.clear()
        refused = aware.send_action({"body_yaw": far})

        assert refused["status"] == "error"
        assert sent == []
        assert "coupling" in _text(refused)

    def test_this_surface_keeps_no_record_of_the_head_yaw_it_commanded(self, rmd: Any) -> None:
        """Why it cannot do the same: the ``look`` RPC stores nothing.

        Measured on the driver's own state rather than asserted in prose, so a
        driver that grows such a record fails here and is told to apply the
        limit rather than leaving this file's scope paragraph stale.
        """
        driver, link = _device_connect(rmd)
        before = set(vars(driver))

        _call(driver, "look", pitch=0.0, roll=0.0, yaw=_inside(MOTION_ENVELOPE_DEG["head_yaw"]))

        assert link.commands, "the look RPC did reach the link"
        assert set(vars(driver)) == before, "a new attribute here would be that record"
        assert _call(driver, "body", yaw=MOTION_ENVELOPE_DEG["body_yaw"])["status"] == "success"


class TestThePremisesThisRestsOn:
    """Measured rather than asserted in prose."""

    def test_the_shared_package_names_both_consumers_as_its_purpose(self) -> None:
        import strands_robots.tools.reachy as shared

        doc = " ".join((shared.__doc__ or "").split())
        assert "the *two* Reachy consumers must agree on and neither owns: the motion envelope" in doc

    def test_the_envelope_ignores_a_key_it_has_no_limit_for(self) -> None:
        """Why the mapping is needed: the RPC's own spelling bounds nothing."""
        assert envelope_error({"pitch": 200.0}, "look") is None
        assert envelope_error({"head_pitch": 200.0}, "look") is not None

    def test_the_shared_module_needs_no_reachy_and_no_daemon(self) -> None:
        """The reason the old exclusion gave is what this refutes."""
        import strands_robots.tools.reachy._reachy_common as common

        tree = ast.parse(pathlib.Path(str(common.__file__)).read_text(encoding="utf-8"))
        imported = {node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)} | {
            alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
        }
        assert imported == {"__future__", "typing", "strands_robots.utils"}, imported

    def test_the_sibling_usable_values_stay_inside_every_bounded_axis(self) -> None:
        """Drift guard: tightening an axis fails here, not in an unrelated sweep.

        ``USABLE_MOTION_VALUES`` drives every parameter of every movement RPC,
        including the bounded ones, so a value outside the tightest axis would be
        refused on travel and read as a finiteness regression.
        """
        tightest = min(MOTION_ENVELOPE_DEG.values())
        numeric = [value for value in USABLE_MOTION_VALUES if isinstance(value, int | float)]
        assert numeric, "expected numeric values to grade"
        for value in numeric:
            assert abs(float(value)) <= tightest, f"{value!r} exceeds the tightest axis (+/-{tightest} deg)"
