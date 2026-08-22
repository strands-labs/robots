"""The loop period every ROS 2 hardware bridge paces a thread with is one shared domain.

Both hardware bridges run a background thread that services inbound commands,
and both let the caller name its cadence: ``HardwareRosBridge``'s ``spin_period``
and ``HardwareRtpsBridge``'s ``poll_period``. The value is a wait budget - handed
to :meth:`threading.Event.wait` on the rclpy bridge and to a
:class:`~strands_robots.mesh.pacing.Ticker` on the RTPS one - so unlike a domain
id it is not a value the transport ever gets to reject; the loop simply runs at
whatever cadence the argument implies.

That makes the failure a hot thread rather than a refused connection.
``0``, a negative and ``nan`` all return from ``wait`` immediately, so the loop
becomes a busy-spin with no bound; ``inf`` raises ``OverflowError`` out of
``wait`` and kills the loop thread outright, leaving a bridge that reported a
successful construction with a command surface that will never deliver again.
Neither is reported: there is no exception where the caller can see one and no
log line attributing the cadence to the argument that set it.

``rclpy`` and ``cyclonedds`` are optional, so every refusal test here runs with
both absent: each guard is placed ahead of its surface's transport probe, which
is what makes that possible and is asserted directly.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import threading
from typing import Any

import numpy as np
import pytest

import strands_robots
import strands_robots.hardware_rtps_bridge as rtps_mod
from strands_robots.hardware_ros_bridge import HardwareRosBridge
from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge
from strands_robots.utils import positive_finite_number_error

#: Values that cannot pace a loop, one per way of failing to name a cadence.
UNUSABLE_PERIODS: list[Any] = [
    0,  # wait returns immediately: a busy-spin
    0.0,
    -1,  # a negative budget is not a shorter wait, it is no wait
    -0.5,
    float("nan"),  # every comparison against it is False
    float("inf"),  # OverflowError out of wait: the loop thread dies
    float("-inf"),
    True,  # int subclass: a silent 1-second period
    False,  # int subclass: a silent busy-spin
    "0.02",  # a numeric string is not a real number
    None,
    [0.02],
    10**400,  # positive and finite, but past the float64 range
]

#: Values that do name a cadence, spanning the fractional and whole cases.
USABLE_PERIODS: list[float] = [0.001, 0.02, 0.5, 1, 2.5]


def _refuses(fn: Any, value: Any) -> bool:
    """Whether ``fn(value)`` refuses ``value`` as a wait budget.

    An ``ImportError`` means the value cleared the guard and the surface then
    found its optional transport missing - an install problem, not a verdict
    about the period - so it counts as accepted.
    """
    try:
        fn(value)
    except ValueError as exc:
        return "spin_period must be" in str(exc) or "poll_period must be" in str(exc)
    except ImportError:
        return False
    return False


#: Every surface that names a loop period, with the parameter it names it by.
SURFACES: list[tuple[str, Any]] = [
    ("HardwareRosBridge(spin_period=)", lambda v: HardwareRosBridge(spin_period=v)),
    ("HardwareRtpsBridge(poll_period=)", lambda v: HardwareRtpsBridge(poll_period=v)),
]
SURFACE_IDS = [name.split("(")[0] for name, _ in SURFACES]


class _RecordingStop:
    """A stop event that records the budget it is asked to wait for.

    Stands in for the bridges' ``threading.Event`` so a loop body can be driven
    for a fixed number of iterations with no wall-clock dependence at all: the
    subject is *which value the loop paces itself with*, not how fast this host
    happens to run.
    """

    def __init__(self, iterations: int) -> None:
        self.remaining = iterations
        self.waits: list[Any] = []

    def is_set(self) -> bool:
        if self.remaining <= 0:
            return True
        self.remaining -= 1
        return False

    def wait(self, timeout: Any = None) -> bool:
        self.waits.append(timeout)
        return False


class _RecordingTicker:
    """A ticker that records the period it was built with instead of sleeping.

    The RTPS poll loop hands its stored period to a ticker rather than to the
    stop event, so this stands where :class:`_RecordingStop` stands for the rclpy
    loop: it makes "which value paces this loop" answerable without a wall clock.
    """

    def __init__(self, period: Any, stop_event: Any = None) -> None:
        self.period = period
        self.stop_event = stop_event
        self.waits = 0
        self.closed = False

    def wait(self) -> bool:
        self.waits += 1
        return False

    def close(self) -> None:
        self.closed = True

    # The poll loop enters the ticker with `with`, so the double matches.
    def __enter__(self) -> _RecordingTicker:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class TestWhyAnUnusablePeriodCannotPaceALoop:
    """Executable premise: what :meth:`threading.Event.wait` does with each value.

    The guard's reason for existing is a property of ``Event.wait``, not an
    opinion about the arguments, so it is measured here rather than asserted in
    prose. Nothing below depends on how fast this host runs: a value that does
    not pace a wait is shown by ``wait`` *returning*, and the one that cannot be
    waited on at all is shown by the exception it raises.
    """

    @pytest.mark.parametrize("value", [0, 0.0, -1, -0.5, float("nan")], ids=repr)
    def test_a_non_pacing_period_returns_from_wait_without_pacing_anything(self, value: Any) -> None:
        """These are the busy-spin values: ``wait`` hands the loop straight back."""
        assert threading.Event().wait(value) is False

    def test_an_infinite_period_raises_out_of_wait(self) -> None:
        """``inf`` does not wait forever - it kills the thread that waits on it."""
        with pytest.raises(OverflowError):
            threading.Event().wait(float("inf"))

    def test_a_boolean_period_is_a_silent_one_second_cadence(self) -> None:
        """``bool`` is an ``int`` subclass, so a bare positivity test admits it.

        The value is bound rather than written into the comparison as a
        literal: a comparison between two literals is decided when it is
        typed, so it would state this premise without measuring it.
        """
        period: Any = True
        assert float(period) == 1.0
        assert period > 0
        # What replaced that comparison does see it.
        assert positive_finite_number_error(period, "poll_period", "HardwareRtpsBridge") is not None

    def test_a_numpy_float32_period_cannot_be_waited_on(self) -> None:
        """Why the ``float()`` conversion after the guard is load-bearing.

        The shared domain accepts any real scalar, and documents a NumPy scalar
        read from a config array as usable - but ``Event.wait`` rejects a
        ``np.float32`` outright, so an accepted value has to be converted before
        the loop can pace itself with it.
        """
        # Annotated ``float | None``, so the value under test is bound through
        # ``Any``: the subject is what the runtime does with a scalar the shared
        # domain accepts, which is not what the annotation describes.
        narrower: Any = np.float32(0.02)
        with pytest.raises(TypeError):
            threading.Event().wait(narrower)
        assert threading.Event().wait(float(np.float32(0.02))) is False


class TestTheStoredPeriodIsTheWholeCadenceOfTheLoop:
    """The stored value is the only thing pacing either command loop.

    This is the link between the guard and the consequence: whatever the
    constructor stores is handed to ``wait`` once per iteration, so a value that
    does not pace a wait does not pace the loop either. Driven for a fixed
    iteration count against a recording stop event, so the assertion is on the
    budget the loop *asks for* rather than on throughput this host achieved.
    """

    def test_the_rtps_poll_loop_paces_itself_with_the_stored_period(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The stored period is what the loop's ticker is built with.

        The loop paces on a ticker rather than on ``_stop.wait(period)``, so the
        budget is spent once at construction instead of once per iteration. The
        claim is unchanged - the stored value is the only thing pacing the loop -
        and it is still checked over a fixed iteration count with no wall clock:
        the ticker is asked to pace every iteration, and it is released after.
        """
        tickers: list[_RecordingTicker] = []

        def _make(period: Any, stop_event: Any = None) -> _RecordingTicker:
            tickers.append(_RecordingTicker(period, stop_event))
            return tickers[-1]

        monkeypatch.setattr(rtps_mod, "Ticker", _make)
        bridge = HardwareRtpsBridge.__new__(HardwareRtpsBridge)
        stop = _RecordingStop(iterations=4)
        bridge._stop = stop  # type: ignore[assignment]  # bounds the loop; no wall-clock wait
        bridge._command_reader = type("_Reader", (), {"take": lambda self, N=10: []})()
        bridge._poll_period = 0.02
        bridge._poll_loop()
        assert len(tickers) == 1, "one ticker per loop, built from the stored period"
        assert tickers[0].period == 0.02
        assert tickers[0].stop_event is stop, "the loop's stop event must reach the ticker"
        assert tickers[0].waits == 4, "the ticker paces every iteration"
        assert tickers[0].closed, "the selector must be released when the loop ends"
        assert stop.waits == [], "the period is no longer spent on an Event.wait"

    def test_the_ros_spin_loop_forwards_the_stored_period_to_the_executor(self) -> None:
        """The rclpy loop spends the period as a ``spin_once`` timeout.

        ``rclpy`` is optional and absent here, so the executor stands in: what
        is asserted is that the caller's value is what the loop budgets each
        iteration with, whichever of the two waits consumes it.
        """
        bridge = HardwareRosBridge.__new__(HardwareRosBridge)
        stop = _RecordingStop(iterations=3)
        bridge._stop = stop  # type: ignore[assignment]
        timeouts: list[Any] = []
        bridge._rclpy = type(  # type: ignore[assignment]
            "_Rclpy", (), {"spin_once": lambda self, node, timeout_sec=None: timeouts.append(timeout_sec)}
        )()
        bridge._node = object()
        bridge._spin_period = 0.02
        bridge._spin_loop()
        assert timeouts == [0.02] * 3


class TestTheSharedDomain:
    """``positive_finite_number_error`` decides which values name a cadence."""

    @pytest.mark.parametrize("value", UNUSABLE_PERIODS, ids=repr)
    def test_a_value_that_cannot_pace_a_loop_is_refused(self, value: Any) -> None:
        error = positive_finite_number_error(value, "poll_period", "Surface")
        assert error is not None
        assert error.startswith("Surface: poll_period must be")

    @pytest.mark.parametrize("value", USABLE_PERIODS, ids=repr)
    def test_a_period_that_paces_a_loop_is_accepted(self, value: float) -> None:
        assert positive_finite_number_error(value, "poll_period", "Surface") is None

    def test_a_period_past_the_float_range_is_refused_for_its_own_reason(self) -> None:
        """``10**400`` is positive and finite, so ``must be > 0`` would be false of it.

        It also used to raise ``OverflowError`` out of the guard's own
        conversion. Covered here because these are new call sites for the
        domain, rather than assumed from the surfaces that already had it.
        """
        error = positive_finite_number_error(10**400, "poll_period", "Surface")
        assert error is not None
        assert "within the range of a 64-bit float" in error


class TestARefusedPeriodReachesNoTransport:
    """Each guard runs before its surface probes for an optional transport.

    Placing it there is what lets the same caller mistake report identically on
    an install with the ``[ros2]`` extra and one without it, and it means no DDS
    or rclpy state is built for a cadence that was never usable.
    """

    @pytest.mark.parametrize("value", UNUSABLE_PERIODS, ids=repr)
    def test_the_rtps_bridge_refuses_before_probing_for_cyclonedds(
        self, value: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _unreachable(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the cyclonedds probe must not be reached")

        monkeypatch.setattr(rtps_mod, "require_optional", _unreachable)
        with pytest.raises(ValueError, match="poll_period must be"):
            HardwareRtpsBridge(poll_period=value)

    def test_a_usable_period_still_reaches_the_cyclonedds_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probed: list[str] = []

        def _record(module: str, **_kwargs: Any) -> Any:
            probed.append(module)
            raise ImportError("cyclonedds absent")

        monkeypatch.setattr(rtps_mod, "require_optional", _record)
        with pytest.raises(ImportError):
            HardwareRtpsBridge(poll_period=0.02)
        assert probed == ["cyclonedds"]

    @pytest.mark.parametrize("value", UNUSABLE_PERIODS, ids=repr)
    def test_the_rclpy_bridge_refuses_before_its_base_constructor_runs(self, value: Any) -> None:
        """``rclpy`` is probed by the base constructor, so the guard precedes it."""
        with pytest.raises(ValueError, match="spin_period must be"):
            HardwareRosBridge(spin_period=value)

    def test_a_usable_period_still_reaches_the_rclpy_probe(self) -> None:
        with pytest.raises(ImportError, match="rclpy"):
            HardwareRosBridge(spin_period=0.02)


class TestARefusedSpinPeriodLeavesTheProcessEnvironmentAlone:
    """The rclpy guard precedes the process-wide ``ROS_DOMAIN_ID`` write.

    That write is global to the process and lands before ``rclpy`` is imported,
    so a construction that can never succeed used to move it anyway: the period
    was read well after the base had already pinned the domain. Refusing first
    means a rejected call has no side effect at all.
    """

    @pytest.mark.parametrize("value", UNUSABLE_PERIODS, ids=repr)
    def test_a_refused_period_does_not_touch_ros_domain_id(self, value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        import os

        with pytest.raises(ValueError, match="spin_period must be"):
            HardwareRosBridge(domain_id=11, spin_period=value)
        assert os.environ["ROS_DOMAIN_ID"] == "7"

    def test_a_usable_period_still_lets_the_domain_be_pinned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        import os

        with pytest.raises(ImportError):
            HardwareRosBridge(domain_id=11, spin_period=0.02)
        assert os.environ["ROS_DOMAIN_ID"] == "11"


class TestBothSurfacesRefuseTheSamePeriods:
    """Neither surface may accept a period the other one refuses.

    The two bridges are alternative transports for the same robot - a caller
    switches between them with ``Robot(ros2_transport=...)`` - so a cadence one
    of them runs a hot thread on is not one the other may accept either.
    """

    @pytest.mark.parametrize("value", UNUSABLE_PERIODS, ids=repr)
    def test_an_unusable_period_is_refused_by_every_surface(self, value: Any) -> None:
        refused = {name: _refuses(fn, value) for name, fn in SURFACES}
        assert all(refused.values()), f"accepted by {[n for n, r in refused.items() if not r]}"

    @pytest.mark.parametrize("value", USABLE_PERIODS, ids=repr)
    def test_a_usable_period_is_refused_by_no_surface(self, value: float) -> None:
        refused = {name: _refuses(fn, value) for name, fn in SURFACES}
        assert not any(refused.values()), f"refused by {[n for n, r in refused.items() if r]}"

    @pytest.mark.parametrize("name,fn", SURFACES, ids=SURFACE_IDS)
    def test_each_surface_names_the_parameter_the_caller_used(self, name: str, fn: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            fn(0)
        param = name.split("(")[1].rstrip("=)")
        assert param in str(excinfo.value)


class TestAnAcceptedPeriodIsStoredAsAConsumableFloat:
    """A NumPy scalar the domain accepts is converted before the loop waits on it.

    The conversion sits after the guard rather than instead of it. Before the
    guard existed it was the whole of the check, which is where ``True`` became a
    1-second period and ``'0.02'`` was quietly parsed; what it does now is the
    part that was always load-bearing - ``Event.wait`` rejects a ``np.float32``,
    so an accepted NumPy scalar would otherwise raise on the loop thread.
    """

    @pytest.mark.parametrize("value", [np.float32(0.02), np.float64(0.02), 1, 0.5], ids=repr)
    def test_an_accepted_period_is_stored_as_a_builtin_float(self, value: Any) -> None:
        assert positive_finite_number_error(value, "poll_period", "Surface") is None
        bridge = HardwareRtpsBridge.__new__(HardwareRtpsBridge)
        bridge._poll_period = float(value)
        assert type(bridge._poll_period) is float
        assert threading.Event().wait(bridge._poll_period) is False


class TestEveryWaitBudgetSurfaceRoutesThroughTheSharedDomain:
    """Structural guard: a period-taking surface guards it or forwards it.

    A surface that stores a caller-supplied loop period without either calling
    :func:`positive_finite_number_error` or handing the value to a surface that
    does is accepting a cadence that may not pace anything. Checked structurally
    so a third bridge cannot ship without joining the rule.

    The scope is a parameter whose name ends in ``_period``, which is what the
    two bridges call theirs and what a third would. It deliberately does not
    reach ``mesh.security.input_frame_slew_violation``'s ``min_interval_s``:
    that is a floor on the interval charged to a joint's move - a denominator in
    a speed comparison - and is never waited on.
    """

    @staticmethod
    def _package_root() -> pathlib.Path:
        """The installed package directory, derived from an imported symbol."""
        return pathlib.Path(inspect.getfile(strands_robots)).parent

    @classmethod
    def _classify(cls, source: str) -> dict[str, tuple[bool, bool]]:
        """Map ``function name -> (calls the guard, forwards the parameter)``."""
        found: dict[str, tuple[bool, bool]] = {}
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            taken = [a for a in args if a.endswith("_period")]
            if not taken:
                continue
            guards = any(
                isinstance(call.func, ast.Name) and call.func.id == "positive_finite_number_error"
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            )
            forwards = any(
                keyword.arg in taken and isinstance(keyword.value, ast.Name) and keyword.value.id in taken
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                for keyword in call.keywords
            )
            found[node.name] = (guards, forwards)
        return found

    def _surfaces(self) -> dict[str, tuple[bool, bool]]:
        surfaces: dict[str, tuple[bool, bool]] = {}
        root = self._package_root()
        for path in sorted(root.rglob("*.py")):
            for name, verdict in self._classify(path.read_text()).items():
                surfaces[f"{path.relative_to(root)}::{name}"] = verdict
        return surfaces

    def test_the_scan_finds_every_known_wait_budget_surface(self) -> None:
        """Non-vacuity: a scan rooted elsewhere would report a clean sweep."""
        assert set(self._surfaces()) == {
            "hardware_ros_bridge.py::__init__",
            "hardware_rtps_bridge.py::__init__",
        }

    def test_every_wait_budget_surface_guards_or_forwards_the_value(self) -> None:
        adrift = {name for name, (guards, forwards) in self._surfaces().items() if not (guards or forwards)}
        assert not adrift, f"these surfaces neither validate nor forward the loop period: {sorted(adrift)}"

    def test_the_scanner_detects_a_surface_that_does_neither(self) -> None:
        """A scanner that matched nothing would pass the sweep vacuously."""
        planted = "def brand_new_bridge(self, *, tick_period: float = 0.02) -> None:\n    self._t = tick_period\n"
        assert self._classify(planted) == {"brand_new_bridge": (False, False)}
