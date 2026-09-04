"""``STRANDS_MESH_STREAM_HZ`` must not be divided straight from the environment.

Two call sites throttle ``publish_step`` against a monotonic clock -- the
hardware control loop and the simulation ``run_policy`` hook -- and both
computed their period as
``1.0 / float(os.environ.get("STRANDS_MESH_STREAM_HZ", "10") or 10)``.
That expression raises ``ZeroDivisionError`` on ``0`` and ``ValueError`` on any
non-numeric value. One of the two runs inside ``HardwareRobot.__init__``, so the
consequence is not "telemetry degraded" but "every ``Robot(..., mode="real")``
construction raises before the serial port is even opened".

``0`` is a realistic input, not an adversarial one. The sibling knob
``STRANDS_MESH_CAMERA_HZ`` advertises non-positive as off -- ``Mesh.start``
tells operators to *set* it to enable frames -- so an operator disabling step
telemetry the same way bricked hardware bring-up.

Both sites now route through :func:`stream_min_period_from_env`, which reuses
the package's shared ``hz_from_env`` domain check and expresses "off" as
``math.inf``. No *finite* elapsed time reaches an infinite period -- but that
is a property of the base as much as of the period, and both sites start
theirs below every clock reading, so that a rollout's first step is due
wherever the platform's monotonic epoch sits. On that step ``inf >= inf``
holds, and the period alone would let exactly one publish past the opt-out.
Each site therefore tests the period for finiteness once, where it resolves
it, and gates the publish on that rather than on the subtraction alone;
``test_a_sentinel_below_every_reading_defeats_the_subtraction`` is the cell
that records why.

:class:`TestEveryThrottleMeasuresItsPeriodOnTheMonotonicClock` is the other half
of that sharing: the period is an elapsed interval, so the base both sites
subtract has to be a clock rather than the date.

:class:`TestNeitherCallSiteDividesTheRawEnvValue` is the structural half. The
behavioural tests would pass again if someone reintroduced the division at a
third site, and the bug was a duplicated expression in the first place, so the
source itself is asserted on.
"""

from __future__ import annotations

import ast
import math
import pathlib

import pytest

from strands_robots.mesh.session import STREAM_HZ, stream_min_period_from_env

#: Modules that hold a ``publish_step`` throttle period.
THROTTLE_SITES = [
    "strands_robots/hardware_robot.py",
    "strands_robots/simulation/mujoco/simulation.py",
]


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


class TestAnUnsetVariableKeepsTheDocumentedDefault:
    """The common case is unchanged by the guard."""

    def test_unset_resolves_to_the_stream_hz_period(self, monkeypatch) -> None:
        monkeypatch.delenv("STRANDS_MESH_STREAM_HZ", raising=False)

        assert stream_min_period_from_env() == pytest.approx(1.0 / STREAM_HZ)

    def test_blank_is_treated_as_unset(self, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", "   ")

        assert stream_min_period_from_env() == pytest.approx(1.0 / STREAM_HZ)

    @pytest.mark.parametrize(("raw", "period"), [("5", 0.2), ("20", 0.05), ("0.5", 2.0)])
    def test_a_positive_rate_becomes_its_period(self, monkeypatch, raw, period) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", raw)

        assert stream_min_period_from_env() == pytest.approx(period)


class TestANonPositiveRateTurnsPublishingOff:
    """``0`` is the operator opt-out, spelled as it is for the camera loop."""

    @pytest.mark.parametrize("raw", ["0", "0.0", "-1", "-0.5"])
    def test_it_resolves_to_an_infinite_period(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", raw)

        assert stream_min_period_from_env() == math.inf

    def test_an_infinite_period_never_lets_the_throttle_fire(self, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", "0")
        period = stream_min_period_from_env()

        # The shape both call sites use: ``now - last >= period``. A day of
        # elapsed time must not clear an infinite period.
        assert not (86_400.0 - 0.0 >= period)

    def test_a_sentinel_below_every_reading_defeats_the_subtraction(self, monkeypatch) -> None:
        """Which is why a call site reads the period, not just the difference.

        A throttle whose "never published" sentinel is below every clock
        reading -- so that its first tick is always due, whatever the platform's
        monotonic epoch is -- turns ``now - last`` into ``inf`` on that tick,
        and ``inf >= inf`` holds. The opt-out therefore cannot be carried by an
        infinite period alone, and the cell above passes only because its base
        is finite.
        """
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", "0")
        period = stream_min_period_from_env()

        assert 86_400.0 - float("-inf") >= period
        assert not math.isfinite(period)


class TestAnUnusableValueIsReportedAndDisablesPublishing:
    """Follows ``_resolve_camera_hz``: warn, then leave the loop off."""

    @pytest.mark.parametrize("raw", ["fast", "10Hz", "", "nan", "inf", "1e999"])
    def test_it_does_not_raise(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", raw)

        # No exception: the point of the change is that a misconfigured knob
        # cannot be the thing that fails a constructor.
        stream_min_period_from_env()

    @pytest.mark.parametrize("raw", ["fast", "nan", "inf", "1e999"])
    def test_it_resolves_to_an_infinite_period(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", raw)

        assert stream_min_period_from_env() == math.inf

    def test_the_warning_names_the_variable_and_the_value(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", "fast")

        with caplog.at_level("WARNING", logger="strands_robots.mesh.session"):
            stream_min_period_from_env()

        # "no step telemetry" is undiagnosable without both halves in the log.
        assert "STRANDS_MESH_STREAM_HZ" in caplog.text
        assert "fast" in caplog.text

    def test_a_non_positive_value_is_not_warned_about(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", "0")

        with caplog.at_level("WARNING", logger="strands_robots.mesh.session"):
            stream_min_period_from_env()

        # 0 is a request, not a mistake; warning about it trains operators to
        # ignore the warning that does mean something.
        assert caplog.text == ""


class TestNeitherCallSiteDividesTheRawEnvValue:
    """The duplicated expression cannot come back unnoticed."""

    @pytest.mark.parametrize("relpath", THROTTLE_SITES)
    def test_the_module_does_not_read_the_knob_directly(self, relpath) -> None:
        source = (_repo_root() / relpath).read_text(encoding="utf-8")
        tree = ast.parse(source)

        # A string literal naming the knob outside a comment means the module
        # resolved it itself instead of calling the shared resolver.
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == "STRANDS_MESH_STREAM_HZ"
        ]

        assert not literals, (
            f"{relpath} names STRANDS_MESH_STREAM_HZ itself; the period belongs to "
            "strands_robots.mesh.session.stream_min_period_from_env, which applies "
            "the shared hz_from_env domain check"
        )

    @pytest.mark.parametrize("relpath", THROTTLE_SITES)
    def test_the_module_calls_the_shared_resolver(self, relpath) -> None:
        source = (_repo_root() / relpath).read_text(encoding="utf-8")

        assert "stream_min_period_from_env()" in source


def _throttle_clock_base(relpath: str) -> tuple[str, str]:
    """Return the throttle comparison and the call its elapsed base reads.

    Locates the one ``<now> - <last> >= <period>`` comparison in the module
    whose right-hand side names the throttle period, then resolves the name on
    the left of the subtraction to the expression it is assigned from. Derived
    from the source rather than listed, so a site that spells its locals
    differently is still read.
    """
    source = (_repo_root() / relpath).read_text(encoding="utf-8")
    tree = ast.parse(source)

    gates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.GtE)
        and isinstance(node.left, ast.BinOp)
        and isinstance(node.left.op, ast.Sub)
        and "stream_min_period" in (ast.get_source_segment(source, node.comparators[0]) or "")
    ]
    assert len(gates) == 1, f"{relpath} holds {len(gates)} publish_step throttle gates, expected exactly 1"

    gate = gates[0]
    base = gate.left.left  # type: ignore[attr-defined]
    assert isinstance(base, ast.Name), (
        f"{relpath} subtracts from something this check cannot resolve: {ast.get_source_segment(source, gate.left)}"
    )
    assigned = [
        ast.get_source_segment(source, node.value) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == base.id for target in node.targets)
    ]
    assert len(assigned) == 1, f"{relpath} assigns {base.id} {len(assigned)} times, expected exactly 1"
    return (ast.get_source_segment(source, gate) or "", assigned[0])


class TestEveryThrottleMeasuresItsPeriodOnTheMonotonicClock:
    """The period is an elapsed interval, so the base cannot be the date.

    ``stream_min_period_from_env`` hands the same period to both sites, and both
    subtract two readings of a clock from each other to decide whether it has
    passed. On ``time.time()`` a backward wall-clock step makes that difference
    negative and the throttle refuses every later tick until the date catches
    up, which for a rollout means the stream stops and the gaps that did land
    stay correctly spaced -- nothing distinguishes it afterwards from a shorter
    run. The sim site read the date; the hardware site did not.

    Asserted structurally because the two sites sit in loops with nothing in
    common -- one commands a servo bus, the other steps a simulator -- so a
    behavioural pin has to be written per site and a third site would ship
    ungraded. This one is derived from the population above.
    """

    @pytest.mark.parametrize("relpath", THROTTLE_SITES)
    def test_the_elapsed_base_is_a_monotonic_reading(self, relpath) -> None:
        gate, base_expr = _throttle_clock_base(relpath)

        assert "monotonic" in base_expr, (
            f"{relpath} decides `{gate}` against `{base_expr}`; the period is an "
            "elapsed interval and belongs on time.monotonic(), which no clock "
            "correction moves"
        )
