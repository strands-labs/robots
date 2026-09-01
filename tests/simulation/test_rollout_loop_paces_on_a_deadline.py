# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A rollout runs for the wall clock it was asked for, whatever a step costs.

``duration`` is documented as wall-clock seconds and ``fast_mode=False`` as
real-time pacing, and both were false by the cost of one step: the loop slept
``1 / control_frequency`` AFTER each step, which is a delay where a rate needs a
deadline. The wall clock a step spends on inference, the physics substeps, a
render for the video and the recorder's frame write was added to the period
instead of subtracted from it, so the loop ran at ``1 / (period + work)``.
Measured on a MuJoCo so101 rollout asking for 2.0 s, before and after:

============================  =========  ========  =======  =========
case                          requested  achieved  Hz       asked Hz
============================  =========  ========  =======  =========
free policy, 50 Hz (before)      2.0 s     2.15 s    46.4      50
free policy, 50 Hz (after)       2.0 s     2.00 s    49.97     50
10 ms inference, 50 Hz (b)       2.0 s     3.15 s    31.8      50
10 ms inference, 50 Hz (a)       2.0 s     2.00 s    49.98     50
30 ms inference, 30 Hz (b)       2.0 s     3.90 s    15.4      30
30 ms inference, 30 Hz (a)       2.0 s     2.00 s    29.98     30
============================  =========  ========  =======  =========

Sim time was exact throughout (2.0 s of integration in every row), so nothing
about the physics or the recorded timebase was wrong - only the wall clock the
caller was promised, and the rate anything watching the rollout saw.

:mod:`strands_robots.mesh.pacing` already owned this argument and the fix: its
``Ticker`` is a deadline, and it DROPS missed deadlines rather than chasing them,
so an overrunning step is followed by a gap rather than by a burst of
back-to-back actions at the arm. The mesh's publish loops were converted for the
same reason (see ``tests/test_mesh_state_loop_rate.py``, whose inventory scans
for the ``stop_event.wait(period)`` spelling of a delay); the rollout loops paced
with the other spelling, ``time.sleep(period)``, and that inventory could not see
them. The source-level cells below cover that second spelling for the simulation
package, because the timing cells - being timing cells - could in principle be
met by luck on a heavily loaded machine.
"""

from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.policy_runner import PolicyRunner
from tests.simulation.test_policy_runner import FakeSim

_PACKAGE = Path(__file__).resolve().parents[2] / "strands_robots"

#: The rollout loops that must pace on a deadline, as (module path, qualified
#: function). Read from source rather than imported: the Isaac backend needs
#: ``isaacsim``, which no CI runner has, and its loop is exactly the same shape.
_ROLLOUT_LOOPS = (
    ("simulation/policy_runner.py", "PolicyRunner.run"),
    ("simulation/mujoco/simulation.py", "MuJoCoSimEngine.run_multi_policy"),
    ("simulation/isaac/simulation.py", "IsaacSimulation.run_multi_policy"),
)


def _busy_wait(seconds: float) -> None:
    """Burn ``seconds`` of wall clock without sleeping.

    A ``time.sleep`` would let the pacer's own wait overlap the work on some
    platforms, and is itself inflated on a host that taxes blocking waits (the
    cost :func:`strands_robots.mesh.pacing.sleep_penalty_s` measures), so the
    step's cost is spent on the CPU exactly as inference spends it.
    """
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def _run(*, n_steps: int, frequency: float, on_frame: Any = None, fast_mode: bool = False) -> float:
    """Drive the real ``PolicyRunner.run`` loop; return the wall clock it took."""
    sim = FakeSim()
    policy = MockPolicy()
    start = time.perf_counter()
    result = PolicyRunner(sim).run(
        "fake_robot",
        policy,
        n_steps=n_steps,
        control_frequency=frequency,
        fast_mode=fast_mode,
        on_frame=on_frame,
    )
    elapsed = time.perf_counter() - start
    assert result["status"] == "success", result
    return elapsed


class TestARolloutTakesTheWallClockItWasAskedFor:
    """The headline behaviour: a step's own cost comes out of the period."""

    def test_the_work_of_a_step_comes_out_of_the_period_not_on_top_of_it(self) -> None:
        """32 ms of per-step work inside a 40 ms period still ticks at 25 Hz.

        The median step gap rather than the total elapsed, for the reason the
        mesh's equivalent cell gives: a median is unmoved by the odd 100 ms of
        descheduling a loaded CI host will hand any thread, while the regression
        it grades is every gap. On the delay pacing this gap was ``period +
        work`` = 72 ms; a deadline pacer absorbs the work and gives 40 ms.
        """
        frequency, n_steps, work_s = 25.0, 20, 0.032
        period = 1.0 / frequency
        stamps: list[float] = []

        def on_frame(_idx: int, _obs: Any, _act: Any) -> None:
            _busy_wait(work_s)
            stamps.append(time.perf_counter())

        _run(n_steps=n_steps, frequency=frequency, on_frame=on_frame)
        gaps = sorted(b - a for a, b in zip(stamps, stamps[1:], strict=False))
        assert gaps, "no steps to measure"
        median_gap = gaps[len(gaps) // 2]
        assert median_gap < period * 1.5, (
            f"median step gap {median_gap * 1000:.1f}ms against a {period * 1000:.0f}ms period with "
            f"{work_s * 1000:.0f}ms of work per step - the work is being added to the period instead "
            "of subtracted from it, so the loop runs at 1 / (period + work). Use mesh.pacing.Ticker."
        )

    def test_a_free_step_still_takes_the_period(self) -> None:
        """The pace is a floor as well as a ceiling: 25 steps at 50 Hz is 0.5 s.

        Without this, "fast" would be an equally passing answer to the cell
        above, and ``fast_mode=False`` would no longer mean anything.
        """
        frequency, n_steps = 50.0, 25
        requested = n_steps / frequency
        elapsed = _run(n_steps=n_steps, frequency=frequency)
        assert elapsed >= requested * 0.9, (
            f"a paced rollout of {n_steps} steps at {frequency:.0f}Hz returned in {elapsed:.3f}s, "
            f"under the {requested:.3f}s its own rate implies - it is not pacing at all."
        )

    def test_fast_mode_is_still_unpaced(self) -> None:
        """``fast_mode=True`` must pay no pace at all, deadline or otherwise."""
        frequency, n_steps = 50.0, 25
        elapsed = _run(n_steps=n_steps, frequency=frequency, fast_mode=True)
        assert elapsed < (n_steps / frequency) * 0.5, (
            f"fast_mode=True took {elapsed:.3f}s for {n_steps} steps at {frequency:.0f}Hz - "
            "it is being paced despite asking not to be."
        )


class TestAnOverrunningStepIsFollowedByAGapNotABurst:
    """A dropped deadline must not be chased.

    Chasing fires several steps back to back to pay off the debt, which on a
    robot means a burst of setpoints microseconds apart - a worse lie about what
    the controller did than the stall itself. This is a property of the chosen
    pacer rather than a regression from the delay, and it is what stops the
    obvious alternative fix (accumulate the debt and sleep the remainder) from
    passing.
    """

    def test_one_slow_step_does_not_fire_a_burst_of_catch_up_steps(self) -> None:
        frequency, n_steps, period = 50.0, 20, 0.02
        stamps: list[float] = []

        def on_frame(step_idx: int, _obs: Any, _act: Any) -> None:
            if step_idx == 5:  # one step worth about five periods
                _busy_wait(period * 5)
            stamps.append(time.perf_counter())

        _run(n_steps=n_steps, frequency=frequency, on_frame=on_frame)
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
        instant = [g for g in gaps if g < 0.002]
        assert len(instant) <= 1, (
            f"{len(instant)} of {len(gaps)} step gaps were under 2ms at a {period * 1000:.0f}ms "
            "period - the loop is chasing the deadlines it missed during the slow step."
        )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/fd is Linux-only")
def test_a_paced_rollout_releases_the_pacers_descriptors() -> None:
    """A Ticker owns a selector and a socketpair, so an unclosed one leaks two.

    An eval loop builds one rollout per episode, so a leak here is a file
    descriptor exhaustion over a long evaluation rather than a one-off.
    """
    fds = Path("/proc/self/fd")
    before = len(os.listdir(fds))
    for _ in range(5):
        _run(n_steps=2, frequency=200.0)
    after = len(os.listdir(fds))
    assert after <= before + 1, (
        f"{after - before} descriptors outlived 5 paced rollouts (before {before}, after {after}) - "
        "the pacer is not being closed."
    )


def _function_source(rel_path: str, qualified: str) -> str:
    """Return the CODE of ``qualified`` in ``rel_path`` - no comments, no docstring.

    Read through :mod:`ast` rather than ``inspect`` because one graded module
    imports ``isaacsim``, and unparsed rather than sliced out of the file because
    a converted loop explains in prose what it stopped doing: the first version
    of this scanner failed on the very comment documenting the fix. A scanner
    that reads its own documentation punishes documenting.
    """
    path = _PACKAGE / rel_path
    tree = ast.parse(path.read_text())
    cls_name, _, fn_name = qualified.rpartition(".")
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != cls_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name == fn_name:
                body = list(item.body)
                if ast.get_docstring(item) is not None:
                    body = body[1:]
                return "\n".join(ast.unparse(stmt) for stmt in body)
    raise AssertionError(f"{qualified} not found in {rel_path}")


@pytest.mark.parametrize(("rel_path", "qualified"), _ROLLOUT_LOOPS, ids=[q for _, q in _ROLLOUT_LOOPS])
def test_every_rollout_loop_paces_through_the_shared_ticker(rel_path: str, qualified: str) -> None:
    """All three rollout loops pace in one place, so one cannot regress alone.

    A loop that quietly kept ``time.sleep(period)`` while its siblings were
    converted is harder to notice than all three being slow, because the one
    that is late looks like the backend that is genuinely slower.
    """
    source = _function_source(rel_path, qualified)
    assert "Ticker(" in source, f"{qualified} does not pace on mesh.pacing.Ticker"
    assert "time.sleep(" not in source, (
        f"{qualified} sleeps a period again - that delay adds the step's work to the period, "
        "so the loop runs at 1 / (period + work). Use mesh.pacing.Ticker."
    )


def test_no_loop_in_the_simulation_package_paces_on_a_rate_derived_sleep() -> None:
    """The inventory: this is only cured if no pacer was missed.

    Scans for the SHAPE - ``time.sleep(x)`` where ``x`` is a name assigned from a
    division by a rate and never from a subtraction - rather than for the one
    spelling that was fixed. Two exclusions carry the whole rule.

    The subtraction: a sleep computed as ``interval - elapsed`` has already taken
    the body's cost off the period, which is what the two rendering pacers do.

    The loop: this does NOT require the call to sit inside a ``while``/``for``,
    and that is the correction that makes it see the site this change fixes. The
    runner's pace lived in a per-step helper called from the loop rather than in
    the loop body, so a version of this check that walked loops named the two
    backends and missed the runner - the same near miss as the mesh inventory,
    which looked for one attribute name and missed the teleop loop whose event is
    spelled differently.

    Scoped to the rollout/render loops of the simulation package. The same shape
    exists in the bounded publish bursts of ``use_rtps`` / ``use_rosbridge``,
    where ``rate`` is an inter-message spacing for a fixed ``count`` rather than
    a control loop promising a duration; those are a separate surface with their
    own contract and are deliberately out of this check's scope.
    """
    offenders: dict[tuple[str, int], str] = {}
    for path in sorted((_PACKAGE / "simulation").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            # Names bound in this function (nested helpers included) to a
            # quotient with no subtraction in it - i.e. a bare 1 / rate period.
            rate_derived: set[str] = set()
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                dumped = ast.dump(node.value)
                if "Div()" in dumped and "Sub()" not in dumped:
                    rate_derived.add(target.id)
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call) or getattr(call.func, "attr", "") != "sleep":
                    continue
                if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
                    continue
                if call.args[0].id in rate_derived:
                    rel = str(path.relative_to(_PACKAGE))
                    offenders[rel, call.lineno] = (
                        f"{rel}:{call.lineno}: sleeps {call.args[0].id}, a period derived from a rate"
                    )
    assert not offenders, (
        "these sleeps pace a loop, so the body's work is added to the period and the loop runs at "
        f"1 / (period + work); pace them on mesh.pacing.Ticker: {sorted(offenders.values())}"
    )
