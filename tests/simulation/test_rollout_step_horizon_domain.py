"""The rollout step horizon must be a whole number of steps.

``run_policy`` / ``start_policy`` / ``run_multi_policy`` take the rollout length
as a step count -- ``n_steps``, or the legacy ``max_steps`` alias -- and all
three resolve it through one shared helper,
``SimEngine._resolve_horizon``, which converts it to a wall-clock duration
(``duration = n_steps / control_frequency``).

That helper used to test the horizon with a bare ``<= 0``, which only sees the
sign. ``n_steps=2.7`` therefore ran two steps and ``n_steps=True`` ran one, each
reported as a successful rollout of a horizon the caller never asked for, while
every sibling count on the same call -- ``n_episodes``, ``action_horizon``,
``control_substeps`` -- and the identically-named ``eval_policy`` step budget
refused both through the shared positive-count domain. One parameter name,
``max_steps``, had two contracts depending on which method it was passed to.

A non-positive ``max_steps`` also reported ``n_steps must be > 0``, naming a
parameter the caller never passed, because the alias was normalized before the
check rather than validated under its own name.

These tests pin the corrected contract: the effective horizon is refused on the
shared count domain, under the name the caller wrote, on every entry point that
resolves one -- and every usable horizon is honoured exactly as before.

"Honoured" is the half that has to be graded on the executed step count, not on
the status string, and the probe values decide whether it can disagree at all.
``USABLE`` held ``[1, 4, 7]``, every one of which round-trips exactly through
``int((n_steps / control_frequency) * control_frequency)`` at the default 50 Hz,
so the honour cell agreed with the code whatever the loop did with the count --
and ``run_multi_policy`` was graded only on the refusal half. Measured on the
multi-robot loop before this suite covered it, two arms in one MuJoCo world:

| ``n_steps`` | ``control_frequency`` | steps executed |
| --- | --- | --- |
| ``1`` | ``49.0`` | **0** - "completed ... 0 synchronized steps" |
| ``2`` | ``49.0`` | **1** |
| ``29`` | ``50.0`` | **28** |
| ``57`` | ``50.0`` | **56** |
| ``113`` | ``50.0`` | **112** |
| ``123`` | ``30.0`` | **122** |
| ``4`` / ``10`` | ``50.0`` | 4 / 10 (exact - the old probe set) |

``_resolve_horizon`` returns the normalized count alongside the duration it
derives from it, and both multi-robot loops discarded the count and recomputed
``int(duration * control_frequency)`` from the float. ``n_steps`` is documented
as the exact horizon and as the parameter that "bypasses the lossy
``int(duration * control_frequency)`` conversion", and one merged frame is
recorded per timestep, so a truncated horizon is also a dataset episode short of
the frames the caller asked for. At 49 Hz it is a rollout that never ran,
reported as a completed success - the degenerate-success shape
``_validate_positive_int`` exists to refuse.

``LOSSY`` below carries those frequencies into the honour cells, and the
structural sweep at the end derives the rule over every rollout loop in the
package from the tree, so the Isaac loop is graded on a runner with no
``isaacsim`` installed rather than inheriting an exemption by being unrunnable.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import numpy as np
import pytest

import strands_robots
from strands_robots.policies import MockPolicy
from strands_robots.simulation import create_simulation
from strands_robots.simulation.base import SimEngine

#: Values no step horizon can be built from. ``0``/negative make the rollout a
#: no-op; ``2.7``/``3.0``/NumPy scalars/``True`` used to be truncated into a
#: different horizon; the rest never reached the loop bound at all. The NumPy
#: and integral-float spellings are refused by ``eval_policy``'s ``max_steps``
#: too, so refusing them here is parity with the sibling budget rather than a
#: new restriction (pinned by the parity test below).
UNUSABLE: list[Any] = [
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(-50, id="very-negative"),
    pytest.param(2.7, id="fractional"),
    pytest.param(3.0, id="integral-float"),
    pytest.param(True, id="bool-true"),
    pytest.param(False, id="bool-false"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param("5", id="numeric-string"),
    pytest.param([5], id="list"),
    pytest.param(np.int64(3), id="numpy-int"),
    pytest.param(np.float64(3.0), id="numpy-float"),
]

#: Horizons the rollout can execute exactly.
USABLE = [1, 4, 7]

#: ``(n_steps, control_frequency)`` pairs where ``n_steps / control_frequency``
#: is not exactly representable, so ``int(duration * control_frequency)`` lands
#: one step below the count. Every value is a perfectly ordinary horizon at a
#: perfectly ordinary rate; ``USABLE`` at the default 50 Hz cannot separate a
#: loop that honours the resolved count from one that re-derives it.
LOSSY = [
    pytest.param(1, 49.0, id="1-at-49hz-truncated-to-zero"),
    pytest.param(29, 50.0, id="29-at-50hz"),
    pytest.param(57, 50.0, id="57-at-50hz"),
    pytest.param(123, 30.0, id="123-at-30hz"),
]

#: The entry points that resolve a step horizon through ``_resolve_horizon``.
HORIZON_PARAMS = ["n_steps", "max_steps"]


@pytest.fixture
def sim():
    s = create_simulation()
    s.create_world()
    s.add_robot("arm1", data_config="so100")
    yield s
    s.cleanup()


def _text(result: dict) -> str:
    return " ".join(block["text"] for block in result.get("content", []) if "text" in block)


def _report(result: dict) -> dict:
    return next(block["json"] for block in result["content"] if "json" in block)


class _CountingPolicy(MockPolicy):
    """MockPolicy that records how many times the rollout queried it.

    Exercises the ``policy_object`` parameter (also newly documented) and makes
    "the refused call never started the rollout" a measurement rather than an
    inference about the status string.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def get_actions(self, observation_dict, instruction, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return await super().get_actions(observation_dict, instruction, **kwargs)


class TestTheStepHorizonIsAWholeNumberOfSteps:
    """An unusable horizon is a caller error, not a truncated rollout."""

    @pytest.mark.parametrize("param", HORIZON_PARAMS)
    @pytest.mark.parametrize("bad", UNUSABLE)
    def test_run_policy_refuses_it(self, sim, param, bad):
        result = sim.run_policy("arm1", policy_provider="mock", **{param: bad})
        assert result["status"] == "error", result
        assert f"{param} must be a positive integer" in _text(result)

    @pytest.mark.parametrize("param", HORIZON_PARAMS)
    @pytest.mark.parametrize("bad", UNUSABLE)
    def test_the_refusal_quotes_the_offending_value(self, sim, param, bad):
        text = _text(sim.run_policy("arm1", policy_provider="mock", **{param: bad}))
        assert repr(bad) in text or str(bad) in text

    @pytest.mark.parametrize("bad", UNUSABLE)
    def test_the_message_is_ascii(self, sim, bad):
        _text(sim.run_policy("arm1", policy_provider="mock", n_steps=bad)).encode("ascii")


class TestTheRefusalNamesTheParameterTheCallerPassed:
    """The legacy alias is validated before it is normalized away.

    Pre-fix ``max_steps`` was rewritten to ``n_steps`` first, so the refusal
    named a parameter the caller had not written.
    """

    @pytest.mark.parametrize("bad", UNUSABLE)
    def test_max_steps_is_named_and_n_steps_is_not(self, sim, bad):
        text = _text(sim.run_policy("arm1", policy_provider="mock", max_steps=bad))
        assert "max_steps must be a positive integer" in text
        assert "n_steps" not in text

    @pytest.mark.parametrize("bad", UNUSABLE)
    def test_n_steps_is_named_when_it_is_the_one_supplied(self, sim, bad):
        text = _text(sim.run_policy("arm1", policy_provider="mock", n_steps=bad))
        assert "n_steps must be a positive integer" in text
        assert "max_steps" not in text

    def test_n_steps_is_the_effective_knob_when_both_are_supplied(self, sim):
        # n_steps wins, so a bad max_steps alongside a usable n_steps is not
        # the value the rollout would have used and must not be reported.
        result = sim.run_policy("arm1", policy_provider="mock", n_steps=3, max_steps=0)
        assert result["status"] == "success", result
        assert _report(result)["n_steps"] == 3


class TestEveryEntryPointResolvingAHorizonSharesTheDomain:
    """``_resolve_horizon`` is the one funnel, so all three refuse alike."""

    @pytest.mark.parametrize("param", HORIZON_PARAMS)
    def test_start_policy_refuses_synchronously(self, sim, param):
        result = sim.start_policy("arm1", policy_provider="mock", **{param: 2.7})
        assert result["status"] == "error", result
        assert f"start_policy: {param} must be a positive integer" in _text(result)
        # A false "started" would also leave the robot marked as running.
        assert "arm1" not in sim._policy_threads

    @pytest.mark.parametrize("param", HORIZON_PARAMS)
    def test_run_multi_policy_refuses(self, sim, param):
        result = sim.run_multi_policy({"arm1": MockPolicy()}, **{param: 2.7})
        assert result["status"] == "error", result
        assert f"run_multi_policy: {param} must be a positive integer" in _text(result)


class TestTheHorizonAgreesWithTheEvalStepBudget:
    """One parameter name must not carry two contracts.

    ``eval_policy``'s ``max_steps`` is the same quantity -- a per-episode step
    cap -- and already went through the shared positive-count domain. These
    assertions are what stop the two drifting apart again.
    """

    @pytest.mark.parametrize("value", [*UNUSABLE, *[pytest.param(v, id=f"usable-{v}") for v in USABLE]])
    def test_run_policy_refuses_a_horizon_exactly_when_eval_policy_does(self, sim, value):
        rollout = sim.run_policy("arm1", policy_provider="mock", n_steps=value)
        budget = sim.eval_policy(robot_name="arm1", policy_provider="mock", n_episodes=1, max_steps=value)
        assert (rollout["status"] == "error") == (budget["status"] == "error"), (
            f"run_policy={rollout['status']} eval_policy={budget['status']} for {value!r}"
        )

    def test_both_surfaces_use_one_wording(self, sim):
        rollout = _text(sim.run_policy("arm1", policy_provider="mock", max_steps=2.7))
        budget = _text(sim.eval_policy(robot_name="arm1", policy_provider="mock", n_episodes=1, max_steps=2.7))
        assert "max_steps must be a positive integer, got 2.7." in rollout
        assert "max_steps must be a positive integer, got 2.7." in budget


class TestUsableHorizonsAreUnchanged:
    """The guard is additive: every horizon that worked still works."""

    @pytest.mark.parametrize("good", USABLE)
    @pytest.mark.parametrize("param", HORIZON_PARAMS)
    def test_the_rollout_executes_exactly_that_many_steps(self, sim, param, good):
        result = sim.run_policy("arm1", policy_provider="mock", fast_mode=True, **{param: good})
        assert result["status"] == "success", result
        assert _report(result)["n_steps"] == good

    @pytest.mark.parametrize(("good", "frequency"), LOSSY)
    @pytest.mark.parametrize("param", HORIZON_PARAMS)
    def test_run_policy_executes_a_lossy_horizon_exactly(self, sim, param, frequency, good):
        # Control for the multi-robot cell below: the single-robot loop forwards
        # the resolved count rather than re-deriving it, so it is already exact
        # at these frequencies.
        result = sim.run_policy(
            "arm1", policy_provider="mock", fast_mode=True, control_frequency=frequency, **{param: good}
        )
        assert result["status"] == "success", result
        assert _report(result)["n_steps"] == good

    @pytest.mark.parametrize("good", USABLE)
    @pytest.mark.parametrize("param", HORIZON_PARAMS)
    def test_run_multi_policy_executes_exactly_that_many_steps(self, sim, param, good):
        result = sim.run_multi_policy({"arm1": MockPolicy()}, **{param: good})
        assert result["status"] == "success", result
        assert _report(result)["steps"] == good

    @pytest.mark.parametrize(("good", "frequency"), LOSSY)
    @pytest.mark.parametrize("param", HORIZON_PARAMS)
    def test_run_multi_policy_executes_a_lossy_horizon_exactly(self, sim, param, frequency, good):
        result = sim.run_multi_policy({"arm1": MockPolicy()}, control_frequency=frequency, **{param: good})
        assert result["status"] == "success", result
        assert _report(result)["steps"] == good

    @pytest.mark.parametrize(("good", "frequency"), LOSSY)
    def test_a_horizon_the_loop_truncated_to_zero_is_not_reported_as_completed(self, sim, good, frequency):
        # The sharpest form of the finding: a rollout that never stepped came
        # back as a completed success, so the report and the world disagreed.
        result = sim.run_multi_policy({"arm1": MockPolicy()}, n_steps=good, control_frequency=frequency)
        assert _report(result)["steps"] > 0, _text(result)

    def test_the_duration_path_needs_no_horizon(self, sim):
        # No step horizon given: duration still resolves the rollout length.
        result = sim.run_policy("arm1", policy_provider="mock", duration=0.1, control_frequency=50.0, fast_mode=True)
        assert result["status"] == "success", result
        assert _report(result)["n_steps"] == 5


class TestARefusedHorizonNeverRunsTheRollout:
    """The guard precedes the loop, so no step and no inference happen."""

    @pytest.mark.parametrize("param", HORIZON_PARAMS)
    def test_the_policy_is_never_queried(self, sim, param):
        policy = _CountingPolicy()
        result = sim.run_policy("arm1", policy_object=policy, **{param: 2.7})
        assert result["status"] == "error", result
        assert policy.calls == 0

    def test_the_counting_policy_really_would_have_been_queried(self, sim):
        # Non-vacuity for the assertion above: a usable horizon does query it.
        policy = _CountingPolicy()
        result = sim.run_policy("arm1", policy_object=policy, n_steps=3, fast_mode=True)
        assert result["status"] == "success", result
        assert policy.calls > 0


class TestTheHorizonParametersAreDiscoverable:
    """A refused value must name a parameter the docstring documents.

    ``n_steps`` / ``max_steps`` carried no ``Args:`` entry at all, so the
    domain enforced above was undiscoverable from the API docs: the ``duration``
    entry told a reader that a step count "wins" over it without either count
    being documented anywhere. ``policy_object`` (recommended by the Returns:
    section) and ``max_onframe_failures`` were undocumented for the same reason.
    """

    @pytest.mark.parametrize("param", ["policy_object", "n_steps", "max_steps", "max_onframe_failures"])
    def test_run_policy_documents_the_parameter(self, param):
        doc = inspect.getdoc(SimEngine.run_policy) or ""
        assert f"\n    {param}:" in doc, f"{param} has no Args: entry"

    def test_every_run_policy_parameter_is_documented(self):
        # The whole signature, so a parameter added later cannot slip in
        # undocumented the way these four did.
        doc = inspect.getdoc(SimEngine.run_policy) or ""
        signature = inspect.signature(SimEngine.run_policy)
        undocumented = [name for name in signature.parameters if name != "self" and f"\n    {name}:" not in doc]
        assert undocumented == [], undocumented


def _package_root() -> pathlib.Path:
    """The installed package directory, derived from an imported symbol.

    A module-level helper rather than a literal, so the structural sweep below
    grades the tree it was imported from and the whole-tree preflight can
    resolve what this file walks.
    """
    return pathlib.Path(inspect.getfile(strands_robots)).parent


def _bound_names(function: ast.AST) -> set[str]:
    """Names bound in ``function`` - parameters and assignment targets."""
    assert isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
    args = function.args
    names = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    return names


def _derives_a_bound_from_the_duration(node: ast.AST) -> bool:
    """Whether ``node``'s value multiplies the duration by the frequency."""
    value = getattr(node, "value", None)
    if not isinstance(node, ast.Assign | ast.AnnAssign) or value is None:
        return False
    for sub in ast.walk(value):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult):
            if {ast.unparse(sub.left), ast.unparse(sub.right)} == {"duration", "control_frequency"}:
                return True
    return False


def _bounds_in(root: pathlib.Path) -> list[tuple[str, bool]]:
    """Return ``(label, honours_the_count)`` for every derived bound."""
    found: list[tuple[str, bool]] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - the package always parses
            continue
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            bounds = [
                n
                for n in ast.walk(function)
                if isinstance(n, ast.Assign | ast.AnnAssign) and _derives_a_bound_from_the_duration(n)
            ]
            if not bounds or "n_steps" not in _bound_names(function):
                continue
            # Assignments reached only when a test on the count says so.
            conditional = {
                id(d)
                for node in ast.walk(function)
                if isinstance(node, ast.If) and "n_steps" in ast.unparse(node.test)
                for d in ast.walk(node)
            }
            for bound in bounds:
                assert bound.value is not None  # guaranteed by the predicate above
                honours = "n_steps" in ast.unparse(bound.value) or id(bound) in conditional
                label = f"{path.relative_to(root)}::{function.name}:{bound.lineno}"
                found.append((label, honours))
    return found


class TestEveryRolloutLoopHonoursTheResolvedStepCount:
    """A loop given a step count must not re-derive one from the duration.

    ``_resolve_horizon`` returns ``(duration, n_steps, error)``: on the horizon
    path the duration it returns is derived AS ``n_steps / control_frequency``,
    so multiplying it back by the frequency is a float round trip, not a
    conversion. The population is derived from the tree - every function that
    assigns a value built from ``duration * control_frequency`` - so a fourth
    rollout loop is graded on arrival instead of being absent from a tuple, and
    the Isaac loop is graded on a runner with no ``isaacsim`` installed.

    Both spellings of honouring the count are accepted, because both ship: the
    conditional expression the multi-robot loops now use and the ``if n_steps is
    not None:`` statement ``PolicyRunner.run`` has always used. Grading one
    spelling would report a tree using the other as clean.
    """

    def test_every_loop_in_the_package_honours_it(self):
        adrift = [label for label, honours in _bounds_in(_package_root()) if not honours]
        assert adrift == [], adrift

    def test_the_sweep_really_reaches_the_rollout_loops(self):
        # Non-vacuity: a sweep that selected nothing would read as a clean tree.
        # Floored well below the three loops shipped so a fourth needs no edit.
        found = _bounds_in(_package_root())
        assert len(found) >= 3, found
        assert any("policy_runner.py" in label for label, _ in found), found
        assert any("isaac" in label or "mujoco" in label for label, _ in found), found

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            pytest.param(
                "    total_steps = n_steps if n_steps is not None else int(duration * control_frequency)",
                True,
                id="conditional-expression",
            ),
            pytest.param(
                "    if n_steps is not None:\n"
                "        total_steps = n_steps\n"
                "    else:\n"
                "        total_steps = int(duration * control_frequency)",
                True,
                id="if-statement",
            ),
            pytest.param("    total_steps = int(duration * control_frequency)", False, id="bare-re-derivation"),
            pytest.param("    total_steps = int(control_frequency * duration)", False, id="bare-operands-reversed"),
        ],
    )
    def test_both_spellings_of_honouring_the_count_are_recognized(self, tmp_path, body, expected):
        header = "def run(policies, duration, control_frequency, n_steps):\n"
        (tmp_path / "loop.py").write_text(header + body + "\n", encoding="utf-8")
        assert [honours for _, honours in _bounds_in(tmp_path)] == [expected]

    def test_a_loop_with_no_step_count_is_outside_the_population(self, tmp_path):
        # ``duration`` is the only horizon some loops take; there is no count to
        # honour there, so selecting it would need an exemption list.
        (tmp_path / "loop.py").write_text(
            "def run(duration, control_frequency):\n    total_steps = int(duration * control_frequency)\n",
            encoding="utf-8",
        )
        assert _bounds_in(tmp_path) == []
