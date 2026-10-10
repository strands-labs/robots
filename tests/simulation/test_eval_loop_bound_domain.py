"""Accepted domain of ``n_episodes`` / ``max_steps`` on ``PolicyRunner.evaluate``.

These are the two bounds of that method's own episode loop. Every public entry
point above it already refuses a non-positive integer for both -
``SimEngine.eval_policy`` through :meth:`SimEngine._validate_positive_int`,
whose docstring names them and states the reason ("an eval over zero episodes,
or episodes of zero length, that fabricate a 0% success rate ... instead of
surfacing the caller's mistake") - and the benchmark route checks the horizon it
reads off the spec at that read.

``PolicyRunner`` applied no domain to either, and it is documented as drivable
directly. Unlike a horizon outside its domain, which degrades a rollout, a loop
bound outside it *removes the evaluation while still reporting one*:

* ``max_steps`` of ``0`` / ``-5`` / ``nan`` returned ``status="success"`` over
  two "completed" episodes with ``success_rate: 0.0``, ``success_measured:
  True`` and **zero applied actions** - and ``success_measured`` is the flag
  added so a ``0.0`` cannot be read as a measurement.
* ``n_episodes`` of ``0`` / ``-5`` reported the same clean ``0.0`` over zero
  episodes; ``success_rate`` is ``n_success / max(n_completed, 1)``, so the
  divisor guard that protects the division also turns "nothing ran" into a rate.
* ``max_steps=inf`` never terminated: ``while steps < max_steps`` has no false
  case, and the episode ran ~20k control steps per second indefinitely. Against
  a real policy each of those is a model inference.
* ``2.7`` / ``True`` truncated to a horizon nobody typed, and ``"3"`` / ``None``
  / a list leaked a bare ``'X' object cannot be interpreted as an integer``
  naming neither the parameter nor the method.

The established rule in this module, stated verbatim by ``_control_substeps``:
"The public entry points reject such a value with a structured error before
reaching the runner; this raise is the guarantee for callers driving
``PolicyRunner`` directly." ``evaluate`` already honors it for ``seed``,
``action_horizon`` (#2080), ``rtc_inference_timeout_s`` and ``control_substeps``.
These were the two remaining knobs of that signature, and the two that bound the
loop.

These tests pin, on the values rather than on wording:

* each bound refusing exactly what the facade refuses, with the shared rule's
  message, before ``sim.reset()``, before any inference and before
  ``set_eval_seed`` touches the process-global RNG - and each usable bound
  running with its accounting unchanged (one cell per bound, every value read),
* ``max_steps`` checked only where it is read (a ``spec=`` call never reads it,
  so refusing it there would reject a value that call ignores),
* and a structural sweep so a future runner surface cannot forget the domain.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
import random
from typing import Any

import numpy as np
import pytest

import strands_robots.simulation.policy_runner as runner_mod
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner
from strands_robots.utils import positive_count_error
from tests._package_ast import parse_source

from .test_benchmark_horizon_domain import _AttributeBenchmark
from .test_policy_runner_async_rtc import _ChunkPolicy, _CountingSim

# Values neither loop bound can be built from. ``0``/negatives/``False`` because
# an evaluation of zero episodes, or episodes of zero length, is not an
# evaluation; ``True`` because ``bool`` is an ``int`` subclass and acts as a
# silent count of 1; ``2.7`` because a fractional bound truncates (``max_steps``)
# or raises (``n_episodes``, consumed as a ``range()`` bound); ``nan``/``inf``
# because a comparison against them is uniformly False / uniformly True; ``"3"``
# / ``None`` / a list because they are not counts at all. The numpy and integral
# float spellings are refused by the facade's own strict-int domain, so including
# them is parity rather than a new narrowing - see
# ``TestEachBoundIsTheFacadesDomain``.
UNUSABLE: list[Any] = [
    0,
    -5,
    True,
    False,
    2.7,
    3.0,
    math.nan,
    math.inf,
    "3",
    None,
    [3],
    np.int64(3),
    np.float64(3.0),
]

# Values a bound CAN be built from: both parameters are plain positive ints.
USABLE: list[int] = [1, 2, 3, 10]

BOUNDS = ("n_episodes", "max_steps")


def _arm() -> tuple[_CountingSim, _ChunkPolicy]:
    """A GL-free engine plus a single-action policy that records its calls."""
    sim = _CountingSim()
    policy = _ChunkPolicy(sim, chunk=1, infer_sleep=0.0)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    return sim, policy


def _evaluate(runner: PolicyRunner, **overrides: Any) -> Any:
    """``evaluate`` with the legacy success path and both bounds overridable.

    A ``success_fn`` is always supplied so a success criterion is genuinely in
    force: without one the payload's ``success_measured`` is ``False`` by design
    and a ``0.0`` rate carries no claim, which is the confusion this domain is
    about. Funnelled through one annotated helper so the deliberately off-type
    bounds are stated once rather than suppressed per call.
    """
    kwargs: dict[str, Any] = {
        "instruction": "",
        "success_fn": lambda obs: False,
        "n_episodes": 2,
        "max_steps": 3,
        "control_frequency": 50.0,
    }
    kwargs.update(overrides)
    return runner.evaluate("arm", kwargs.pop("policy"), **kwargs)


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    """The json block of a result dict, or ``{}``."""
    for block in result.get("content", []):
        if "json" in block:
            return dict(block["json"])
    return {}


class TestEachBoundIsTheFacadesDomain:
    """One rule, so a bound refused for ``eval_policy`` cannot be run here.

    Each cell reads every value through a fresh engine: a refused value raises
    the shared rule's message before any side effect (``inf`` included - the
    bound is the applied-action count, not a clock, so zero applied actions is
    a complete proof of termination), and a usable one runs exactly
    ``n_episodes * max_steps`` actions.
    """

    @pytest.mark.parametrize("bound", BOUNDS)
    def test_the_runner_refuses_exactly_what_the_facade_refuses(self, bound: str) -> None:
        for index, value in enumerate([*UNUSABLE, *USABLE]):
            refused = SimEngine._validate_positive_int(value, bound, "eval_policy") is not None
            assert refused is (index < len(UNUSABLE)), f"facade verdict for {bound}={value!r}"
            sim, policy = _arm()
            if refused:
                with pytest.raises(ValueError) as exc:
                    _evaluate(PolicyRunner(sim), policy=policy, **{bound: value})
                assert str(exc.value) == positive_count_error(value, bound, "PolicyRunner.evaluate")
                assert (sim.send_count, policy.infer_starts) == (0, []), f"{bound}={value!r} ran"
                continue
            bounds = {"n_episodes": 2, "max_steps": 3, bound: value}
            payload = _payload(_evaluate(PolicyRunner(sim), policy=policy, **bounds))
            assert payload["episodes_completed"] == bounds["n_episodes"]
            assert payload["avg_steps"] == float(bounds["max_steps"])
            assert payload["success_measured"] is True
            assert sim.send_count == bounds["n_episodes"] * bounds["max_steps"]

    def test_the_episode_count_is_reported_first_when_both_are_unusable(self) -> None:
        """Deliberate: the outer bound of the loop is decided before the inner."""
        sim, policy = _arm()
        with pytest.raises(ValueError, match="n_episodes"):
            _evaluate(PolicyRunner(sim), policy=policy, n_episodes=0, max_steps=0)

    def test_the_process_global_rng_is_not_reseeded(self) -> None:
        """``set_eval_seed`` reseeds Python / NumPy / torch for the whole process."""
        for bound in BOUNDS:
            random.seed(1234)
            expected = [random.random() for _ in range(3)]
            random.seed(1234)
            sim, policy = _arm()
            with pytest.raises(ValueError):
                _evaluate(PolicyRunner(sim), policy=policy, seed=99, **{bound: 0})
            assert [random.random() for _ in range(3)] == expected, bound

    def test_the_documented_defaults_are_inside_the_domain(self) -> None:
        params = inspect.signature(PolicyRunner.evaluate).parameters
        for bound in BOUNDS:
            default = params[bound].default
            assert positive_count_error(default, bound, "PolicyRunner.evaluate") is None


class TestMaxStepsIsCheckedOnlyWhereItIsRead:
    """A ``spec=`` call takes its horizon off the benchmark and never reads this.

    Refusing the parameter there would reject a value the call ignores, so the
    check is gated on ``spec is None``. Effectiveness is knowable at this entry
    point because ``spec`` is a parameter of the same signature.
    """

    def test_the_spec_path_does_not_take_this_parameter_at_all(self) -> None:
        assert "max_steps" not in inspect.signature(PolicyRunner._evaluate_with_spec).parameters

    def test_an_unusable_step_cap_is_not_refused_alongside_a_spec(self) -> None:
        unread: list[Any] = [0, -5, math.inf, None]
        for value in unread:
            sim, policy = _arm()
            result = PolicyRunner(sim).evaluate(
                "arm",
                policy,
                spec=_AttributeBenchmark(2),
                n_episodes=1,
                max_steps=value,
                control_frequency=50.0,
            )
            assert (result["status"], sim.send_count) == ("success", 2), f"max_steps={value!r}"

    def test_the_specs_own_horizon_is_still_refused_at_its_read(self) -> None:
        sim, policy = _arm()
        result = PolicyRunner(sim).evaluate(
            "arm", policy, spec=_AttributeBenchmark(0), n_episodes=1, control_frequency=50.0
        )
        assert result["status"] == "error"
        assert "max_steps must be a positive integer" in str(result["content"])
        assert sim.send_count == 0

    def test_the_episode_count_is_refused_on_the_spec_path_too(self) -> None:
        """``n_episodes`` IS forwarded there, so its check is unconditional."""
        refused: list[Any] = [0, 2.7]
        for value in refused:
            sim, policy = _arm()
            with pytest.raises(ValueError, match="n_episodes"):
                PolicyRunner(sim).evaluate(
                    "arm", policy, spec=_AttributeBenchmark(2), n_episodes=value, control_frequency=50.0
                )


class TestEveryRunnerSurfaceOwnsItsLoopBounds:
    """A future runner surface taking either bound must validate what it takes."""

    @staticmethod
    def _public_surfaces(source: str) -> dict[str, set[str]]:
        """Public ``PolicyRunner`` methods mapped to the loop bounds they declare."""
        tree = parse_source(source)
        found: dict[str, set[str]] = {}
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.name != "PolicyRunner":
                continue
            for fn in cls.body:
                if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_"):
                    continue
                declared = {a.arg for a in fn.args.args + fn.args.kwonlyargs} & set(BOUNDS)
                if declared:
                    found[fn.name] = declared
        return found

    @staticmethod
    def _guarded(source: str, method: str) -> set[str]:
        """Bounds ``method`` passes to the shared count domain."""
        tree = parse_source(source)
        guarded: set[str] = set()
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.name != "PolicyRunner":
                continue
            for fn in cls.body:
                if not isinstance(fn, ast.FunctionDef) or fn.name != method:
                    continue
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "positive_count_error"
                        and node.args
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id in BOUNDS
                    ):
                        guarded.add(node.args[0].id)
        return guarded

    def _source(self) -> str:
        return pathlib.Path(inspect.getfile(runner_mod)).read_text(encoding="utf-8")

    def test_the_expected_surfaces_are_discovered(self) -> None:
        """Non-vacuity: the sweep really reads this module's public surfaces."""
        assert self._public_surfaces(self._source()) == {"evaluate": {"n_episodes", "max_steps"}}

    def test_each_public_surface_validates_every_bound_it_declares(self) -> None:
        source = self._source()
        adrift = {
            name: sorted(declared - self._guarded(source, name))
            for name, declared in self._public_surfaces(source).items()
            if declared - self._guarded(source, name)
        }
        assert not adrift, f"loop bounds reaching the episode loop unvalidated: {adrift}"

    def test_the_sweep_detects_a_planted_surface(self) -> None:
        """Otherwise an empty result would look like coverage."""
        planted = self._source() + (
            "\n\nclass PolicyRunner:  # a synthetic second definition, parsed not imported\n"
            "    def sweep(self, robot_name, policy, *, n_episodes=10, max_steps=300):\n"
            '        """A surface that forgot the domain."""\n'
            "        return {}\n"
        )
        assert self._public_surfaces(planted)["sweep"] == {"n_episodes", "max_steps"}
        assert self._guarded(planted, "sweep") == set()
