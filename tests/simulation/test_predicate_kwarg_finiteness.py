# mypy: disable-error-code="arg-type"
"""A predicate-DSL kwarg that is not a finite number is refused at compile time.

Every numeric kwarg in a benchmark spec is coerced with a bare ``float(...)``
inside its predicate factory and then closed over, so before this guard a
``nan``/``inf`` threshold or weight compiled clean and only surfaced in the
evaluated result. Measured on the pre-fix tree:

* ``dense_reward: [{predicate: constant, value: .inf}]`` compiled, and a 10-step
  episode reported ``cumulative_reward = inf`` -> ``avg_reward = inf`` under
  ``status="success"``, silently poisoning whatever consumes the score.
* ``success: {all: [{predicate: body_above_z, body: cube, z: .nan}]}`` compiled
  into a clause that can never be satisfied (every comparison against ``nan`` is
  ``False``), so the rollout burned its whole step budget reporting an honest
  miss - the exact failure mode a typo'd body name is probed against the live sim
  to prevent (``can_resolve_body``), reached by a route that was not checked.
* ``value: "abc"`` escaped as a bare ``ValueError: could not convert string to
  float: 'abc'``, naming neither the predicate nor the field.

The guard lives in :func:`make_predicate` rather than in the spec compiler
because that is the only choke point every predicate call passes through:
``staged_reward`` compiles its per-stage ``reward`` / ``advance_when`` calls by
calling back into ``make_predicate``, so a guard in ``_compile_call`` would have
left nested stage calls unchecked (measured: a nested ``value: .inf`` still
produced ``on_step reward = inf``).

The registry-wide test below is the drift guard: it derives its cases from
``PREDICATE_REGISTRY`` and the factories' own parameter annotations, so a
predicate added later is covered by declaring its params - there is no separate
table to fall out of step with the registry.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark, compile_stop_when
from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    make_predicate,
    register_predicate,
)

INF = float("inf")
NAN = float("nan")

# Annotation -> a valid stand-in, so a parametrized case can supply every OTHER
# required param of a factory while poisoning exactly one.
# Strictly positive, because some params carry a narrower domain of their own
# (``base_velocity_tracking``'s ``tracking_sigma`` must be > 0) and this test is
# about finiteness, not about re-deriving each factory's own range checks.
_VALID_FOR_ANNOTATION: dict[str, Any] = {
    "str": "body_x",
    "float": 1.0,
    "int": 1,
    "bool": False,
    "str | None": None,
    "list[float]": [1.0, 1.0, 1.0],
    "list[int]": [1, 1, 1],
    "list[str]": ["body_x", "body_y"],
    "tuple[float, ...]": (1.0, 1.0, 1.0),
}
# Params whose domain is a nested predicate-call structure, not a number.
# ``staged_reward``'s stages are pinned explicitly further down.
_STRUCTURAL_ANNOTATIONS = frozenset({"list[Any]"})

_NUMERIC_ANNOTATIONS = frozenset({"float", "int", "list[float]", "list[int]", "tuple[float, ...]"})


def _numeric_params() -> list[tuple[str, str, str]]:
    """Every ``(predicate, param, annotation)`` in the registry with a numeric domain."""
    cases: list[tuple[str, str, str]] = []
    for name, factory in PREDICATE_REGISTRY.items():
        for param, annotation in getattr(factory, "__annotations__", {}).items():
            if param == "return":
                continue
            if str(annotation) in _NUMERIC_ANNOTATIONS:
                cases.append((name, param, str(annotation)))
    return sorted(cases)


def _kwargs_for(name: str) -> dict[str, Any]:
    """Valid kwargs covering every annotated param of *name*'s factory."""
    factory = PREDICATE_REGISTRY[name]
    kwargs: dict[str, Any] = {}
    for param, annotation in getattr(factory, "__annotations__", {}).items():
        if param == "return":
            continue
        kwargs[param] = _VALID_FOR_ANNOTATION[str(annotation)]
    return kwargs


def _spec(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"name": "finiteness-fixture", "default_robot": "so100", "max_steps": 10}
    base.update(over)
    return base


class _StubSim:
    """Fake SimEngine with one resolvable body, in the MuJoCo result shape."""

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        return {"status": "success", "position": [0.0, 0.0, 0.5], "quaternion": [1.0, 0.0, 0.0, 0.0]}

    def get_observation(self) -> dict[str, Any]:
        return {}


# --------------------------------------------------------------------------
# Registry-wide drift guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "param", "annotation"), _numeric_params())
@pytest.mark.parametrize("bad", [INF, -INF, NAN], ids=["inf", "-inf", "nan"])
def test_every_numeric_predicate_kwarg_refuses_a_non_finite_value(
    name: str, param: str, annotation: str, bad: float
) -> None:
    kwargs = _kwargs_for(name)
    kwargs[param] = [bad, 0.0, 0.0] if annotation.startswith(("list", "tuple")) else bad
    with pytest.raises(ValueError, match=f"{param}"):
        make_predicate(name, **kwargs)


@pytest.mark.parametrize(("name", "param", "annotation"), _numeric_params())
def test_every_numeric_predicate_kwarg_still_accepts_a_finite_value(name: str, param: str, annotation: str) -> None:
    """Control for the guard above: the valid domain is unchanged."""
    make_predicate(name, **_kwargs_for(name))


def test_no_registry_param_annotation_escapes_the_guard_unclassified() -> None:
    """A param annotation that is neither numeric nor a known non-numeric shape fails here.

    Forces a decision when a predicate is added with e.g. ``float | None`` or a
    ``Decimal`` threshold: either it joins the numeric sets in
    ``predicates._SCALAR_NUMBER_ANNOTATIONS`` / ``_NUMBER_SEQUENCE_ANNOTATIONS``
    and gets checked, or it is added here as explicitly non-numeric. Without this
    the new param would silently pass unvalidated.
    """
    known = _NUMERIC_ANNOTATIONS | _STRUCTURAL_ANNOTATIONS | {"str", "bool", "str | None", "list[str]"}
    unclassified = {
        (name, param, str(annotation))
        for name, factory in PREDICATE_REGISTRY.items()
        for param, annotation in getattr(factory, "__annotations__", {}).items()
        if param != "return" and str(annotation) not in known
    }
    assert not unclassified, f"unclassified predicate param annotations: {sorted(unclassified)}"


def test_the_guard_reads_the_same_annotations_the_valid_stand_ins_cover() -> None:
    """Every non-structural annotation in the registry has a valid stand-in above."""
    missing = {
        str(annotation)
        for factory in PREDICATE_REGISTRY.values()
        for param, annotation in getattr(factory, "__annotations__", {}).items()
        if param != "return"
        and str(annotation) not in _STRUCTURAL_ANNOTATIONS
        and str(annotation) not in _VALID_FOR_ANNOTATION
    }
    assert not missing, f"no valid stand-in for annotations: {sorted(missing)}"


# --------------------------------------------------------------------------
# Consequence pins, one per authoring surface
# --------------------------------------------------------------------------


def test_a_dense_reward_term_with_a_non_finite_weight_is_refused_not_scored_as_inf() -> None:
    with pytest.raises(ValueError, match="value must be a finite number"):
        DeclarativeBenchmark.from_dict(_spec(dense_reward=[{"predicate": "constant", "value": INF}]))


def test_a_finite_dense_reward_term_still_sums_into_the_episode_reward() -> None:
    """Control: the fix refuses a poisoned score, it does not stop scoring."""
    bench = DeclarativeBenchmark.from_dict(_spec(dense_reward=[{"predicate": "constant", "value": 2.0}]))
    reward = bench.on_step(_StubSim(), {}, {}).reward
    assert reward == pytest.approx(2.0)
    assert math.isfinite(reward)


def test_a_success_clause_with_a_non_finite_threshold_is_refused_not_unsatisfiable() -> None:
    with pytest.raises(ValueError, match="z must be a finite number"):
        DeclarativeBenchmark.from_dict(
            _spec(success={"all": [{"predicate": "body_above_z", "body": "cube", "z": NAN}]})
        )


def test_a_failure_clause_with_a_non_finite_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="z must be a finite number"):
        DeclarativeBenchmark.from_dict(
            _spec(failure={"any": [{"predicate": "body_below_z", "body": "cube", "z": INF}]})
        )


def test_a_stop_when_clause_with_a_non_finite_threshold_is_refused_before_the_rollout() -> None:
    """Same domain as the spec clauses: a clause that can never fire is not armed."""
    with pytest.raises(ValueError, match="z must be a finite number"):
        compile_stop_when({"predicate": "body_above_z", "body": "cube", "z": NAN})


@pytest.mark.parametrize("param", ["min", "max"])
def test_a_region_bound_with_a_non_finite_component_is_refused(param: str) -> None:
    """The ``list[float]`` half of the domain: one poisoned component is enough."""
    call = {"predicate": "inside_region", "body": "cube", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    call[param] = [0.0, NAN, 0.0]
    with pytest.raises(ValueError, match=f"'{param}' must contain finite numbers"):
        DeclarativeBenchmark.from_dict(_spec(success={"all": [call]}))


# --------------------------------------------------------------------------
# The nested staged_reward path - why the guard is not in the spec compiler
# --------------------------------------------------------------------------


def test_a_nested_stage_reward_call_with_a_non_finite_kwarg_is_refused() -> None:
    """A stage's ``reward`` never passes through ``_compile_call``."""
    with pytest.raises(ValueError, match="value must be a finite number"):
        DeclarativeBenchmark.from_dict(
            _spec(
                dense_reward=[
                    {"predicate": "staged_reward", "stages": [{"reward": {"predicate": "constant", "value": INF}}]}
                ]
            )
        )


def test_a_nested_advance_when_call_with_a_non_finite_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="z must be a finite number"):
        DeclarativeBenchmark.from_dict(
            _spec(
                dense_reward=[
                    {
                        "predicate": "staged_reward",
                        "stages": [
                            {
                                "reward": {"predicate": "constant", "value": 1.0},
                                "advance_when": {"predicate": "body_above_z", "body": "cube", "z": NAN},
                            },
                            {"reward": {"predicate": "constant", "value": 1.0}},
                        ],
                    }
                ]
            )
        )


def test_a_stage_bonus_that_is_not_finite_is_refused() -> None:
    """``bonus`` is not a registry factory param, so it carries the domain itself."""
    with pytest.raises(ValueError, match=r"stage\[0\]: bonus must be a finite number"):
        DeclarativeBenchmark.from_dict(
            _spec(
                dense_reward=[
                    {
                        "predicate": "staged_reward",
                        "stages": [{"reward": {"predicate": "constant", "value": 1.0}, "bonus": INF}],
                    }
                ]
            )
        )


def test_a_finite_stage_bonus_is_still_accepted() -> None:
    bench = DeclarativeBenchmark.from_dict(
        _spec(
            dense_reward=[
                {
                    "predicate": "staged_reward",
                    "stages": [{"reward": {"predicate": "constant", "value": 1.0}, "bonus": 5.0}],
                }
            ]
        )
    )
    assert math.isfinite(bench.on_step(_StubSim(), {}, {}).reward)


# --------------------------------------------------------------------------
# Domain edges: what the guard must NOT change
# --------------------------------------------------------------------------


def test_a_non_numeric_kwarg_now_names_the_predicate_and_the_field() -> None:
    """Pre-fix this escaped as a bare "could not convert string to float: 'abc'"."""
    with pytest.raises(ValueError, match="predicate 'constant': value must be a finite number, got 'abc'"):
        DeclarativeBenchmark.from_dict(_spec(dense_reward=[{"predicate": "constant", "value": "abc"}]))


def test_a_bool_is_refused_where_a_number_belongs() -> None:
    """``True`` is an ``int`` subclass and would act as a silent ``1.0`` threshold."""
    with pytest.raises(ValueError, match="z must be a finite number"):
        make_predicate("body_above_z", body="cube", z=True)


def test_a_bool_annotated_param_is_untouched_by_the_numeric_guard() -> None:
    make_predicate("body_on", body_a="a", body_b="b", require_contact=True)


def test_an_optional_robot_selector_is_untouched_by_the_numeric_guard() -> None:
    make_predicate("base_below_z", z=0.1, robot=None)


def test_a_negative_value_is_accepted_because_the_domain_is_signed() -> None:
    """A reward weight and a coordinate are both legitimately negative."""
    bench = DeclarativeBenchmark.from_dict(_spec(dense_reward=[{"predicate": "constant", "value": -3.5}]))
    assert bench.on_step(_StubSim(), {}, {}).reward == pytest.approx(-3.5)


def test_a_numpy_scalar_threshold_is_accepted() -> None:
    """Matches the other sim setters: a NumPy real scalar is a number here."""
    np = pytest.importorskip("numpy")
    make_predicate("body_above_z", body="cube", z=np.float32(0.2))


def test_an_int_threshold_is_accepted() -> None:
    make_predicate("body_above_z", body="cube", z=0)


def test_a_predicate_registered_without_annotations_is_exempt() -> None:
    """``register_predicate`` is documented as opting into an unsandboxed factory."""

    def _factory(threshold):  # type: ignore[no-untyped-def] # noqa: ANN001, ANN202 - the point of the test
        return lambda _sim: bool(threshold)

    name = "test_unannotated_exempt_from_finiteness"
    register_predicate(name, _factory)
    try:
        assert make_predicate(name, threshold=NAN) is not None
    finally:
        PREDICATE_REGISTRY.pop(name, None)
