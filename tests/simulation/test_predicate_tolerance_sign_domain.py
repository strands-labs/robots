"""A predicate-DSL tolerance kwarg is refused when it is negative.

Every tolerance in the predicate DSL is compared against a distance, an absolute
difference or a squared magnitude - quantities that are never negative - so a
negative tolerance is not a looser bound, it is an unsatisfiable one. Measured on
the pre-fix tree with a cube resting on a tray (``distance_less_than`` satisfied
at ``threshold=0.30``):

* ``threshold=-0.30`` compiled clean and evaluated ``False`` at every pose, so a
  ``success`` clause built from it could never be satisfied and the rollout burned
  its whole step budget reporting an honest miss. That is the same outcome a
  ``nan`` threshold produces (see ``test_predicate_kwarg_finiteness``), reached by
  a route that was not checked - and unlike ``nan`` it reads as a *looser* bound.
* ``body_inside(xy_tol=-0.20)``, ``body_inside(z_tol=-0.20)`` and
  ``body_on(xy_tol=-0.20)`` behaved the same way: each was ``True`` at the
  positive value and permanently ``False`` at the negated one.
* ``body_upright`` and ``base_tipped`` already refused ``tol < 0`` inside their
  factories, so those two tolerance params were held to this domain and every
  other one the registry declares was not. The count is deliberately not stated:
  the sweep below derives its cases from the registry, so a predicate registered
  later adds its own and any number written here would go stale.

Signed params keep both signs, and the tests below pin that: ``body_on``'s
``z_offset`` is a signed offset that a caller legitimately sets negative (and
which still evaluates ``True`` there), ``base_velocity``'s ``vx`` is a velocity
component whose sign is a direction, and ``body_below_z``'s ``z`` is a
coordinate. A guard that constrained every numeric kwarg would break all three.

The registry-wide test is the drift guard: it derives its cases from
``PREDICATE_REGISTRY`` and the factories' own parameter annotations and names, so
a predicate added later is covered by naming its tolerance the way every shipped
one does - there is no separate table to fall out of step with the registry.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    make_predicate,
    register_predicate,
)

_NUMERIC_ANNOTATIONS = frozenset({"float", "int"})


def _names_a_tolerance(param: str) -> bool:
    """The contract, restated in the test: a whole-name or ``_tol``-suffix match.

    Deliberately a local copy rather than an import of the module's own
    predicate, so this file states the domain it pins instead of asserting the
    implementation against itself. ``test_module_rule_matches_the_pinned_rule``
    below is what keeps the two from drifting.
    """
    return param in ("tol", "threshold") or param.endswith("_tol")


def _numeric_params(factory: Any) -> dict[str, str]:
    """Map every param the factory annotates as a scalar number to that annotation."""
    annotations = getattr(factory, "__annotations__", {})
    return {
        p.name: str(annotations.get(p.name, ""))
        for p in inspect.signature(factory).parameters.values()
        if str(annotations.get(p.name, "")) in _NUMERIC_ANNOTATIONS
    }


# A placeholder for each annotation the shipped factories declare on a param with
# no default, keyed on the WHOLE annotation. Exact rather than a substring test
# for the same reason the tolerance rule below is: ``"str" in "list[str]"`` is
# true, so a substring match hands a container param a bare string, and
# :func:`strands_robots.utils.name_list_error` refuses one (a string is iterable
# per character, so reading it as a list of names is the mistake that domain
# exists to catch). An annotation this table does not name is refused rather than
# guessed at - see ``test_every_probe_value_has_the_shape_its_annotation_declares``
# and ``test_every_graded_predicate_can_actually_be_built``.
_PROBE_VALUES: dict[str, Any] = {
    "float": lambda param: 0.25,
    "int": lambda param: 1,
    "str": lambda param: f"probe_{param}",
    "list[str]": lambda param: [f"probe_{param}"],
    "list[float]": lambda param: [0.0, 0.0, 0.0],
}


def _buildable_kwargs(factory: Any) -> dict[str, Any]:
    """Kwargs that let *factory* be constructed, without asserting on any value.

    Body/container names resolve to nothing in a bare sim, which every predicate
    already degrades to ``False`` for, so a placeholder name is enough to reach
    the factory. Only params without a default need supplying, and each is
    supplied per its exact annotation via :data:`_PROBE_VALUES`; a param whose
    annotation is unknown is left out so the factory reports the missing argument
    itself rather than receiving a guessed value.
    """
    annotations = getattr(factory, "__annotations__", {})
    kwargs: dict[str, Any] = {}
    for p in inspect.signature(factory).parameters.values():
        if p.default is not inspect.Parameter.empty:
            continue
        probe = _PROBE_VALUES.get(str(annotations.get(p.name, "")))
        if probe is not None:
            kwargs[p.name] = probe(p.name)
    return kwargs


def _tolerance_cases() -> list[tuple[str, str]]:
    """Every ``(predicate, tolerance param)`` pair the shipped registry declares."""
    cases: list[tuple[str, str]] = []
    for name in sorted(PREDICATE_REGISTRY):
        for param in _numeric_params(PREDICATE_REGISTRY[name]):
            if _names_a_tolerance(param):
                cases.append((name, param))
    return cases


# A registry that stopped declaring tolerances would make every parametrized case
# below vacuous, so the sweep's own reach is asserted rather than assumed.
_MINIMUM_TOLERANCE_PARAMS = 6


class TestEveryDeclaredToleranceIsHeldNonNegative:
    def test_the_sweep_reaches_the_shipped_tolerance_params(self):
        cases = _tolerance_cases()
        assert len(cases) >= _MINIMUM_TOLERANCE_PARAMS, (
            f"the tolerance sweep found only {cases}; with fewer than "
            f"{_MINIMUM_TOLERANCE_PARAMS} pairs the parametrized cases below prove nothing"
        )
        # The two the factories already refused, and the four they did not.
        assert ("body_upright", "tol") in cases
        assert ("base_tipped", "tol") in cases
        assert ("distance_less_than", "threshold") in cases
        assert ("body_inside", "xy_tol") in cases
        assert ("body_inside", "z_tol") in cases
        assert ("body_on", "xy_tol") in cases

    @pytest.mark.parametrize(("name", "param"), _tolerance_cases())
    def test_a_negative_tolerance_is_refused(self, name, param):
        kwargs = _buildable_kwargs(PREDICATE_REGISTRY[name])
        kwargs[param] = -0.25
        with pytest.raises(ValueError, match=param):
            make_predicate(name, **kwargs)

    @pytest.mark.parametrize(("name", "param"), _tolerance_cases())
    def test_the_refusal_names_the_domain_and_the_value(self, name, param):
        kwargs = _buildable_kwargs(PREDICATE_REGISTRY[name])
        kwargs[param] = -0.25
        with pytest.raises(ValueError) as excinfo:
            make_predicate(name, **kwargs)
        message = str(excinfo.value)
        assert name in message
        assert ">= 0" in message
        assert "-0.25" in message

    @pytest.mark.parametrize(("name", "param"), _tolerance_cases())
    def test_zero_stays_accepted(self, name, param):
        """``0`` is the boundary the two shipped factory guards already admit.

        ``body_upright``/``base_tipped`` have refused ``tol < 0`` - not ``<= 0`` -
        since they were written, so admitting ``0`` keeps the whole family on one
        domain. Whether a zero-width tolerance is itself worth refusing is a
        separate question about those two shipped guards, not this one.
        """
        kwargs = _buildable_kwargs(PREDICATE_REGISTRY[name])
        kwargs[param] = 0.0
        assert callable(make_predicate(name, **kwargs))


class TestSignedParamsKeepBothSigns:
    """The no-overreach half: a guard over every numeric kwarg would break these."""

    def test_a_negative_body_on_z_offset_is_accepted_and_can_be_true(self, tmp_path):
        pytest.importorskip("mujoco")
        from strands_robots import Robot

        sim = Robot("so101", mode="sim", mesh=False)
        try:
            # Hoisted out of the asserts: these build the scene the verdict is
            # read from, and ``python -O`` would drop the calls with the checks.
            tray = sim.add_object(name="tray", shape="box", position=[0.15, 0.0, 0.005], size=[0.2, 0.2, 0.01])
            cube = sim.add_object(name="cube", shape="box", position=[0.15, 0.0, 0.03], size=[0.05, 0.05, 0.05])
            assert tray["status"] == "success", tray
            assert cube["status"] == "success", cube
            sim.step(200)
            # ``z_offset`` is signed: a caller lowers it to accept a body that
            # sits slightly below the reference and still counts as "on" it.
            pred = make_predicate("body_on", body_a="cube", body_b="tray", xy_tol=0.2, z_offset=-0.05)
            assert pred(sim) is True
        finally:
            sim.destroy()

    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("body_below_z", {"body": "cube", "z": -0.5}),
            ("body_above_z", {"body": "cube", "z": -0.5}),
            ("joint_below", {"joint": "j", "value": -1.0}),
            ("base_beyond_x", {"x": -2.0}),
            ("base_yaw_beyond", {"yaw": -1.0}),
            ("base_velocity", {"vx": -0.5}),
            ("base_height", {"target": 0.7, "weight": -1.0}),
            ("body_on", {"body_a": "a", "body_b": "b", "z_offset": -0.05}),
            ("constant", {"value": -1.0}),
        ],
    )
    def test_a_negative_signed_param_is_accepted(self, name, kwargs):
        assert callable(make_predicate(name, **kwargs))

    def test_no_signed_param_is_classified_as_a_tolerance(self):
        signed = {"z", "x", "y", "yaw", "value", "target", "weight", "vx", "vy", "wz", "z_offset", "tracking_sigma"}
        misread = sorted(p for p in signed if _names_a_tolerance(p))
        assert misread == [], f"the tolerance rule claims signed params {misread}"


class TestTheDomainComposesWithTheFinitenessGuard:
    def test_a_non_finite_tolerance_still_reports_finiteness(self):
        """The pre-existing half is unchanged and is reported first.

        ``nan`` is not negative, so the sign check must not displace the
        finiteness reason a caller acts on.
        """
        with pytest.raises(ValueError, match="finite"):
            make_predicate("distance_less_than", body_a="a", body_b="b", threshold=math.nan)

    def test_a_tolerance_on_a_later_registered_predicate_is_covered(self):
        """The rule is read from the param name, so it needs no registry edit."""

        def _factory(body: str, tol: float = 0.1):
            def check(_sim):
                return False

            return check

        register_predicate("probe_tolerance_domain", _factory)
        try:
            with pytest.raises(ValueError, match="tol"):
                make_predicate("probe_tolerance_domain", body="cube", tol=-0.1)
            assert callable(make_predicate("probe_tolerance_domain", body="cube", tol=0.1))
        finally:
            PREDICATE_REGISTRY.pop("probe_tolerance_domain", None)


class TestABooleanIsAnsweredBeforeTheCoercion:
    """A boolean numeric kwarg is refused by the finiteness guard, not coerced.

    The sign check reads ``float(value)``, which it may only do because
    :func:`~strands_robots.utils.finite_number_error` ran first and refuses
    ``bool`` and ``numpy.bool_`` explicitly. Without that ordering ``tol=True``
    would be written as a tolerance of ``1.0`` under ``status="success"`` - a
    ``bool`` is an ``int`` subclass, so ``float(True)`` is not an error.

    That delegation is the premise ``_kwarg_domain_error``'s entry in
    ``tests/simulation/test_input_validators_refuse_a_boolean.py`` states in
    prose, and nothing there checks it. It is pinned here instead, so the
    exemption rests on a measurement: deleting the ``finite_number_error`` call
    makes every refusal case below report ``DID NOT RAISE`` - ``tol=True``
    accepted as a tolerance of ``1.0`` - while the ``still accepted`` cases keep
    passing. Stated without counts on purpose, since the registry-derived sweep
    grows the refusal cases whenever a predicate declares a new tolerance.

    What these deliberately do *not* claim is an ordering. Moving the sign check
    above the finiteness guard changes no verdict here, because the sign check
    only refuses a *negative* value and ``float(True)`` is ``1.0``: a boolean
    falls through it either way and is answered by the shared domain. The
    substantive premise is that the delegation happens at all.

    They pass on both trees, and that is the point. The refusal is not new -
    ``finite_number_error`` has always held this domain - but a coercion now sits
    behind it, so the refusal has to stay.
    """

    @pytest.mark.parametrize("value", [True, False, np.True_, np.False_])
    @pytest.mark.parametrize(("name", "param"), _tolerance_cases())
    def test_a_boolean_tolerance_is_refused(self, name, param, value):
        kwargs = _buildable_kwargs(PREDICATE_REGISTRY[name])
        kwargs[param] = value
        with pytest.raises(ValueError, match="finite") as excinfo:
            make_predicate(name, **kwargs)
        # Answered by the finiteness reason, not the sign reason: the coercion the
        # sign check performs is never reached.
        assert ">= 0" not in str(excinfo.value)
        assert param in str(excinfo.value)

    @pytest.mark.parametrize("value", [True, np.True_])
    def test_a_boolean_signed_param_is_refused_too(self, value):
        """The domain is the whole numeric kwarg set, not just the tolerances."""
        with pytest.raises(ValueError, match="finite"):
            make_predicate("body_below_z", body="cube", z=value)

    @pytest.mark.parametrize("value", [1, np.int64(1), 1.0, 0])
    def test_a_value_equal_to_a_boolean_is_still_accepted(self, value):
        """The gate keys on the type, not the value.

        ``1`` coerces to the same ``1.0`` ``True`` would have written, so this
        separates "refuses a boolean" from "refuses anything equal to one".
        """
        assert callable(make_predicate("body_upright", body="cube", tol=value))


class TestTheRuleIsMatchedOnAWholeNameNotASubstring:
    """A param that merely contains the letters keeps its signed domain.

    These pass on both trees: they fail only if the whole-name/``_tol``-suffix
    rule is ever widened into a substring match, which would claim params like
    ``tolerance_scale`` or ``control`` that carry no such domain.
    """

    @pytest.mark.parametrize("param", ["tolerance", "protocol", "toleration", "threshold_scale", "control"])
    def test_a_param_whose_name_merely_contains_the_letters_accepts_a_negative(self, param):
        def _factory(body: str, **extra: float):
            def check(_sim):
                return False

            return check

        # The domain is read from ``__annotations__`` by param name, so declaring
        # the annotation is what puts this param in scope for the guard; the
        # variadic sink is only there to let the factory accept it.
        _factory.__annotations__[param] = "float"
        register_predicate(f"probe_substring_{param}", _factory)
        try:
            assert callable(make_predicate(f"probe_substring_{param}", body="cube", **{param: -1.0}))
        finally:
            PREDICATE_REGISTRY.pop(f"probe_substring_{param}", None)

    def test_every_probe_value_has_the_shape_its_annotation_declares(self):
        """The annotation is matched whole too: a container param gets a container.

        ``"str" in "list[str]"``, so a substring test on the annotation hands a
        ``list[str]`` param a bare string. The name-list domain refuses one, which
        makes every tolerance case on such a predicate unbuildable - the sweep
        stops grading the tolerances it was pointed at instead of reporting on
        them. Read from the registry so a predicate added later is checked too.
        """
        wrong: list[str] = []
        for name in sorted(PREDICATE_REGISTRY):
            factory = PREDICATE_REGISTRY[name]
            annotations = getattr(factory, "__annotations__", {})
            for param, value in _buildable_kwargs(factory).items():
                annotation = str(annotations.get(param, ""))
                if annotation.startswith("list[") and not isinstance(value, list):
                    wrong.append(f"{name}.{param}: {annotation} got {value!r}")
                elif annotation == "str" and not isinstance(value, str):
                    wrong.append(f"{name}.{param}: {annotation} got {value!r}")
                elif annotation in _NUMERIC_ANNOTATIONS and isinstance(value, str | list):
                    wrong.append(f"{name}.{param}: {annotation} got {value!r}")
        assert wrong == [], f"the probe value does not match the declared annotation for {wrong}"

    def test_every_graded_predicate_can_actually_be_built(self):
        """The sweep must reach every factory it points a tolerance case at.

        A required param whose annotation :data:`_PROBE_VALUES` does not name gets
        no value, so the factory raises about the missing argument. That is a loud
        failure rather than a wrong one, but it still stops the tolerance from
        being graded, so it is asserted here instead of surfacing case by case.
        """
        unbuildable: list[str] = []
        for name, _param in _tolerance_cases():
            factory = PREDICATE_REGISTRY[name]
            required = {
                p.name
                for p in inspect.signature(factory).parameters.values()
                if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            }
            missing = sorted(required - set(_buildable_kwargs(factory)))
            if missing:
                unbuildable.append(f"{name} (no probe value for {missing})")
        assert unbuildable == [], f"the sweep cannot build {unbuildable}, so their tolerances go ungraded"

    def test_module_rule_matches_the_pinned_rule(self):
        """The one assertion that names the new helper - the two rules must agree."""
        from strands_robots.simulation.predicates import _is_tolerance_param

        every_param = {p for name in PREDICATE_REGISTRY for p in inspect.signature(PREDICATE_REGISTRY[name]).parameters}
        disagree = sorted(p for p in every_param if _is_tolerance_param(p) != _names_a_tolerance(p))
        assert disagree == [], f"the module's rule and this file's rule disagree about {disagree}"
