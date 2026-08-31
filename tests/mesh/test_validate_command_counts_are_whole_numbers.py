"""A wire-side count is refused when it is fractional, never truncated.

``validate_command`` in ``strands_robots/mesh/security.py`` coerces four
caller-supplied integers through one helper, ``_coerce_int``: ``step.steps``,
and ``policy_port`` / ``action_horizon`` / ``n_steps`` on ``start`` /
``execute``. That helper's docstring says it "mirrors the defences in
``_coerce_float``", and it did mirror two of them -- the ``math.isfinite`` check
and the coercion-error wrap -- while missing the property that makes
``_coerce_float`` safe at all: it returns the caller's value or refuses it, and
never a third number.

``int(...)`` rounds toward zero, so the helper answered with a value nobody
sent. Three consequences, every one of them silent, all measured through the
public validator on a payload that has been through ``json.dumps`` /
``json.loads`` (Python renders an integer held in a float as ``2.0``, so a float
here is an ordinary wire shape rather than a contrived one):

* **A count the caller never asked for was honoured.** ``{"steps": 2.5}``
  validated as ``2``, and ``{"policy_port": 5556.7}`` as ``5556`` -- a port is an
  exact address, so that is a different endpoint, not a rounded magnitude.
* **The ceiling stopped refusing.** ``int(...)`` runs *before* the bounds
  compare, so every one of the four fields carried a value from above ``hi`` to
  exactly ``hi`` and accepted it, while the integer one step further was refused
  by name. ``{"n_steps": 10000000.5}`` was accepted as ``10000000`` and
  ``{"n_steps": 10000001}`` was refused. Same intent, two answers, decided by
  how the JSON number was spelled. The comment above those bounds says they
  "match the ``SimEngine.run_policy`` surface so a wire-side ``tell()`` cannot
  drive the runner to absurd frequencies / step counts", so a value over the cap
  is exactly what they exist to refuse.
* **A refusal named a number the caller never wrote.** ``{"steps": 0.9}``
  reported ``steps=0 out of bounds [1, 10000]`` -- the verdict was right and the
  quoted value was invented by the coercion.

The split this pins is the one the simulation side already applies to the same
quantity. ``SimEngine.step`` / ``run_policy`` resolve their step horizon through
``strands_robots.utils.positive_whole_number_error``, which accepts an integral
real and refuses a fractional one; ``tests/simulation/test_rollout_step_horizon_domain.py``
and ``tests/simulation/mujoco/test_control_substeps_validation.py`` are that
contract, and the second is titled for it -- a count must be "honored or
rejected, never silently clamped". So ``3.0`` stays accepted here (nothing is
lost in coercing it, and refusing it would break wire payloads for no gain) and
``2.5`` is refused under its own name.

Why the existing wire-side suites did not see it: the two that grade these
helpers cover the other edges. ``test_validate_command_finite_numerics`` pins
NaN / inf, and ``test_validate_command_numeric_coercion_none_handling`` pins
``None`` and type rejection -- including ``bool``, refused precisely because it
"must not be accepted as a count". Its ``step.steps`` type case parametrised
over ``str`` / ``list`` / ``dict`` and recorded the float behaviour as an aside:
"a float is accepted and truncated, matching ``int(...)`` semantics". That
sentence was the contract nobody had questioned; it has been corrected with this
change.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import pathlib
from typing import Any

import pytest

from strands_robots.mesh import security
from strands_robots.mesh.security import ValidationError, validate_command
from strands_robots.utils import positive_whole_number_error

# Every field ``validate_command`` routes through ``_coerce_int``, with the
# minimal payload that reaches it and the ``hi`` bound declared at its call
# site. ``TestEveryWireCountIsHeldToTheDomain`` derives the same set from the
# source, so a fifth call site added later fails this file rather than
# inheriting an exemption by being absent from a hand-written list.
_INT_FIELDS: dict[str, int] = {
    "steps": 10_000,
    "policy_port": 65_535,
    "action_horizon": 10_000,
    "n_steps": 10_000_000,
}

# A ``_coerce_float`` field, as the over-reach control: a fractional wait budget
# or rate is perfectly usable and must stay accepted.
_FLOAT_FIELD_HI: dict[str, float] = {"control_frequency": 2000.0, "duration": 3600.0}


def _payload(field: str, value: Any) -> dict[str, Any]:
    """Return the smallest command that carries *field* to its coercion.

    Routed through ``json.dumps`` / ``json.loads`` so the value under test is
    the shape a peer's payload really arrives as rather than a Python literal.
    """
    if field == "steps":
        cmd: dict[str, Any] = {"action": "step", "steps": value}
    else:
        cmd = {
            "action": "start",
            "instruction": "go",
            "policy_provider": "mock",
            field: value,
        }
    return json.loads(json.dumps(cmd))


def _validated(field: str, value: Any) -> Any:
    """Return what the validator made of *value*, or raise ``ValidationError``."""
    return validate_command(_payload(field, value))[field]


class TestAFractionalCountIsRefusedNotTruncated:
    """The regression: a value with a fractional part is not answered with another."""

    @pytest.mark.parametrize("field", sorted(_INT_FIELDS))
    def test_a_fractional_count_is_refused(self, field: str) -> None:
        with pytest.raises(ValidationError, match=f"{field} must be a whole number"):
            _validated(field, 2.5)

    @pytest.mark.parametrize("field", sorted(_INT_FIELDS))
    def test_the_refusal_quotes_the_value_the_caller_sent(self, field: str) -> None:
        with pytest.raises(ValidationError) as caught:
            _validated(field, 2.5)
        assert "2.5" in str(caught.value)

    def test_a_port_is_never_answered_with_a_different_port(self) -> None:
        """A port is an exact address; truncating it names another endpoint."""
        with pytest.raises(ValidationError, match="policy_port must be a whole number"):
            _validated("policy_port", 5556.7)


class TestTheCeilingRefusesWhicheverWayTheNumberIsSpelled:
    """A value over ``hi`` is refused whether it is spelled as a float or an int.

    This is the half that made the bound stop working: the coercion ran first,
    so a fractional value above the cap was carried down to the cap and
    accepted.
    """

    @pytest.mark.parametrize("field", sorted(_INT_FIELDS))
    def test_a_fractional_value_over_the_ceiling_is_refused(self, field: str) -> None:
        over = _INT_FIELDS[field] + 0.5
        with pytest.raises(ValidationError):
            _validated(field, over)

    @pytest.mark.parametrize("field", sorted(_INT_FIELDS))
    def test_it_is_not_accepted_as_the_ceiling(self, field: str) -> None:
        """The specific failure: the answer used to be ``hi`` itself."""
        hi = _INT_FIELDS[field]
        try:
            answered = _validated(field, hi + 0.5)
        except ValidationError:
            return
        pytest.fail(f"{field}={hi + 0.5} was accepted as {answered!r}, carried into range by the coercion")

    @pytest.mark.parametrize("field", sorted(_INT_FIELDS))
    def test_the_integer_one_step_further_is_still_refused_by_the_bound(self, field: str) -> None:
        """Unchanged: the bounds compare still owns the integer case, by name."""
        hi = _INT_FIELDS[field]
        with pytest.raises(ValidationError, match="out of bounds"):
            _validated(field, hi + 1)

    @pytest.mark.parametrize("field", sorted(_INT_FIELDS))
    def test_the_ceiling_itself_is_still_accepted(self, field: str) -> None:
        assert _validated(field, _INT_FIELDS[field]) == _INT_FIELDS[field]


class TestABelowFloorRefusalNamesTheCallersOwnValue:
    """``{"steps": 0.9}`` used to report ``steps=0``, a number nobody sent."""

    def test_the_message_carries_the_supplied_value(self) -> None:
        with pytest.raises(ValidationError) as caught:
            _validated("steps", 0.9)
        assert "0.9" in str(caught.value)

    def test_the_message_does_not_invent_the_truncated_one(self) -> None:
        with pytest.raises(ValidationError) as caught:
            _validated("steps", 0.9)
        assert "steps=0 " not in str(caught.value)


class TestAnIntegralFloatIsStillAccepted:
    """Over-reach control: ``3.0`` loses nothing in coercion, so it is honoured.

    ``json.dumps`` renders an integer held in a Python float as ``3.0``, so
    refusing this would refuse ordinary wire payloads for no gain. It is also
    the accepted side of the same split the simulation surface applies.
    """

    @pytest.mark.parametrize("field", sorted(_INT_FIELDS))
    def test_an_integral_float_coerces_to_its_integer(self, field: str) -> None:
        assert _validated(field, 4.0) == 4

    @pytest.mark.parametrize("field", sorted(_INT_FIELDS))
    def test_a_plain_int_is_unchanged(self, field: str) -> None:
        assert _validated(field, 4) == 4

    def test_a_negative_zero_float_is_still_refused_by_the_floor(self) -> None:
        """``-0.0`` is integral, so it reaches the bound and the floor refuses it."""
        with pytest.raises(ValidationError, match="out of bounds"):
            _validated("steps", -0.0)


class TestTheFloatSiblingIsUnchanged:
    """Over-reach control: a fractional *continuous* value is still usable."""

    @pytest.mark.parametrize("field", sorted(_FLOAT_FIELD_HI))
    def test_a_fractional_value_is_accepted(self, field: str) -> None:
        assert _validated(field, 12.5) == 12.5

    @pytest.mark.parametrize("field", sorted(_FLOAT_FIELD_HI))
    def test_a_fractional_value_over_the_ceiling_was_already_refused(self, field: str) -> None:
        """``float(...)`` never truncated, which is why this half always worked."""
        with pytest.raises(ValidationError, match="out of bounds"):
            _validated(field, _FLOAT_FIELD_HI[field] + 0.5)


class TestTheEarlierGuardsStillOwnTheirCases:
    """The new check sits after the type and finiteness guards, not in front."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_count_is_still_refused_as_non_finite(self, bad: float) -> None:
        """``nan.is_integer()`` is False, so an earlier guard must answer first."""
        with pytest.raises(ValidationError, match="steps must be finite"):
            _validated("steps", bad)

    def test_a_bool_is_still_refused_by_type(self) -> None:
        with pytest.raises(ValidationError, match="steps must be an integer, got bool"):
            _validated("steps", True)

    def test_a_string_is_still_refused_by_type(self) -> None:
        with pytest.raises(ValidationError, match="steps must be an integer, got str"):
            _validated("steps", "3")


class TestTheGuardSitsAfterTheFinitenessCheck:
    """Structural: the ordering the ``nan`` case above depends on, read off source.

    ``float("nan").is_integer()`` is False, so a whole-number check placed first
    would answer a non-finite value with the wrong reason. The behavioural cell
    covers that for ``steps``; this states the ordering itself so a reordering
    fails by name rather than only where a payload happens to reach it.
    """

    def test_both_checks_are_present(self) -> None:
        """Fails by name pre-fix, rather than as a ``ValueError`` from ``index``."""
        source = inspect.getsource(security._coerce_int)
        assert "must be finite" in source
        assert "must be a whole number" in source, "the whole-number guard is absent"

    def test_the_finiteness_check_precedes_the_whole_number_check(self) -> None:
        source = inspect.getsource(security._coerce_int)
        assert "must be a whole number" in source, "the whole-number guard is absent"
        assert source.index("must be finite") < source.index("must be a whole number")


class TestTheSplitIsTheOneTheSimulationSurfaceApplies:
    """Premise: the shared domain the sim uses makes exactly this cut.

    ``SimEngine.step`` / ``run_policy`` resolve a step horizon through
    ``positive_whole_number_error``. Grading the validator's verdicts against
    that domain's is what makes "the same quantity now answers the same way on
    both surfaces" a measurement rather than a claim, and it needs no simulator.
    """

    @pytest.mark.parametrize("value", [2.5, 0.9, 10_000.5])
    def test_the_shared_domain_refuses_what_the_validator_now_refuses(self, value: float) -> None:
        assert positive_whole_number_error(value, "steps", "step") is not None
        with pytest.raises(ValidationError):
            _validated("steps", value)

    @pytest.mark.parametrize("value", [3, 4.0, 10_000])
    def test_the_shared_domain_accepts_what_the_validator_accepts(self, value: float) -> None:
        assert positive_whole_number_error(value, "steps", "step") is None
        assert _validated("steps", value) == int(value)

    def test_the_domain_is_the_whole_number_one_and_not_the_strict_int_one(self) -> None:
        """Non-vacuity: the two candidate shared domains disagree on ``4.0``.

        ``positive_count_error`` refuses an integral float. Naming which domain
        the verdicts match is what makes the comparison above load-bearing.
        """
        from strands_robots.utils import positive_count_error

        assert positive_whole_number_error(4.0, "steps", "step") is None
        assert positive_count_error(4.0, "steps", "step") is not None


class TestEveryWireCountIsHeldToTheDomain:
    """Derived: the graded set is read off ``validate_command``'s own source.

    A fifth ``_coerce_int`` call site added later is refused here by name rather
    than inheriting an exemption by being absent from ``_INT_FIELDS``.
    """

    @staticmethod
    def _coerced_int_fields() -> set[str]:
        source = pathlib.Path(security.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        found: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "validate_command":
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                target = call.func
                if not isinstance(target, ast.Name) or target.id != "_coerce_int":
                    continue
                if call.args and isinstance(call.args[0], ast.Constant):
                    field = call.args[0].value
                    if isinstance(field, str):
                        found.add(field)
        return found

    def test_the_scan_finds_the_fields_this_file_grades(self) -> None:
        assert self._coerced_int_fields() == set(_INT_FIELDS)

    def test_the_scan_is_not_empty(self) -> None:
        """Non-vacuity: a scan that matched nothing would report clean."""
        assert len(self._coerced_int_fields()) >= 4

    def test_the_helper_refuses_a_fractional_value_directly(self) -> None:
        """Unit-level, so the guard is graded even for a field with no payload."""
        with pytest.raises(ValidationError, match="probe must be a whole number"):
            security._coerce_int("probe", 2.5, lo=1, hi=10, default=None)

    def test_the_helper_accepts_an_integral_float_directly(self) -> None:
        assert security._coerce_int("probe", 4.0, lo=1, hi=10, default=None) == 4

    def test_an_integral_float_is_exactly_what_is_integer_reports(self) -> None:
        """The predicate, stated locally so the guard is not its own oracle."""
        assert (4.0).is_integer() is True
        assert (2.5).is_integer() is False
        assert math.isnan(float("nan")) is True
