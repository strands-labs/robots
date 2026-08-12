"""The simulation input validators refuse a boolean instead of coercing it to 1.0 (#1842).

#1837 settled the actuator command and #1841 the runtime state writers, putting
the predicate behind one owner (:func:`strands_robots.utils.is_boolean`). Six
input validators in ``strands_robots/simulation/`` were not routed through it,
and they failed in two distinct ways.

**Mode A - no boolean check at all.** ``_coerce_finite_vector`` and
``_normalize_gravity`` coerced with a bare ``float()``, so every spelling of a
boolean arrived as a silent ``1.0`` under ``status="success"``::

    raycast(origin=[True, 0, 0])   -> success, cast from x=1.0
    set_gravity(True)              -> success, gravity [0, 0, +1.0] - pointing *up*
    set_gravity([True, 0, 0])      -> success, gravity [1.0, 0, 0]
    set_obs_noise(std=True)        -> success, a noise sigma of 1.0
    randomize(mass=(True, 2.0))    -> success, a scale range of (1.0, 2.0)

``_normalize_gravity`` refused a ``numpy.bool_`` *scalar* only incidentally:
``numpy.bool_`` is not registered as ``numbers.Real``, so it missed the scalar
branch, fell through to the vector path and failed ``len()`` - surfacing as
``'gravity' must be a 3-element list of numbers (len() of unsized object)``,
which describes neither the value nor the reason.

**Mode B - the rule was documented and ``isinstance(x, bool)`` did not implement
it.** ``_validate_timestep`` and ``_validate_mass`` both already said a bool is
"rejected explicitly since ``True`` would act as a silent 1-second step" / "1 kg
body" - and used ``isinstance(value, bool)``, which ``numpy.bool_`` is not a
subclass of. So the guard held for a hand-typed literal and vanished for the
spelling computed code actually produces (``gripper > 0.5``)::

    set_timestep(np.True_)               -> success, dt = 1.0 s
    set_body_properties(mass=np.True_)   -> success, a 1 kg body

A ``dt`` of one second is not a mis-sized step, it is a different simulation.

Every boolean assertion here fails on pre-fix code, where the call returned
``status="success"``. The ``still_accepted`` tests are the over-reach controls:
``1``, ``np.uint8(1)``, ``np.int64(1)`` and a 0-d numeric array coerce to the
same ``1.0`` the boolean would have written, so they separate "refuses a
boolean" from "refuses anything equal to 1" - the gate keys on the type, not the
value. ``nan`` / ``inf`` and non-numeric values keep their own distinct
messages; the bool gate is additive.

:class:`TestTheBooleanDomainIsStructurallyClosed` is what makes this the last
pass over the question rather than a fourth: it enumerates the input validators
and fails when a new one coerces without consulting the shared predicate.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import pathlib
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation import base as sim_base
from strands_robots.simulation.base import (
    _BOOLEAN_WORLD_REASON,
    SimEngine,
    finite_non_negative_error,
    randomization_range_error,
)
from strands_robots.utils import is_boolean

# Every spelling of a boolean that can reach a validator: a python bool, a numpy
# boolean scalar (what ``gripper > 0.5`` produces), and a 0-d boolean array (what
# ``np.array(True)`` or a reduction produces). numpy.bool_ is not a bool
# subclass, so an isinstance-only gate catches the first two and misses the rest -
# which is exactly the Mode B defect this pins.
_BOOLEANS = [True, False, np.True_, np.bool_(False), np.array(True)]
_BOOLEAN_IDS = ["True", "False", "np_true", "np_false", "zero_d_array"]

# The subset that must be refused where the domain also excludes zero (a
# timestep, a mass): ``False`` and ``np.False_`` would be refused by the
# positivity check anyway, so pinning them there would not distinguish the bool
# gate from the pre-existing one.
_TRUTHY_BOOLEANS = [True, np.True_, np.array(True)]
_TRUTHY_IDS = ["True", "np_true", "zero_d_array"]

# Numeric values a caller legitimately passes. ``1``, ``np.uint8(1)`` and
# ``np.int64(1)`` are the load-bearing entries: they coerce to the same 1.0 the
# boolean would have written.
_POSITIVE_NUMBERS = [1, 0.5, 2, np.float64(0.3), np.int64(1), np.uint8(1), np.float32(0.7), np.array(0.7)]
_POSITIVE_IDS = ["int_1", "float", "int_2", "np_float", "np_int64_1", "np_uint8_1", "np_float32", "zero_d_array"]

_ANY_NUMBERS = [0, 1, -1, 0.5, -1.25, np.float64(0.3), np.int64(1), np.uint8(1), np.array(0.7)]
_ANY_IDS = ["int_0", "int_1", "int_neg1", "float", "neg_float", "np_float", "np_int64_1", "np_uint8_1", "zero_d_array"]

_NON_FINITE = [float("nan"), float("inf"), float("-inf")]
_NON_FINITE_IDS = ["nan", "inf", "neg_inf"]


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


# ---------------------------------------------------------------------------
# Mode A: the two validators with no boolean check at all
# ---------------------------------------------------------------------------


class TestGravityRefusesABoolean:
    """``_normalize_gravity`` backs ``set_gravity`` and ``create_world`` on both backends.

    Exercised directly so the domain is pinned where no simulator is installed;
    the sim-level classes below prove the public methods route through it.
    """

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_a_boolean_scalar_is_refused(self, value: Any) -> None:
        """``set_gravity(True)`` was a +1 m/s^2 gravity pointing up, under success."""
        components, err = SimEngine._normalize_gravity(value, "set_gravity")
        assert err is not None, f"{value!r} was accepted as a gravity scalar"
        assert components is None, "a refused gravity must normalize nothing"

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    @pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
    def test_a_boolean_component_is_refused_on_every_axis(self, value: Any, axis: int) -> None:
        vector: list[Any] = [0.0, 0.0, -9.81]
        vector[axis] = value
        components, err = SimEngine._normalize_gravity(vector, "set_gravity")
        assert err is not None, f"{value!r} was accepted as gravity component {axis}"
        assert components is None, "the normalization is all-or-nothing"

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_the_refusal_names_the_parameter_and_carries_the_reason(self, value: Any) -> None:
        _, err = SimEngine._normalize_gravity(value, "create_world", param="gravity")
        assert err is not None
        text = _text(err)
        assert "create_world" in text
        assert "'gravity'" in text
        assert "not a bool" in text, "the message must distinguish a bool from a plain non-number"
        assert _BOOLEAN_WORLD_REASON in text, "the refusal must carry the reason, not just the rejection"

    def test_a_numpy_boolean_scalar_is_no_longer_refused_as_a_length_problem(self) -> None:
        """The pre-fix message named a component count for a value that has none.

        ``numpy.bool_`` is not ``numbers.Real``, so it missed the scalar branch
        and reached ``len()``, which raises ``len() of unsized object`` - reported
        as "must be a 3-element list of numbers". A caller reading that would go
        looking for a length bug in a scalar.
        """
        _, err = SimEngine._normalize_gravity(np.True_, "set_gravity")
        assert err is not None
        text = _text(err)
        assert "not a bool" in text
        assert "3-element" not in text, "a scalar bool is not a component-count problem"
        assert "unsized" not in text, "no binding-level phrasing should reach the caller"

    # A 0-d numeric array is absent: it is not numbers.Real either, so it takes
    # the vector path and fails len() - see
    # test_a_zero_d_numeric_array_scalar_is_still_a_length_error.
    @pytest.mark.parametrize("value", _ANY_NUMBERS[:-1], ids=_ANY_IDS[:-1])
    def test_numeric_scalars_are_still_accepted(self, value: Any) -> None:
        components, err = SimEngine._normalize_gravity(value, "set_gravity")
        assert err is None, f"{value!r} was refused: {err}"
        assert components == [0.0, 0.0, float(value)]

    def test_a_zero_d_numeric_array_scalar_is_still_a_length_error(self) -> None:
        """Unchanged by this fix, and pinned so the asymmetry is on the record.

        ``np.array(0.7)`` is not ``numbers.Real``, so it misses the scalar branch
        and reaches ``len()`` exactly as ``np.True_`` used to. The boolean case is
        now answered before that point; widening the *numeric* scalar branch to
        accept a 0-d array is a separate question and is deliberately not part of
        this change.
        """
        components, err = SimEngine._normalize_gravity(np.array(0.7), "set_gravity")
        assert err is not None
        assert components is None
        assert "not a bool" not in _text(err), "a numeric 0-d array is not a bool"

    @pytest.mark.parametrize("value", _ANY_NUMBERS, ids=_ANY_IDS)
    def test_numeric_components_are_still_accepted(self, value: Any) -> None:
        components, err = SimEngine._normalize_gravity([value, 0.0, -9.81], "set_gravity")
        assert err is None, f"{value!r} was refused: {err}"
        assert components == [float(value), 0.0, -9.81]

    def test_one_is_accepted_though_it_is_what_true_would_have_written(self) -> None:
        """Separates "refuses a boolean" from "refuses anything equal to 1"."""
        components, err = SimEngine._normalize_gravity(1, "set_gravity")
        assert err is None
        assert components == [0.0, 0.0, 1.0]

    def test_earth_gravity_is_still_accepted(self) -> None:
        components, err = SimEngine._normalize_gravity([0.0, 0.0, -9.81], "create_world")
        assert err is None
        assert components == [0.0, 0.0, -9.81]

    @pytest.mark.parametrize("value", _NON_FINITE, ids=_NON_FINITE_IDS)
    def test_a_non_finite_component_keeps_its_own_message(self, value: Any) -> None:
        """The bool gate is additive - it must not relabel a nan as a bool."""
        _, err = SimEngine._normalize_gravity([0.0, 0.0, value], "set_gravity")
        assert err is not None
        assert "not a bool" not in _text(err)
        assert "finite" in _text(err)

    def test_a_wrong_length_vector_keeps_its_own_message(self) -> None:
        _, err = SimEngine._normalize_gravity([0.0, -9.81], "set_gravity")
        assert err is not None
        assert "not a bool" not in _text(err)
        assert "3-element" in _text(err)


class TestNoiseMagnitudesAndRandomizationRangesRefuseABoolean:
    """A sigma or a scale bound of ``1`` is a distribution, not a flag."""

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_a_boolean_magnitude_is_refused(self, value: Any) -> None:
        reason = finite_non_negative_error(value, "std", "set_obs_noise")
        assert reason is not None, f"{value!r} was accepted as a noise magnitude"
        assert "not a bool" in reason
        assert _BOOLEAN_WORLD_REASON in reason

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    @pytest.mark.parametrize("position", [0, 1], ids=["lo", "hi"])
    def test_a_boolean_range_bound_is_refused(self, value: Any, position: int) -> None:
        bounds: list[Any] = [0.5, 2.0]
        bounds[position] = value
        reason = randomization_range_error(tuple(bounds), "mass")
        assert reason is not None, f"{value!r} was accepted as range bound {position}"
        assert "not bools" in reason
        assert _BOOLEAN_WORLD_REASON in reason

    @pytest.mark.parametrize("value", _POSITIVE_NUMBERS, ids=_POSITIVE_IDS)
    def test_numeric_magnitudes_are_still_accepted(self, value: Any) -> None:
        assert finite_non_negative_error(value, "std", "set_obs_noise") is None

    def test_zero_noise_is_still_accepted(self) -> None:
        """``0`` disables the noise and is the documented way to do so."""
        assert finite_non_negative_error(0, "std", "set_obs_noise") is None
        assert finite_non_negative_error(0.0, "std", "set_obs_noise") is None

    def test_a_numeric_range_is_still_accepted(self) -> None:
        assert randomization_range_error((0.8, 1.2), "mass") is None
        assert randomization_range_error((np.float64(0.8), np.int64(1)), "mass") is None

    def test_the_pre_existing_range_domain_is_unchanged(self) -> None:
        """The bool gate must not displace the bounds checks it sits in front of."""
        assert "non-negative" in str(randomization_range_error((-1.0, 2.0), "mass"))
        assert "exceeds upper bound" in str(randomization_range_error((3.0, 1.0), "mass"))
        assert "finite" in str(randomization_range_error((0.0, float("inf")), "mass"))
        assert "pair of numbers" in str(randomization_range_error("wide", "mass"))

    @pytest.mark.parametrize("value", _NON_FINITE, ids=_NON_FINITE_IDS)
    def test_a_non_finite_magnitude_keeps_its_own_message(self, value: Any) -> None:
        reason = finite_non_negative_error(value, "std", "set_obs_noise")
        assert reason is not None
        assert "not a bool" not in reason

    def test_a_negative_magnitude_keeps_its_own_message(self) -> None:
        reason = finite_non_negative_error(-0.1, "std", "set_obs_noise")
        assert reason is not None
        assert "not a bool" not in reason


# ---------------------------------------------------------------------------
# Mode B: the rule was documented, isinstance(x, bool) did not implement it
# ---------------------------------------------------------------------------


class TestTimestepRefusesEveryBooleanSpellingNotJustThePythonOne:
    @pytest.mark.parametrize("value", _TRUTHY_BOOLEANS, ids=_TRUTHY_IDS)
    def test_a_boolean_timestep_is_refused(self, value: Any) -> None:
        """``np.True_`` was a 1-second dt under ``status="success"``."""
        err = SimEngine._validate_timestep(value, "set_timestep")
        assert err is not None, f"{value!r} was accepted as a timestep"
        assert "not a bool" in _text(err)
        assert _BOOLEAN_WORLD_REASON in _text(err)

    @pytest.mark.parametrize("value", _TRUTHY_BOOLEANS, ids=_TRUTHY_IDS)
    def test_the_refusal_names_the_method_and_the_parameter(self, value: Any) -> None:
        err = SimEngine._validate_timestep(value, "create_world", param="default_timestep")
        assert err is not None
        assert "create_world" in _text(err)
        assert "default_timestep" in _text(err)

    @pytest.mark.parametrize("value", _POSITIVE_NUMBERS, ids=_POSITIVE_IDS)
    def test_positive_numeric_timesteps_are_still_accepted(self, value: Any) -> None:
        assert SimEngine._validate_timestep(value, "set_timestep") is None

    def test_the_default_timestep_is_still_accepted(self) -> None:
        assert SimEngine._validate_timestep(0.002, "set_timestep") is None

    def test_one_second_is_still_accepted_when_asked_for_as_a_number(self) -> None:
        """The gate refuses the *type*, not the quantity - ``1`` remains legal."""
        assert SimEngine._validate_timestep(1, "set_timestep") is None
        assert SimEngine._validate_timestep(1.0, "set_timestep") is None

    @pytest.mark.parametrize("value", [0, 0.0, -1, -0.002, *_NON_FINITE, "fast", None])
    def test_the_pre_existing_domain_keeps_its_own_message(self, value: Any) -> None:
        err = SimEngine._validate_timestep(value, "set_timestep")
        assert err is not None
        assert "not a bool" not in _text(err)
        assert "must be a finite positive number" in _text(err)


class TestMassRefusesEveryBooleanSpellingNotJustThePythonOne:
    @pytest.mark.parametrize("value", _TRUTHY_BOOLEANS, ids=_TRUTHY_IDS)
    def test_a_boolean_mass_is_refused(self, value: Any) -> None:
        """``np.True_`` was a 1 kg body under ``status="success"``."""
        err = SimEngine._validate_mass(value, "set_body_properties")
        assert err is not None, f"{value!r} was accepted as a mass"
        assert "not a bool" in _text(err)
        assert _BOOLEAN_WORLD_REASON in _text(err)

    @pytest.mark.parametrize("value", _POSITIVE_NUMBERS, ids=_POSITIVE_IDS)
    def test_positive_numeric_masses_are_still_accepted(self, value: Any) -> None:
        assert SimEngine._validate_mass(value, "set_body_properties") is None

    def test_a_one_kilogram_mass_is_still_accepted(self) -> None:
        assert SimEngine._validate_mass(1, "set_body_properties") is None
        assert SimEngine._validate_mass(1.0, "set_body_properties") is None

    @pytest.mark.parametrize("value", [0, 0.0, -1, *_NON_FINITE, "heavy", None])
    def test_the_pre_existing_domain_keeps_its_own_message(self, value: Any) -> None:
        err = SimEngine._validate_mass(value, "set_body_properties")
        assert err is not None
        assert "not a bool" not in _text(err)


# ---------------------------------------------------------------------------
# The MuJoCo vector chokepoint
# ---------------------------------------------------------------------------


class TestTheVectorCoercionRefusesABooleanComponent:
    """``_coerce_finite_vector`` is one gate for seven call sites.

    A raycast origin and direction, a geom size and friction, an rgba colour,
    and each ray of a ``multi_raycast`` batch all coerce through it, so the check
    belongs here rather than at each caller.
    """

    @staticmethod
    def _coerce(*args: Any, **kwargs: Any) -> Any:
        from strands_robots.simulation.mujoco.physics import _coerce_finite_vector

        return _coerce_finite_vector(*args, **kwargs)

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    @pytest.mark.parametrize(
        ("name", "method"),
        [
            ("origin", "raycast"),
            ("direction", "raycast"),
            ("friction", "set_geom_properties"),
            ("size", "set_geom_properties"),
            ("color", "add_object"),
        ],
    )
    def test_a_boolean_component_is_refused_on_every_surface(self, value: Any, name: str, method: str) -> None:
        floats, err = self._coerce([value, 0.5, 0.5], name, method)
        assert err is not None, f"{value!r} was accepted as a {name} component"
        assert floats is None, "a refused vector must coerce nothing"
        assert method in _text(err)
        assert f"'{name}'" in _text(err)

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    @pytest.mark.parametrize("index", [0, 1, 2], ids=["first", "middle", "last"])
    def test_a_boolean_anywhere_in_the_vector_refuses_the_whole_vector(self, value: Any, index: int) -> None:
        vector: list[Any] = [0.1, 0.2, 0.3]
        vector[index] = value
        floats, err = self._coerce(vector, "origin", "raycast")
        assert err is not None, f"{value!r} accepted at index {index}"
        assert floats is None, "a partial coercion would cast a ray the caller never asked for"

    def test_the_refusal_carries_the_vector_reason_not_the_state_writer_one(self) -> None:
        """The reason must name the quantities this helper actually guards.

        ``_BOOLEAN_STATE_REASON`` names radians, rad/s and newtons, which are
        the units of the joint writers - not of a coordinate, an extent or a
        colour channel.
        """
        from strands_robots.simulation.base import _BOOLEAN_STATE_REASON
        from strands_robots.utils import BOOLEAN_VECTOR_REASON

        _, err = self._coerce([True, 0.0, 0.0], "origin", "raycast")
        assert err is not None
        text = _text(err)
        assert "not a bool" in text
        assert BOOLEAN_VECTOR_REASON in text
        assert _BOOLEAN_STATE_REASON not in text
        assert "radian" not in text, "a raycast origin is not measured in radians"

    @pytest.mark.parametrize("value", _ANY_NUMBERS, ids=_ANY_IDS)
    def test_numeric_components_are_still_accepted(self, value: Any) -> None:
        floats, err = self._coerce([value, 0.0, 0.0], "origin", "raycast")
        assert err is None, f"{value!r} was refused: {err}"
        assert floats is not None
        assert floats[0] == pytest.approx(float(value))

    def test_a_unit_vector_is_still_accepted(self) -> None:
        """``[1, 0, 0]`` is the component-wise value ``True`` would have written."""
        floats, err = self._coerce([1, 0, 0], "direction", "raycast")
        assert err is None
        assert floats == [1.0, 0.0, 0.0]

    def test_an_all_zero_vector_is_still_accepted(self) -> None:
        floats, err = self._coerce([0, 0, 0], "friction", "set_geom_properties")
        assert err is None
        assert floats == [0.0, 0.0, 0.0]

    def test_a_numpy_vector_is_still_accepted(self) -> None:
        floats, err = self._coerce(np.array([0.1, 0.2, 0.3]), "origin", "raycast")
        assert err is None
        assert floats == pytest.approx([0.1, 0.2, 0.3])

    def test_an_opaque_white_colour_is_still_accepted(self) -> None:
        """An rgba of all ones is the colour most likely to look like four bools."""
        floats, err = self._coerce([1.0, 1.0, 1.0, 1.0], "color", "add_object", accepted_lengths=(3, 4))
        assert err is None
        assert floats == [1.0, 1.0, 1.0, 1.0]

    @pytest.mark.parametrize("value", _NON_FINITE, ids=_NON_FINITE_IDS)
    def test_a_non_finite_component_keeps_its_own_message(self, value: Any) -> None:
        _, err = self._coerce([value, 0.0, 0.0], "origin", "raycast")
        assert err is not None
        assert "not a bool" not in _text(err)
        assert "finite" in _text(err)

    def test_a_non_numeric_component_keeps_its_own_message(self) -> None:
        _, err = self._coerce(["a", 0.0, 0.0], "origin", "raycast")
        assert err is not None
        assert "not a bool" not in _text(err)

    def test_the_pre_existing_bound_and_length_checks_are_unchanged(self) -> None:
        _, err = self._coerce([-1.0, 0.5, 0.5], "friction", "set_geom_properties", min_value=0.0)
        assert err is not None
        assert "not a bool" not in _text(err)
        _, err = self._coerce([0.1, 0.2], "origin", "raycast", accepted_lengths=(3,), layout="x, y, z")
        assert err is not None
        assert "not a bool" not in _text(err)


# ---------------------------------------------------------------------------
# One predicate, and a guard that keeps it that way
# ---------------------------------------------------------------------------


class TestOnePredicateNotSix:
    """Every validator answers "is this a boolean" the same way."""

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_the_shared_predicate_reports_every_boolean_spelling(self, value: Any) -> None:
        assert is_boolean(value) is True

    @pytest.mark.parametrize("value", _ANY_NUMBERS, ids=_ANY_IDS)
    def test_the_shared_predicate_reports_no_number_as_boolean(self, value: Any) -> None:
        assert is_boolean(value) is False

    @pytest.mark.parametrize(
        "validator",
        [
            "randomization_range_error",
            "finite_non_negative_error",
        ],
    )
    def test_the_module_level_validators_use_the_shared_predicate(self, validator: str) -> None:
        source = inspect.getsource(getattr(sim_base, validator))
        assert "is_boolean" in source, f"{validator} must not re-decide what a boolean is"

    @pytest.mark.parametrize("validator", ["_validate_timestep", "_validate_mass", "_normalize_gravity"])
    def test_the_engine_validators_use_the_shared_predicate(self, validator: str) -> None:
        source = inspect.getsource(getattr(SimEngine, validator))
        assert "is_boolean" in source, f"{validator} must not re-decide what a boolean is"

    @pytest.mark.parametrize("validator", ["_validate_timestep", "_validate_mass", "_normalize_gravity"])
    def test_no_validator_narrows_the_question_back_to_isinstance(self, validator: str) -> None:
        """The Mode B defect itself: ``isinstance(x, bool)`` misses ``numpy.bool_``.

        Pinned structurally as well as behaviourally, because the behavioural
        tests would pass again if someone re-added the narrow check *alongside*
        the shared one, leaving two answers in one function.

        Matched on the parsed tree rather than the text, so the comment that
        names the check it replaced does not read as the check itself.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(SimEngine, validator))))
        narrow = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
            and ast.unparse(node.args[1]) == "bool"
        ]
        assert not narrow, f"{validator} narrows the boolean question back to {narrow}"

    def test_the_vector_coercion_uses_the_shared_predicate(self) -> None:
        from strands_robots.simulation.mujoco import physics

        source = inspect.getsource(physics._coerce_finite_vector)
        assert "is_boolean" in source

    def test_the_documented_intent_still_matches_the_code(self) -> None:
        """The Mode B docstrings promised a refusal the code did not deliver."""
        for validator in ("_validate_timestep", "_validate_mass"):
            doc = inspect.getdoc(getattr(SimEngine, validator)) or ""
            assert "bool" in doc, f"{validator} should keep stating the rule it enforces"


# Names an input validator may carry without consulting the predicate, each with
# the reason it is not an input domain. Anything else that coerces caller input
# with a bare float() must route through is_boolean.
_NOT_AN_INPUT_DOMAIN = {
    # The bool case on this path is its sibling _boolean_action_error's job
    # (#1841); this one answers only the nan/inf question.
    "_non_finite_action_error": "boolean handled by _boolean_action_error on the same path",
    # These compare rates that positive_whole_number_error has already accepted,
    # so a boolean is refused before either is reached.
    "dataset_rate_mismatch_reason": "compares already-validated rates",
    "rollout_rate_mismatch_reason": "compares already-validated rates",
    # Reads a pose back off the USD stage - not caller input.
    "_prim_body_state": "reads state out of the engine",
    # Measures the distance between a target _validate_move_to_args has already
    # coerced (its position runs through coerce_pose_vector, which refuses a
    # boolean component) and the engine-owned robot base position - a boolean
    # cannot reach the float() here.
    "_workspace_sanity_error": "measures an already-coerced target against engine state",
}

_GUARDED_VALIDATORS = {
    "randomization_range_error",
    "finite_non_negative_error",
    "_validate_timestep",
    "_validate_mass",
    "_normalize_gravity",
    "_coerce_finite_vector",
}


def _discovered_validators() -> dict[str, str]:
    """Every input-validation coercion under ``strands_robots/simulation/``.

    The structural signature of one: it returns a *reason* rather than raising
    (``str | None``, ``dict[str, Any] | None``, or a
    ``(value | None, error | None)`` pair, which is the structured-error tool
    contract) and it coerces with a bare ``float()``. That pairing is what makes
    a function an input domain, and it is also exactly what lets a boolean
    through as ``1.0``.
    """
    package = pathlib.Path(inspect.getfile(sim_base)).parent
    validator_returns = (
        "str | None",
        "dict[str, Any] | None",
        "tuple[list[float] | None",
        "tuple[float | None",
    )
    found: dict[str, str] = {}
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            returns = ast.unparse(node.returns) if node.returns else ""
            if not any(returns.startswith(prefix) for prefix in validator_returns):
                continue
            coerces = any(
                isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "float"
                for call in ast.walk(node)
            )
            if not coerces:
                continue
            found[node.name] = ast.unparse(node)
    return found


class TestTheBooleanDomainIsStructurallyClosed:
    """A new coercion cannot reopen the hole three PRs have now closed.

    #1837 fixed the actuator command, #1841 the state writers, #1842 these six.
    Each pass found the *next* helper that coerced with a bare ``float()``,
    because nothing enumerated them. This does.
    """

    def test_every_discovered_validator_is_guarded_or_explained(self) -> None:
        discovered = _discovered_validators()
        unaccounted = set(discovered) - _GUARDED_VALIDATORS - set(_NOT_AN_INPUT_DOMAIN)
        assert not unaccounted, (
            "these coerce caller input with a bare float() and neither consult "
            f"utils.is_boolean nor are listed as out of domain: {sorted(unaccounted)}. "
            "A boolean reaching one of them is written as 1.0 under status='success'. "
            "Route it through is_boolean, or add it to _NOT_AN_INPUT_DOMAIN with a reason."
        )

    def test_every_guarded_validator_actually_consults_the_predicate(self) -> None:
        discovered = _discovered_validators()
        for name in sorted(_GUARDED_VALIDATORS):
            assert name in discovered, f"{name} is pinned as guarded but no longer discovered - update the pin"
            assert "is_boolean" in discovered[name], f"{name} stopped consulting the shared predicate"

    def test_the_scan_is_not_vacuous(self) -> None:
        """A scan that finds nothing would pass every assertion above."""
        discovered = _discovered_validators()
        assert len(discovered) >= len(_GUARDED_VALIDATORS)
        assert _GUARDED_VALIDATORS <= set(discovered)

    def test_every_documented_exemption_is_still_real(self) -> None:
        """An exemption for a function that no longer exists hides a new gap."""
        discovered = _discovered_validators()
        stale = set(_NOT_AN_INPUT_DOMAIN) - set(discovered)
        assert not stale, f"exemptions no longer match a discovered validator: {sorted(stale)}"

    def test_the_guard_detects_a_hand_rolled_validator(self) -> None:
        """The guard must fail on the thing it exists to catch, or it proves nothing."""
        source = '''
def newly_added_scale_error(value: Any, param: str) -> str | None:
    """A validator that coerces caller input without consulting the predicate."""
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return f"{param} must be a number"
    return None
'''
        tree = ast.parse(source)
        node = tree.body[0]
        assert isinstance(node, ast.FunctionDef)
        returns = ast.unparse(node.returns) if node.returns else ""
        assert returns == "str | None", "the fake must match the signature the scan keys on"
        coerces = any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "float"
            for call in ast.walk(node)
        )
        assert coerces, "the fake must coerce with a bare float()"
        assert "is_boolean" not in ast.unparse(node), "the fake must be unguarded"
        assert node.name not in _GUARDED_VALIDATORS
        assert node.name not in _NOT_AN_INPUT_DOMAIN


# ---------------------------------------------------------------------------
# Sim-level: the public methods route through the validators above
# ---------------------------------------------------------------------------

requires_mujoco = pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="mujoco not installed",
)

# Inline scene XML - no network dependency on robot model repos.
_SCENE_XML = """
<mujoco model="bool_validator_scene">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.2">
      <freejoint/>
      <geom name="block" type="box" size="0.05 0.05 0.05" rgba="0.3 0.3 0.8 1" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def scene_sim(tmp_path):
    """A simulation with one free body, reproducing #1842's public surfaces."""
    from strands_robots.simulation import Simulation

    path = tmp_path / "bool_validator_scene.xml"
    path.write_text(_SCENE_XML)

    sim = Simulation()
    result = sim.create_world(timestep=0.002)
    assert result["status"] == "success", f"create_world failed: {result}"
    yield sim
    sim.destroy()


@requires_mujoco
class TestCreateWorldRefusesABoolean:
    @pytest.mark.parametrize("value", _TRUTHY_BOOLEANS, ids=_TRUTHY_IDS)
    def test_a_boolean_timestep_is_refused(self, value: Any) -> None:
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            result = sim.create_world(timestep=value)
            assert result["status"] == "error", f"{value!r} built a world with a 1-second dt"
            assert "not a bool" in _text(result)
            # A refused create_world must not leave a half-built world behind.
            assert sim.step(1)["status"] == "error"
        finally:
            sim.destroy()

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_a_boolean_gravity_is_refused(self, value: Any) -> None:
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            result = sim.create_world(gravity=value)
            assert result["status"] == "error", f"{value!r} built a world with gravity 1.0"
            assert "not a bool" in _text(result)
        finally:
            sim.destroy()


@requires_mujoco
class TestTheWorldSettersRefuseABooleanAndWriteNothing:
    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_a_boolean_gravity_leaves_opt_gravity_untouched(self, scene_sim, value: Any) -> None:
        before = [float(g) for g in scene_sim._world._model.opt.gravity]
        result = scene_sim.set_gravity(value)
        assert result["status"] == "error", f"{value!r} was applied as a gravity"
        assert "not a bool" in _text(result)
        assert [float(g) for g in scene_sim._world._model.opt.gravity] == before

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_a_boolean_gravity_component_leaves_opt_gravity_untouched(self, scene_sim, value: Any) -> None:
        before = [float(g) for g in scene_sim._world._model.opt.gravity]
        result = scene_sim.set_gravity([value, 0.0, -9.81])
        assert result["status"] == "error"
        assert [float(g) for g in scene_sim._world._model.opt.gravity] == before

    @pytest.mark.parametrize("value", _TRUTHY_BOOLEANS, ids=_TRUTHY_IDS)
    def test_a_boolean_timestep_leaves_opt_timestep_untouched(self, scene_sim, value: Any) -> None:
        before = float(scene_sim._world._model.opt.timestep)
        result = scene_sim.set_timestep(value)
        assert result["status"] == "error", f"{value!r} was applied as a dt"
        assert "not a bool" in _text(result)
        assert float(scene_sim._world._model.opt.timestep) == before

    def test_a_numeric_gravity_is_still_applied(self, scene_sim) -> None:
        result = scene_sim.set_gravity([0.0, 0.0, -1.62])
        assert result["status"] == "success", result
        assert float(scene_sim._world._model.opt.gravity[2]) == pytest.approx(-1.62)

    def test_a_numeric_scalar_gravity_is_still_applied(self, scene_sim) -> None:
        result = scene_sim.set_gravity(-3.72)
        assert result["status"] == "success", result
        assert float(scene_sim._world._model.opt.gravity[2]) == pytest.approx(-3.72)

    def test_a_numeric_timestep_is_still_applied(self, scene_sim) -> None:
        result = scene_sim.set_timestep(0.001)
        assert result["status"] == "success", result
        assert float(scene_sim._world._model.opt.timestep) == pytest.approx(0.001)

    def test_a_one_second_timestep_is_still_applied_when_asked_for_numerically(self, scene_sim) -> None:
        """The refusal is of the type, not the quantity ``True`` would have set."""
        result = scene_sim.set_timestep(1.0)
        assert result["status"] == "success", result
        assert float(scene_sim._world._model.opt.timestep) == pytest.approx(1.0)


@requires_mujoco
class TestRaycastRefusesABooleanComponent:
    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    @pytest.mark.parametrize("param", ["origin", "direction"])
    def test_a_boolean_component_is_refused(self, scene_sim, value: Any, param: str) -> None:
        kwargs: dict[str, Any] = {"origin": [0.0, 0.0, 1.0], "direction": [0.0, 0.0, -1.0]}
        kwargs[param] = [value, 0.0, 0.0]
        result = scene_sim.raycast(**kwargs)
        assert result["status"] == "error", f"{value!r} was cast as a {param}"
        assert "not a bool" in _text(result)

    def test_a_numeric_ray_is_still_cast(self, scene_sim) -> None:
        result = scene_sim.raycast(origin=[0.0, 0.0, 1.0], direction=[0.0, 0.0, -1.0])
        assert result["status"] == "success", result

    def test_a_unit_axis_ray_is_still_cast(self, scene_sim) -> None:
        """``[1, 0, 0]`` is componentwise what a boolean would have written."""
        result = scene_sim.raycast(origin=[1, 0, 1], direction=[0, 0, -1])
        assert result["status"] == "success", result
