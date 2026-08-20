"""A domain-randomization axis flag must be a boolean, not anything truthy.

``randomize`` takes one flag per axis, and each selects a *posture* - run this
axis or leave it alone - rather than scaling a quantity. Every non-empty string
is truthy, so ``"false"`` / ``"no"`` / ``"off"`` / ``"0"`` - the spellings an
operator reaches for when turning an axis off - turned that axis **on**, and the
call reported the axis applied:

* On MuJoCo, ``randomize(randomize_physics="false", randomize_positions="false")``
  ran both axes. Those two default to ``False`` precisely because they cannot be
  undone: the physics axis rewrites ``body_mass`` / ``body_inertia`` /
  ``geom_friction`` and the position axis rewrites ``model.qpos0`` - the pose a
  ``reset`` restores - so the misread persists for the rest of the scene's life
  and recompiling is the only way back.
* On Newton the three flags are stored in the randomization spec through
  ``bool()``, so ``randomize_colors="false"`` was stored as ``True`` and the
  axis re-applied on **every** later rebuild, not just the call that misread it.
* ``0`` / ``""`` / ``None`` / ``[]`` took the other branch without ever being a
  declared spelling of it.

Newton's MuJoCo-parity refusal branches on the same flag, so it inherited the
misread: ``randomize_positions="false"`` was answered with "randomize_positions
is not supported by the Newton backend yet ... Use the MuJoCo backend for
object-position randomization" - an unsupported-axis verdict for a caller who
had asked *not* to randomize positions.

A misspelled axis *name* was already refused with the valid list, which the
``**kwargs``-typed :meth:`~strands_robots.simulation.base.SimEngine.randomize`
base signature requires "so a misspelled axis cannot report success while
leaving that axis untouched". These tests pin the same guarantee for the *value*
that name carries, on :func:`~strands_robots.utils.boolean_flag_error`, and pin
that both backends that implement the method reach it - the numeric knobs in the
same signature have been checked on a shared domain across both backends since
the physics defect they caused.
"""

from __future__ import annotations

import inspect
import threading
import types
from typing import Any

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.newton.randomization import (  # noqa: E402
    DomainRandomizationMixin as NewtonRandomization,
)
from strands_robots.utils import boolean_flag_error  # noqa: E402

#: Values no posture can be read from. The four string spellings and the
#: non-zero number are the ones that selected the *enabled* branch; ``0`` /
#: ``""`` / ``None`` / ``[]`` silently took the other one without ever being a
#: declared spelling of it.
UNUSABLE_FLAGS: list[Any] = ["false", "no", "off", "0", "true", 1, 0, "", None, []]

#: The subset that reads as "off" to a human and as "on" to ``if flag:``.
TRUTHY_NON_BOOLEANS: list[Any] = ["false", "no", "off", "0", 1, float("nan")]

#: The MuJoCo axis flags, in signature order. Derived from the live signature
#: rather than copied, so an axis added later is graded without an edit.
MUJOCO_AXIS_FLAGS = [
    name
    for name, param in inspect.signature(Simulation.randomize).parameters.items()
    if param.annotation is bool or param.annotation == "bool"
]

#: Newton declares three of the four; ``randomize_positions`` reaches it through
#: ``**kwargs`` for the MuJoCo-parity refusal.
NEWTON_DECLARED_AXIS_FLAGS = [
    name
    for name, param in inspect.signature(NewtonRandomization.randomize).parameters.items()
    if param.annotation is bool or param.annotation == "bool"
]

CUBE_Z = 0.30


@pytest.fixture
def sim():
    """A MuJoCo world holding one dynamic 1 kg cube hovering at 0.30 m."""
    s = Simulation(tool_name="randomize_axis_flag_domain", mesh=False)
    created = s.create_world(gravity=[0, 0, -9.81])
    assert created["status"] == "success", _text(created)
    added = s.add_object(name="cube", shape="box", size=[0.06, 0.06, 0.06], position=[0.0, 0.0, CUBE_Z], mass=1.0)
    assert added["status"] == "success", _text(added)
    yield s
    s.cleanup()


def _text(result: dict) -> str:
    return " ".join(block["text"] for block in result["content"] if "text" in block)


def _newton_engine() -> Any:
    """A stand-in ``self`` carrying only what ``randomize`` reads.

    Newton's runtime is not importable here, and ``randomize`` reaches none of
    it: the axis flags, the ranges and the seed are all decided before the
    rebuild that would need it. Annotated ``Any`` because the stand-in
    deliberately satisfies the method's reads rather than the engine's type.
    """
    engine = types.SimpleNamespace(_world=object(), _lock=threading.RLock(), _dr={}, _dr_applied={})
    engine._rebuild = lambda: None
    return engine


def _fingerprint(sim: Simulation) -> dict[str, np.ndarray]:
    """Every model array an axis of ``randomize`` writes."""
    assert sim._world is not None and sim._world._model is not None
    model = sim._world._model
    return {
        "qpos0": np.array(model.qpos0, copy=True),
        "geom_rgba": np.array(model.geom_rgba, copy=True),
        "geom_friction": np.array(model.geom_friction, copy=True),
        "body_mass": np.array(model.body_mass, copy=True),
        "light_pos": np.array(model.light_pos, copy=True),
    }


def _assert_untouched(sim: Simulation, before: dict[str, np.ndarray]) -> None:
    after = _fingerprint(sim)
    for field, expected in before.items():
        assert np.array_equal(after[field], expected), f"a refused randomize wrote {field}"


class TestTheSharedDomainOwnsTheSpellings:
    """The refused set is the shared domain's, not a list copied into here."""

    @pytest.mark.parametrize("value", UNUSABLE_FLAGS)
    def test_every_listed_spelling_is_refused_by_the_shared_domain(self, value):
        assert boolean_flag_error(value, "randomize_colors", "randomize") is not None

    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_a_real_boolean_is_accepted_by_the_shared_domain(self, value):
        assert boolean_flag_error(value, "randomize_colors", "randomize") is None

    def test_the_scan_found_the_axis_flags_it_grades(self):
        """A clean run must mean the flags were graded, not that none was found."""
        assert MUJOCO_AXIS_FLAGS == [
            "randomize_colors",
            "randomize_lighting",
            "randomize_physics",
            "randomize_positions",
        ]
        assert NEWTON_DECLARED_AXIS_FLAGS == ["randomize_colors", "randomize_lighting", "randomize_physics"]


class TestMujocoRefusesAnAxisFlagItCannotRead:
    """Every declared axis flag is checked before the first model write."""

    @pytest.mark.parametrize("param", MUJOCO_AXIS_FLAGS)
    @pytest.mark.parametrize("value", UNUSABLE_FLAGS)
    def test_an_unusable_axis_flag_is_refused_with_nothing_applied(self, sim, param, value):
        before = _fingerprint(sim)
        result = sim.randomize(**{param: value}, seed=7)
        assert result["status"] == "error", f"{param}={value!r} was honored: {_text(result)}"
        assert param in _text(result)
        _assert_untouched(sim, before)

    @pytest.mark.parametrize("value", TRUTHY_NON_BOOLEANS)
    def test_the_destructive_axes_are_not_run_by_a_spelling_of_off(self, sim, value):
        """Pre-fix: both axes ran and the call reported them applied.

        These two default to ``False`` because undoing them means recompiling
        the scene: the physics axis rewrites mass/inertia/friction and the
        position axis rewrites ``qpos0``, the pose a ``reset`` restores.
        """
        before = _fingerprint(sim)
        result = sim.randomize(
            randomize_colors=False,
            randomize_lighting=False,
            randomize_physics=value,
            randomize_positions=value,
            seed=7,
        )
        assert result["status"] == "error"
        _assert_untouched(sim, before)
        assert np.array_equal(_fingerprint(sim)["body_mass"], before["body_mass"])

    @pytest.mark.parametrize("value", TRUTHY_NON_BOOLEANS)
    def test_the_lighting_axis_is_checked_before_it_resolves_its_reference(self, sim, value):
        """The lighting refusal branches on this flag, so the flag comes first.

        Pre-fix a truthy non-boolean sent the call into the authored-light
        lookup, whose own refusal then advises ``randomize_lighting=False`` -
        the value the caller believes they passed.
        """
        before = _fingerprint(sim)
        result = sim.randomize(randomize_colors=False, randomize_lighting=value, seed=7)
        assert result["status"] == "error"
        assert "randomize_lighting" in _text(result)
        assert "must be a boolean" in _text(result)
        _assert_untouched(sim, before)


class TestNewtonRefusesAnAxisFlagItCannotRead:
    """Newton stores its spec through ``bool()``, so a misread would persist."""

    @pytest.mark.parametrize("param", NEWTON_DECLARED_AXIS_FLAGS)
    @pytest.mark.parametrize("value", UNUSABLE_FLAGS)
    def test_an_unusable_axis_flag_is_refused_with_no_spec_stored(self, param, value):
        engine = _newton_engine()
        result = NewtonRandomization.randomize(engine, **{param: value})
        assert result["status"] == "error", f"{param}={value!r} was honored: {_text(result)}"
        assert param in _text(result)
        assert engine._dr == {}, "a refused randomize stored a randomization spec"

    @pytest.mark.parametrize("value", TRUTHY_NON_BOOLEANS)
    def test_the_parity_refusal_does_not_inherit_the_misread(self, value):
        """Pre-fix: refused as an unsupported axis the caller had opted out of."""
        engine = _newton_engine()
        result = NewtonRandomization.randomize(
            engine, randomize_colors=False, randomize_lighting=False, randomize_positions=value
        )
        assert result["status"] == "error"
        text = _text(result)
        assert "must be a boolean" in text, text
        assert "not supported by the Newton backend" not in text, text
        assert engine._dr == {}


class TestTheDocumentedSpellingsAreUnchanged:
    """Controls: what already worked keeps working, on both backends."""

    def test_mujoco_false_leaves_every_axis_alone(self, sim):
        before = _fingerprint(sim)
        result = sim.randomize(
            randomize_colors=False,
            randomize_lighting=False,
            randomize_physics=False,
            randomize_positions=False,
            seed=7,
        )
        assert result["status"] == "success"
        _assert_untouched(sim, before)

    def test_mujoco_true_still_applies_the_axis(self, sim):
        before = _fingerprint(sim)
        result = sim.randomize(randomize_colors=True, randomize_lighting=False, seed=7)
        assert result["status"] == "success"
        assert not np.array_equal(_fingerprint(sim)["geom_rgba"], before["geom_rgba"])

    def test_newton_false_stores_exactly_false(self):
        engine = _newton_engine()
        result = NewtonRandomization.randomize(
            engine, randomize_colors=False, randomize_lighting=False, randomize_physics=False
        )
        assert result["status"] == "success"
        assert engine._dr["randomize_colors"] is False
        assert engine._dr["randomize_lighting"] is False
        assert engine._dr["randomize_physics"] is False

    def test_newton_still_refuses_the_unsupported_axis_when_it_is_asked_for(self):
        engine = _newton_engine()
        result = NewtonRandomization.randomize(engine, randomize_positions=True)
        assert result["status"] == "error"
        assert "not supported by the Newton backend" in _text(result)

    def test_a_misspelled_axis_name_still_names_the_valid_parameters(self, sim):
        """The guarantee that already existed, at the level above the value."""
        result = sim.randomize(randomize_postions=True, seed=7)
        assert result["status"] == "error"
        text = _text(result)
        assert "randomize_postions" in text
        assert "randomize_positions" in text, "the refusal must name the parameter that was meant"

    def test_the_numeric_knobs_keep_their_own_domain(self, sim):
        """A range is a quantity, not a posture: it keeps the numeric verdict."""
        result = sim.randomize(
            randomize_colors=False, randomize_lighting=False, randomize_physics=True, mass_range=(0.0, 0.0)
        )
        assert result["status"] == "error"
        assert "mass_range" in _text(result)
        assert "must be a boolean" not in _text(result)


class TestBothBackendsReachTheSameDomain:
    """The accepted spellings must not diverge between the two backends."""

    @pytest.mark.parametrize("value", TRUTHY_NON_BOOLEANS)
    def test_a_shared_axis_flag_is_refused_identically(self, sim, value):
        mujoco_result = sim.randomize(randomize_physics=value)
        newton_result = NewtonRandomization.randomize(_newton_engine(), randomize_physics=value)
        assert mujoco_result["status"] == newton_result["status"] == "error"
        for text in (_text(mujoco_result), _text(newton_result)):
            assert "randomize_physics" in text
            assert "must be a boolean" in text

    @pytest.mark.parametrize(
        "randomize",
        [Simulation.randomize, NewtonRandomization.randomize],
        ids=["mujoco", "newton"],
    )
    def test_each_implementation_calls_the_shared_helper(self, randomize):
        assert "boolean_flag_error" in inspect.getsource(randomize)
