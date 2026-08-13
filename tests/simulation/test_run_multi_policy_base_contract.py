"""The ``run_multi_policy`` base contract and its shared validation helpers.

Promoting ``run_multi_policy`` from a MuJoCo-only method to the
:class:`~strands_robots.simulation.base.SimEngine` base (#2157, step 1 of the
#2122 Isaac parity work) has two halves, each pinned here:

* **The base default is a documented refusal.** A backend without a
  synchronized multi-robot loop must return the structured not-supported
  error naming its own class - never ``AttributeError`` (there was no
  contract), never a silent per-robot fallback (which would interleave
  recording frames and break the merged-frame contract the docstring
  documents). Newton inherits this today; MuJoCo and Isaac override it.

* **The validation/normalization is shared, not duplicated.** The
  backend-independent first phase of the MuJoCo loop - empty-``policies``
  rejection, ``instructions`` normalization, the distinct-instructions
  one-task-per-frame warning, and per-robot ``action_horizon``
  normalization - lives on the base as
  :meth:`~strands_robots.simulation.base.SimEngine._validate_multi_policies`,
  :meth:`~strands_robots.simulation.base.SimEngine._normalize_multi_policy_instructions`
  and
  :meth:`~strands_robots.simulation.base.SimEngine._normalize_multi_policy_horizons`,
  so the upcoming Isaac implementation cannot drift from MuJoCo's refusal
  texts. The behavioural coverage through the MuJoCo entry point stays in
  ``tests/simulation/mujoco/test_run_multi_policy_no_recording.py`` and
  ``tests/simulation/test_run_policy_horizon_validation.py``; these tests
  drive the helpers directly so a future backend gets the same verdicts
  without needing a compiled world.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import pytest

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_TWO_ROBOTS: dict[str, Any] = {"alpha": object(), "beta": object()}


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


# --------------------------------------------------------------------------- #
# The base default: a structured refusal, not a fallback                      #
# --------------------------------------------------------------------------- #
class TestTheBaseDefaultRefuses:
    """A backend without the synchronized loop says so in the tool envelope."""

    @pytest.mark.parametrize("engine_cls", [NewtonSimEngine])
    def test_a_backend_without_an_override_returns_the_structured_error(self, engine_cls: type[SimEngine]) -> None:
        """``__new__`` skeleton: the refusal must precede any engine state."""
        engine = engine_cls.__new__(engine_cls)
        result = engine.run_multi_policy(policies={"alpha": MockPolicy()}, n_steps=2)
        assert result["status"] == "error"
        assert engine_cls.__name__ in _text(result)
        assert "synchronized" in _text(result)
        assert "multi-robot" in _text(result)

    def test_the_refusal_names_the_concrete_class_not_the_base(self) -> None:
        """The error must point at the backend the caller holds."""
        engine = NewtonSimEngine.__new__(NewtonSimEngine)
        text = _text(engine.run_multi_policy(policies={"alpha": MockPolicy()}))
        assert text.startswith("run_multi_policy: NewtonSimEngine does not implement")
        assert not text.startswith("run_multi_policy: SimEngine ")

    def test_isaac_overrides_the_default(self) -> None:
        """The Isaac synchronized loop (#2158) must not fall through to the refusal."""
        assert IsaacSimulation.run_multi_policy is not SimEngine.run_multi_policy

    def test_isaac_signature_extends_the_base_contract(self) -> None:
        """Isaac keeps the base parameters (names, order, defaults) and adds only
        the keyword-only ``reset_between`` (its #1895 forward-compat guard)."""
        base = inspect.signature(SimEngine.run_multi_policy).parameters
        override = inspect.signature(IsaacSimulation.run_multi_policy).parameters
        assert list(override)[: len(base)] == list(base)
        assert {n: p.default for n, p in base.items()} == {n: override[n].default for n in base}
        extras = {n: p for n, p in override.items() if n not in base}
        assert list(extras) == ["reset_between"]
        assert extras["reset_between"].kind is inspect.Parameter.KEYWORD_ONLY
        assert extras["reset_between"].default is False

    def test_mujoco_overrides_the_default(self) -> None:
        """The reference implementation must not fall through to the refusal."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        assert MuJoCoSimEngine.run_multi_policy is not SimEngine.run_multi_policy

    def test_mujoco_signature_matches_the_base_contract(self) -> None:
        """Parameter names, order and defaults agree between base and override.

        Compared by name and default rather than full ``inspect.signature``
        equality because the annotations are strings under ``from __future__
        import annotations`` and the two modules spell the ``Policy`` forward
        reference differently.
        """
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        base = inspect.signature(SimEngine.run_multi_policy).parameters
        override = inspect.signature(MuJoCoSimEngine.run_multi_policy).parameters
        assert list(base) == list(override)
        assert {n: p.default for n, p in base.items()} == {n: p.default for n, p in override.items()}


# --------------------------------------------------------------------------- #
# _validate_multi_policies                                                    #
# --------------------------------------------------------------------------- #
class TestEmptyPoliciesRejection:
    def test_an_empty_mapping_is_refused_with_the_pinned_text(self) -> None:
        result = SimEngine._validate_multi_policies({}, "run_multi_policy")
        assert result is not None
        assert _text(result) == "run_multi_policy: 'policies' is empty."

    def test_a_populated_mapping_passes(self) -> None:
        assert SimEngine._validate_multi_policies(_TWO_ROBOTS, "run_multi_policy") is None

    def test_the_method_name_prefixes_the_refusal(self) -> None:
        """A future backend entry point reports the refusal under its own name."""
        result = SimEngine._validate_multi_policies({}, "other_driver")
        assert result is not None
        assert _text(result).startswith("other_driver: ")


# --------------------------------------------------------------------------- #
# _normalize_multi_policy_instructions                                        #
# --------------------------------------------------------------------------- #
class TestInstructionNormalization:
    def test_a_single_string_broadcasts_to_every_robot(self) -> None:
        instr_map, err = SimEngine._normalize_multi_policy_instructions(_TWO_ROBOTS, "pick", "run_multi_policy")
        assert err is None
        assert instr_map == {"alpha": "pick", "beta": "pick"}

    def test_a_partial_mapping_defaults_omitted_robots_to_empty(self) -> None:
        instr_map, err = SimEngine._normalize_multi_policy_instructions(
            _TWO_ROBOTS, {"alpha": "pour"}, "run_multi_policy"
        )
        assert err is None
        assert instr_map == {"alpha": "pour", "beta": ""}

    def test_a_key_naming_no_driven_robot_is_refused(self) -> None:
        instr_map, err = SimEngine._normalize_multi_policy_instructions(
            _TWO_ROBOTS, {"ghost": "pour"}, "run_multi_policy"
        )
        assert instr_map is None
        assert err is not None
        assert "run_multi_policy: instructions names a robot not driven by this call" in _text(err)
        assert "ghost" in _text(err)

    @pytest.mark.parametrize("bad", [["pick", "hold"], 12345, None], ids=["list", "int", "none"])
    def test_a_non_string_non_mapping_value_is_refused(self, bad: Any) -> None:
        instr_map, err = SimEngine._normalize_multi_policy_instructions(_TWO_ROBOTS, bad, "run_multi_policy")
        assert instr_map is None
        assert err is not None
        assert "run_multi_policy: 'instructions' must be a string" in _text(err)
        assert type(bad).__name__ in _text(err)

    def test_distinct_instructions_warn_but_still_normalize(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.base"):
            instr_map, err = SimEngine._normalize_multi_policy_instructions(
                _TWO_ROBOTS, {"alpha": "pour", "beta": "catch"}, "run_multi_policy"
            )
        assert err is None
        assert instr_map == {"alpha": "pour", "beta": "catch"}
        assert any("distinct per-robot instructions" in rec.message for rec in caplog.records)

    def test_the_warning_is_attributed_to_the_callers_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        """A backend passes its own module logger so the warning names the loop
        that records the frame, not this shared helper."""
        backend_logger = logging.getLogger("tests.fake_backend.run_multi_policy")
        with caplog.at_level(logging.WARNING, logger="tests.fake_backend.run_multi_policy"):
            SimEngine._normalize_multi_policy_instructions(
                _TWO_ROBOTS, {"alpha": "pour", "beta": "catch"}, "run_multi_policy", warn_logger=backend_logger
            )
        warned = [rec for rec in caplog.records if "distinct per-robot instructions" in rec.message]
        assert [rec.name for rec in warned] == ["tests.fake_backend.run_multi_policy"]

    def test_a_shared_instruction_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """One task per frame is exactly representable; nothing to surface."""
        with caplog.at_level(logging.WARNING):
            _, err = SimEngine._normalize_multi_policy_instructions(
                _TWO_ROBOTS, {"alpha": "pick", "beta": "pick"}, "run_multi_policy"
            )
        assert err is None
        assert not any("distinct per-robot instructions" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# _normalize_multi_policy_horizons                                            #
# --------------------------------------------------------------------------- #
class TestHorizonNormalization:
    def test_a_single_int_broadcasts_to_every_robot(self) -> None:
        horizon_map, err = SimEngine._normalize_multi_policy_horizons(_TWO_ROBOTS, 4, "run_multi_policy")
        assert err is None
        assert horizon_map == {"alpha": 4, "beta": 4}

    def test_a_partial_mapping_defaults_omitted_robots(self) -> None:
        horizon_map, err = SimEngine._normalize_multi_policy_horizons(
            _TWO_ROBOTS, {"alpha": 2}, "run_multi_policy", default_horizon=8
        )
        assert err is None
        assert horizon_map == {"alpha": 2, "beta": 8}

    def test_a_key_naming_no_driven_robot_is_refused(self) -> None:
        horizon_map, err = SimEngine._normalize_multi_policy_horizons(_TWO_ROBOTS, {"ghost": 4}, "run_multi_policy")
        assert horizon_map is None
        assert err is not None
        assert "run_multi_policy: action_horizon names a robot not driven by this call" in _text(err)

    @pytest.mark.parametrize("bad", [0, -5, 2.7, True, "x", float("nan"), None], ids=repr)
    def test_a_non_positive_int_scalar_is_refused(self, bad: Any) -> None:
        horizon_map, err = SimEngine._normalize_multi_policy_horizons(_TWO_ROBOTS, bad, "run_multi_policy")
        assert horizon_map is None
        assert err is not None
        assert "run_multi_policy: action_horizon must be a positive integer" in _text(err)

    @pytest.mark.parametrize("bad", [0, -5, 2.7, True], ids=repr)
    def test_a_non_positive_int_mapping_entry_is_refused_naming_the_entry(self, bad: Any) -> None:
        horizon_map, err = SimEngine._normalize_multi_policy_horizons(_TWO_ROBOTS, {"alpha": bad}, "run_multi_policy")
        assert horizon_map is None
        assert err is not None
        assert "run_multi_policy: action_horizon['alpha'] must be a positive integer" in _text(err)

    def test_an_empty_mapping_keeps_the_default_for_every_robot(self) -> None:
        """A mapping is an override layer, so an empty one is legitimate."""
        horizon_map, err = SimEngine._normalize_multi_policy_horizons(
            _TWO_ROBOTS, {}, "run_multi_policy", default_horizon=8
        )
        assert err is None
        assert horizon_map == {"alpha": 8, "beta": 8}

    def test_a_full_mapping_overrides_every_robot(self) -> None:
        horizon_map, err = SimEngine._normalize_multi_policy_horizons(
            _TWO_ROBOTS, {"alpha": 1, "beta": 10}, "run_multi_policy"
        )
        assert err is None
        assert horizon_map == {"alpha": 1, "beta": 10}
