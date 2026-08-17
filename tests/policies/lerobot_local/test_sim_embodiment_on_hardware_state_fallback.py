# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A sim embodiment driven from real hardware must pack the arm's own readings.

``PackStateProcessorStep.observation()`` composes ``observation.state`` from the
embodiment's declared ``state_keys``. A SIM embodiment declares the simulator's
joint naming -- ``so101`` declares ``['1' .. '6']`` -- while the same arm driven
as REAL hardware reports lerobot ``SOFollower`` scalars named ``'<motor>.pos'``.
None of the declared keys is then present, so without a fallback the step would
hand the model an observation carrying no ``observation.state`` at all and
``embodiment="so101"`` would work only in sim.

The step therefore falls back to the hardware ``.pos`` keys, and that fallback
carries five contracts nothing exercised:

* the values are bound POSITIONALLY, so they must land in the arm's motor order
  -- the ordering :func:`~strands_robots.policies.lerobot_local.embodiment.hardware_pos_keys`
  documents as part of its contract;
* the emitted vector is the arm's readings, NOT the all-zero vector the
  zero-fill loop immediately above has just built for the absent sim keys;
* hardware ``.pos`` values are already in the model's training units, so they
  are packed RAW -- the sim-radian to model-degree conversion further down must
  not also run, or every joint is converted twice;
* the consumed ``.pos`` keys leave the observation while every other key
  (cameras, extra motors) passes through untouched;
* the model's declared width is still reconciled through
  :func:`~strands_robots.policies.lerobot_local.embodiment.reconcile_dim` under
  the embodiment's own ``dim_policy``, and a ``.pos`` set too small to fill the
  declared keys yields no partial state.

Together these are the whole of "``embodiment="so101"`` works on the physical
arm too, not only in sim".
"""

import logging
from typing import Any

import numpy as np
import pytest

pytest.importorskip("lerobot")

import strands_robots.policies.lerobot_local.embodiment as E

# lerobot ``SOFollower`` reports its motors in bus order; the positional sim
# ``state_keys`` are index-aligned to it, which is what makes the fallback sound.
MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# Deliberately non-monotonic and mixed-sign so a re-ordering (alphabetical, say)
# cannot coincide with the expected vector.
HW_VALUES = [11.0, -22.0, 33.0, -4.0, 55.0, 12.5]
SIM_KEYS_SO101 = ["1", "2", "3", "4", "5", "6"]


def _hardware_observation(*, motors: int = 6, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """An observation shaped like ``SOFollower.get_observation()``.

    Insertion order is motor order, which is the ordering the fallback binds to.
    """
    obs: dict[str, Any] = {f"{MOTOR_ORDER[i]}.pos": HW_VALUES[i] for i in range(motors)}
    if extra:
        obs.update(extra)
    return obs


def _step(state_keys: list[str], expected_dim: int, **kw: Any) -> Any:
    """Build the registered step, skipping when lerobot's processor framework is absent."""
    step_cls = E.register_pack_state_step()
    if step_cls is None:
        pytest.skip("lerobot processor framework unavailable")
    kw.setdefault("dim_policy", "pad")
    kw.setdefault("state_units", "degrees")
    return step_cls(state_keys=list(state_keys), expected_dim=expected_dim, **kw)


@pytest.fixture(autouse=True)
def _reset_warn_dedup() -> None:
    """The missing-key warning is deduplicated process-wide; start each test clean."""
    E._WARNED_STATE_KEY_MISMATCH.clear()


class TestSimEmbodimentDrivenFromHardware:
    def test_the_declared_sim_keys_are_absent_from_a_hardware_observation(self) -> None:
        """Premise: the fixture really does reach the no-declared-key branch.

        Without this the assertions below could pass through the ordinary
        packing path and say nothing about the fallback.
        """
        obs = _hardware_observation()
        assert not any(k in obs for k in SIM_KEYS_SO101)
        assert E.hardware_pos_keys(obs) == [f"{m}.pos" for m in MOTOR_ORDER]

    def test_hardware_readings_pack_in_motor_order(self) -> None:
        """Each motor's reading lands at its own model index."""
        state = _step(SIM_KEYS_SO101, 6).observation(_hardware_observation())["observation.state"]
        np.testing.assert_allclose(state.numpy(), HW_VALUES, atol=1e-5)

    def test_the_state_is_the_arm_reading_not_the_zero_fill_vector(self) -> None:
        """The zero-fill loop appends 0.0 for every absent declared key before
        this branch runs; the emitted vector must be the arm's readings rather
        than that all-zero placeholder."""
        state = _step(SIM_KEYS_SO101, 6).observation(_hardware_observation())["observation.state"]
        assert not np.allclose(state.numpy(), np.zeros(6))

    def test_the_fallback_does_not_warn_about_the_absent_sim_keys(self, caplog: Any) -> None:
        """Binding the hardware keys is a supported configuration, not a
        degradation, so it must not emit the missing-state-key warning."""
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.lerobot_local.embodiment"):
            _step(SIM_KEYS_SO101, 6).observation(_hardware_observation())
        assert [r for r in caplog.records if "absent from the observation" in r.getMessage()] == []
        assert not E._WARNED_STATE_KEY_MISMATCH

    def test_consumed_pos_keys_leave_and_every_other_key_passes_through(self) -> None:
        """The bound ``.pos`` keys are replaced by ``observation.state``; a camera
        and a motor beyond the declared width are untouched."""
        obs = _hardware_observation(motors=6, extra={"observation.images.front": "FRAME", "aux.pos": 7.0})
        out = _step(SIM_KEYS_SO101, 6).observation(obs)
        assert [k for k in out if k.endswith(".pos")] == ["aux.pos"]
        assert out["observation.images.front"] == "FRAME"
        assert "observation.state" in out

    def test_values_are_packed_raw_even_when_the_embodiment_declares_degrees(self) -> None:
        """Hardware ``.pos`` is already in the model's training units, so the
        sim-radian to degree conversion below this branch must not also apply.
        Mid-points and a gripper range are configured precisely so that a
        conversion would be visible."""
        step = _step(
            SIM_KEYS_SO101,
            6,
            state_units="degrees",
            gripper_index=5,
            gripper_joint_range=[0.0, 1.75],
            joint_mids=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        state = step.observation(_hardware_observation())["observation.state"]
        np.testing.assert_allclose(state.numpy(), HW_VALUES, atol=1e-5)

    @pytest.mark.parametrize(
        ("expected_dim", "dim_policy", "expected"),
        [
            (0, "pad", HW_VALUES),
            (6, "pad", HW_VALUES),
            (8, "pad", [*HW_VALUES, 0.0, 0.0]),
            (4, "truncate", HW_VALUES[:4]),
        ],
    )
    def test_the_declared_width_is_reconciled_under_the_embodiment_policy(
        self, expected_dim: int, dim_policy: str, expected: list[float]
    ) -> None:
        """The model's declared width still governs, via the shared
        ``reconcile_dim``; ``expected_dim=0`` means "no declared width"."""
        step = _step(SIM_KEYS_SO101, expected_dim, dim_policy=dim_policy)
        state = step.observation(_hardware_observation())["observation.state"]
        np.testing.assert_allclose(state.numpy(), expected, atol=1e-5)

    def test_a_strict_policy_still_refuses_a_width_mismatch(self) -> None:
        """``dim_policy="strict"`` is not weakened by taking the fallback."""
        with pytest.raises(ValueError, match="observation.state dim 6 != model expected 8"):
            _step(SIM_KEYS_SO101, 8, dim_policy="strict").observation(_hardware_observation())

    def test_too_few_pos_keys_leaves_the_observation_untouched(self) -> None:
        """A ``.pos`` set that cannot fill the declared keys yields no partial
        state: the observation passes through so a clearer downstream error can
        fire instead of a silently short vector."""
        obs = _hardware_observation(motors=5)
        out = _step(SIM_KEYS_SO101, 6).observation(dict(obs))
        assert "observation.state" not in out
        assert out == obs

    def test_the_shipped_so101_config_binds_a_hardware_observation(self) -> None:
        """End to end through the real registry entry: the sim embodiment a
        caller names as ``embodiment="so101"`` produces the arm's own state."""
        emb = E.load_embodiment("so101")
        assert emb.state_keys == SIM_KEYS_SO101, "fixture assumes the shipped sim joint naming"
        step = _step(
            emb.state_keys,
            len(emb.state_keys),
            dim_policy=emb.dim_policy,
            state_units=emb.state_units,
        )
        state = step.observation(_hardware_observation())["observation.state"]
        np.testing.assert_allclose(state.numpy(), HW_VALUES, atol=1e-5)
