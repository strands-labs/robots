"""A rollout ``duration`` that resolves to no control step is refused.

``duration`` is one factor of the horizon: with no ``n_steps`` the rollout runs
``int(duration * control_frequency)`` control steps. ``_validate_duration``
judged the factor rather than the product - it delegated to
``positive_finite_number_error``, which reads ``duration`` alone - so every
duration shorter than one control period was positive, finite, and resolved to
zero steps. Measured on a MuJoCo ``so101`` rollout before the fix, at the
library default ``control_frequency=50.0``:

    run_policy(duration=0.01)  ->  status="success", "0 steps", sim_t=0.000s
                                   "Video requested but 0 frames captured"

That is verbatim the outcome ``_validate_duration``'s own docstring says it
exists to prevent ("reported as ``status="success"`` for a rollout that never
queried the policy and never stepped physics - and, when a ``video`` was
requested, wrote no MP4 while still claiming success"). The guard prevented it
only for the half of the domain visible in one factor's sign, and which side of
the line a duration falls on is not a property of the duration:
``duration=1.0`` is a full second and still resolves to no step at 0.5 Hz.

The same reading error is already recorded one parameter over, on the step-count
path: ``_resolve_horizon``'s docstring notes that "a bare ``<= 0`` test only saw
the sign", which let ``n_steps=2.7`` run two steps. This is that error on the
duration path, where the sign belongs to only one of the two factors.

Refused rather than floored to one step, for the reason ``_resolve_horizon``
gives: a horizon that cannot be honored as asked is a caller error, not a value
to substitute silently. A fractional duration that does produce steps stays
usable - the boundary is only at zero - which is what the accepted rows below
pin.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.base import SimEngine  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

#: Minimal single-hinge arm, so the behavioural rows below run a real rollout
#: without depending on a packaged robot description.
ARM_XML = """
<mujoco model="one_joint_arm">
  <worldbody>
    <body name="link" pos="0 0 0.1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <position name="hinge_act" joint="hinge" kp="10"/>
  </actuator>
</mujoco>
"""

#: ``(duration, control_frequency)`` pairs whose product is under one step.
#: Every one is a positive finite number the value domain accepts, and the last
#: two are whole seconds - a duration is not short in isolation.
NO_STEP = [
    pytest.param(0.01, 50.0, id="0.01s-at-50Hz"),
    pytest.param(0.019, 50.0, id="0.019s-just-under-one-period"),
    pytest.param(0.5, 1.0, id="half-a-second-at-1Hz"),
    pytest.param(1.0, 0.5, id="a-whole-second-at-0.5Hz"),
    pytest.param(2.0, 0.25, id="two-seconds-at-0.25Hz"),
]

#: ``(duration, control_frequency, steps)`` triples that DO produce a horizon,
#: including the exact boundary (one period) and a fractional duration that
#: truncates without emptying.
SOME_STEPS = [
    pytest.param(0.02, 50.0, 1, id="exactly-one-period"),
    pytest.param(0.021, 50.0, 1, id="a-hair-over-one-period"),
    pytest.param(2.0, 0.5, 1, id="one-step-at-0.5Hz"),
    pytest.param(2.5, 62.5, 156, id="fractional-and-truncating"),
    pytest.param(10.0, 50.0, 500, id="the-library-defaults"),
]


def _text(result: dict[str, Any]) -> str:
    return " ".join(block["text"] for block in result["content"] if "text" in block)


@pytest.fixture
def arm_path(tmp_path):
    path = tmp_path / "one_joint_arm.xml"
    path.write_text(ARM_XML)
    return str(path)


@pytest.fixture
def sim(arm_path):
    s = Simulation(tool_name="test_rollout_duration_must_produce_a_control_step", mesh=False)
    assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
    assert s.add_robot("arm", urdf_path=arm_path)["status"] == "success"
    yield s
    s.cleanup(policy_stop_timeout=2.0)


class TestTheGuardJudgesTheProduct:
    """The verdict follows ``duration * control_frequency``, not the sign."""

    @pytest.mark.parametrize("duration, rate", NO_STEP)
    def test_a_horizon_of_no_steps_is_refused(self, duration: float, rate: float) -> None:
        assert SimEngine._validate_duration(duration, "run_policy", rate) is not None

    @pytest.mark.parametrize("duration, rate, steps", SOME_STEPS)
    def test_a_horizon_of_at_least_one_step_is_accepted(self, duration: float, rate: float, steps: int) -> None:
        assert int(duration * rate) == steps, "the row states the horizon the consumer will run"
        assert SimEngine._validate_duration(duration, "run_policy", rate) is None

    def test_the_refusal_names_both_factors_and_the_remedy(self) -> None:
        """The caller cannot act on a message naming only the value they passed."""
        refusal = SimEngine._validate_duration(0.01, "run_policy", 50.0)

        assert refusal is not None
        assert _text(refusal) == (
            "run_policy: duration=0.01 at control_frequency=50 resolves to 0 control steps, "
            "so the rollout would query no policy and step no physics. Raise duration to at "
            "least 0.02, or pass n_steps to give the horizon as a step count."
        )

    def test_the_value_domain_keeps_its_own_message(self) -> None:
        """A value no rate could rescue is still refused as the value error it is.

        The hardware guard pins this exact text as the shared one
        (``tests/test_hardware_task_duration_guard.py``), so the horizon
        condition must sit after the value domain rather than in front of it.
        """
        refusal = SimEngine._validate_duration(0, "run_policy", 50.0)

        assert refusal is not None
        assert _text(refusal) == "run_policy: duration must be > 0, got 0."


class TestABooleanNeverReachesTheProduct:
    """Each factor is answered by its own domain before the multiplication.

    The horizon test coerces with ``float()``, and ``float(True)`` is ``1.0`` - a
    silent one-second span or a silent 1 Hz rate. Neither is reachable: the value
    domain above refuses a boolean ``duration`` by name, and the rate is refused
    one call earlier by ``_validate_positive_frequency``, which every rollout
    entry point runs first. ``tests/simulation/test_input_validators_refuse_a_boolean.py``
    exempts this guard from its ``float()`` sweep on exactly that basis, so these
    two rows are what that exemption stands on.
    """

    @pytest.mark.parametrize("spelling", [True, np.True_], ids=["bool", "numpy-bool"])
    def test_a_boolean_duration_is_refused_as_a_value(self, spelling: Any) -> None:
        refusal = SimEngine._validate_duration(spelling, "run_policy", 50.0)

        assert refusal is not None
        assert _text(refusal) == f"run_policy: duration must be > 0, got {spelling!r}."

    @pytest.mark.parametrize("spelling", [True, np.True_], ids=["bool", "numpy-bool"])
    def test_a_boolean_rate_is_refused_before_this_guard_is_reached(self, spelling: Any) -> None:
        assert SimEngine._validate_positive_frequency(spelling, "run_policy") is not None


class TestTheRolloutSurfacesRefuseIt:
    """The refusal reaches the caller of a real rollout, not just the guard.

    Asserted on what the caller receives rather than by comparing against the
    guard's own envelope, so these rows fail on the outcome - a success reported
    for an empty rollout - on any tree where the horizon is not judged.
    """

    def test_run_policy_refuses_it_and_writes_no_video(self, sim, tmp_path) -> None:
        """The empty rollout used to be reported as a success with no MP4."""
        video = tmp_path / "rollout.mp4"
        result = sim.run_policy(
            robot_name="arm",
            policy_provider="mock",
            duration=0.01,
            control_frequency=50.0,
            video={"path": str(video), "fps": 30, "width": 64, "height": 48},
        )

        assert result["status"] == "error", _text(result)
        assert "0 control steps" in _text(result)
        assert "control_frequency=50" in _text(result), "the other factor has to be named"
        assert not video.exists()

    def test_run_policy_still_runs_the_shortest_honorable_horizon(self, sim, tmp_path) -> None:
        """One control period is one step, and its frame reaches the MP4."""
        video = tmp_path / "one_step.mp4"
        result = sim.run_policy(
            robot_name="arm",
            policy_provider="mock",
            duration=0.02,
            control_frequency=50.0,
            video={"path": str(video), "fps": 30, "width": 64, "height": 48},
        )

        assert result["status"] == "success"
        assert "1 steps" in _text(result)
        assert video.exists() and video.stat().st_size > 0

    def test_start_policy_refuses_it_before_the_thread_is_submitted(self, sim) -> None:
        """The background surface resolves the same horizon, so it shares the guard."""
        result = sim.start_policy(robot_name="arm", policy_provider="mock", duration=0.01, control_frequency=50.0)

        assert result["status"] == "error", _text(result)
        assert "start_policy" in _text(result) and "0 control steps" in _text(result)
        assert sim._active_policy_robots() == [], "no rollout may be left running behind a refusal"
