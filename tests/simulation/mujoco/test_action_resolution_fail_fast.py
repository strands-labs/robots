"""PolicyRunner action-resolution diagnostics.

Covers two related behaviours that turn a silently-broken rollout into an
actionable signal:

* Fail-fast: when EVERY action step in the opening probe window drives zero
  actuators (none of the policy's emitted keys resolve to any of the robot's
  actuators), the rollout can never move the robot. The runner raises at the
  probe boundary instead of running the whole episode (and every remaining
  inference call + recording write). Pre-fix the error surfaced only at episode
  end, so an ``n_episodes`` x N-step eval burned the full budget first.

* Per-actuator resolution stats: the result json reports
  ``action_resolution_rate`` (fraction of steps each actuator was driven) and
  ``partial_action_failure_rate`` (mean fraction of the robot's DOF never
  driven). A policy that drives only 1 of 6 joints returns ``status=success``
  with ``action_errors=0`` yet a ``partial_action_failure_rate`` of ~0.83, so a
  silently under-actuated rollout is visible instead of looking like a clean
  success with a zero success-rate.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.base import Policy
from strands_robots.simulation.mujoco.simulation import Simulation


class _FixedKeysPolicy(Policy):
    """Policy that ignores ``set_robot_state_keys`` and emits a fixed key map.

    Models a misconfigured external policy whose output keys are pinned to the
    wrong embodiment. The chunk is length 1 so the runner re-queries (and so
    re-applies the same keys) every control step.
    """

    def __init__(self, action: dict[str, float]) -> None:
        self._action = action

    @property
    def provider_name(self) -> str:
        return "fixed_keys_test"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        # Intentionally ignore the correct keys.
        pass

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [dict(self._action)]


@pytest.fixture
def sim():
    s = Simulation(mesh=False)
    s.create_world()
    s.add_robot("so101")
    # so101 actuator/joint names are ["1".."6"]; assert so the tests below
    # stay valid if the asset's naming ever changes.
    assert s.robot_joint_names("so101") == ["1", "2", "3", "4", "5", "6"]
    yield s
    s.cleanup()


class TestFailFastOnTotalUnresolved:
    def test_raises_within_probe_window_when_all_keys_unresolved(self, sim):
        """100% unresolved keys -> error within the first 3 steps, not at end."""
        policy = _FixedKeysPolicy({"shoulder_pan": 0.5})  # no such actuator on so101
        result = sim.run_policy(
            robot_name="so101",
            policy_object=policy,
            n_steps=50,  # would run 50 steps pre-fix; must bail at step 3
            control_frequency=20.0,
            fast_mode=True,
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        # The error must name the offending keys and point at the fix.
        assert "shoulder_pan" in text
        assert "get_features" in text
        # Crucially: it bailed in the probe window, it did NOT run all 50 steps.
        json_block = next((c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c), None)
        # The error path carries the rtc telemetry json; n_steps may be absent
        # there, so assert the bail happened by checking the message wording and
        # that no "50 steps" full-run completion text is present.
        assert "50 steps" not in text
        # Probe window is 3 -> the message reports exactly 3 probed steps.
        assert "first 3 action steps" in text
        if json_block is not None and "n_steps" in json_block:
            assert json_block["n_steps"] <= 3


class TestPartialResolutionStats:
    def test_partial_drive_reports_resolution_rate_and_does_not_fail_fast(self, sim):
        """Driving 1 of 6 joints every step -> success with ~0.83 failure rate."""
        policy = _FixedKeysPolicy({"1": 0.5})  # valid actuator, 1 of 6
        result = sim.run_policy(
            robot_name="so101",
            policy_object=policy,
            n_steps=6,
            control_frequency=20.0,
            fast_mode=True,
        )
        assert result["status"] == "success", result
        payload = next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)
        # Ran to completion (fail-fast did NOT fire: joint "1" resolves each step).
        assert payload["n_steps"] == 6
        # action_errors is a step-level status count; joint "1" resolves every
        # step so no send_action returned error -> 0.
        assert payload["action_errors"] == 0
        rates = payload["action_resolution_rate"]
        assert rates["1"] == 1.0
        assert all(rates[j] == 0.0 for j in ["2", "3", "4", "5", "6"])
        # 5 of 6 actuators never driven -> mean undriven fraction ~ 0.8333.
        assert payload["partial_action_failure_rate"] == pytest.approx(5 / 6, abs=1e-3)

    def test_full_drive_reports_zero_partial_failure(self, sim):
        """A policy driving all 6 joints -> partial_action_failure_rate == 0.0."""
        from strands_robots.policies.mock import MockPolicy

        result = sim.run_policy(
            robot_name="so101",
            policy_object=MockPolicy(),
            n_steps=5,
            control_frequency=20.0,
            fast_mode=True,
        )
        assert result["status"] == "success", result
        payload = next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)
        assert payload["partial_action_failure_rate"] == 0.0
        rates = payload["action_resolution_rate"]
        assert set(rates) == {"1", "2", "3", "4", "5", "6"}
        assert all(v == 1.0 for v in rates.values())


class _FixedVectorPolicy(Policy):
    """Policy that emits a fixed-length numeric action vector every step.

    Models a policy trained for a different embodiment whose action head width
    does not match the target robot's joint count - e.g. a 7-DOF checkpoint
    driving a 6-DOF arm by positional vector. A numeric vector binds
    positionally to every joint, so a length mismatch is a structural failure
    the runner cannot partially resolve.
    """

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    @property
    def provider_name(self) -> str:
        return "fixed_vector_test"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        pass

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        # A bare numeric vector is a valid runner action (it binds positionally
        # to every joint); the Policy ABC types the chunk as dicts, so silence
        # the element-type check for this intentionally-vector chunk.
        return [list(self._vector)]  # type: ignore[list-item]


class TestFailFastOnActionVectorShapeMismatch:
    """A numeric action vector whose length never matches the robot's joint
    count drives zero actuators on every step. ``send_action`` rejects it with a
    plain shape-mismatch error that carries no per-key breakdown (unlike a dict
    with unresolvable names), so the runner must still count the step as a 100%
    failure and bail in the probe window - not silently run the full episode
    with an arm that never moves.
    """

    @pytest.mark.parametrize(
        "vector",
        [
            pytest.param([0.1, 0.2, 0.3], id="too-short"),
            pytest.param([0.1] * 9, id="too-long"),
        ],
    )
    def test_vector_length_mismatch_fails_fast_within_probe_window(self, sim, vector):
        policy = _FixedVectorPolicy(vector)  # so101 has 6 joints
        result = sim.run_policy(
            robot_name="so101",
            policy_object=policy,
            n_steps=50,  # would run 50 steps pre-fix; must bail at step 3
            control_frequency=20.0,
            fast_mode=True,
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        # Bailed in the probe window, did NOT run the full 50-step episode.
        assert "first 3 action steps" in text
        assert "50 steps" not in text
        # The diagnostic points at the embodiment fix and names valid joints.
        assert "get_features" in text
        assert "1" in text and "6" in text


class TestFailFastOnDictVectorValuedKey:
    """A dict action whose VALUE is non-scalar (a vector on one key, e.g. a
    policy emitting ``{"base_velocity": [vx, vy, omega]}`` or a wrongly-shaped
    per-joint chunk) is rejected atomically by ``send_action`` with a plain
    scalar-value error that carries NO per-key breakdown - unlike a dict whose
    KEYS are unresolvable, which enumerates ``unresolved_keys``.

    The key itself may be a perfectly valid actuator name; it is the value shape
    that is wrong, so nothing on that step drives the robot. The runner must
    still count the whole step as a 100% failure - crediting the dict's own keys
    as unresolved from the unstructured error - so the fail-fast probe surfaces a
    structurally-dead rollout instead of silently running the full episode with
    an arm that never moves. Pre-pin, this unstructured-error branch (dict form,
    no json breakdown) was the one action-failure path with no coverage.
    """

    def test_dict_with_vector_valued_key_fails_fast_within_probe_window(self, sim):
        # "1" IS a valid so101 actuator, but its value is a 2-vector, not a
        # scalar target, so send_action rejects the whole action with no
        # per-key breakdown.
        policy = _FixedKeysPolicy({"1": [0.1, 0.2]})
        result = sim.run_policy(
            robot_name="so101",
            policy_object=policy,
            n_steps=50,  # would run 50 steps if the dead step were miscounted
            control_frequency=20.0,
            fast_mode=True,
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        # Bailed in the probe window; did NOT run the full 50-step episode.
        assert "first 3 action steps" in text
        assert "50 steps" not in text
        # The unstructured error is attributed to the dict's own key: the
        # runner records "1" as unresolved for the dead step so the diagnostic
        # names the offending key rather than reporting an empty key set.
        assert "'1'" in text
        assert "get_features" in text
