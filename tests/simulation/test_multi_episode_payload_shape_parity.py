"""``run_policy``'s payload keeps its documented shape at every episode count.

``run_policy``'s docstring names the fields an agent reads out of the ``json``
block, and tells it which to gate on: "Gate on ``partial_action_failure_rate``
and the binding-degradation flags below to decide whether a rollout is worth
anything", because a rollout that drove a subset of the robot's actuators is
deliberately ``status="success"``. Two of those fields are documented for a
LOOP specifically - ``policy_load_cache_hit`` (``False`` on episode 2+ means
the caller rebuilt the policy) and ``policy_resident_rss_mb`` ("across a loop,
that it stays resident") - so the multi-episode payload is the one that has to
carry them.

The aggregate the ``n_episodes > 1`` path returns is built separately from the
single-rollout payload, and every field it did not restate was absent from it.
The gate then read its healthy default on the shape used to collect data: the
same policy on the same robot for the same horizon reported the binding
degradation at ``n_episodes=1`` and reported nothing at ``n_episodes=3``, both
``status="success"``.

These tests pin the parity that closes it:

* the documented gate reaches the SAME verdict at any episode count;
* every field the single-rollout payload reads off the shared policy object is
  carried into the aggregate - and that set is derived from the runner's own
  payload build rather than listed here, so a seventh such field is graded on
  arrival;
* the identity fields are constant, because they are the call's own inputs;
* per-episode action health stays in the ``episodes`` records, which is the
  route the docstring points at - a rate has no single aggregate an N-episode
  call can report without choosing a summary, so this deliberately does NOT
  require one.
"""

from __future__ import annotations

import ast
import inspect
import os
from typing import Any

import pytest

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation import create_simulation  # noqa: E402
from strands_robots.simulation import policy_runner as policy_runner_mod  # noqa: E402

# The robot has six actuators; the policy below drives one of them, so a
# healthy-looking report is measurably wrong rather than merely unspecific.
ROBOT = "arm1"
N_ACTUATORS = 6
# The one joint the policy below drives, of the six this robot carries.
DRIVEN_JOINT = "Rotation"
INSTRUCTION = "drive one joint"

# Fields whose aggregate is a summary choice rather than a value the call
# already knows. They stay per-episode, and the tests below pin that.
PER_EPISODE_ONLY = ("action_errors", "action_resolution_rate", "partial_action_failure_rate")


@pytest.fixture
def sim():
    s = create_simulation()
    s.create_world()
    s.add_robot(ROBOT, data_config="so100")
    yield s
    s.cleanup()


class DegradedBindingPolicy(Policy):
    """Drives 1 of the robot's 6 actuators and reports degraded binding.

    The flags are what a real policy sets when it could not bind the
    observation to the model's inputs by name: a camera routed positionally, or
    ``observation.state`` composed from the observation's own scalar keys. The
    load telemetry is set too, so the loop-specific reading of
    ``policy_load_cache_hit`` has a value to carry.
    """

    def __init__(self) -> None:
        self.positional_fallback_used = True
        self.generic_state_keys_used = True
        self.missing_state_keys_used = False
        self.load_time_s = 1.25
        self.load_cache_hit = True

    @property
    def provider_name(self) -> str:
        return "degraded_binding"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        return None

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{DRIVEN_JOINT: 0.2} for _ in range(8)]


def _policy_derived_payload_fields() -> set[str]:
    """Payload keys the single-rollout build reads off the shared policy object.

    Derived from the runner's own source rather than listed here: a key whose
    value expression reads ``policy`` (or the process RSS taken at result time)
    has one value for the whole call, so an N-episode aggregate can report it
    without choosing a summary. A seventh such field added to the single-rollout
    payload is therefore graded by these tests on arrival.

    Returns:
        The key names, read out of the runner module's payload dict literals.
    """
    tree = ast.parse(inspect.getsource(policy_runner_mod))
    fields: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            # Skip the envelope wrappers, whose nested payload merely contains
            # these expressions; only a leaf value is a field of its own.
            if isinstance(value, (ast.Dict, ast.List)):
                continue
            source = ast.unparse(value)
            if "getattr(policy," in source or "process_rss_mb(" in source:
                fields.add(key.value)
    return fields


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    for block in result.get("content", []) or []:
        if isinstance(block, dict) and isinstance(block.get("json"), dict):
            return block["json"]
    raise AssertionError(f"no json block in {result}")


def _run(sim: Any, n_episodes: int) -> dict[str, Any]:
    result = sim.run_policy(
        robot_name=ROBOT,
        policy_object=DegradedBindingPolicy(),
        instruction=INSTRUCTION,
        n_steps=12,
        control_frequency=50.0,
        n_episodes=n_episodes,
    )
    assert result["status"] == "success", result
    return _json_block(result)


def _documented_gate(report: dict[str, Any]) -> str:
    """The verdict ``run_policy``'s docstring tells a caller to compute.

    Read with ``.get`` and the healthy default exactly as a caller does, so an
    absent field is scored the way it is actually scored in the wild rather
    than raising.

    Args:
        report: The ``json`` block of a ``run_policy`` result.

    Returns:
        ``"degraded"`` or ``"healthy"``.
    """
    degraded = (
        float(report.get("partial_action_failure_rate", 0.0)) > 0.0
        or bool(report.get("positional_fallback_used", False))
        or bool(report.get("generic_state_keys_used", False))
        or bool(report.get("missing_state_keys_used", False))
    )
    return "degraded" if degraded else "healthy"


class TestTheDocumentedGateReadsTheSameAtEveryEpisodeCount:
    """The gate the docstring names must not depend on ``n_episodes``."""

    def test_a_degraded_rollout_reads_degraded_at_both_episode_counts(self, sim):
        single = _run(sim, 1)
        multi = _run(sim, 3)
        assert _documented_gate(single) == "degraded", single
        assert _documented_gate(multi) == _documented_gate(single), (
            f"the same policy on the same robot reads "
            f"{_documented_gate(single)!r} at n_episodes=1 and "
            f"{_documented_gate(multi)!r} at n_episodes=3; the gate "
            f"run_policy documents flips with the episode count. Missing from "
            f"the aggregate: {sorted(set(single) - set(multi))}"
        )

    def test_the_binding_flags_carry_their_values_not_just_their_keys(self, sim):
        multi = _run(sim, 3)
        assert multi["positional_fallback_used"] is True, multi
        assert multi["generic_state_keys_used"] is True, multi
        assert multi["missing_state_keys_used"] is False, multi


class TestEveryPolicyDerivedFieldIsCarried:
    """A field with one value for the whole call is reported for the whole call."""

    def test_the_scan_finds_the_policy_derived_fields(self):
        """Premise: a clean result below must not come from an empty scan."""
        fields = _policy_derived_payload_fields()
        assert {
            "positional_fallback_used",
            "generic_state_keys_used",
            "missing_state_keys_used",
            "policy_load_time_s",
            "policy_load_cache_hit",
            "policy_resident_rss_mb",
        } <= fields, fields

    def test_the_aggregate_carries_every_policy_derived_field(self, sim):
        derived = _policy_derived_payload_fields()
        single = _run(sim, 1)
        multi = _run(sim, 3)
        expected = derived & set(single)
        assert expected, single
        assert expected <= set(multi), (
            f"the single-rollout payload reads {sorted(expected)} off the "
            f"shared policy object; the n_episodes=3 aggregate omits "
            f"{sorted(expected - set(multi))}"
        )

    def test_the_identity_fields_are_the_calls_own_inputs(self, sim):
        multi = _run(sim, 3)
        assert multi["robot_name"] == ROBOT, multi
        assert multi["instruction"] == INSTRUCTION, multi
        assert multi["policy"] == DegradedBindingPolicy.__name__, multi


class TestTheAggregateDoesNotSummariseWhatItCannot:
    """Per-episode action health stays per episode - the documented route."""

    def test_per_episode_records_carry_the_action_health(self, sim):
        multi = _run(sim, 3)
        episodes = multi["episodes"]
        assert len(episodes) == 3, multi
        for record in episodes:
            for field in PER_EPISODE_ONLY:
                assert field in record, (field, sorted(record))
            # 1 of 6 actuators driven, so the per-episode rate is the harm the
            # aggregate cannot summarise - it must still be readable here.
            assert record["partial_action_failure_rate"] == pytest.approx(1.0 - 1.0 / N_ACTUATORS, abs=1e-3), record

    def test_the_aggregate_reports_no_summarised_action_rate(self, sim):
        """Aggregating a rate is a summary choice, so none is invented here."""
        multi = _run(sim, 3)
        for field in PER_EPISODE_ONLY:
            assert field not in multi, (
                f"{field} appeared on the aggregate; summarising a per-step "
                f"rate across episodes is a choice this contract leaves to the "
                f"caller, which reads it per episode from 'episodes'"
            )


class TestTheSingleEpisodeShapeIsUnchanged:
    """The historical fast-path payload must not move."""

    def test_the_single_payload_carries_no_aggregate_only_field(self, sim):
        single = _run(sim, 1)
        for field in ("episodes", "stopped_reasons", "total_steps", "video_paths"):
            assert field not in single, (field, sorted(single))

    def test_the_single_payload_still_reports_its_own_action_health(self, sim):
        single = _run(sim, 1)
        for field in PER_EPISODE_ONLY:
            assert field in single, (field, sorted(single))
        assert single["partial_action_failure_rate"] == pytest.approx(1.0 - 1.0 / N_ACTUATORS, abs=1e-3), single

    def test_the_episode_count_fields_are_exact_at_both_counts(self, sim):
        single = _run(sim, 1)
        multi = _run(sim, 3)
        assert single["n_episodes_requested"] == 1
        assert single["n_episodes_completed"] == 1
        assert multi["n_episodes_requested"] == 3
        assert multi["n_episodes_completed"] == 3
        assert multi["steps_used"] == multi["total_steps"]
