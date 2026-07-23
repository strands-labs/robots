"""Shape-normalization + unit-conversion contract for action-tensor decoding.

``LerobotLocalPolicy._tensor_to_action_dicts`` turns a raw policy action tensor
into the per-step ``{actuator_key: value}`` dicts the sim consumes. It must:

* accept 1-D (single step), 2-D (horizon, dim) and 3-D (batch, horizon, dim)
  tensors, and any higher rank by flattening to one step;
* cap the emitted chunk at ``execution_horizon`` (the consumer's re-query
  interval), never the raw trained chunk length;
* map values onto ``robot_state_keys`` by index; and
* convert model action units to sim units when the embodiment declares
  ``action_units="degrees"`` (SO-arm checkpoints emit degrees; MuJoCo joints
  are radians), and stay a no-op for the default ``"native"`` embodiment.

These pin the tensor-shape and unit-conversion behavior that the index->name
mapping depends on; the sibling ``test_action_diagnostics`` module covers the
dim-mismatch / near-zero warnings.
"""

from __future__ import annotations

import math

import torch

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


def _policy(execution_horizon: int = 8) -> LerobotLocalPolicy:
    # actions_per_step drives execution_horizon when RTC is off (the default).
    policy = LerobotLocalPolicy(actions_per_step=execution_horizon)
    policy.set_robot_state_keys(["j1", "j2", "j3"])
    return policy


def test_1d_tensor_is_a_single_step():
    policy = _policy()
    # exactly-representable float32 values so the dict compares cleanly.
    result = policy._tensor_to_action_dicts(torch.tensor([0.5, -1.5, 2.0]))
    assert result == [{"j1": 0.5, "j2": -1.5, "j3": 2.0}]


def test_2d_tensor_maps_each_horizon_row_in_order():
    policy = _policy()
    tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    result = policy._tensor_to_action_dicts(tensor)
    assert result == [
        {"j1": 0.0, "j2": 1.0, "j3": 2.0},
        {"j1": 3.0, "j2": 4.0, "j3": 5.0},
    ]


def test_2d_tensor_chunk_is_capped_at_execution_horizon():
    # execution_horizon=2, but the model emits a 4-step chunk: only the first
    # two steps are returned (the consumer re-queries after execution_horizon).
    policy = _policy(execution_horizon=2)
    tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    result = policy._tensor_to_action_dicts(tensor)
    assert len(result) == 2
    assert result[0] == {"j1": 0.0, "j2": 1.0, "j3": 2.0}
    assert result[1] == {"j1": 3.0, "j2": 4.0, "j3": 5.0}


def test_3d_batched_tensor_uses_first_batch_and_slices_horizon():
    # Shape [batch=2, horizon=3, dim=3]. Only batch 0 is decoded, capped at
    # execution_horizon. Batch 1 must not leak into the result.
    policy = _policy(execution_horizon=2)
    batch0 = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    batch1 = [[90.0, 90.0, 90.0]] * 3  # sentinel: must be ignored
    tensor = torch.tensor([batch0, batch1])
    result = policy._tensor_to_action_dicts(tensor)
    assert len(result) == 2  # horizon sliced to execution_horizon
    assert result[0] == {"j1": 0.0, "j2": 1.0, "j3": 2.0}
    assert result[1] == {"j1": 3.0, "j2": 4.0, "j3": 5.0}
    assert all(90.0 not in d.values() for d in result)


def test_higher_rank_tensor_flattens_to_a_single_step():
    # A 4-D tensor has no defined batch/horizon semantics: it is flattened and
    # decoded as one step (values beyond the key count are dropped).
    policy = _policy()
    tensor = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    result = policy._tensor_to_action_dicts(tensor)
    assert result == [{"j1": 0.0, "j2": 1.0, "j3": 2.0}]


def test_native_embodiment_does_not_convert_units():
    policy = _policy()
    policy._embodiment = EmbodimentMap(
        name="native_test",
        state_keys=["j1", "j2", "j3"],
        action_keys=["j1", "j2", "j3"],
        action_units="native",
    )
    result = policy._tensor_to_action_dicts(torch.tensor([90.0, 180.0, 45.0]))
    # native = passthrough, no deg->rad scaling.
    assert result == [{"j1": 90.0, "j2": 180.0, "j3": 45.0}]


def test_degrees_embodiment_converts_model_action_to_sim_radians():
    # A degrees embodiment (SO-arm checkpoint convention) must convert the
    # model's degree action into the sim's radian joint units before mapping.
    policy = _policy()
    policy._embodiment = EmbodimentMap(
        name="degrees_test",
        state_keys=["j1", "j2", "j3"],
        action_keys=["j1", "j2", "j3"],
        action_units="degrees",
    )
    result = policy._tensor_to_action_dicts(torch.tensor([90.0, 180.0, 45.0]))
    assert len(result) == 1
    got = result[0]
    assert math.isclose(got["j1"], math.pi / 2, rel_tol=1e-6)
    assert math.isclose(got["j2"], math.pi, rel_tol=1e-6)
    assert math.isclose(got["j3"], math.pi / 4, rel_tol=1e-6)
