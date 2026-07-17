"""Deterministic unit tests for the from-scratch RL trainer (PPO).

These run in CI: they need ``torch`` (and ``mujoco`` for the env-contract /
smoke-train cases) but no model downloads and no convergence assumptions. The
end-to-end convergence proof lives in ``tests_integ/training/test_ppo_reach.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from strands_robots.training import create_trainer, list_trainers
from strands_robots.training.base import TrainSpec

torch = pytest.importorskip("torch")


def test_ppo_registered_and_created() -> None:
    assert "ppo" in list_trainers()
    trainer = create_trainer("ppo")
    assert trainer.provider_name == "ppo"


def test_empirical_normalization_converges_to_true_stats() -> None:
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    torch.manual_seed(0)
    norm = EmpiricalNormalization(3, device="cpu")
    norm.train()
    data = torch.randn(20000, 3) * torch.tensor([2.0, 5.0, 0.5]) + torch.tensor([1.0, -3.0, 10.0])
    for i in range(0, data.shape[0], 256):
        norm(data[i : i + 256])

    assert torch.allclose(norm.mean, torch.tensor([1.0, -3.0, 10.0]), atol=0.1)
    assert torch.allclose(norm.std, torch.tensor([2.0, 5.0, 0.5]), atol=0.15)

    # In eval mode the statistics freeze and whitening is deterministic.
    norm.eval()
    before = norm.mean.clone()
    norm(torch.ones(100, 3) * 999.0)
    assert torch.allclose(norm.mean, before)
    whitened = norm(torch.full((4, 3), 1.0))
    assert torch.allclose(whitened, norm(torch.full((4, 3), 1.0)))


def test_compute_gae_hand_computed() -> None:
    from strands_robots.training.rl.ppo import compute_gae

    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    next_values = torch.zeros(3)
    dones = torch.tensor([0.0, 0.0, 1.0])
    terminated = torch.zeros(3)
    adv, ret = compute_gae(rewards, values, next_values, dones, terminated, gamma=1.0, lam=1.0)
    # undiscounted cumulative future reward with the trace reset at the boundary
    assert torch.allclose(adv, torch.tensor([3.0, 2.0, 1.0]))
    assert torch.allclose(ret, adv + values)


def test_compute_gae_bootstraps_timeout_not_terminal() -> None:
    from strands_robots.training.rl.ppo import compute_gae

    rewards = torch.tensor([1.0])
    values = torch.zeros(1)
    next_values = torch.tensor([5.0])
    dones = torch.tensor([1.0])
    # time-out: bootstrap with next_value -> delta = 1 + 0.9 * 5
    adv_to, _ = compute_gae(rewards, values, next_values, dones, torch.tensor([0.0]), gamma=0.9, lam=0.95)
    assert torch.allclose(adv_to, torch.tensor([5.5]))
    # real terminal: no bootstrap -> delta = 1
    adv_term, _ = compute_gae(rewards, values, next_values, dones, torch.tensor([1.0]), gamma=0.9, lam=0.95)
    assert torch.allclose(adv_term, torch.tensor([1.0]))


def test_validate_rejects_bad_specs() -> None:
    from strands_robots.training.rl import RLTrainSpec

    trainer = create_trainer("ppo")

    # A plain (non-RL) spec is rejected with a clear message.
    problems = trainer.validate(TrainSpec(output_dir="/tmp/x"))
    assert any("RLTrainSpec" in p for p in problems)

    # Missing env_factory.
    problems = trainer.validate(RLTrainSpec(output_dir="/tmp/x"))
    assert any("env_factory" in p for p in problems)

    # Vectorized envs ARE now supported by PPO (VecSimEnv path): num_envs > 1
    # must NOT be rejected (only the missing env_factory above is the problem).
    problems = trainer.validate(RLTrainSpec(output_dir="/tmp/x", num_envs=4))
    assert not any("num_envs" in p for p in problems), problems
    # But num_envs < 1 is still invalid.
    problems = trainer.validate(RLTrainSpec(output_dir="/tmp/x", env_factory=lambda: None, num_envs=0))  # type: ignore[arg-type,return-value]
    assert any("num_envs must be >= 1" in p for p in problems)

    # rollout_steps must divide into num_mini_batches.
    problems = trainer.validate(RLTrainSpec(output_dir="/tmp/x", rollout_steps=10, num_mini_batches=3))
    assert any("divisible" in p for p in problems)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"output_dir": ""}, "output_dir is required"),
        ({"total_timesteps": 0}, "total_timesteps must be > 0"),
        ({"total_timesteps": -1}, "total_timesteps must be > 0"),
        ({"rollout_steps": 0}, "rollout_steps must be > 0"),
        ({"num_envs": 0}, "num_envs must be >= 1"),
        ({"rollout_steps": 10, "num_mini_batches": 3}, "divisible"),
        ({"num_mini_batches": 0}, "divisible"),
    ],
)
def test_validate_flags_each_out_of_range_field(overrides: dict[str, Any], expected: str) -> None:
    """Each numeric guard in the PPO preflight names the offending field.

    ``validate`` is a pure, read-only preflight the ``train`` entry point runs
    before touching torch or the env, so a misconfigured spec fails with an
    actionable message instead of a deep stack trace inside ``setup``. One
    otherwise-valid spec per case isolates a single bad field; a non-None
    ``env_factory`` keeps the env-factory guard quiet so only the field under
    test is exercised. Mirrors the FastSAC preflight contract test.
    """
    from strands_robots.training.rl import RLTrainSpec

    trainer = create_trainer("ppo")
    base: dict[str, Any] = {"output_dir": "/tmp/x", "env_factory": lambda: None}
    base.update(overrides)
    problems = trainer.validate(RLTrainSpec(**base))
    assert any(expected in p for p in problems), problems


def test_train_rejects_non_rl_spec() -> None:
    trainer = create_trainer("ppo")
    result = trainer.train(TrainSpec(output_dir="/tmp/x"))
    assert result.status == "error"
    assert "RLTrainSpec" in result.message


# --- env-contract + smoke train (need MuJoCo) ---

pytest.importorskip("mujoco")


def _elbow_reward(engine):  # type: ignore[no-untyped-def]
    return -abs(float(engine.get_observation(skip_images=True)["Elbow"]) - 0.2)


def _make_reach_env():  # type: ignore[no-untyped-def]
    import strands_robots as sr
    from strands_robots.training.rl import SimEnv

    engine = sr.Robot("so100", mode="sim")
    return SimEnv(
        engine,
        actor_obs_keys=["Elbow", "Elbow.vel"],
        reward_terms=[_elbow_reward],
        action_dim=6,
        max_episode_steps=10,
    )


def test_simenv_contract() -> None:
    env = _make_reach_env()
    assert env.num_actor_obs == 2
    assert env.num_critic_obs == 2
    assert env.num_actions == 6

    obs = env.reset()
    assert obs["actor_obs"].shape == (1, 2)
    assert obs["critic_obs"].shape == (1, 2)

    action = torch.zeros(1, 6)
    next_obs, reward, done, info = env.step(action)
    assert next_obs["actor_obs"].shape == (1, 2)
    assert reward.shape == (1,)
    assert done.shape == (1,)
    assert torch.isfinite(reward).all()
    assert "time_out" in info


def test_simenv_rejects_unknown_obs_key() -> None:
    import strands_robots as sr
    from strands_robots.training.rl import SimEnv

    engine = sr.Robot("so100", mode="sim")
    with pytest.raises(KeyError):
        SimEnv(engine, actor_obs_keys=["NotAJoint"], reward_terms=[lambda _s: 0.0], action_dim=6)


def test_ppo_smoke_train_produces_loadable_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from strands_robots.training.rl import RLTrainSpec

    trainer = create_trainer("ppo")
    spec = RLTrainSpec(
        env_factory=_make_reach_env,
        output_dir=str(tmp_path),
        total_timesteps=20 * 3,
        rollout_steps=20,
        num_mini_batches=4,
        num_learning_epochs=2,
        seed=0,
    )
    assert trainer.validate(spec) == []

    result = trainer.train(spec)
    assert result.status == "success"
    assert result.checkpoint_dir is not None

    policy_pt = os.path.join(result.checkpoint_dir, "policy.pt")
    assert os.path.isfile(policy_pt)
    assert result.exported_model == policy_pt

    state = torch.load(policy_pt, weights_only=False)
    assert "actor_critic" in state and "actor_norm" in state

    with open(os.path.join(result.checkpoint_dir, "policy_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["num_actions"] == 6
    assert meta["actor_obs_keys"] == ["Elbow", "Elbow.vel"]
    assert meta["joint_names"]  # non-empty

    # latest_checkpoint discovers the saved artifact.
    assert trainer.latest_checkpoint(str(tmp_path)) == result.checkpoint_dir


def test_setup_reconciles_env_device_to_learner_device() -> None:
    """Regression: the learner device is authoritative over the env device.

    On a GPU host the learner (actor-critic, normalizers, rollout buffers)
    resolves to ``cuda`` while ``SimEnv`` keeps its default ``cpu`` device, so
    every forward pass over the env's observations raised "Expected all tensors
    to be on the same device, but found at least two devices, cuda:0 and cpu".
    ``setup`` must reconcile the env onto the learner device. The mismatch is
    reproduced on CPU-only CI with the storage-free ``meta`` device, so this
    guards the contract without requiring a GPU.
    """
    from strands_robots.training.rl import RLTrainSpec
    from strands_robots.training.rl.ppo import PpoTrainer

    def factory():  # type: ignore[no-untyped-def]
        env = _make_reach_env()
        # Mimic an env constructed on a different device than the learner.
        env.device = torch.device("meta")
        return env

    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=factory,
        output_dir="/tmp/ppo_device_reconcile",
        device="cpu",
        rollout_steps=4,
        num_mini_batches=2,
    )
    trainer.setup(spec)

    assert trainer.env.device == trainer.device
    assert trainer._obs["actor_obs"].device == trainer.device
    assert trainer._obs["critic_obs"].device == trainer.device
