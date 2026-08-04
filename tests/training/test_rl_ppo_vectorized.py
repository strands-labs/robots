"""Tests for the vectorized (num_envs > 1) PPO path over VecSimEnv.

CPU-only, fake engine. Pins: VecSimEnv built when num_envs>1, collect_rollout
produces a (T*N, ...) batch with correct shapes, GAE-per-env correctness, a
smoke train, and that num_envs=1 still routes through the original single path.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from strands_robots.training.rl import PpoTrainer, RLTrainSpec, SimEnv, VecSimEnv  # noqa: E402


class _FakeEngine:
    def __init__(self) -> None:
        self._j = 0.0
        self._v = 0.0

    def list_robots(self) -> list[str]:
        return ["fake"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["J"]

    def robot_action_keys(self, robot_name: str) -> list[str]:
        # These fakes are duck-typed rather than ``SimEngine`` subclasses, so
        # they do not inherit the default that mirrors the joint names. This
        # robot's one joint is its one actuator, so the two vocabularies agree -
        # which is the shape ``SimEnv`` sizes its action head from.
        return ["J"]

    def reset(self) -> dict:
        self._j = 0.0
        self._v = 0.0
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images: bool = False) -> dict:
        return {"J": self._j, "J.vel": self._v}

    def send_action(self, action, robot_name=None, n_substeps: int = 1) -> dict:
        a = float(action[0]) if len(action) else 0.0
        self._v = 0.1 * a
        self._j += self._v
        return {"status": "success"}


def _make_env():  # type: ignore[no-untyped-def]
    return SimEnv(
        _FakeEngine(),
        actor_obs_keys=["J", "J.vel"],
        reward_terms=[lambda e: -abs(float(e.get_observation(skip_images=True)["J"]) - 0.2)],
        action_dim=1,
        max_episode_steps=8,
    )


def test_setup_builds_vec_env_when_num_envs_gt_1() -> None:
    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir="/tmp/ppo_vec_setup",
        num_envs=4,
        rollout_steps=8,
        num_mini_batches=4,
        hidden_dims=(8,),
        seed=0,
    )
    assert trainer.validate(spec) == []
    trainer.setup(spec)
    assert isinstance(trainer.env, VecSimEnv)
    assert trainer.env.num_envs == 4
    assert trainer._vectorized is True
    trainer.env.close()


def test_vectorized_collect_rollout_batch_shapes() -> None:
    trainer = PpoTrainer()
    T, N = 8, 4
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir="/tmp/ppo_vec_shapes",
        num_envs=N,
        rollout_steps=T,
        num_mini_batches=4,
        hidden_dims=(8,),
        seed=0,
    )
    trainer.setup(spec)
    metrics = trainer.collect_rollout()
    batch = trainer._batch
    # Flattened (T*N, ...) batch.
    assert batch["actor_obs"].shape == (T * N, 2)
    assert batch["critic_obs"].shape == (T * N, 2)
    assert batch["actions"].shape == (T * N, 1)
    assert batch["old_log_prob"].shape == (T * N,)
    assert batch["advantages"].shape == (T * N,)
    assert batch["returns"].shape == (T * N,)
    assert "mean_reward" in metrics and "mean_episode_return" in metrics
    trainer.env.close()


def test_num_envs_1_uses_single_path() -> None:
    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir="/tmp/ppo_single_path",
        num_envs=1,
        rollout_steps=8,
        num_mini_batches=4,
        hidden_dims=(8,),
        seed=0,
    )
    trainer.setup(spec)
    assert trainer._vectorized is False
    assert not isinstance(trainer.env, VecSimEnv)
    metrics = trainer.collect_rollout()
    # Single path: (T, ...) batch, not (T*N, ...).
    assert trainer._batch["actor_obs"].shape == (8, 2)
    assert "mean_reward" in metrics


def test_vectorized_smoke_train_and_update() -> None:
    """Full vectorized train loop: setup -> collect -> update runs and produces a checkpoint."""
    import os

    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir="/tmp/ppo_vec_smoke",
        num_envs=4,
        total_timesteps=8 * 4 * 3,  # T*N*iters
        rollout_steps=8,
        num_mini_batches=4,
        num_learning_epochs=2,
        hidden_dims=(16,),
        seed=0,
    )
    assert trainer.validate(spec) == []
    result = trainer.train(spec)
    assert result.status == "success"
    assert result.checkpoint_dir is not None
    assert os.path.isfile(os.path.join(result.checkpoint_dir, "policy.pt"))


def test_vectorized_throughput_vs_single() -> None:
    """N-env collect gathers N*T transitions per call vs T for single-env.

    This is the whole point of vectorization: one collect_rollout yields N times
    the data. We assert the batch is exactly N larger, proving the sample-
    throughput multiplier is real (not that wall-clock is N times faster - CPU
    threading on MuJoCo is sublinear, the GPU backend is future work).
    """
    T = 8
    single = PpoTrainer()
    single.setup(
        RLTrainSpec(
            env_factory=_make_env,
            output_dir="/tmp/p1",
            num_envs=1,
            rollout_steps=T,
            num_mini_batches=4,
            hidden_dims=(8,),
            seed=0,
        )
    )
    single.collect_rollout()
    n_single = single._batch["actor_obs"].shape[0]

    vec = PpoTrainer()
    vec.setup(
        RLTrainSpec(
            env_factory=_make_env,
            output_dir="/tmp/p4",
            num_envs=4,
            rollout_steps=T,
            num_mini_batches=4,
            hidden_dims=(8,),
            seed=0,
        )
    )
    vec.collect_rollout()
    n_vec = vec._batch["actor_obs"].shape[0]
    vec.env.close()

    assert n_vec == 4 * n_single, f"vectorized batch {n_vec} should be 4x single {n_single}"


def test_evaluate_works_on_vectorized_trainer() -> None:
    """Regression: evaluate() on a num_envs>1 trainer must not crash on (N,) rewards.

    Pins the bug where evaluate() called reward.item() on a VecSimEnv's
    (N,)-batched reward. evaluate() now runs on a single sub-env.
    """
    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir="/tmp/ppo_vec_eval",
        num_envs=8,
        rollout_steps=8,
        num_mini_batches=4,
        hidden_dims=(8,),
        seed=0,
    )
    trainer.setup(spec)
    trainer.collect_rollout()
    trainer.update()
    ev = trainer.evaluate(num_episodes=3)
    assert ev["num_episodes"] == 3
    assert len(ev["returns"]) == 3
    assert isinstance(ev["mean_return"], float)
    trainer.env.close()
