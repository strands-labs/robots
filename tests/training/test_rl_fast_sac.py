"""Deterministic unit tests for the off-policy RL trainer (FastSAC).

These run in CI: they need ``torch`` (and ``mujoco`` for the env-contract /
smoke-train cases) but no model downloads and no convergence assumptions. The
end-to-end convergence proof lives in ``tests_integ/training/test_fastsac_reach.py``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from strands_robots.training import create_trainer, list_trainers
from strands_robots.training.base import TrainSpec

torch = pytest.importorskip("torch")


def test_fast_sac_registered_and_created() -> None:
    assert "fast_sac" in list_trainers()
    trainer = create_trainer("fast_sac")
    assert trainer.provider_name == "fast_sac"


def test_replay_buffer_add_and_sample_shapes() -> None:
    from strands_robots.training.rl import SimpleReplayBuffer

    buf = SimpleReplayBuffer(capacity=10, num_actor_obs=2, num_critic_obs=3, num_actions=4, device="cpu")
    assert len(buf) == 0
    for i in range(4):
        buf.add(
            actor_obs=torch.full((1, 2), float(i)),
            critic_obs=torch.full((1, 3), float(i)),
            action=torch.full((1, 4), float(i)),
            reward=torch.tensor([float(i)]),
            next_actor_obs=torch.full((1, 2), float(i) + 0.5),
            next_critic_obs=torch.full((1, 3), float(i) + 0.5),
            done=torch.tensor([0.0]),
        )
    assert len(buf) == 4

    batch = buf.sample(8)  # with replacement -> can exceed stored count
    assert batch["actor_obs"].shape == (8, 2)
    assert batch["critic_obs"].shape == (8, 3)
    assert batch["actions"].shape == (8, 4)
    assert batch["rewards"].shape == (8, 1)
    assert batch["next_actor_obs"].shape == (8, 2)
    assert batch["dones"].shape == (8, 1)
    # rewards are drawn from the four stored integer values.
    assert set(batch["rewards"].reshape(-1).tolist()) <= {0.0, 1.0, 2.0, 3.0}


def test_replay_buffer_is_circular() -> None:
    from strands_robots.training.rl import SimpleReplayBuffer

    buf = SimpleReplayBuffer(capacity=3, num_actor_obs=1, num_critic_obs=1, num_actions=1, device="cpu")
    for i in range(5):  # overflow capacity -> oldest two overwritten
        buf.add(
            actor_obs=torch.tensor([[float(i)]]),
            critic_obs=torch.tensor([[float(i)]]),
            action=torch.tensor([[0.0]]),
            reward=torch.tensor([float(i)]),
            next_actor_obs=torch.tensor([[float(i)]]),
            next_critic_obs=torch.tensor([[float(i)]]),
            done=torch.tensor([0.0]),
        )
    assert len(buf) == 3
    # only the three most recent rewards (2, 3, 4) survive.
    assert set(buf._rewards.reshape(-1).tolist()) == {2.0, 3.0, 4.0}


def test_replay_buffer_rejects_bad_capacity_and_empty_sample() -> None:
    from strands_robots.training.rl import SimpleReplayBuffer

    with pytest.raises(ValueError):
        SimpleReplayBuffer(capacity=0, num_actor_obs=1, num_critic_obs=1, num_actions=1)
    buf = SimpleReplayBuffer(capacity=2, num_actor_obs=1, num_critic_obs=1, num_actions=1)
    with pytest.raises(ValueError):
        buf.sample(4)


def test_validate_rejects_bad_specs() -> None:
    from strands_robots.training.rl import RLTrainSpec

    trainer = create_trainer("fast_sac")

    # A plain (non-RL) spec is rejected with a clear message.
    problems = trainer.validate(TrainSpec(output_dir="/tmp/x"))
    assert any("RLTrainSpec" in p for p in problems)

    # Missing env_factory.
    problems = trainer.validate(RLTrainSpec(output_dir="/tmp/x"))
    assert any("env_factory" in p for p in problems)

    # Vectorized envs are not supported by the MuJoCo backend yet.
    problems = trainer.validate(RLTrainSpec(output_dir="/tmp/x", num_envs=4))
    assert any("single-env" in p for p in problems)

    # learning_starts must be >= batch_size so the first update has a full batch.
    problems = trainer.validate(RLTrainSpec(output_dir="/tmp/x", batch_size=256, learning_starts=10))
    assert any("learning_starts" in p for p in problems)

    # tau out of range.
    problems = trainer.validate(RLTrainSpec(output_dir="/tmp/x", tau=2.0))
    assert any("tau" in p for p in problems)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"output_dir": ""}, "output_dir is required"),
        ({"total_timesteps": 0}, "total_timesteps must be > 0"),
        ({"rollout_steps": 0}, "rollout_steps must be > 0"),
        ({"buffer_size": 0}, "buffer_size must be > 0"),
        ({"batch_size": 0}, "batch_size must be > 0"),
        ({"gradient_steps": 0}, "gradient_steps must be > 0"),
    ],
)
def test_validate_flags_each_out_of_range_field(overrides: dict[str, Any], expected: str) -> None:
    """Each numeric guard in the FastSAC preflight names the offending field.

    ``validate`` is a pure, read-only preflight the ``train`` entry point runs
    before touching torch or the env, so a misconfigured spec fails with an
    actionable message instead of a deep stack trace. One otherwise-valid spec
    per case isolates a single bad field; a non-None ``env_factory`` keeps the
    env-factory guard quiet so only the field under test is exercised.
    """
    from strands_robots.training.rl import RLTrainSpec

    trainer = create_trainer("fast_sac")
    base: dict[str, Any] = {"output_dir": "/tmp/x", "env_factory": lambda: None}
    base.update(overrides)
    problems = trainer.validate(RLTrainSpec(**base))
    assert any(expected in p for p in problems), problems


def test_train_rejects_non_rl_spec() -> None:
    trainer = create_trainer("fast_sac")
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


def test_sac_actor_log_prob_finite_under_saturation() -> None:
    """tanh-squash log-prob correction must stay finite even at the bounds."""
    from strands_robots.training.rl import RLTrainSpec
    from strands_robots.training.rl.fast_sac import _build_actor_critic

    spec = RLTrainSpec(hidden_dims=(16,))
    ac = _build_actor_critic(num_actor_obs=3, num_critic_obs=3, num_actions=2, spec=spec)
    obs = torch.full((8, 3), 50.0)  # drive the pre-squash mean far out -> tanh saturates
    action, log_prob = ac.sample(obs)
    assert action.shape == (8, 2)
    assert log_prob.shape == (8, 1)
    assert torch.isfinite(action).all()
    assert torch.isfinite(log_prob).all()
    assert (action.abs() <= 1.0).all()


def test_fast_sac_smoke_train_produces_loadable_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from strands_robots.training.rl import RLTrainSpec

    trainer = create_trainer("fast_sac")
    spec = RLTrainSpec(
        env_factory=_make_reach_env,
        output_dir=str(tmp_path),
        total_timesteps=40,
        rollout_steps=10,
        learning_starts=16,
        batch_size=16,
        gradient_steps=2,
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
    assert "actor_critic" in state and "actor_norm" in state and "log_alpha" in state

    with open(os.path.join(result.checkpoint_dir, "policy_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["provider"] == "fast_sac"
    assert meta["num_actions"] == 6
    assert meta["actor_obs_keys"] == ["Elbow", "Elbow.vel"]
    assert meta["joint_names"]  # non-empty

    assert trainer.latest_checkpoint(str(tmp_path)) == result.checkpoint_dir


def test_setup_reconciles_env_device_to_learner_device() -> None:
    """The learner device is authoritative over the env device (GPU-host guard).

    Mirrors the PPO regression: on a GPU host the learner resolves to ``cuda``
    while ``SimEnv`` keeps its default ``cpu`` device, so observation tensors
    would mix devices. ``setup`` must reconcile the env onto the learner device.
    Reproduced on CPU with the storage-free ``meta`` device.
    """
    from strands_robots.training.rl import RLTrainSpec
    from strands_robots.training.rl.fast_sac import FastSacTrainer

    def factory():  # type: ignore[no-untyped-def]
        env = _make_reach_env()
        env.device = torch.device("meta")
        return env

    trainer = FastSacTrainer()
    spec = RLTrainSpec(
        env_factory=factory,
        output_dir="/tmp/sac_device_reconcile",
        device="cpu",
        rollout_steps=4,
        batch_size=16,
        learning_starts=16,
    )
    trainer.setup(spec)

    assert trainer.env.device == trainer.device
    assert trainer._obs["actor_obs"].device == trainer.device
    assert trainer.buffer.device == trainer.device


def test_setup_without_autotune_uses_fixed_alpha() -> None:
    """``autotune_alpha=False`` pins alpha as a constant, not a learnable temp.

    With autotuning off the learner must not build a temperature optimizer and
    ``log_alpha`` must be a plain (non-grad) tensor, so ``alpha`` stays at
    ``init_alpha`` for the whole run rather than drifting toward the entropy
    target. This is the SAC-without-auto-entropy contract users opt into when
    they want a hand-tuned temperature.
    """
    from strands_robots.training.rl import RLTrainSpec
    from strands_robots.training.rl.fast_sac import FastSacTrainer

    trainer = FastSacTrainer()
    spec = RLTrainSpec(
        env_factory=_make_reach_env,
        output_dir="/tmp/sac_fixed_alpha",
        device="cpu",
        autotune_alpha=False,
        init_alpha=0.5,
        rollout_steps=4,
        batch_size=16,
        learning_starts=16,
    )
    trainer.setup(spec)

    assert trainer.autotune_alpha is False
    assert trainer.log_alpha.requires_grad is False
    assert not hasattr(trainer, "alpha_optimizer")
    assert trainer.alpha.item() == pytest.approx(0.5)


def test_update_is_noop_until_buffer_reaches_batch_size() -> None:
    """``update`` returns zero-loss metrics while the buffer is below a batch.

    The learner cannot form a full minibatch until the replay buffer holds at
    least ``batch_size`` transitions, so a freshly-set-up trainer (empty buffer)
    must short-circuit to zero losses rather than sampling an undersized batch.
    """
    from strands_robots.training.rl import RLTrainSpec
    from strands_robots.training.rl.fast_sac import FastSacTrainer

    trainer = FastSacTrainer()
    spec = RLTrainSpec(
        env_factory=_make_reach_env,
        output_dir="/tmp/sac_warmup_noop",
        device="cpu",
        rollout_steps=4,
        batch_size=16,
        learning_starts=16,
    )
    trainer.setup(spec)
    assert trainer.buffer.size < spec.batch_size  # nothing collected yet

    metrics = trainer.update()
    assert metrics["critic_loss"] == 0.0
    assert metrics["actor_loss"] == 0.0
    assert metrics["latest_loss"] == 0.0
    assert metrics["alpha"] == pytest.approx(trainer.alpha.item())
