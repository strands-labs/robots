"""Unit tests for ``BaseRLAlgo.evaluate()`` - the deterministic eval peer of train().

CPU-only: a tiny fake ``SimEngine`` drives a deterministic 1-DOF env so the test
needs neither mujoco nor model downloads. Pins the eval contract: deterministic
rollout, frozen normalizer, schema, success_rate, and the not-setup guard.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from strands_robots.training.rl import PpoTrainer, RLTrainSpec, SimEnv  # noqa: E402


class _FakeEngine:
    """Minimal SimEngine stand-in: one joint ``J`` integrated by the action.

    ``J`` starts at 0.0; each ``send_action([a])`` moves it by ``0.1 * a``.
    ``J.vel`` reports the last delta. Enough to exercise reset/step/observe and
    a success predicate without any physics backend.
    """

    def __init__(self) -> None:
        self._j = 0.0
        self._vel = 0.0

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
        self._vel = 0.0
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images: bool = False) -> dict:
        return {"J": self._j, "J.vel": self._vel}

    def send_action(self, action, robot_name=None, n_substeps: int = 1) -> dict:
        a = float(action[0]) if len(action) else 0.0
        self._vel = 0.1 * a
        self._j += self._vel
        return {"status": "success"}


def _make_env():  # type: ignore[no-untyped-def]
    eng = _FakeEngine()
    return SimEnv(
        eng,
        actor_obs_keys=["J", "J.vel"],
        reward_terms=[lambda e: -abs(float(e.get_observation(skip_images=True)["J"]) - 0.2)],
        action_dim=1,
        max_episode_steps=5,
    )


def _make_env_with_success():  # type: ignore[no-untyped-def]
    eng = _FakeEngine()
    return SimEnv(
        eng,
        actor_obs_keys=["J", "J.vel"],
        reward_terms=[lambda e: 1.0],
        action_dim=1,
        max_episode_steps=5,
        # Always-successful predicate: terminates on the first step.
        success_fn=lambda e: True,
    )


def test_evaluate_schema_and_determinism(tmp_path) -> None:  # type: ignore[no-untyped-def]
    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir=str(tmp_path),
        rollout_steps=4,
        num_mini_batches=2,
        hidden_dims=(8,),
        seed=0,
    )
    trainer.setup(spec)

    out = trainer.evaluate(num_episodes=3)
    # Schema.
    for k in (
        "num_episodes",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "mean_length",
        "success_rate",
        "returns",
    ):
        assert k in out, f"missing key {k}"
    assert out["num_episodes"] == 3
    assert len(out["returns"]) == 3
    assert out["mean_length"] == 5.0  # no success_fn -> always times out at max_episode_steps
    assert out["success_rate"] == 0.0

    # Determinism: same weights -> identical eval returns.
    out2 = trainer.evaluate(num_episodes=3)
    assert out["returns"] == out2["returns"]


def test_evaluate_success_rate_counts_terminals(tmp_path) -> None:  # type: ignore[no-untyped-def]
    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env_with_success,
        output_dir=str(tmp_path),
        rollout_steps=4,
        num_mini_batches=2,
        hidden_dims=(8,),
        seed=0,
    )
    trainer.setup(spec)
    out = trainer.evaluate(num_episodes=4)
    # success_fn always True -> every episode terminates on step 1.
    assert out["success_rate"] == 1.0
    assert out["mean_length"] == 1.0


def test_evaluate_requires_spec_when_not_setup() -> None:
    trainer = PpoTrainer()
    with pytest.raises(ValueError, match="setup"):
        trainer.evaluate(num_episodes=2)


def test_evaluate_rejects_bad_episode_count(tmp_path) -> None:  # type: ignore[no-untyped-def]
    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir=str(tmp_path),
        rollout_steps=4,
        num_mini_batches=2,
        hidden_dims=(8,),
    )
    trainer.setup(spec)
    with pytest.raises(ValueError, match="num_episodes"):
        trainer.evaluate(num_episodes=0)


def test_evaluate_loads_checkpoint_fresh_instance(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Train briefly, checkpoint, then evaluate from a FRESH trainer + checkpoint.
    t1 = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir=str(tmp_path),
        total_timesteps=4 * 3,
        rollout_steps=4,
        num_mini_batches=2,
        num_learning_epochs=1,
        hidden_dims=(8,),
        seed=0,
    )
    result = t1.train(spec)
    assert result.status == "success"

    t2 = PpoTrainer()
    out = t2.evaluate(spec=spec, num_episodes=2)  # discovers latest checkpoint under output_dir
    assert out["num_episodes"] == 2
    assert len(out["returns"]) == 2


def test_evaluate_works_for_sac_too(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """evaluate() lives on BaseRLAlgo, so FastSAC inherits it unchanged.

    SAC's deterministic action is ``tanh(mean)`` via the same ``act_inference``
    contract PPO uses, so no per-subclass override is needed.
    """
    from strands_robots.training.rl import FastSacTrainer

    trainer = FastSacTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env_with_success,
        output_dir=str(tmp_path),
        rollout_steps=4,
        batch_size=16,
        learning_starts=16,
        hidden_dims=(8,),
        seed=0,
    )
    trainer.setup(spec)
    out = trainer.evaluate(num_episodes=3)
    assert out["num_episodes"] == 3
    assert out["success_rate"] == 1.0  # success_fn always True
    assert out["mean_length"] == 1.0


def test_evaluate_restores_train_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """evaluate() must be side-effect-free w.r.t. train/eval mode.

    Regression guard: a train -> evaluate -> train continuation needs the
    actor-critic (and obs normalizer) returned to training mode, else
    BatchNorm/Dropout running stats silently freeze on resumed training.
    """
    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir=str(tmp_path),
        rollout_steps=4,
        num_mini_batches=2,
        hidden_dims=(8,),
        seed=0,
    )
    trainer.setup(spec)

    # Force a known training mode, then evaluate, then assert it is restored.
    trainer.actor_critic.train()
    assert trainer.actor_critic.training is True
    actor_norm = getattr(trainer, "actor_norm", None)
    if actor_norm is not None:
        actor_norm.train()

    trainer.evaluate(num_episodes=2)

    assert trainer.actor_critic.training is True, "actor_critic left in eval() after evaluate()"
    if actor_norm is not None:
        assert actor_norm.training is True, "actor_norm left in eval() after evaluate()"

    # And the inverse: an eval-mode caller stays in eval mode.
    trainer.actor_critic.eval()
    trainer.evaluate(num_episodes=1)
    assert trainer.actor_critic.training is False, "evaluate() wrongly flipped a model that was in eval()"
