"""Train an SO-100 arm to reach a target joint angle with FastSAC, from scratch.

The off-policy peer of ``examples/training/train_ppo_reach.py``: no demonstration
dataset, just a reward function. FastSAC (Soft Actor-Critic) replays past
transitions from a buffer, so it reaches the target in far fewer environment
steps than on-policy PPO. The deterministic (mean) policy is rolled out at the
end and the joint trajectory is reported.

Run (headless on a GPU box, or plain CPU - FastSAC here trains fine on CPU)::

    MUJOCO_GL=egl python examples/training/train_fastsac_reach.py

Requires the ``[sim-mujoco]`` extra plus ``torch``. The same recipe generalizes
to locomotion / whole-body control by swapping the robot and the reward terms
(see ``docs/training/rl.md``).
"""

from __future__ import annotations

import tempfile

import strands_robots as sr
from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec, SimEnv

TARGET_ELBOW = 0.2  # radians, within the SO-100 Elbow reachable range


def elbow_reach_reward(engine: object) -> float:
    """Dense reach reward: negative distance of the Elbow joint from the target.

    A reward term is any ``Callable[[SimEngine], float]``; this one reads the
    current ``Elbow`` joint angle and rewards closing the gap to ``TARGET_ELBOW``.
    """
    elbow = engine.get_observation(skip_images=True)["Elbow"]  # type: ignore[attr-defined]
    return -abs(float(elbow) - TARGET_ELBOW)


def make_env() -> SimEnv:
    """Build the reach environment: observe the Elbow, reward closing on TARGET."""
    engine = sr.Robot("so100", mode="sim")
    return SimEnv(
        engine,
        actor_obs_keys=["Elbow", "Elbow.vel"],
        reward_terms=[elbow_reach_reward],
        action_dim=6,
        max_episode_steps=50,
    )


def main() -> None:
    trainer = create_trainer("fast_sac")
    output_dir = tempfile.mkdtemp(prefix="fastsac_reach_")
    spec = RLTrainSpec(
        env_factory=make_env,
        output_dir=output_dir,
        total_timesteps=50 * 80,
        rollout_steps=50,
        learning_starts=500,
        batch_size=256,
        gradient_steps=50,
        buffer_size=50_000,
        learning_rate=3e-4,
        tau=0.01,
        seed=0,
    )

    problems = trainer.validate(spec)
    if problems:
        raise SystemExit(f"spec invalid: {problems}")

    result = trainer.train(spec)
    print(f"status={result.status}")
    print(f"checkpoint={result.checkpoint_dir}")
    print(f"exported policy={result.exported_model}")
    print(f"final metrics={ {k: round(v, 4) if isinstance(v, float) else v for k, v in result.metrics.items()} }")


if __name__ == "__main__":
    main()
