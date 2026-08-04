"""Gym-style single-environment adapter over a ``SimEngine`` for RL training.

``SimEnv`` turns the simulation's observe / act / reward primitives into the
``reset -> step`` contract on-policy RL needs, reusing the existing reward-term
DSL (:data:`strands_robots.simulation.predicates.RewardTerm`) for the reward
signal. It produces batched tensors with a leading ``num_envs == 1`` axis so the
trainer code is shaped the same way it would be for a future vectorized backend
(IsaacGym / MJWarp), where only the env count changes.

The observation contract is the holosoma ``actor_obs_keys`` / ``critic_obs_keys``
split: the actor sees ``actor_obs_keys`` (deployable on hardware), the critic may
additionally see privileged simulation-only keys via ``critic_obs_keys``
(asymmetric actor-critic). Each key is a scalar entry of
``SimEngine.get_observation`` (e.g. a joint position ``"Elbow"`` or velocity
``"Elbow.vel"``); the vector is the keys concatenated in the given order.

The action contract is the other half: ``step`` sends a numeric vector, which
``SimEngine.send_action`` binds positionally to
``robot_action_keys(robot_name)``. Those are the robot's *actuators*, which are
not always its joints - a tendon-driven gripper is one actuator over two finger
joints, and the Newton backend's floating base is a joint with no commandable
scalar - so the action head is sized from the action keys and a checkpoint
records them as ``action_keys``. Pass ``action_dim`` to override the width.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strands_robots.simulation.base import SimEngine
    from strands_robots.simulation.predicates import RewardTerm


class SimEnv:
    """Single-environment RL wrapper around a :class:`SimEngine`.

    Args:
        engine: A live ``SimEngine`` with at least one robot added.
        actor_obs_keys: Ordered observation keys the policy (actor) sees. Each
            must be a scalar key returned by ``engine.get_observation``.
        reward_terms: Reward-term callables (``RewardTerm`` = ``SimEngine ->
            float``); the step reward is their sum. Build them with the
            predicate DSL (e.g. ``_joint_progress``, ``_distance_neg``).
        action_dim: Size of the action vector sent to ``engine.send_action``.
            Defaults to ``len(engine.robot_action_keys(robot_name))`` - the
            actuator count ``send_action`` binds a vector against, which is not
            always the robot's joint count.
        robot_name: Robot to observe / drive. Defaults to the engine's first
            registered robot.
        critic_obs_keys: Optional privileged keys appended to the critic
            observation (asymmetric actor-critic). Defaults to ``actor_obs_keys``
            (symmetric).
        max_episode_steps: Steps before the episode is truncated (time-out).
        action_scale: Scalar multiplier applied to actions before sending.
        n_substeps: Physics control substeps per env step. The action is a
            position target; the PD controller needs several substeps to track
            it, so a single substep barely moves the arm. Default 5.
        success_fn: Optional predicate; when it returns ``True`` the episode
            terminates (a genuine terminal, not a time-out).
        reset_fn: Optional callable run on ``engine`` at each reset (e.g. to
            randomize the target). Defaults to ``engine.reset()``.
        device: Torch device for the returned tensors.
        skip_images: Pass ``skip_images=True`` to ``get_observation`` (scalar
            state only) - the default, since the obs keys are scalar.
    """

    def __init__(
        self,
        engine: SimEngine,
        actor_obs_keys: Sequence[str],
        reward_terms: Sequence[RewardTerm],
        *,
        action_dim: int | None = None,
        robot_name: str | None = None,
        critic_obs_keys: Sequence[str] | None = None,
        max_episode_steps: int = 200,
        action_scale: float = 1.0,
        n_substeps: int = 5,
        success_fn: Callable[[SimEngine], bool] | None = None,
        reset_fn: Callable[[SimEngine], None] | None = None,
        device: torch.device | str = "cpu",
        skip_images: bool = True,
    ) -> None:
        if not actor_obs_keys:
            raise ValueError("actor_obs_keys must be a non-empty ordered sequence of observation keys")
        if not reward_terms:
            raise ValueError("reward_terms must be a non-empty sequence of RewardTerm callables")
        self.engine = engine
        self.actor_obs_keys = list(actor_obs_keys)
        self.critic_obs_keys = list(critic_obs_keys) if critic_obs_keys is not None else list(actor_obs_keys)
        self.reward_terms = list(reward_terms)
        self.max_episode_steps = int(max_episode_steps)
        self.action_scale = float(action_scale)
        if int(n_substeps) < 1:
            raise ValueError(f"n_substeps must be >= 1, got {n_substeps}")
        self.n_substeps = int(n_substeps)
        self.success_fn = success_fn
        self.reset_fn = reset_fn
        self.device = torch.device(device)
        self.skip_images = skip_images

        names = engine.list_robots()
        self.robot_name = robot_name or (names[0] if names else None)
        if action_dim is not None:
            self.num_actions = int(action_dim)
        elif self.robot_name is not None:
            # ``robot_action_keys``, not ``robot_joint_names``: ``step`` sends a
            # numeric vector, and ``send_action`` binds a vector positionally to
            # the action keys. The two lists coincide only when a robot's
            # actuator set matches its joint set; a tendon gripper (one actuator
            # driving two finger joints) or a Newton floating base (a 6-DoF free
            # joint with no scalar target) makes the joint list wider, and a
            # vector of that width is refused - so every step wrote no target
            # while the reward was still collected.
            self.num_actions = len(engine.robot_action_keys(self.robot_name))
        else:
            raise ValueError("action_dim must be given when the engine has no registered robot")

        self._step_count = 0
        # Validate obs keys up front so a typo fails loudly here, not mid-rollout.
        obs = self._raw_obs()
        missing = [k for k in set(self.actor_obs_keys + self.critic_obs_keys) if k not in obs]
        if missing:
            raise KeyError(f"observation keys not produced by the engine: {sorted(missing)}; available: {sorted(obs)}")
        self.num_actor_obs = len(self.actor_obs_keys)
        self.num_critic_obs = len(self.critic_obs_keys)

    def _raw_obs(self) -> dict[str, Any]:
        return self.engine.get_observation(robot_name=self.robot_name, skip_images=self.skip_images)

    def _vector(self, obs: dict[str, Any], keys: list[str]) -> torch.Tensor:
        vals = [float(obs[k]) for k in keys]
        return torch.tensor(vals, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _obs_dict(self) -> dict[str, torch.Tensor]:
        obs = self._raw_obs()
        return {
            "actor_obs": self._vector(obs, self.actor_obs_keys),
            "critic_obs": self._vector(obs, self.critic_obs_keys),
        }

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset the episode and return the initial ``{actor_obs, critic_obs}``.

        Stateful reward terms (any term exposing a zero-arg ``reset()``, e.g. a
        ``staged_reward`` phase machine) are reset here so per-episode state
        (current phase, awarded bonuses) does not leak across episodes. Plain
        stateless function terms have no ``reset`` and are left untouched.
        """
        if self.reset_fn is not None:
            self.reset_fn(self.engine)
        else:
            self.engine.reset()
        for term in self.reward_terms:
            term_reset = getattr(term, "reset", None)
            if callable(term_reset):
                term_reset()
        self._step_count = 0
        return self._obs_dict()

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Apply ``action`` for one control step.

        Args:
            action: Action tensor, shape ``(1, num_actions)`` or
                ``(num_actions,)``.

        Returns:
            ``(obs_dict, reward, done, info)`` where ``reward`` and ``done`` have
            shape ``(1,)`` and ``info`` carries ``time_out`` (truncation that
            should be value-bootstrapped, not treated as a terminal state).
        """
        act = action.detach().reshape(-1).to("cpu").numpy().astype(np.float64) * self.action_scale
        self.engine.send_action(act.tolist(), robot_name=self.robot_name, n_substeps=self.n_substeps)
        self._step_count += 1

        reward = sum(term(self.engine) for term in self.reward_terms)
        terminated = bool(self.success_fn(self.engine)) if self.success_fn is not None else False
        time_out = self._step_count >= self.max_episode_steps
        done = terminated or time_out

        obs = self._obs_dict()
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([done], dtype=torch.float32, device=self.device)
        info: dict[str, Any] = {"time_out": time_out and not terminated, "terminated": terminated}
        # No auto-reset: the caller resets on ``done`` AFTER reading the terminal
        # observation, so on-policy GAE can bootstrap the truncated value from the
        # true terminal obs (a time-out is value-bootstrapped, a real terminal is
        # not). See ``PpoTrainer.collect_rollout``.
        return obs, reward_t, done_t, info

    def close(self) -> None:
        """Release env resources. No-op: the engine/Robot lifecycle is owned by
        the caller (mirrors ``GymSimEnv.close`` / ``VecSimEnv.close`` so SimEnv
        and VecSimEnv present one interface to the trainers)."""
        return None
