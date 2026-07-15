"""Gymnasium adapter exposing a :class:`SimEnv` as a standard ``gymnasium.Env``.

``SimEnv`` is our bespoke RL adapter: it emits ``{actor_obs, critic_obs}`` dicts
of ``(1, D)`` torch tensors and a ``(obs, reward, done, info)`` 4-tuple shaped for
the from-scratch PPO / FastSAC trainers. That is deliberately NOT the gymnasium
contract, so external RL libraries (Stable-Baselines3, RLlib, CleanRL) cannot
consume it directly.

``GymSimEnv`` is the thin, lossless bridge: it wraps ONE ``SimEnv`` and presents
the gymnasium 5-tuple API (``reset -> (obs, info)``;
``step -> (obs, reward, terminated, truncated, info)``) over plain NumPy. Its one
job beyond shape-shuffling is to split our env's done signal CORRECTLY:

    - ``terminated``: a genuine task terminal (``info["terminated"]`` from
      ``SimEnv.success_fn``) - the episode ended because the goal/failure was
      reached, so the value backup MUST NOT bootstrap past it.
    - ``truncated``: a time-out (``info["time_out"]``) - the episode was cut at
      ``max_episode_steps``; a correct learner SHOULD bootstrap the truncated
      value.

Conflating these is the single most common silent RL bug (the SB3 ``TimeLimit``
footgun); doing the split here, at the source, means any downstream library gets
it right for free.

The actor observation is the gym observation (what a deployable policy sees); the
privileged critic observation is passed through ``info["critic_obs"]`` so an
asymmetric-actor-critic consumer can still reach it without polluting the
policy's observation space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.utils import require_optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strands_robots.training.rl.env import SimEnv


def _make_gym_env_class() -> type:
    """Build the ``GymSimEnv`` class (gymnasium required at call time).

    gymnasium is an optional dep (the ``[sim]`` extra pulls it in), so the class
    is defined inside this factory to keep ``import strands_robots.training.rl``
    free of a hard gymnasium dependency. :func:`GymSimEnv` (below) calls this.
    """
    import gymnasium as gym
    from gymnasium import spaces

    class GymSimEnv(gym.Env):
        """A single ``SimEnv`` presented as a ``gymnasium.Env`` (NumPy, 5-tuple).

        Args:
            sim_env: The wrapped :class:`SimEnv`.
            action_low / action_high: Symmetric action-space bounds. ``SimEnv``
                multiplies the raw action by ``action_scale`` and forwards it, so
                the natural normalized action range is ``[-1, 1]``; override for
                an unnormalized joint-target space.

        Notes:
            - Observation space is the ACTOR observation (``Box``), the
              deployable view. The critic observation is returned via
              ``info["critic_obs"]`` every step/reset.
            - Observation bounds are ``[-inf, inf]`` (joint positions / velocities
              are unbounded a priori); SB3 handles this with a running normalizer.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            sim_env: SimEnv,
            *,
            action_low: float = -1.0,
            action_high: float = 1.0,
        ) -> None:
            super().__init__()
            self._env = sim_env
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(sim_env.num_actor_obs,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=float(action_low),
                high=float(action_high),
                shape=(sim_env.num_actions,),
                dtype=np.float32,
            )

        # --- helpers -------------------------------------------------------

        @staticmethod
        def _to_numpy_obs(obs_dict: dict[str, Any]) -> np.ndarray:
            """Flatten the ``(1, D)`` actor-obs tensor to a ``(D,)`` float32 array."""
            actor = obs_dict["actor_obs"]
            arr = actor.detach().reshape(-1).to("cpu").numpy().astype(np.float32)
            return arr

        @staticmethod
        def _critic_info(obs_dict: dict[str, Any]) -> dict[str, Any]:
            critic = obs_dict["critic_obs"]
            return {"critic_obs": critic.detach().reshape(-1).to("cpu").numpy().astype(np.float32)}

        # --- gymnasium API -------------------------------------------------

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            """Reset the wrapped env; return ``(actor_obs_array, info)``."""
            super().reset(seed=seed)
            obs_dict = self._env.reset()
            return self._to_numpy_obs(obs_dict), self._critic_info(obs_dict)

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            """Advance one control step; return the gymnasium 5-tuple.

            ``terminated`` is the genuine task terminal; ``truncated`` is the
            time-out. The two are split from ``SimEnv``'s ``info`` so a
            downstream learner bootstraps truncations but not terminals.
            """
            import torch

            act = torch.as_tensor(
                np.asarray(action, dtype=np.float32).reshape(-1),
                dtype=torch.float32,
                device=self._env.device,
            ).unsqueeze(0)
            obs_dict, reward_t, _done_t, info = self._env.step(act)

            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("time_out", False))
            reward = float(reward_t.item())

            out_info: dict[str, Any] = self._critic_info(obs_dict)
            # Carry through any extra keys SimEnv emitted (without clobbering critic_obs).
            for k, v in info.items():
                out_info.setdefault(k, v)
            return self._to_numpy_obs(obs_dict), reward, terminated, truncated, out_info

        def render(self) -> None:  # pragma: no cover - no render mode advertised
            """No-op: this wrapper advertises no render mode (always returns ``None``)."""
            return None

        def close(self) -> None:
            """No-op: SimEnv teardown is owned by the caller / Robot, not this wrapper."""
            # SimEnv does not own engine teardown (the caller / Robot owns it),
            # so there is nothing to release here.
            return None

    return GymSimEnv


def GymSimEnv(sim_env: SimEnv, **kwargs: Any) -> Any:  # noqa: N802 - factory mimics a class
    """Wrap a :class:`SimEnv` as a ``gymnasium.Env``.

    Thin entry point that builds the gymnasium-backed class on first use (so the
    module imports without gymnasium installed) and returns an instance.

    Args:
        sim_env: The :class:`SimEnv` to wrap.
        **kwargs: Forwarded to the underlying ``GymSimEnv`` (``action_low`` /
            ``action_high``).

    Returns:
        A ``gymnasium.Env`` instance presenting ``sim_env`` over the standard
        NumPy 5-tuple API.
    """
    require_optional("gymnasium", purpose="GymSimEnv (strands_robots.training.rl.gym_env)")
    cls = _make_gym_env_class()
    return cls(sim_env, **kwargs)


__all__ = ["GymSimEnv"]
