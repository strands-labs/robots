"""Vectorized multi-environment wrapper over N independent :class:`SimEnv`.

The single-env :class:`~strands_robots.training.rl.env.SimEnv` emits ``(1, D)``
tensors by design ("only the env count changes"). ``VecSimEnv`` is the realisation
of that promise for the CPU/MuJoCo backend: it owns N independent ``SimEnv``
(each its own engine), steps them through ONE reused thread pool, and stacks the
results to ``(N, D)`` so the from-scratch PPO / FastSAC trainers can collect N
trajectories per step instead of one.

Autoreset semantics (matching gymnasium's vector API and what on-policy GAE
needs): when env ``i`` reports ``done`` on a step, the TERMINAL observation is
captured into ``infos[i]["terminal_obs"]`` BEFORE the env is reset, and the
returned ``obs[i]`` is the FRESH post-reset observation. The trainer bootstraps
the value of a truncation from ``terminal_obs`` (a time-out is value-bootstrapped,
a real terminal is not), so this capture is load-bearing for correctness, not a
convenience.

A future GPU-batched backend (Newton warp ``replicate`` + a batched
``get_observations``) can implement this SAME interface as a single engine driving
N worlds, so the trainer code written against ``VecSimEnv`` does not change when
the backend does.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strands_robots.training.rl.env import SimEnv


class VecSimEnv:
    """N independent :class:`SimEnv` presented as one ``(N, D)``-batched env.

    Args:
        env_factory: Zero-arg callable returning a fresh :class:`SimEnv`. Called
            ``num_envs`` times. (Same contract the trainers already use for the
            single-env path, so existing factories work unchanged.)
        num_envs: Number of parallel environments. Must be >= 1.
        device: Torch device for the stacked tensors. ``None`` inherits the
            first sub-env's device.
        max_workers: Thread-pool size for stepping sub-envs. ``None`` uses
            ``min(num_envs, 8)`` - MuJoCo releases the GIL during ``mj_step`` so
            threads give real parallelism on the physics call. One executor is
            created and reused for the env's lifetime (never per-step).

    Raises:
        ValueError: ``num_envs < 1`` or the sub-envs disagree on obs/action dims.
    """

    def __init__(
        self,
        env_factory: Callable[[], SimEnv],
        num_envs: int,
        *,
        device: torch.device | str | None = None,
        max_workers: int | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = int(num_envs)
        self.envs: list[SimEnv] = [env_factory() for _ in range(self.num_envs)]

        first = self.envs[0]
        self.num_actor_obs = first.num_actor_obs
        self.num_critic_obs = first.num_critic_obs
        self.num_actions = first.num_actions
        # Fail loudly if sub-envs are not homogeneous - a heterogeneous batch
        # would silently misalign the stacked tensors.
        for i, e in enumerate(self.envs[1:], start=1):
            if (e.num_actor_obs, e.num_critic_obs, e.num_actions) != (
                self.num_actor_obs,
                self.num_critic_obs,
                self.num_actions,
            ):
                raise ValueError(
                    f"VecSimEnv sub-env {i} dims "
                    f"({e.num_actor_obs}, {e.num_critic_obs}, {e.num_actions}) "
                    f"differ from env 0 ({self.num_actor_obs}, {self.num_critic_obs}, {self.num_actions})"
                )

        self.device = torch.device(device) if device is not None else first.device
        # Reconcile every sub-env onto the batch device so the per-env (1, D)
        # tensors stack without a device mismatch.
        for e in self.envs:
            e.device = self.device

        self._max_workers = max_workers if max_workers is not None else min(self.num_envs, 8)
        # ONE executor for the whole lifetime (AGENTS.md: never create executors
        # in the hot loop). Only spin up a pool when there is real concurrency.
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=self._max_workers) if self.num_envs > 1 else None
        )

    # --- metadata proxies (delegate to sub-env 0) ----------------------------
    # The trainer's save_checkpoint reads these off ``self.env``; they are
    # identical across homogeneous sub-envs, so env 0 is authoritative. This
    # lets a VecSimEnv stand in anywhere a SimEnv is expected for metadata.

    @property
    def actor_obs_keys(self) -> list[str]:
        """Actor observation keys, proxied from sub-env 0 (identical across homogeneous sub-envs)."""
        return self.envs[0].actor_obs_keys

    @property
    def critic_obs_keys(self) -> list[str]:
        """Critic observation keys, proxied from sub-env 0 (identical across homogeneous sub-envs)."""
        return self.envs[0].critic_obs_keys

    @property
    def engine(self) -> Any:
        """Simulation engine of sub-env 0 (authoritative for homogeneous sub-envs)."""
        return self.envs[0].engine

    @property
    def robot_name(self) -> str | None:
        """Robot name of sub-env 0, or ``None`` if unset (identical across homogeneous sub-envs)."""
        return self.envs[0].robot_name

    # --- helpers -------------------------------------------------------------

    def _stack(self, per_env: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Stack a list of per-env ``{actor_obs:(1,Da), critic_obs:(1,Dc)}`` to ``(N, D)``."""
        actor = torch.cat([d["actor_obs"] for d in per_env], dim=0)
        critic = torch.cat([d["critic_obs"] for d in per_env], dim=0)
        return {"actor_obs": actor, "critic_obs": critic}

    def _map(self, fn: Callable[[int], Any]) -> list[Any]:
        """Run ``fn(i)`` for every env, in parallel when a pool exists, order-preserving."""
        if self._executor is None:
            return [fn(i) for i in range(self.num_envs)]
        return list(self._executor.map(fn, range(self.num_envs)))

    # --- API -----------------------------------------------------------------

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset ALL sub-envs; return stacked ``{actor_obs:(N,Da), critic_obs:(N,Dc)}``."""
        per_env = self._map(lambda i: self.envs[i].reset())
        return self._stack(per_env)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Step all envs with ``actions`` shape ``(N, A)``; autoreset on done.

        Returns:
            ``(obs, rewards, dones, infos)`` where ``obs`` is the stacked
            ``(N, D)`` post-step (post-autoreset) observation, ``rewards`` and
            ``dones`` are ``(N,)``, and ``infos`` is a length-N list. On a done
            env, ``infos[i]["terminal_obs"]`` holds the pre-reset terminal
            ``{actor_obs, critic_obs}`` (each ``(1, D)``) for value bootstrapping,
            and ``infos[i]`` carries ``time_out`` / ``terminated``.
        """
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"actions batch {actions.shape[0]} != num_envs {self.num_envs}")

        def _one(i: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
            obs_i, reward_i, done_i, info_i = self.envs[i].step(actions[i : i + 1])
            if bool(done_i.item()):
                # Capture the TRUE terminal obs before autoreset clobbers it.
                info_i = dict(info_i)
                info_i["terminal_obs"] = obs_i
                obs_i = self.envs[i].reset()
            return obs_i, reward_i, done_i, info_i

        results = self._map(_one)
        obs = self._stack([r[0] for r in results])
        rewards = torch.cat([r[1] for r in results], dim=0)  # each (1,) -> (N,)
        dones = torch.cat([r[2] for r in results], dim=0)
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

    def close(self) -> None:
        """Shut down the thread pool. Engine teardown is the caller's concern."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


__all__ = ["VecSimEnv"]
