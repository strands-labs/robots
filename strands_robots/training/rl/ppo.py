"""PPO - on-policy from-scratch RL trainer for the ``SimEngine`` env interface.

Proximal Policy Optimization with Generalized Advantage Estimation, a clipped
policy surrogate, a clipped value loss, and running observation normalization.
It trains a Gaussian-MLP actor-critic on a :class:`~strands_robots.training.rl.env.SimEnv`
from a reward function (the reward-term DSL), so a locomotion / reach / WBC policy
can be learned in MuJoCo with no demonstration dataset.

The algorithm is the standard RSL-RL / Holosoma PPO (BSD-3-Clause,
https://github.com/amazon-far/holosoma) adapted to the single-environment MuJoCo
backend; the clipped-surrogate, clipped-value, GAE, and advantage-normalization
math are ported directly. Selected via ``create_trainer("ppo")``.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from strands_robots.training.base import TrainSpec
from strands_robots.training.rl.base_algo import BaseRLAlgo, RLTrainSpec
from strands_robots.utils import require_optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> Any:
    """Build a Tanh-activated MLP ``in_dim -> *hidden -> out_dim``."""
    import torch.nn as nn

    layers: list[Any] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def _build_actor_critic(num_actor_obs: int, num_critic_obs: int, num_actions: int, spec: RLTrainSpec) -> Any:
    """Construct the Gaussian-MLP ``ActorCritic`` module (torch required)."""
    import torch
    import torch.nn as nn
    from torch.distributions import Normal

    class ActorCritic(nn.Module):
        """Gaussian-MLP actor + value critic with a learned action std."""

        def __init__(self) -> None:
            super().__init__()
            self.actor = _mlp(num_actor_obs, spec.hidden_dims, num_actions)
            self.critic = _mlp(num_critic_obs, spec.hidden_dims, 1)
            self.log_std = nn.Parameter(torch.ones(num_actions) * float(torch.log(torch.tensor(spec.init_noise_std))))

        def _distribution(self, actor_obs: torch.Tensor) -> Normal:
            mean = self.actor(actor_obs)
            std = torch.exp(self.log_std).expand_as(mean)
            return Normal(mean, std)

        def act(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> dict[str, torch.Tensor]:
            """Sample an action for rollout collection; returns ``{action, log_prob, value, mean}``."""
            dist = self._distribution(actor_obs)
            action = dist.sample()
            return {
                "action": action,
                "log_prob": dist.log_prob(action).sum(-1),
                "value": self.critic(critic_obs).squeeze(-1),
                "mean": dist.mean,
            }

        def evaluate(
            self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, action: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            """Re-score stored actions under the current policy for the PPO update; returns ``{log_prob, value, entropy}``."""
            dist = self._distribution(actor_obs)
            return {
                "log_prob": dist.log_prob(action).sum(-1),
                "value": self.critic(critic_obs).squeeze(-1),
                "entropy": dist.entropy().sum(-1),
            }

        def act_inference(self, actor_obs: torch.Tensor) -> torch.Tensor:
            """Deterministic (mean) action - the deployable policy."""
            return self.actor(actor_obs)

    return ActorCritic()


def compute_gae(
    rewards: Any,
    values: Any,
    next_values: Any,
    dones: Any,
    terminated: Any,
    gamma: float,
    lam: float,
) -> tuple[Any, Any]:
    """Generalized Advantage Estimation over one rollout (all args 1-D tensors).

    Truncated steps (``dones`` set but ``terminated`` clear, i.e. a time-out) are
    value-bootstrapped via ``next_values``; real terminals zero the bootstrap; the
    advantage trace resets at every episode boundary (``dones``).

    Args:
        rewards: Per-step reward, shape ``(T,)``.
        values: Critic value of the step observation, shape ``(T,)``.
        next_values: Critic value of the resulting observation, shape ``(T,)``.
        dones: 1.0 at any episode boundary (terminal OR time-out), shape ``(T,)``.
        terminated: 1.0 only on real terminals (not time-outs), shape ``(T,)``.
        gamma: Discount factor.
        lam: GAE-lambda.

    Returns:
        ``(advantages, returns)`` tensors, each shape ``(T,)``; ``returns =
        advantages + values``.
    """
    import torch

    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - terminated[t]
        delta = rewards[t] + gamma * next_values[t] * next_nonterminal - values[t]
        last_adv = delta + gamma * lam * (1.0 - dones[t]) * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


class PpoTrainer(BaseRLAlgo):
    """Proximal Policy Optimization trainer (``provider_name == "ppo"``)."""

    @property
    def provider_name(self) -> str:
        """Registry key for this trainer (``"ppo"``)."""
        return "ppo"

    def validate(self, spec: TrainSpec) -> list[str]:
        """Preflight an :class:`RLTrainSpec` for a PPO run (pure / read-only)."""
        problems = self._security_problems(spec)
        if not isinstance(spec, RLTrainSpec):
            problems.append(f"ppo requires an RLTrainSpec, got {type(spec).__name__}")
            return problems
        if spec.env_factory is None:
            problems.append("env_factory is required (a zero-arg callable returning a SimEnv)")
        if not spec.output_dir:
            problems.append("output_dir is required")
        if spec.total_timesteps <= 0:
            problems.append(f"total_timesteps must be > 0, got {spec.total_timesteps}")
        if spec.rollout_steps <= 0:
            problems.append(f"rollout_steps must be > 0, got {spec.rollout_steps}")
        if spec.num_envs < 1:
            problems.append(f"num_envs must be >= 1, got {spec.num_envs}")
        if spec.num_mini_batches <= 0 or spec.rollout_steps % spec.num_mini_batches != 0:
            problems.append(
                f"rollout_steps ({spec.rollout_steps}) must be divisible by num_mini_batches ({spec.num_mini_batches})"
            )
        return problems

    def setup(self, spec: RLTrainSpec) -> None:
        """Build env, actor-critic, optimizer, normalizers, and rollout storage."""
        require_optional("torch", purpose="PPO RL training (strands_robots.training.rl.ppo)")
        import torch

        from strands_robots.training.rl.normalization import EmpiricalNormalization

        self.spec = spec
        self.device = torch.device(spec.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if spec.seed is not None:
            torch.manual_seed(spec.seed)

        if spec.env_factory is None:  # pragma: no cover - guarded by validate()
            raise ValueError("env_factory is required")
        # num_envs == 1 keeps the single SimEnv path (zero behavioural change);
        # num_envs > 1 wraps N independent SimEnv in a VecSimEnv that emits
        # (N, D) batches. ``self._vectorized`` selects the collect_rollout path.
        self._vectorized = spec.num_envs > 1
        if self._vectorized:
            from strands_robots.training.rl.vec_env import VecSimEnv

            self.env = VecSimEnv(spec.env_factory, spec.num_envs, device=self.device)
        else:
            self.env = spec.env_factory()
        # The learner (actor-critic, normalizers, rollout buffers) lives on
        # self.device. The env emits observation / reward / done tensors on its
        # own device, which defaults to CPU. On a GPU host the learner resolves
        # to cuda while the env stays on CPU, so every actor/critic forward pass
        # mixes cuda and cpu tensors and raises "Expected all tensors to be on
        # the same device". The learner device is authoritative: reconcile the
        # env onto it so observations are built on the right device at the
        # source (no per-step host<->device copies).
        if self.env.device != self.device:
            self.env.device = self.device
        self.steps_per_iter = spec.rollout_steps * spec.num_envs

        self.actor_critic = _build_actor_critic(
            self.env.num_actor_obs, self.env.num_critic_obs, self.env.num_actions, spec
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=spec.learning_rate)

        self.actor_norm = EmpiricalNormalization(self.env.num_actor_obs, self.device) if spec.normalize_obs else None
        self.critic_norm = EmpiricalNormalization(self.env.num_critic_obs, self.device) if spec.normalize_obs else None
        self._obs = self.env.reset()
        self._batch: dict[str, torch.Tensor] = {}

    def _norm_actor(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.actor_norm(x, update=update) if self.actor_norm is not None else x

    def _norm_critic(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.critic_norm(x, update=update) if self.critic_norm is not None else x

    def collect_rollout(self) -> dict[str, float]:
        """Roll out ``rollout_steps`` transitions and compute GAE returns/advantages.

        Dispatches to the vectorized path when ``setup`` built a ``VecSimEnv``
        (num_envs > 1), else the original single-env path (unchanged).
        """
        if getattr(self, "_vectorized", False):
            return self._collect_rollout_vectorized()
        return self._collect_rollout_single()

    def _collect_rollout_single(self) -> dict[str, float]:
        """Single-env rollout (num_envs == 1). Behaviourally identical to the original."""
        import torch

        from strands_robots.training.rl.env import SimEnv

        assert isinstance(self.env, SimEnv)  # non-vectorized path
        spec = self.spec
        T = spec.rollout_steps
        obs_buf, cobs_buf, act_buf, logp_buf = [], [], [], []
        val_buf, rew_buf, done_buf, term_buf, nextval_buf = [], [], [], [], []

        self.actor_critic.train()
        ep_returns: list[float] = []
        running_return = 0.0
        for _ in range(T):
            actor_obs = self._norm_actor(self._obs["actor_obs"])
            critic_obs = self._norm_critic(self._obs["critic_obs"])
            with torch.no_grad():
                out = self.actor_critic.act(actor_obs, critic_obs)
            next_obs, reward, done, info = self.env.step(out["action"])

            obs_buf.append(actor_obs)
            cobs_buf.append(critic_obs)
            act_buf.append(out["action"])
            logp_buf.append(out["log_prob"])
            val_buf.append(out["value"])
            rew_buf.append(reward)
            done_buf.append(done)
            terminated_flag = float(info["terminated"])
            term_buf.append(torch.tensor([terminated_flag], dtype=torch.float32, device=self.device))

            with torch.no_grad():
                next_value = self.actor_critic.critic(self._norm_critic(next_obs["critic_obs"], update=False)).squeeze(
                    -1
                )
            nextval_buf.append(next_value)

            running_return += float(reward.item())
            if bool(done.item()):
                ep_returns.append(running_return)
                running_return = 0.0
                self._obs = self.env.reset()
            else:
                self._obs = next_obs

        obs = torch.cat(obs_buf, 0)
        cobs = torch.cat(cobs_buf, 0)
        actions = torch.cat(act_buf, 0)
        old_logp = torch.cat(logp_buf, 0)
        values = torch.cat(val_buf, 0)
        rewards = torch.cat(rew_buf, 0)
        dones = torch.cat(done_buf, 0)
        terminated = torch.cat(term_buf, 0)
        next_values = torch.cat(nextval_buf, 0)

        advantages, returns = compute_gae(rewards, values, next_values, dones, terminated, spec.gamma, spec.lam)
        if spec.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self._batch = {
            "actor_obs": obs,
            "critic_obs": cobs,
            "actions": actions,
            "old_log_prob": old_logp,
            "values": values,
            "returns": returns,
            "advantages": advantages,
        }
        mean_return = float(sum(ep_returns) / len(ep_returns)) if ep_returns else float(rewards.sum().item())
        return {"mean_reward": float(rewards.mean().item()), "mean_episode_return": mean_return}

    def _collect_rollout_vectorized(self) -> dict[str, float]:
        """Vectorized rollout over a ``VecSimEnv`` (num_envs == N > 1).

        Collects ``(T, N, ...)`` transitions, bootstraps each step's next-value
        from the per-env resulting observation (using the captured pre-reset
        ``terminal_obs`` on a done env so a truncation is value-bootstrapped from
        its TRUE terminal state, not the post-reset state), computes GAE per env
        on the ``(T, N)`` tensors (``compute_gae`` broadcasts over the env axis),
        then flattens to ``(T*N, ...)`` for the same minibatch update the
        single-env path uses. Asymmetric actor/critic, normalizers, and the
        checkpoint contract are unchanged.
        """
        import torch

        from strands_robots.training.rl.vec_env import VecSimEnv

        assert isinstance(self.env, VecSimEnv)  # vectorized path
        spec = self.spec
        T = spec.rollout_steps
        N = self.env.num_envs
        obs_buf, cobs_buf, act_buf, logp_buf = [], [], [], []
        val_buf, rew_buf, done_buf, term_buf, nextval_buf = [], [], [], [], []

        self.actor_critic.train()
        # Per-env running returns + completed-episode returns for logging.
        running = torch.zeros(N, device=self.device)
        ep_returns: list[float] = []

        for _ in range(T):
            actor_obs = self._norm_actor(self._obs["actor_obs"])  # (N, Da)
            critic_obs = self._norm_critic(self._obs["critic_obs"])  # (N, Dc)
            with torch.no_grad():
                out = self.actor_critic.act(actor_obs, critic_obs)
            next_obs, reward, done, infos = self.env.step(out["action"])  # reward,done (N,)

            obs_buf.append(actor_obs)
            cobs_buf.append(critic_obs)
            act_buf.append(out["action"])  # (N, A)
            logp_buf.append(out["log_prob"])  # (N,)
            val_buf.append(out["value"])  # (N,)
            rew_buf.append(reward)  # (N,)
            done_buf.append(done)  # (N,)

            # Per-env terminated flag (real terminal, not time-out).
            term_flags = torch.tensor(
                [float(info.get("terminated", False)) for info in infos],
                dtype=torch.float32,
                device=self.device,
            )
            term_buf.append(term_flags)

            # next-value: for a done env, bootstrap from the captured TERMINAL
            # critic obs (pre-reset); for a live env, from the resulting obs.
            # VecSimEnv put fresh post-reset obs into next_obs[i] on done, so we
            # must pull the terminal obs back out of infos[i]["terminal_obs"].
            critic_next = next_obs["critic_obs"].clone()  # (N, Dc)
            for i, info in enumerate(infos):
                term_obs = info.get("terminal_obs")
                if term_obs is not None:
                    critic_next[i] = term_obs["critic_obs"].reshape(-1)
            with torch.no_grad():
                next_value = self.actor_critic.critic(self._norm_critic(critic_next, update=False)).squeeze(-1)  # (N,)
            nextval_buf.append(next_value)

            running = running + reward
            done_mask = done.bool()
            if bool(done_mask.any()):
                for i in range(N):
                    if bool(done_mask[i].item()):
                        ep_returns.append(float(running[i].item()))
                        running[i] = 0.0
            self._obs = next_obs

        # Stack to (T, N, ...).
        obs = torch.stack(obs_buf, 0)  # (T, N, Da)
        cobs = torch.stack(cobs_buf, 0)
        actions = torch.stack(act_buf, 0)  # (T, N, A)
        old_logp = torch.stack(logp_buf, 0)  # (T, N)
        values = torch.stack(val_buf, 0)  # (T, N)
        rewards = torch.stack(rew_buf, 0)  # (T, N)
        dones = torch.stack(done_buf, 0)  # (T, N)
        terminated = torch.stack(term_buf, 0)  # (T, N)
        next_values = torch.stack(nextval_buf, 0)  # (T, N)

        # GAE per env: compute_gae broadcasts the recurrence over the N axis.
        advantages, returns = compute_gae(
            rewards, values, next_values, dones, terminated, spec.gamma, spec.lam
        )  # (T, N)
        if spec.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten (T, N, ...) -> (T*N, ...) for the shared minibatch update.
        Da, Dc, A = self.env.num_actor_obs, self.env.num_critic_obs, self.env.num_actions
        self._batch = {
            "actor_obs": obs.reshape(T * N, Da),
            "critic_obs": cobs.reshape(T * N, Dc),
            "actions": actions.reshape(T * N, A),
            "old_log_prob": old_logp.reshape(T * N),
            "values": values.reshape(T * N),
            "returns": returns.reshape(T * N),
            "advantages": advantages.reshape(T * N),
        }
        mean_return = float(sum(ep_returns) / len(ep_returns)) if ep_returns else float(rewards.sum().item() / N)
        return {"mean_reward": float(rewards.mean().item()), "mean_episode_return": mean_return}

    def update(self) -> dict[str, float]:
        """PPO epochs over the rollout batch: clipped surrogate + clipped value loss."""
        import torch
        import torch.nn as nn

        spec = self.spec
        batch = self._batch
        n = batch["actor_obs"].shape[0]
        mb_size = n // spec.num_mini_batches

        tot_surrogate, tot_value, tot_entropy, n_updates = 0.0, 0.0, 0.0, 0
        for _ in range(spec.num_learning_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb_size):
                idx = perm[start : start + mb_size]
                ev = self.actor_critic.evaluate(
                    batch["actor_obs"][idx], batch["critic_obs"][idx], batch["actions"][idx]
                )
                ratio = torch.exp(ev["log_prob"] - batch["old_log_prob"][idx])
                adv = batch["advantages"][idx]
                surrogate = -adv * ratio
                surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - spec.clip_param, 1.0 + spec.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                value = ev["value"]
                returns = batch["returns"][idx]
                old_values = batch["values"][idx]
                value_clipped = old_values + (value - old_values).clamp(-spec.clip_param, spec.clip_param)
                value_loss = torch.max((value - returns).pow(2), (value_clipped - returns).pow(2)).mean()

                entropy = ev["entropy"].mean()
                loss = surrogate_loss + spec.value_loss_coef * value_loss - spec.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), spec.max_grad_norm)
                self.optimizer.step()

                tot_surrogate += float(surrogate_loss.item())
                tot_value += float(value_loss.item())
                tot_entropy += float(entropy.item())
                n_updates += 1

        return {
            "surrogate_loss": tot_surrogate / max(1, n_updates),
            "value_loss": tot_value / max(1, n_updates),
            "entropy": tot_entropy / max(1, n_updates),
            "latest_loss": tot_value / max(1, n_updates),
        }

    def _checkpoint_dir(self, output_dir: str) -> str:
        return os.path.join(output_dir, "checkpoints", "last")

    def save_checkpoint(self, output_dir: str, iteration: int | None = None) -> str:
        """Save the actor-critic, normalizers, and a deployable-policy metadata file."""
        import torch

        ckpt_dir = self._checkpoint_dir(output_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        state: dict[str, Any] = {
            "actor_critic": self.actor_critic.state_dict(),
            "iteration": iteration,
            "provider": self.provider_name,
        }
        if self.actor_norm is not None:
            state["actor_norm"] = self.actor_norm.state_dict()
        if self.critic_norm is not None:
            state["critic_norm"] = self.critic_norm.state_dict()
        torch.save(state, os.path.join(ckpt_dir, "policy.pt"))

        meta = {
            "provider": self.provider_name,
            "num_actor_obs": self.env.num_actor_obs,
            "num_critic_obs": self.env.num_critic_obs,
            "num_actions": self.env.num_actions,
            "actor_obs_keys": self.env.actor_obs_keys,
            # ``action_keys``, not a joint list: the field names what the
            # ``num_actions`` outputs above it drive, so it must be the same
            # vocabulary ``send_action`` binds a vector against. A tendon
            # gripper's actuator has no matching joint name at all, and a
            # Newton floating base is a joint with no commandable scalar.
            "action_keys": (self.env.engine.robot_action_keys(self.env.robot_name) if self.env.robot_name else []),
            "hidden_dims": list(self.spec.hidden_dims),
            "iteration": iteration,
        }
        with open(os.path.join(ckpt_dir, "policy_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return ckpt_dir

    def latest_checkpoint(self, output_dir: str) -> str | None:
        """Return the checkpoint dir holding ``policy.pt`` under ``output_dir``."""
        ckpt = self._checkpoint_dir(output_dir)
        return ckpt if os.path.isfile(os.path.join(ckpt, "policy.pt")) else None

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Return the loadable policy artifact (``policy.pt``) for inference."""
        return os.path.join(checkpoint_dir, "policy.pt")

    @property
    def hardware_floor(self) -> dict[str, Any]:
        """PPO on MuJoCo trains fine on CPU; no GPU floor."""
        return {"min_gpus": 0, "min_vram_gb": 0, "multinode": False}
