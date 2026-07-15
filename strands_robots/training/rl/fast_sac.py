"""FastSAC - off-policy from-scratch RL trainer for the ``SimEngine`` env.

Soft Actor-Critic: an off-policy, sample-efficient peer of the on-policy
:class:`~strands_robots.training.rl.ppo.PpoTrainer`. It trains a tanh-squashed
Gaussian-MLP actor and twin Q critics (clipped double-Q, Polyak-averaged target
critics) with automatic entropy-temperature tuning, replaying transitions from a
:class:`~strands_robots.training.rl.replay_buffer.SimpleReplayBuffer`. Like PPO
it learns from a reward function alone (the reward-term DSL) on a
:class:`~strands_robots.training.rl.env.SimEnv`, so a reach / locomotion / WBC
policy can be trained in MuJoCo with no demonstration dataset.

Off-policy means the on-policy ``BaseRLAlgo.train()`` loop does not fit, so this
overrides :meth:`train` with the standard SAC schedule (random warmup -> per-step
gradient updates from the replay buffer) while keeping the same
``setup -> collect_rollout -> update -> save_checkpoint`` hooks and the
``policy.pt`` + ``policy_meta.json`` checkpoint contract as PPO. Selected via
``create_trainer("fast_sac")``.

The SAC math (clipped double-Q target, tanh-squashed log-prob correction,
automatic temperature) is the standard Haarnoja et al. formulation, adapted from
the Amazon FAR Holosoma FastSAC (BSD-3-Clause,
https://github.com/amazon-far/holosoma), re-homed onto the single-environment
MuJoCo backend.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from strands_robots.training.base import TrainResult, TrainSpec
from strands_robots.training.rl.base_algo import BaseRLAlgo, RLTrainSpec
from strands_robots.utils import require_optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from strands_robots.training.rl.env import SimEnv

# tanh-squashed Gaussian: clamp log_std into a stable range and bound the
# log(1 - tanh(u)^2) correction away from -inf at the squash saturation.
_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0
_LOG_PROB_EPS = 1e-6


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> Any:
    """Build a ReLU-activated MLP ``in_dim -> *hidden -> out_dim``."""
    import torch.nn as nn

    layers: list[Any] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def _build_actor_critic(num_actor_obs: int, num_critic_obs: int, num_actions: int, spec: RLTrainSpec) -> Any:
    """Construct the SAC ``ActorCritic`` module: tanh-Gaussian actor + twin Q critics."""
    import torch
    import torch.nn as nn

    class SacActorCritic(nn.Module):
        """tanh-squashed Gaussian actor + twin Q critics with target copies."""

        def __init__(self) -> None:
            super().__init__()
            # Actor outputs (mean, log_std) for the pre-squash Gaussian.
            self.actor = _mlp(num_actor_obs, spec.hidden_dims, 2 * num_actions)
            # Twin critics Q(critic_obs, action) -> scalar (clipped double-Q).
            self.q1 = _mlp(num_critic_obs + num_actions, spec.hidden_dims, 1)
            self.q2 = _mlp(num_critic_obs + num_actions, spec.hidden_dims, 1)
            self.q1_target = _mlp(num_critic_obs + num_actions, spec.hidden_dims, 1)
            self.q2_target = _mlp(num_critic_obs + num_actions, spec.hidden_dims, 1)
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
            for p in self.q1_target.parameters():
                p.requires_grad_(False)
            for p in self.q2_target.parameters():
                p.requires_grad_(False)

        def _mean_log_std(self, actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean, log_std = self.actor(actor_obs).chunk(2, dim=-1)
            log_std = torch.clamp(log_std, _LOG_STD_MIN, _LOG_STD_MAX)
            return mean, log_std

        def sample(self, actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Reparameterized tanh-Gaussian sample; returns ``(action, log_prob)``.

            ``log_prob`` includes the tanh change-of-variables correction and is
            summed over action dimensions, shape ``(B, 1)``.
            """
            mean, log_std = self._mean_log_std(actor_obs)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            u = normal.rsample()
            action = torch.tanh(u)
            log_prob = normal.log_prob(u) - torch.log(1.0 - action.pow(2) + _LOG_PROB_EPS)
            return action, log_prob.sum(-1, keepdim=True)

        def act_inference(self, actor_obs: torch.Tensor) -> torch.Tensor:
            """Deterministic (mean) action - the deployable policy."""
            mean, _ = self._mean_log_std(actor_obs)
            return torch.tanh(mean)

        def q_values(self, critic_obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Twin-critic Q-values ``(Q1, Q2)`` for a state-action pair (clipped double-Q)."""
            x = torch.cat([critic_obs, action], dim=-1)
            return self.q1(x), self.q2(x)

        def q_target(self, critic_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            """Conservative target value ``min(Q1_target, Q2_target)`` for the TD backup."""
            x = torch.cat([critic_obs, action], dim=-1)
            return torch.min(self.q1_target(x), self.q2_target(x))

    return SacActorCritic()


class FastSacTrainer(BaseRLAlgo):
    """Soft Actor-Critic trainer (``provider_name == "fast_sac"``)."""

    @property
    def provider_name(self) -> str:
        """Registry key for this trainer (``"fast_sac"``)."""
        return "fast_sac"

    def validate(self, spec: TrainSpec) -> list[str]:
        """Preflight an :class:`RLTrainSpec` for a FastSAC run (pure / read-only)."""
        problems = self._security_problems(spec)
        if not isinstance(spec, RLTrainSpec):
            problems.append(f"fast_sac requires an RLTrainSpec, got {type(spec).__name__}")
            return problems
        if spec.env_factory is None:
            problems.append("env_factory is required (a zero-arg callable returning a SimEnv)")
        if not spec.output_dir:
            problems.append("output_dir is required")
        if spec.total_timesteps <= 0:
            problems.append(f"total_timesteps must be > 0, got {spec.total_timesteps}")
        if spec.rollout_steps <= 0:
            problems.append(f"rollout_steps must be > 0, got {spec.rollout_steps}")
        if spec.num_envs != 1:
            problems.append(f"the MuJoCo backend is single-env (num_envs must be 1), got {spec.num_envs}")
        if spec.buffer_size <= 0:
            problems.append(f"buffer_size must be > 0, got {spec.buffer_size}")
        if spec.batch_size <= 0:
            problems.append(f"batch_size must be > 0, got {spec.batch_size}")
        if spec.gradient_steps <= 0:
            problems.append(f"gradient_steps must be > 0, got {spec.gradient_steps}")
        if not 0.0 < spec.tau <= 1.0:
            problems.append(f"tau must be in (0, 1], got {spec.tau}")
        if spec.learning_starts < spec.batch_size:
            problems.append(
                f"learning_starts ({spec.learning_starts}) must be >= batch_size ({spec.batch_size}) "
                "so the first gradient step can sample a full batch"
            )
        return problems

    def setup(self, spec: RLTrainSpec) -> None:
        """Build env, actor + twin critics, optimizers, temperature, and replay buffer."""
        require_optional("torch", purpose="FastSAC RL training (strands_robots.training.rl.fast_sac)")
        import torch

        from strands_robots.training.rl.normalization import EmpiricalNormalization
        from strands_robots.training.rl.replay_buffer import SimpleReplayBuffer

        self.spec = spec
        self.device = torch.device(spec.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if spec.seed is not None:
            torch.manual_seed(spec.seed)

        if spec.env_factory is None:  # pragma: no cover - guarded by validate()
            raise ValueError("env_factory is required")
        self.env: SimEnv = spec.env_factory()
        # The learner device is authoritative over the env device (see PpoTrainer
        # for the cross-device mismatch this guards against on a GPU host).
        if self.env.device != self.device:
            self.env.device = self.device
        self.steps_per_iter = spec.rollout_steps * spec.num_envs

        self.actor_critic = _build_actor_critic(
            self.env.num_actor_obs, self.env.num_critic_obs, self.env.num_actions, spec
        ).to(self.device)

        actor_params = list(self.actor_critic.actor.parameters())
        critic_params = list(self.actor_critic.q1.parameters()) + list(self.actor_critic.q2.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=spec.learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=spec.learning_rate)

        # Automatic entropy temperature (alpha): optimize log_alpha against the
        # target entropy (default -num_actions, the SAC heuristic).
        self.target_entropy = (
            float(spec.target_entropy) if spec.target_entropy is not None else -float(self.env.num_actions)
        )
        self.autotune_alpha = spec.autotune_alpha
        if self.autotune_alpha:
            self.log_alpha = torch.tensor(
                float(torch.log(torch.tensor(spec.init_alpha))), device=self.device, requires_grad=True
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=spec.alpha_lr)
        else:
            self.log_alpha = torch.tensor(float(torch.log(torch.tensor(spec.init_alpha))), device=self.device)

        self.actor_norm = EmpiricalNormalization(self.env.num_actor_obs, self.device) if spec.normalize_obs else None
        self.critic_norm = EmpiricalNormalization(self.env.num_critic_obs, self.device) if spec.normalize_obs else None
        self.buffer = SimpleReplayBuffer(
            spec.buffer_size, self.env.num_actor_obs, self.env.num_critic_obs, self.env.num_actions, self.device
        )
        self._obs = self.env.reset()
        self._collected_steps = 0
        self._ep_return = 0.0
        self._recent_returns: list[float] = []

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy temperature (``exp(log_alpha)``)."""
        return self.log_alpha.exp()

    def _norm_actor(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.actor_norm(x, update=update) if self.actor_norm is not None else x

    def _norm_critic(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.critic_norm(x, update=update) if self.critic_norm is not None else x

    def collect_rollout(self) -> dict[str, float]:
        """Step the env ``rollout_steps`` times, pushing transitions to the buffer.

        Before ``learning_starts`` transitions are stored the actions are drawn
        uniformly at random (exploration warmup); afterwards they are sampled
        from the stochastic SAC actor. Stores the *terminal* done flag (a time-out
        truncation is recorded as not-done so its value is bootstrapped).
        """
        import torch

        spec = self.spec
        self.actor_critic.train()
        ep_returns: list[float] = []
        step_rewards: list[float] = []
        for _ in range(spec.rollout_steps):
            actor_obs = self._norm_actor(self._obs["actor_obs"])
            if self.buffer.size < spec.learning_starts:
                action = torch.rand(1, self.env.num_actions, device=self.device) * 2.0 - 1.0
            else:
                with torch.no_grad():
                    action, _ = self.actor_critic.sample(actor_obs)
            next_obs, reward, done, info = self.env.step(action)

            # Bootstrap through time-outs: only a genuine terminal stops the value
            # backup. ``info["terminated"]`` is the real terminal (not a time-out).
            terminal = torch.tensor([[float(info["terminated"])]], dtype=torch.float32, device=self.device)
            self.buffer.add(
                self._obs["actor_obs"],
                self._obs["critic_obs"],
                action,
                reward,
                next_obs["actor_obs"],
                next_obs["critic_obs"],
                terminal,
            )
            self._collected_steps += 1
            r = float(reward.item())
            step_rewards.append(r)
            self._ep_return += r
            if bool(done.item()):
                ep_returns.append(self._ep_return)
                self._ep_return = 0.0
                self._obs = self.env.reset()
            else:
                self._obs = next_obs

        if ep_returns:
            self._recent_returns = ep_returns
        mean_return = float(sum(ep_returns) / len(ep_returns)) if ep_returns else float(sum(step_rewards))
        return {
            "mean_reward": float(sum(step_rewards) / max(1, len(step_rewards))),
            "mean_episode_return": mean_return,
            "buffer_size": float(self.buffer.size),
        }

    def update(self) -> dict[str, float]:
        """Run ``gradient_steps`` SAC updates from the replay buffer.

        Each update does a clipped double-Q critic step, a delayed-style actor
        step against the min-Q minus entropy, an automatic temperature step, and
        a Polyak target-critic update. Returns averaged loss metrics; a no-op
        (empty-ish metrics) until the buffer holds at least ``batch_size``.
        """
        import torch
        import torch.nn.functional as F

        spec = self.spec
        if self.buffer.size < spec.batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": float(self.alpha.item()), "latest_loss": 0.0}

        tot_critic, tot_actor, tot_alpha_loss, tot_entropy = 0.0, 0.0, 0.0, 0.0
        for _ in range(spec.gradient_steps):
            batch = self.buffer.sample(spec.batch_size)
            actor_obs = self._norm_actor(batch["actor_obs"], update=False)
            critic_obs = self._norm_critic(batch["critic_obs"], update=False)
            next_actor_obs = self._norm_actor(batch["next_actor_obs"], update=False)
            next_critic_obs = self._norm_critic(batch["next_critic_obs"], update=False)
            rewards = batch["rewards"]
            dones = batch["dones"]

            # --- critic update: clipped double-Q target with entropy bonus ---
            with torch.no_grad():
                next_action, next_logp = self.actor_critic.sample(next_actor_obs)
                q_next = self.actor_critic.q_target(next_critic_obs, next_action) - self.alpha * next_logp
                target_q = rewards + spec.gamma * (1.0 - dones) * q_next
            q1, q2 = self.actor_critic.q_values(critic_obs, batch["actions"])
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # --- actor update: maximize min-Q minus entropy temperature term ---
            new_action, logp = self.actor_critic.sample(actor_obs)
            q1_pi, q2_pi = self.actor_critic.q_values(critic_obs, new_action)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha.detach() * logp - min_q_pi).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # --- temperature update (automatic entropy tuning) ---
            if self.autotune_alpha:
                alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                tot_alpha_loss += float(alpha_loss.item())

            # --- Polyak target-critic update ---
            with torch.no_grad():
                for p, tp in zip(self.actor_critic.q1.parameters(), self.actor_critic.q1_target.parameters()):
                    tp.mul_(1.0 - spec.tau).add_(spec.tau * p)
                for p, tp in zip(self.actor_critic.q2.parameters(), self.actor_critic.q2_target.parameters()):
                    tp.mul_(1.0 - spec.tau).add_(spec.tau * p)

            tot_critic += float(critic_loss.item())
            tot_actor += float(actor_loss.item())
            tot_entropy += float(-logp.mean().item())

        g = spec.gradient_steps
        return {
            "critic_loss": tot_critic / g,
            "actor_loss": tot_actor / g,
            "alpha_loss": tot_alpha_loss / g,
            "alpha": float(self.alpha.item()),
            "entropy": tot_entropy / g,
            "latest_loss": tot_critic / g,
        }

    def train(self, spec: TrainSpec) -> TrainResult:
        """Off-policy SAC loop: setup -> [collect_rollout -> update]* -> save.

        Overrides the on-policy ``BaseRLAlgo.train``. ``spec`` MUST be an
        :class:`RLTrainSpec`; :meth:`validate` is called first and fails closed.
        Updates run only after the buffer passes ``learning_starts``.
        """
        if not isinstance(spec, RLTrainSpec):
            return TrainResult(
                status="error",
                job_id="",
                message=f"{self.provider_name} requires an RLTrainSpec, got {type(spec).__name__}",
            )
        problems = self.validate(spec)
        if problems:
            return TrainResult(status="error", job_id="", message="validation failed: " + "; ".join(problems))

        self.setup(spec)
        steps_per_iter = max(1, self.steps_per_iter)
        num_iters = max(1, spec.total_timesteps // steps_per_iter)

        job_id = f"{self.provider_name}-{id(self):x}"
        last_metrics: dict[str, Any] = {}
        ckpt_dir: str | None = None
        for it in range(num_iters):
            rollout_metrics = self.collect_rollout()
            loss_metrics = self.update() if self.buffer.size >= spec.learning_starts else {}
            last_metrics = {**rollout_metrics, **loss_metrics, "iteration": it + 1}
            if spec.log_interval and (it % spec.log_interval == 0 or it == num_iters - 1):
                ckpt_dir = self.save_checkpoint(spec.output_dir, iteration=it + 1)
        if ckpt_dir is None:
            ckpt_dir = self.save_checkpoint(spec.output_dir, iteration=num_iters)

        last_metrics.setdefault("latest_step", self._collected_steps)
        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=ckpt_dir,
            exported_model=self.export(spec, ckpt_dir),
            metrics=last_metrics,
            message=f"{self.provider_name}: {num_iters} iterations x {steps_per_iter} steps complete",
        )

    def _checkpoint_dir(self, output_dir: str) -> str:
        return os.path.join(output_dir, "checkpoints", "last")

    def save_checkpoint(self, output_dir: str, iteration: int | None = None) -> str:
        """Save the actor-critic, normalizers, temperature, and policy metadata.

        Writes the same ``policy.pt`` + ``policy_meta.json`` contract as
        ``PpoTrainer`` so a single checkpoint loader serves both RL backends.
        """
        import torch

        ckpt_dir = self._checkpoint_dir(output_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        state: dict[str, Any] = {
            "actor_critic": self.actor_critic.state_dict(),
            "log_alpha": self.log_alpha.detach(),
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
            "joint_names": (self.env.engine.robot_joint_names(self.env.robot_name) if self.env.robot_name else []),
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
        """FastSAC on MuJoCo trains fine on CPU; no GPU floor."""
        return {"min_gpus": 0, "min_vram_gb": 0, "multinode": False}
