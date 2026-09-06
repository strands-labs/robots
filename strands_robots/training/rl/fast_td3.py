"""FastTD3 - off-policy from-scratch RL trainer for the ``SimEngine`` env.

Twin Delayed DDPG: the deterministic-actor peer of
:class:`~strands_robots.training.rl.fast_sac.FastSacTrainer`. It trains a
tanh-bounded deterministic MLP actor against twin Q critics with clipped
double-Q targets, target policy smoothing (clipped Gaussian noise on the target
action), delayed actor / target-network updates (``policy_delay``), and Polyak
averaging, replaying transitions from a
:class:`~strands_robots.training.rl.replay_buffer.SimpleReplayBuffer`. Like SAC
it learns from a reward function alone (the reward-term DSL) on a
:class:`~strands_robots.training.rl.env.SimEnv`, so a reach / locomotion / WBC
policy can be trained in MuJoCo with no demonstration dataset.

Where SAC explores through its stochastic actor, a deterministic actor explores
only through the Gaussian ``exploration_noise_std`` added at collection - after
a uniform-random warmup of ``learning_starts`` steps - so that scale (and the
two smoothing scales) is preflighted rather than trusted: zero silently removes
the mechanism and the run still reports success.

Unlike the single-env FastSAC, collection is vectorized: ``num_envs > 1``
steps N independent envs through a
:class:`~strands_robots.training.rl.vec_env.VecSimEnv` and pushes N transitions
per tick, bootstrapping a done env's stored next-observation from the captured
pre-reset ``infos[i]["terminal_obs"]`` exactly as the vectorized PPO path
bootstraps its next-value - off-policy replay has no rollout tensor to reshape,
so the buffer absorbs the extra envs without changing the update.

Off-policy means the on-policy ``BaseRLAlgo.train()`` loop does not fit, so this
overrides :meth:`train` with the same schedule as FastSAC (random warmup ->
per-step gradient updates from the replay buffer) while keeping the same
``setup -> collect_rollout -> update -> save_checkpoint`` hooks and the
``policy.pt`` + ``policy_meta.json`` checkpoint contract as PPO / FastSAC.
Selected via ``create_trainer("fast_td3")``.

The TD3 math (clipped double-Q, target policy smoothing, delayed policy
updates) is the standard Fujimoto et al. formulation, benchmarked against the
FastTD3 project (https://github.com/younggyoseo/FastTD3, arXiv:2505.22642) and
re-homed onto the strands-robots ``SimEnv`` / ``VecSimEnv`` backend.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, cast

from strands_robots.training.base import TrainResult, TrainSpec
from strands_robots.training.rl.base_algo import BaseRLAlgo, RLTrainSpec
from strands_robots.utils import positive_count_error, require_optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from strands_robots.training.rl.env import SimEnv


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
    """Construct the TD3 ``ActorCritic`` module: deterministic actor + twin Q critics.

    Every network that has a Polyak target carries it here - the actor as well
    as both critics, because target policy smoothing samples around the *target*
    policy's action, not the live one.
    """
    import torch
    import torch.nn as nn

    class Td3ActorCritic(nn.Module):
        """tanh-bounded deterministic actor + twin Q critics, each with a target copy."""

        def __init__(self) -> None:
            super().__init__()
            # Actor outputs a pre-tanh action; tanh bounds it to [-1, 1], the
            # same action interval the SAC actor squashes into, so the two
            # off-policy backends drive SimEnv with one action contract.
            self.actor = _mlp(num_actor_obs, spec.hidden_dims, num_actions)
            self.actor_target = _mlp(num_actor_obs, spec.hidden_dims, num_actions)
            # Twin critics Q(critic_obs, action) -> scalar (clipped double-Q).
            self.q1 = _mlp(num_critic_obs + num_actions, spec.hidden_dims, 1)
            self.q2 = _mlp(num_critic_obs + num_actions, spec.hidden_dims, 1)
            self.q1_target = _mlp(num_critic_obs + num_actions, spec.hidden_dims, 1)
            self.q2_target = _mlp(num_critic_obs + num_actions, spec.hidden_dims, 1)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
            for target in (self.actor_target, self.q1_target, self.q2_target):
                for p in target.parameters():
                    p.requires_grad_(False)

        def act_inference(self, actor_obs: torch.Tensor) -> torch.Tensor:
            """Deterministic bounded action - the deployable policy."""
            return torch.tanh(self.actor(actor_obs))

        def act_target(self, actor_obs: torch.Tensor) -> torch.Tensor:
            """Target policy's bounded action, for the smoothed TD backup."""
            return torch.tanh(self.actor_target(actor_obs))

        def q_values(self, critic_obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Twin-critic Q-values ``(Q1, Q2)`` for a state-action pair (clipped double-Q)."""
            x = torch.cat([critic_obs, action], dim=-1)
            return self.q1(x), self.q2(x)

        def q_target(self, critic_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            """Conservative target value ``min(Q1_target, Q2_target)`` for the TD backup."""
            x = torch.cat([critic_obs, action], dim=-1)
            return torch.min(self.q1_target(x), self.q2_target(x))

    return Td3ActorCritic()


class FastTd3Trainer(BaseRLAlgo):
    """Twin Delayed DDPG trainer (``provider_name == "fast_td3"``)."""

    @property
    def provider_name(self) -> str:
        """Registry key for this trainer (``"fast_td3"``)."""
        return "fast_td3"

    def validate(self, spec: TrainSpec) -> list[str]:
        """Preflight an :class:`RLTrainSpec` for a FastTD3 run (pure / read-only)."""
        problems = self._security_problems(spec)
        problems.extend(self._learning_rate_problems(spec))
        problems.extend(self._seed_problems(spec))
        if not isinstance(spec, RLTrainSpec):
            problems.append(f"fast_td3 requires an RLTrainSpec, got {type(spec).__name__}")
            return problems
        if spec.env_factory is None:
            problems.append("env_factory is required (a zero-arg callable returning a SimEnv)")
        if not spec.output_dir:
            problems.append("output_dir is required")
        # gamma discounts the return this backend optimizes; the arithmetic that
        # consumes it never judges it, so the shared interval domain does.
        problems.extend(self._discount_factor_problems(spec))
        # policy_delay is the modulus that decides whether the actor and the
        # target networks move at all; a modulus the test never satisfies trains
        # the critics for the whole run while the deployable actor never takes a
        # gradient step, under a successful result.
        problems.extend(self._policy_delay_problems(spec))
        # The three noise scales are a deterministic policy's only exploration
        # and the critic target's only smoothing; zero silently removes the
        # mechanism, a negative scale is silently the identical distribution,
        # and a non-finite one poisons the actions or the TD target.
        problems.extend(self._td3_noise_problems(spec))
        # total_timesteps and rollout_steps are the two caller-supplied factors of
        # this loop's own bound, max(1, total_timesteps // (rollout_steps *
        # num_envs)). The max() clamp means a local <= 0 test cannot bound them:
        # it reads True, a fraction below one iteration, nan and inf as a single
        # iteration under a successful run.
        problems.extend(self._rl_run_size_problems(spec))
        # num_envs is the third factor of that same product. Which *counts* are
        # usable is per-backend - this one collects through VecSimEnv, so any
        # positive count is, like PPO and unlike the single-env FastSAC - but
        # that a count is what the field must hold is not, and it is the same
        # ``positive_count_error`` domain the two factors above use, for the
        # same reason: it is a factor of a ``range()`` bound, and it also sizes
        # the warmup batch of uniform actions drawn per tick. Over the values
        # the domain accepts, every positive ``int`` parallelizes, so there is
        # no further count to test here.
        if (error := positive_count_error(spec.num_envs, "num_envs", self.provider_name)) is not None:
            problems.append(error)
        # buffer_size, batch_size and gradient_steps are the replay loop's three
        # caller-supplied counts: the buffer capacity, the transitions sampled
        # per gradient step, and the TD3 updates per iteration. Each is consumed
        # directly as a count (a tensor capacity, a sample size, a range() bound),
        # so a local <= 0 test is weaker than the shared count domain - it admits
        # True as a degenerate size (a one-slot buffer that never fills, a batch
        # of one), lets every fraction / non-finite value raise inside setup or
        # the update loop after the env, networks, optimizers and buffer are
        # built, and raises TypeError itself on a string or None.
        problems.extend(self._rl_replay_problems(spec))
        # hidden_dims is the shape of every network this backend builds - the
        # actor and the critics alike. The expansion loop judges nothing and
        # nn.Linear accepts a width of zero, which makes the activation after it
        # empty and the next layer's output its bias alone: the policy stops
        # being a function of the observation, and the run reports success while
        # exporting a deployable checkpoint whose actor is one fixed action.
        problems.extend(self._network_width_problems(spec))
        # device is spent by torch.device itself, which judges nothing: every
        # network, buffer and rollout tensor is placed on the result. "gpu" and
        # "cuda:abc" raise out of setup after the preflight passed, and a
        # non-str ordinal constructs on any host and then dies at the first
        # .to() with "invalid device ordinal" - the same spec training fine on a
        # box with more GPUs.
        problems.extend(self._spec_device_problems(spec))
        # tau is the rate at which the target critics track the online ones,
        # spent as tp.mul_(1.0 - spec.tau).add_(spec.tau * p) per mirrored pair,
        # so it decides whether a separate target network exists at all. A bare
        # interval comparison could not carry that: bool is an int subclass, so
        # True read as the interval's maximum - the hard update tp = p, a target
        # network that is a copy of the online one, measured as an exactly zero
        # online-to-target gap in the checkpoint of a run that reported success -
        # and a numeric string, None or a list raised TypeError out of the
        # comparison itself, from a validate documented to return its problems.
        # The interval is unchanged, and is the one the on-policy gamma and lam
        # gates cite as the precedent they generalize; it is now shared with them
        # rather than duplicated between this backend and its sibling.
        problems.extend(self._polyak_coefficient_problems(spec))
        # learning_starts >= batch_size is a relation between two counts, so BOTH
        # operands are asked of the shared count domain and the relation only of
        # two values that are counts - a non-finite learning_starts makes ``<``
        # answer False (every comparison against nan is False, and inf is below
        # no int), so without the domain the relation passes and both consumers
        # then read a value that is not a count: ``collect_rollout`` tests
        # ``buffer.size < learning_starts`` to decide the random warmup and
        # ``train`` tests ``buffer.size >= learning_starts`` to decide whether
        # ``update()`` runs at all, so nan skips the warmup and takes zero
        # gradient steps while inf warms up forever and takes zero gradient
        # steps - a run that reports success having learned nothing. See
        # FastSAC's identical guard for the full measured reasoning; the two
        # off-policy backends share the warmup contract verbatim.
        learning_starts_error = positive_count_error(spec.learning_starts, "learning_starts", self.provider_name)
        if learning_starts_error is not None:
            problems.append(learning_starts_error)
        elif (
            positive_count_error(spec.batch_size, "batch_size", self.provider_name) is None
            and spec.learning_starts < spec.batch_size
        ):
            problems.append(
                f"learning_starts ({spec.learning_starts}) must be >= batch_size ({spec.batch_size}) "
                "so the first gradient step can sample a full batch"
            )
        # log_interval is this loop's checkpoint cadence - the modulus of the one
        # test that decides whether an intermediate checkpoint is written - so it
        # answers the same question save_freq does for a supervised run and takes
        # the same shared domain. The modulus judges it not at all: nan never
        # satisfies it and silently keeps only the final checkpoint of a
        # successful run, True writes one every iteration, a fraction is a
        # silently different cadence, and a str raises out of the loop after
        # setup has built the env, the networks and the optimizers.
        problems.extend(self._rl_checkpoint_interval_problems(spec))
        return problems

    def setup(self, spec: RLTrainSpec) -> None:
        """Build env(s), actor + twin critics (with targets), optimizers, and replay buffer."""
        require_optional("torch", purpose="FastTD3 RL training (strands_robots.training.rl.fast_td3)")
        import torch

        from strands_robots.training.rl.normalization import EmpiricalNormalization
        from strands_robots.training.rl.replay_buffer import SimpleReplayBuffer

        self.spec = spec
        self.device = torch.device(spec.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if spec.seed is not None:
            torch.manual_seed(spec.seed)

        if spec.env_factory is None:  # pragma: no cover - guarded by validate()
            raise ValueError("env_factory is required")
        # num_envs == 1 keeps the single SimEnv path (the FastSAC shape);
        # num_envs > 1 wraps N independent SimEnv in a VecSimEnv that emits
        # (N, D) batches. ``self._vectorized`` selects the collect_rollout path.
        self._vectorized = spec.num_envs > 1
        if self._vectorized:
            from strands_robots.training.rl.vec_env import VecSimEnv

            self.env = VecSimEnv(spec.env_factory, spec.num_envs, device=self.device)
        else:
            self.env = spec.env_factory()
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

        self.actor_norm = EmpiricalNormalization(self.env.num_actor_obs, self.device) if spec.normalize_obs else None
        self.critic_norm = EmpiricalNormalization(self.env.num_critic_obs, self.device) if spec.normalize_obs else None
        self.buffer = SimpleReplayBuffer(
            spec.buffer_size, self.env.num_actor_obs, self.env.num_critic_obs, self.env.num_actions, self.device
        )
        self._obs = self.env.reset()
        self._collected_steps = 0
        # The delayed-update counter: persistent across update() calls so the
        # ``update_count % policy_delay`` cadence spans iterations rather than
        # restarting inside each one.
        self._update_count = 0
        self._ep_return = 0.0
        self._ep_returns_vec = torch.zeros(spec.num_envs, device=self.device)
        self._recent_returns: list[float] = []

    def _norm_actor(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.actor_norm(x, update=update) if self.actor_norm is not None else x

    def _norm_critic(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.critic_norm(x, update=update) if self.critic_norm is not None else x

    def _explore_action(self, actor_obs: torch.Tensor, batch: int) -> torch.Tensor:
        """Collection-time action for a ``(batch, num_actor_obs)`` observation.

        Uniform random over the action interval before ``learning_starts``
        transitions are stored (the exploration warmup); afterwards the
        deterministic actor's action plus clipped Gaussian exploration noise -
        the only exploration a deterministic policy has.
        """
        import torch

        spec = self.spec
        if self.buffer.size < spec.learning_starts:
            return torch.rand(batch, self.env.num_actions, device=self.device) * 2.0 - 1.0
        with torch.no_grad():
            action = self.actor_critic.act_inference(actor_obs)
            noise = torch.randn_like(action) * spec.exploration_noise_std
            return (action + noise).clamp(-1.0, 1.0)

    def collect_rollout(self) -> dict[str, float]:
        """Step the env(s) ``rollout_steps`` times, pushing transitions to the buffer.

        Dispatches to the vectorized path when ``setup`` built a ``VecSimEnv``
        (num_envs > 1), else the single-env path (the FastSAC shape). Both
        store the *terminal* done flag (a time-out truncation is recorded as
        not-done so its value is bootstrapped).
        """
        if getattr(self, "_vectorized", False):
            return self._collect_rollout_vectorized()
        return self._collect_rollout_single()

    def _collect_rollout_single(self) -> dict[str, float]:
        """Single-env collection (num_envs == 1), one transition per tick.

        ``SimEnv`` does not auto-reset, so on ``done`` the stored
        next-observation is already the TRUE terminal observation and the reset
        happens here, after the push - the same ordering FastSAC uses.
        """
        import torch

        # Not an isinstance narrow: the single-env contract is duck-typed (the
        # truncation-contract tests drive it with a scripted fake, exactly as
        # they drive FastSAC's), so the cast records which of the two step
        # shapes this path consumes without refusing a conforming fake.
        env = cast("SimEnv", self.env)
        spec = self.spec
        self.actor_critic.train()
        ep_returns: list[float] = []
        step_rewards: list[float] = []
        for _ in range(spec.rollout_steps):
            actor_obs = self._norm_actor(self._obs["actor_obs"])
            action = self._explore_action(actor_obs, batch=1)
            next_obs, reward, done, info = env.step(action)

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
                self._obs = env.reset()
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

    def _collect_rollout_vectorized(self) -> dict[str, float]:
        """Vectorized collection over a ``VecSimEnv``: N transitions per tick.

        ``VecSimEnv`` auto-resets a done sub-env and returns the FRESH
        post-reset observation in ``next_obs[i]``, stashing the pre-reset
        terminal observation in ``infos[i]["terminal_obs"]`` - so the stored
        next-observation for a done env is pulled back out of the info, the
        same capture the vectorized PPO path bootstraps its next-value from.
        Storing the post-reset observation instead would back the TD target of
        the episode's last action with the value of a fresh episode's first
        state - a bootstrap across the reset boundary that nothing reports.
        """
        import torch

        from strands_robots.training.rl.vec_env import VecSimEnv

        assert isinstance(self.env, VecSimEnv)  # vectorized path
        spec = self.spec
        N = self.env.num_envs
        self.actor_critic.train()
        ep_returns: list[float] = []
        reward_sum = 0.0
        reward_count = 0
        for _ in range(spec.rollout_steps):
            actor_obs = self._norm_actor(self._obs["actor_obs"])  # (N, Da)
            action = self._explore_action(actor_obs, batch=N)  # (N, A)
            next_obs, reward, done, infos = self.env.step(action)  # reward, done (N,)

            for i, info in enumerate(infos):
                term_obs = info.get("terminal_obs")
                if term_obs is not None:
                    next_actor_i = term_obs["actor_obs"]
                    next_critic_i = term_obs["critic_obs"]
                else:
                    next_actor_i = next_obs["actor_obs"][i : i + 1]
                    next_critic_i = next_obs["critic_obs"][i : i + 1]
                terminal = torch.tensor([[float(info["terminated"])]], dtype=torch.float32, device=self.device)
                self.buffer.add(
                    self._obs["actor_obs"][i : i + 1],
                    self._obs["critic_obs"][i : i + 1],
                    action[i : i + 1],
                    reward[i : i + 1],
                    next_actor_i,
                    next_critic_i,
                    terminal,
                )
            self._collected_steps += N
            reward_sum += float(reward.sum().item())
            reward_count += N
            self._ep_returns_vec = self._ep_returns_vec + reward
            done_mask = done.bool()
            if bool(done_mask.any()):
                for i in range(N):
                    if bool(done_mask[i].item()):
                        ep_returns.append(float(self._ep_returns_vec[i].item()))
                        self._ep_returns_vec[i] = 0.0
            self._obs = next_obs

        if ep_returns:
            self._recent_returns = ep_returns
        mean_return = float(sum(ep_returns) / len(ep_returns)) if ep_returns else reward_sum / max(1, N)
        return {
            "mean_reward": reward_sum / max(1, reward_count),
            "mean_episode_return": mean_return,
            "buffer_size": float(self.buffer.size),
        }

    def update(self) -> dict[str, float]:
        """Run ``gradient_steps`` TD3 updates from the replay buffer.

        Each update does a clipped double-Q critic step against a smoothed
        target (clipped Gaussian noise on the target policy's action), and -
        every ``policy_delay``-th step - a delayed deterministic-policy-gradient
        actor step against Q1 followed by a Polyak update of the actor and both
        critic targets. Returns averaged loss metrics; a no-op (zero-loss
        metrics) until the buffer holds at least ``batch_size``.
        """
        import torch
        import torch.nn.functional as F

        spec = self.spec
        if self.buffer.size < spec.batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0, "latest_loss": 0.0}

        tot_critic, tot_actor, n_actor = 0.0, 0.0, 0
        for _ in range(spec.gradient_steps):
            batch = self.buffer.sample(spec.batch_size)
            actor_obs = self._norm_actor(batch["actor_obs"], update=False)
            critic_obs = self._norm_critic(batch["critic_obs"], update=False)
            next_actor_obs = self._norm_actor(batch["next_actor_obs"], update=False)
            next_critic_obs = self._norm_critic(batch["next_critic_obs"], update=False)
            rewards = batch["rewards"]
            dones = batch["dones"]

            # --- critic update: clipped double-Q against a smoothed target ---
            with torch.no_grad():
                noise = (torch.randn_like(batch["actions"]) * spec.target_noise_std).clamp(
                    -spec.target_noise_clip, spec.target_noise_clip
                )
                next_action = (self.actor_critic.act_target(next_actor_obs) + noise).clamp(-1.0, 1.0)
                q_next = self.actor_critic.q_target(next_critic_obs, next_action)
                target_q = rewards + spec.gamma * (1.0 - dones) * q_next
            q1, q2 = self.actor_critic.q_values(critic_obs, batch["actions"])
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            tot_critic += float(critic_loss.item())

            # --- delayed actor + target update (the "Delayed" of TD3) ---
            self._update_count += 1
            if self._update_count % spec.policy_delay == 0:
                q1_pi, _ = self.actor_critic.q_values(critic_obs, self.actor_critic.act_inference(actor_obs))
                actor_loss = -q1_pi.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                tot_actor += float(actor_loss.item())
                n_actor += 1

                # Polyak: every network with a target copy moves together, so
                # the smoothed backup keeps sampling around a slow policy.
                with torch.no_grad():
                    for live, target in (
                        (self.actor_critic.actor, self.actor_critic.actor_target),
                        (self.actor_critic.q1, self.actor_critic.q1_target),
                        (self.actor_critic.q2, self.actor_critic.q2_target),
                    ):
                        for p, tp in zip(live.parameters(), target.parameters()):
                            tp.mul_(1.0 - spec.tau).add_(spec.tau * p)

        g = spec.gradient_steps
        return {
            "critic_loss": tot_critic / g,
            "actor_loss": tot_actor / max(1, n_actor),
            "actor_updates": float(n_actor),
            "latest_loss": tot_critic / g,
        }

    def train(self, spec: TrainSpec) -> TrainResult:
        """Off-policy TD3 loop: setup -> [collect_rollout -> update]* -> save.

        Overrides the on-policy ``BaseRLAlgo.train``. ``spec`` MUST be an
        :class:`RLTrainSpec`; :meth:`validate` is called first and fails closed.
        Updates run only after the buffer passes ``learning_starts``. The env
        built by :meth:`setup` is closed in ``finally`` when the run leaves
        this method - see :meth:`BaseRLAlgo._close_env` for the ownership rule
        and why a later ``evaluate`` on the same instance still works.
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

        try:
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
        finally:
            self._close_env()

    def _checkpoint_dir(self, output_dir: str) -> str:
        return os.path.join(output_dir, "checkpoints", "last")

    def save_checkpoint(self, output_dir: str, iteration: int | None = None) -> str:
        """Save the actor-critic (targets included), normalizers, and policy metadata.

        Writes the same ``policy.pt`` + ``policy_meta.json`` contract as
        ``PpoTrainer`` / ``FastSacTrainer`` so a single checkpoint loader
        serves every RL backend. The target networks travel inside the
        actor-critic state dict, so the shared
        :meth:`BaseRLAlgo.load_checkpoint` restores them with no override.
        """
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
        """FastTD3 on MuJoCo trains fine on CPU; no GPU floor."""
        return {"min_gpus": 0, "min_vram_gb": 0, "multinode": False}
