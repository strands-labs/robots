---
description: Train a policy from scratch with reinforcement learning - PPO over the SimEngine env interface, driven by a reward function instead of a dataset.
---

# Reinforcement learning (from scratch)

The [`Trainer`](overview.md) family post-tunes a policy *from a dataset*.
Reinforcement learning is the other half: train a policy *from a reward
function* by interacting with a simulation, with no demonstration data. This is
the only path to a locomotion / whole-body-control policy where no expert
trajectories exist.

RL trainers live in `strands_robots.training.rl` and are selected through the
**same** `create_trainer` factory:

```python
from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec

trainer = create_trainer("ppo")   # on-policy PPO, from-scratch RL
```

## The pieces

| Component | Role |
|---|---|
| [`BaseRLAlgo`](#baserlalgo) | Abstract RL trainer; peer of supervised `Trainer`. Lifecycle `setup -> collect_rollout -> update -> save_checkpoint`. |
| [`RLTrainSpec`](#rltrainspec) | Reward-driven training spec (extends `TrainSpec`). |
| [`PpoTrainer`](#ppo) | Proximal Policy Optimization (on-policy, GAE, clipped surrogate + value). |
| [`FastSacTrainer`](#fastsac) | Soft Actor-Critic (off-policy, replay buffer, twin Q critics, auto-tuned entropy). |
| `SimpleReplayBuffer` | Off-policy transition store (fixed-capacity ring buffer). |
| [`SimEnv`](#simenv) | Gym-style `reset -> step` adapter over a `SimEngine`. |
| `EmpiricalNormalization` | Running observation normalizer (whitens inputs for stable training). Statistics update only in training mode - `eval()` freezes them at both entry points (`forward` and `update`), so an exported policy whitens deterministically. Both entry points want a batched `(batch, num_obs)` tensor and refuse anything else by name; pass `obs.unsqueeze(0)` for a single observation. A refused batch leaves the estimator untouched - the sample count is committed only once the fold succeeds. |

## SimEnv

`SimEnv` wraps a `SimEngine` into the `reset -> step` contract, building the
observation vector from named `get_observation` keys and the step reward from
any reward terms you pass (each a `Callable[[SimEngine], float]`). It uses the holosoma
`actor_obs_keys` / `critic_obs_keys` split: the actor sees only deployable
observations, while the critic may additionally see privileged simulation-only
keys (asymmetric actor-critic).

```python
import strands_robots as sr
from strands_robots.training.rl import SimEnv

TARGET = 0.2

def elbow_reach_reward(engine) -> float:        # a RewardTerm = SimEngine -> float
    elbow = engine.get_observation(skip_images=True)["Elbow"]
    return -abs(float(elbow) - TARGET)

def make_env() -> SimEnv:
    engine = sr.Robot("so100", mode="sim")
    return SimEnv(
        engine,
        actor_obs_keys=["Elbow", "Elbow.vel"],   # what the policy sees
        reward_terms=[elbow_reach_reward],        # dense reward
        action_dim=6,
        max_episode_steps=50,
    )
```

### Numeric arguments

Every numeric `SimEnv` stores is checked at construction, against the shared
domain for its kind, before the engine is read:

| Argument | Domain | Why |
|---|---|---|
| `action_scale` | positive finite | It multiplies every action sent. `0` disconnects the policy from the robot, a negative value inverts every commanded DOF, and `nan`/`inf` make each command unsendable - `send_action` refuses a non-finite action, and `step` discards that status, so the rollout banks its full return having moved nothing. |
| `max_episode_steps` | positive whole number | `0` or below reports a time-out on the first step, so every episode ends before it begins - and reports it as a *truncation*, which on-policy GAE value-bootstraps. |
| `n_substeps` | positive whole number | The action is a position target; the PD controller needs several substeps to track it. This is `send_action`'s own domain for the parameter, which `SimEnv` forwards it to. |
| `action_dim` | positive integer, or `None` | `None` is the documented "size the head from the robot's action keys" spelling. A width of `0` gives the policy no outputs. Narrower than the two above because it sizes the trainers' action head, where an integral float raises rather than being coerced. |

An unusable value raises `ValueError` naming the class and the argument
(`SimEnv: action_scale must be > 0, got 0.0.`). Each domain is the one its
consumer can honor, so nothing is refused here that the code downstream accepts:
a fractional scale (`0.25`) and a scale read from a config array
(`np.float32(0.25)`) are fine, and so is a whole-number count spelled `50.0` or
`np.int64(50)` - all normalized to the plain `float`/`int` the attribute
advertises.

## PPO

```python
from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec

trainer = create_trainer("ppo")
spec = RLTrainSpec(
    env_factory=make_env,          # a zero-arg callable returning a SimEnv
    output_dir="/tmp/ppo_reach",
    total_timesteps=250 * 150,
    rollout_steps=250,             # on-policy batch horizon per update
    num_mini_batches=4,
    num_learning_epochs=5,
    learning_rate=1e-3,
    gamma=0.99, lam=0.95, clip_param=0.2,
    init_noise_std=0.8,
    seed=0,
)

problems = trainer.validate(spec)  # pure preflight (no side effects)
assert not problems
result = trainer.train(spec)       # setup -> (collect_rollout -> update)* -> save
print(result.metrics)              # mean_reward, mean_episode_return, surrogate_loss, value_loss
```

`train()` writes a checkpoint under `output_dir/checkpoints/last/`:

- `policy.pt` - the actor-critic + observation-normalizer state (the loadable
  artifact returned by `result.exported_model`).
- `policy_meta.json` - deployable-policy metadata: `num_actions`,
  `actor_obs_keys`, `action_keys`, `hidden_dims`.

`action_keys` names what the `num_actions` outputs drive, in order, and is the
robot's `robot_action_keys` - the vocabulary `send_action` binds an action
vector against. It is not the joint list: a tendon-driven gripper is one
actuator over two finger joints (so a Panda has 9 joints and 8 action keys), and
the Newton backend's floating base is a joint with no commandable scalar. `SimEnv`
sizes `num_actions` from the same list, so `len(action_keys) == num_actions`
always holds. Pass `action_dim` to `SimEnv` to override the width.

`PpoTrainer` trains fine on CPU (its `hardware_floor` declares no GPU
requirement); MuJoCo stepping dominates, not the network.

### Device selection

The learner (actor-critic, normalizers, rollout buffers) is placed on
`RLTrainSpec.device`, defaulting to `cuda` when available and `cpu` otherwise.
The learner device is authoritative: on a GPU host `setup()` reconciles the
`SimEnv` onto it so observations, rewards, and dones are built on the same
device as the network (no cross-device tensor mismatch and no per-step
host-to-device copies). Pass `device="cpu"` explicitly to keep everything on
CPU even on a GPU machine.

`validate()` refuses a `device` no torch build can parse, before `setup()` builds
anything, on the same domain the `lerobot_train` tool and `LerobotTrainer` apply
to the value they hand to lerobot. Only the *spelling* is graded, never
availability: `device="cuda"` on a CPU-only host is a valid spec, because a
queued or containerised run is written on one machine and executed on another. A
non-string is refused without consulting torch at all -- `torch.device(1)`
constructs on any host and then fails at the first `.to()` with `CUDA error:
invalid device ordinal`, so asking torch about an ordinal would make one spec
launch on a multi-GPU box and abort on a single-GPU one. Leaving `device` unset
(or falsy) still selects the documented default above.

## FastSAC

`FastSacTrainer` is the **off-policy** trainer: it keeps a replay buffer of past
transitions and reuses each one across many gradient steps, so it reaches a
target in far fewer environment steps than on-policy PPO (at the cost of more
compute per step). It trains a tanh-squashed Gaussian actor and twin Q critics
(clipped double-Q) with Polyak-averaged target critics and an automatically
tuned entropy temperature, and writes the **same** `policy.pt` +
`policy_meta.json` checkpoint as PPO.

```python
from strands_robots.training import create_trainer
from strands_robots.training.rl import RLTrainSpec

trainer = create_trainer("fast_sac")
spec = RLTrainSpec(
    env_factory=make_env,          # same SimEnv contract as PPO
    output_dir="/tmp/fastsac_reach",
    total_timesteps=50 * 80,
    rollout_steps=50,              # env steps collected per iteration
    learning_starts=500,           # random-action warmup before the first update
    batch_size=256,                # transitions sampled per gradient step
    gradient_steps=50,             # SAC updates per iteration
    buffer_size=50_000,            # replay-buffer capacity
    learning_rate=3e-4,
    gamma=0.99, tau=0.01,          # discount + Polyak target-critic coefficient
    seed=0,
)
result = trainer.train(spec)       # setup -> (collect_rollout -> update)* -> save
print(result.metrics)              # mean_reward, critic_loss, actor_loss, alpha, entropy
```

The off-policy fields on `RLTrainSpec` (`buffer_size`, `batch_size`,
`learning_starts`, `gradient_steps`, `tau`, `autotune_alpha`, `init_alpha`,
`alpha_lr`, `target_entropy`) are read only by SAC; on-policy PPO ignores them.
`target_entropy` defaults to `-num_actions` (the SAC heuristic) when left
`None`. Like PPO, `FastSacTrainer` trains fine on CPU.

The three fields of the temperature block are each preflighted, on two different
domains. `init_alpha` (the temperature's starting value) and `alpha_lr` (the rate
that moves it) must be positive and finite, because the first reaches
`torch.log` and the second is an optimizer's learning rate. `target_entropy` -
the constant the temperature is optimized *toward* - is instead any **finite real
of either sign**, checked against the same shared domain as the on-policy loss
weights: the field defaults to `-num_actions`, so every reading of it is negative
and no endpoint is decidable, while a value that is not a finite real has no
reading at all. Over a 40-timestep run, `validate()` used to return `[]` for all
of them: `nan`, `inf` and `-inf` made `alpha_loss` non-finite, and since `alpha`
scales the entropy term of both the critic's TD target and the actor loss, the
next rollout raised `ValueError` from inside `torch.distributions.Normal` about a
tensor of `nan` policy means - naming that distribution's parameter rather than
the field, after the env, both networks and a full rollout had been built.
`target_entropy=True` was worse than a crash: `bool` is an `int` subclass, so it
is a target entropy of `+1.0` where the field's own default is `-6.0` for that
env, and the run reported `status="success"` while checkpointing
`log_alpha == -0.0018031001091003418` against the honored run's
`-0.001801646314561367` - it demonstrably drove the temperature somewhere else. A
list or a dict raised `TypeError` out of the `float()` coercion in `setup`.

`None` is exempt from that domain rather than refused by it: unlike `init_alpha`
and `alpha_lr`, which are annotated `float` with concrete defaults, this field is
annotated `float | None` and its `None` is the documented request for the
heuristic. And the check is not conditioned on `autotune_alpha` even though only
the tuning branch spends the value, because the coercion that reads it is
unconditional - with tuning off a list raises the same `TypeError`, while `nan`
reaches a successful run that simply never spends it.

## BaseRLAlgo

`BaseRLAlgo` is the abstract RL trainer - a `Trainer` subclass, so RL flows
through the same `create_trainer` / `validate` / `export` contract while adding
the RL lifecycle hooks `setup`, `collect_rollout`, `update`, and
`save_checkpoint`. The default `train()` runs the standard on-policy loop over
those hooks; an off-policy algorithm overrides `train()` with a replay-buffer
loop while keeping the same hooks and checkpoint format.

`evaluate(spec=None, checkpoint_dir=None, num_episodes=10)` is the eval peer of
`train()`: it rolls out the deterministic (mean) action with gradients disabled
and observation normalization frozen, so its numbers are what a deployed
`policy.pt` would produce. It returns `num_episodes`, `mean_return`,
`std_return`, `min_return`, `max_return`, `mean_length`, `success_rate` (the
fraction of episodes that ended on a genuine terminal via the env's
`success_fn`, not a time-out) and the per-episode `returns`.

`num_episodes` must be a positive integer, checked against the same shared count
domain as `total_timesteps` / `rollout_steps` / `num_envs`: it is the `range()`
bound of the episode loop, the denominator of the reported `success_rate`, and is
echoed back as the `num_episodes` field, so a value that is not a count cannot be
honored by any of the three. `evaluate()` is side-effect-free with respect to
train/eval mode on **every** exit, including a raising one - the actor-critic and
the observation normalizer are returned to the mode they were in before the call,
so a `train -> evaluate -> train` continuation resumes with the running
statistics still learning.

## RLTrainSpec

`RLTrainSpec` extends `TrainSpec`. RL ignores the dataset fields
(`dataset_root` etc.) and reads `env_factory`, `total_timesteps`,
`rollout_steps`, `num_envs`, the PPO hyperparameters (`gamma`, `lam`,
`clip_param`, `num_learning_epochs`, `num_mini_batches`, `entropy_coef`,
`value_loss_coef`, `max_grad_norm`, `hidden_dims`, `init_noise_std`), the
off-policy SAC fields (`buffer_size`, `batch_size`, `learning_starts`,
`gradient_steps`, `tau`, `autotune_alpha`, `init_alpha`, `alpha_lr`,
`target_entropy`), plus the universal `output_dir` / `learning_rate` / `seed`.

`total_timesteps` and `rollout_steps` must each be a positive integer, checked by
`validate()` on both backends against the same shared domain the supervised
backends use for `steps` / `global_batch_size`. They are the two caller-supplied
factors of the one loop bound each backend derives -
`num_iters = max(1, total_timesteps // (rollout_steps * num_envs))`, iterated as
`range(num_iters)` - and because the bound is *derived* through that `max(1, ...)`
clamp, a value that is merely "not non-positive" is not enough. Over a 16-step run
with `rollout_steps=4`, `True`, `0.5`, `nan` and `inf` each used to report
`status="success"` after exactly **one** iteration, writing a checkpoint and
announcing `"1 iterations x 4 steps complete"` for a run asked to be tens of
thousands of steps long (`inf // 4` is `nan`, and `max(1, nan)` is `1` - `nan`
compares false against everything). A fraction that does divide -
`100.5 // 4 == 25.0` - instead raised a bare `TypeError` out of `range()` after
the environment, the networks, the optimizers and (FastSAC) the replay buffer had
been built, and a string or `None` raised out of the comparison itself, from a
preflight documented to *return* problems. `rollout_steps=True` was worse than a
short run because it changes the run's shape rather than its length: it makes
`steps_per_iter` one, so FastSAC ran 16 single-step iterations instead of 4 of 4,
and PPO normalized advantages over a length-one batch - whose standard deviation
is `nan` - and failed inside torch's `Normal` constraint.

`num_envs`, the third factor of `steps_per_iter`, is checked by each backend
separately rather than against that shared domain, because the accepted sets
differ: PPO parallelizes and accepts any count `>= 1`, while the MuJoCo-backed
FastSAC is single-env and requires exactly `1`.

`hidden_dims` must be a sequence of positive integer layer widths, checked by
`validate()` on all three RL backends - each builds every network it trains by
expanding the same field (the on-policy actor and critic; off-policy the actor,
its Polyak target and all four Q heads). Nothing judged the widths, and
`nn.Linear` does not either: a width of **zero** is a legal layer. `torch` only
warns ("Initializing zero-element tensors is a no-op"), the layer emits an empty
activation, and the layer after it therefore emits its bias alone - so the
network's output stops being a function of the observation. Measured over a full
`train()` on each backend, `hidden_dims=(16, 0)` returned a **bit-identical
action** for an all-zero observation and an all-`50` one, while `train()`
reported `status="success"` with a real `actor_loss` and exported
`policy.pt` + `policy_meta.json`: a deployable checkpoint whose actor commands
one fixed action in every state the robot can reach. A negative or non-integer
width instead raised out of `torch` after the environment, the networks and the
optimizers were built, and `np.int64` trained to completion and then lost the run
at the save, because `save_checkpoint` JSON-encodes the field.

The **empty** sequence stays accepted: it is the honest spelling of a linear
policy (input straight to the output layer), and its action still varies with the
observation. So the domain is per element rather than on the length, and a
problem names the offending index (`hidden_dims[1] must be a positive integer`).
A value that is not a sequence of widths at all - a bare `int`, `None`, a `str`,
or a one-shot generator, which would be consumed by the first network built and
leave every later critic a different shape - is refused as a whole.

`gamma` must be a finite number in the closed interval `[0, 1]`, checked by
`validate()` on both backends. It is the one coefficient both of them read (PPO
discounts the GAE recursion with it, FastSAC its target-Q bootstrap) and a
discounted return is a geometric series, so a value above 1 makes that series
diverge in the rollout horizon rather than merely being large - the run would
train on the inflated advantages, report success and write a checkpoint. Both
endpoints are inside the domain: `gamma=1` is the undiscounted episodic return
and `gamma=0` a myopic agent that optimizes the immediate reward only. The
FastSAC `tau` is bounded the same way, in `(0, 1]`.

`num_learning_epochs` must be a positive integer, checked by `validate()` on the
on-policy backend only (FastSAC optimizes per gradient step from a replay buffer
and has no epoch loop). It is the loop bound of the entire optimizer step, so a
non-positive value takes *no* gradient step while the run still collects its
rollouts, writes a checkpoint and reports success - with losses of `0.0`,
because the update averages its accumulators through `max(1, n_updates)`. `True`
is likewise a silent single epoch, and a non-integer raises a bare `TypeError`
out of `range()` after the environment and the networks are already built.

FastSAC builds two optimizers from two learning-rate fields - `learning_rate`
for the actor and both critics, `alpha_lr` for the entropy temperature - so
when `autotune_alpha` is set `alpha_lr` must be a positive finite number too,
checked by the same `validate()`. `alpha_lr=0` builds the temperature
optimizer and never moves it, so the temperature stays at `init_alpha` and the
automatic tuning the spec asked for silently does not happen; `inf` sends it
to an infinity on the first step, and because the temperature multiplies the
log-probability in the actor loss the resulting checkpoint holds non-finite
parameters. Both previously reported success. It is inert when
`autotune_alpha=False`, which builds no temperature optimizer.

`init_alpha` - the temperature that rate moves - must be a positive finite
number, checked by the same `validate()`. FastSAC stores the temperature's
*logarithm* (`log_alpha = log(init_alpha)`, with `alpha = exp(log_alpha)`
scaling the entropy term in both the critic target and the actor loss), so only
a positive finite value has a usable logarithm. `init_alpha=0` makes
`log(0) = -inf` and the temperature exactly `0`, which removes the entropy term
from both losses - the run is no longer the maximum-entropy algorithm that was
asked for - and no finite gradient step moves an infinity, so `autotune_alpha`
cannot recover it; the run previously reported success and checkpointed
`log_alpha = -inf`, so the unusable temperature outlived it. `True` was a silent
temperature of exactly `1.0`, and a negative value, `nan` or `inf` poisoned the
actor loss and raised `ValueError` from `torch.distributions.Normal` about a
tensor of `nan` policy means - naming that distribution's parameter rather than
the field, and only once `train()` was already under way. Unlike `alpha_lr`
this check is **not** conditioned on `autotune_alpha`: the field is read on both
branches, and with tuning off it is the temperature for the whole run.

`max_grad_norm` must be a positive number a 64-bit float can represent, checked
by `validate()` on the on-policy backend only (`clip_grad_norm_` appears in
`rl/ppo.py` and nowhere else). It is the last thing that touches a gradient
before the optimizer steps, and `clip_grad_norm_` scales every gradient by
`max_norm / total_norm` without judging the bound, so two values used to be
honored silently and wrongly:
`max_grad_norm=0` scaled every gradient to zero, and the run reported success
with a checkpoint bit-identical to a never-trained one; a **negative** bound
negated the scaling ratio, so the update became gradient *ascent* on the loss
and moved the parameters away from the objective, also under a successful run.
`True` was a silent bound of one and `"1.0"` was silently coerced, while `nan`,
`None` and a list raised from inside `torch` mid-update.

`inf` is inside the domain: it is the field's only spelling of *do not clip*, and
`clip_grad_norm_` honors it by leaving every gradient untouched. A real that no
64-bit float stands for is refused with the range as its reason rather than the
sign, and `validate()` reports it like any other unusable bound instead of
raising: Python integers are arbitrary-precision, so `10**400` is one request
away.

`clip_param` must be a positive number too, checked by the same `validate()` on
the on-policy backend only (`spec.clip_param` appears in `rl/ppo.py` and nowhere
else). It is the half-width of the trust region PPO is named for, read twice per
mini-batch - once to clip the policy ratio and once to clip the value loss - and
`torch.clamp` judges it not at all, so every unusable value produced a finite,
successful, deployable run whose objective was not the configured one:

- `nan` **silently removes the trust region.** Both clipped terms become `nan`,
  so `torch.max(surrogate, surrogate_clipped)` returns `nan` - but its gradient
  flows to the *unclipped* branch, because every comparison against `nan` is
  false. Measured over a seeded 60-step run, the resulting checkpoint is
  bit-identical to an unclipped one (parameter sum `140.1735330768706262`) while
  `surrogate_loss`, `value_loss` and `latest_loss` are all reported as `nan`.
- A **negative** half-width is not a window: `1 - c` exceeds `1 + c`, so the
  clamp bounds are inverted and it returns a constant regardless of the ratio,
  and the reported surrogate loss changes sign. `0` is the same failure at the
  boundary - the value clip becomes `clamp(-0, 0)`, so `value_clipped` is exactly
  `old_values` and the critic's clipped branch is a constant.
- `True` was a silent half-width of one, five times the shipped `0.2`, and
  `"0.2"`, `None` and a list raised `TypeError` from inside the update loop.

`inf` is inside this field's domain for the same reason as `max_grad_norm`'s -
`clamp(ratio, -inf, inf)` returns the ratio unchanged, so it is the field's only
spelling of *do not clip* - and the two bounds share one domain helper rather
than a copy each.

`log_interval` must be a whole number of iterations, checked by `validate()` on
all three backends. **It is the RL run's checkpoint cadence, not a logging one**:
no RL module emits a progress log, and the field is read in exactly one
expression - `it % log_interval == 0` - which decides whether `save_checkpoint`
runs for that iteration. So it answers for an RL run what
[`TrainSpec.save_freq`](overview.md) answers for a supervised one
(RL reads this field and ignores that one), and it takes the same shared
step-cadence domain, because the modulus judges the value no more than lerobot's
does. Measured on the loop over 20 iterations, against the `[1, 6, 11, 16, 20]`
an `int` cadence of `5` writes:

| `log_interval` | Checkpoints written | Reading |
|---|---|---|
| `5` | 5, at 1/6/11/16/20 | the requested schedule |
| `True` | **20** | a modulus of one - one every iteration |
| `2.5` | 5, at 1/6/11/16/20 | the schedule of `5`, not of `2.5` |
| `0.3` | **2**, at 1 and 20 | the periodic checkpoints are gone |
| `nan` | **1**, at 20 | silently the *disabled* mode |
| `inf` | **2**, at 1 and 20 | ditto |
| `"5"` | **0** | `TypeError` out of `train()`, after `setup` |
| `0` | 1, at 20 | the supported "no intermediate checkpoints" mode |
| `-5` | 5, at 1/6/11/16/20 | the cadence of its magnitude |

`nan` is the worst of them: it satisfies the truthiness guard, never satisfies
the modulus, and so a long run that asked to checkpoint every few iterations
keeps only its final one - under `status="success"`. For RL those intermediate
checkpoints are not a convenience: return is non-monotonic in training, so the
deployable policy is often an earlier iteration, and a run that silently kept
only its last cannot be recovered without training again.

Only the *type* is graded, so this domain has no floor: `0` disables the periodic
checkpoints and leaves the end-of-run fallback to write exactly one final
checkpoint. A negative is accepted for the same reason and is *not* the disabled
mode here - unlike lerobot's `save_freq > 0` test, this loop guards on bare
truthiness - so which spelling disables is the loop's business, not the domain's.

## Worked example

`examples/training/train_ppo_reach.py` (on-policy) and `examples/training/train_fastsac_reach.py`
(off-policy) both train the SO-100 `Elbow` joint to a target angle in MuJoCo
from scratch and print the resulting checkpoint path. The MuJoCo backend is
single-environment (`num_envs == 1`); vectorized backends for
massively-parallel rollouts are tracked separately.


## Result

PPO trained from scratch on CPU (no dataset, reward only) closes the reach
gap over 150 iterations and the deterministic policy drives the `Elbow` joint
to the target:

![PPO reach learning curve](../assets/ppo_reach_curve.png)

![PPO reach rollout](../assets/ppo_reach_demo.gif)

FastSAC (off-policy) reaches the same target in far fewer environment steps,
reusing replayed transitions; the deterministic policy drives the `Elbow`
joint onto the target (0.19 rad vs. a 0.20 target):

![FastSAC reach learning curve](../assets/fastsac_reach_curve.png)

![FastSAC reach rollout](../assets/fastsac_reach_demo.gif)
