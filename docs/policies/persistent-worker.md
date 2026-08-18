# Persistent worker - load once, reuse everywhere

Loading a VLA / LeRobot checkpoint is the dominant cost in a multi-episode
rollout: a MolmoAct2 SO-100/101 build reads ~1300 weight files into GPU memory,
on the order of a minute or two. A loop that rebuilds the policy per episode -
the natural shape of an LLM tool loop that re-calls `run_policy` with
`policy_provider=...` - pays that full cost every episode and its resident
memory oscillates as the model is loaded and dropped.

`strands_robots.policies` provides a synchronous persistent worker so the model
is loaded **once** and reused with zero reload across every subsequent rollout.

## The pattern

```python
from strands_robots.policies import PersistentPolicy

# Build the policy once (loads weights, warms the process-level model cache).
policy = PersistentPolicy("lerobot_local", pretrained_name_or_path="allenai/MolmoAct2-SO100_101")

for _ in range(20):
    sim.run_policy(robot_name="so101_follower", policy_object=policy)  # zero reload
    sim.save_episode()
    sim.reset()
```

`PersistentPolicy` is a thin, thread-safe wrapper around any provider. It builds
the underlying policy in its constructor and delegates the full `Policy`
contract transparently - chunk-shape introspection (`execution_horizon`,
`is_chunk_emitting`, `actions_per_step`, `supports_rtc`), the RTC delay /
control-frequency hooks, `reset(seed=...)`, and the load telemetry
(`load_time_s`, `load_cache_hit`) all forward to the wrapped policy, so the
runtime drives it exactly as it would the bare policy. `eval_policy` already
builds the policy once outside its episode loop; `PersistentPolicy` extends the
same guarantee to *separately issued* `run_policy` calls (LLM tool loops,
sequential benchmarks).

## Cache controls

The model cache is process-wide, so even without `PersistentPolicy` two
instances of the same `(checkpoint, device)` share resident weights. The
agent-facing controls make that explicit:

```python
from strands_robots.policies import preload, list_cached, evict

# Warm the cache ahead of a run; reports the load cost and resident memory.
info = preload("lerobot_local", pretrained_name_or_path="allenai/MolmoAct2-SO100_101")
# {'policy': <PersistentPolicy>, 'load_time_s': 84.2, 'load_cache_hit': False,
#  'resident_rss_mb': 14803.1, 'rss_delta_mb': 12911.4}

list_cached()
# [{'namespace': 'molmoact2', 'pretrained_name_or_path': 'allenai/MolmoAct2-SO100_101',
#   'device': 'cuda', 'policy_class': '...'}]

evict("allenai/MolmoAct2-SO100_101")  # free one checkpoint before loading another
evict()                                # free everything
```

`preload(...)["policy"]` is a ready `PersistentPolicy` - pass it straight to
`run_policy(policy_object=...)`.

## Telemetry

`run_policy` and `eval_policy` carry three load-telemetry fields in their JSON
result block so the saving is observable end to end:

| field | meaning |
| --- | --- |
| `policy_load_time_s` | seconds the weight load took, on a monotonic clock (`0.0` on a cache hit) |
| `policy_load_cache_hit` | whether the heavy `from_pretrained` read was skipped |
| `policy_resident_rss_mb` | process RSS in MB at result time (`None` if unmeasurable) |

A `policy_load_cache_hit == False` on episode 2+ of a loop is the smell that the
caller rebuilt the policy instead of reusing `policy_object=` - an agent can
self-correct on it. `policy_resident_rss_mb` staying flat across episodes
confirms the model is resident and not oscillating.

## Memory-safety caveats

The cached object is the SAME live `nn.Module` shared across every wrapper that
requests the same `(checkpoint, device)` key. LeRobot policies hold per-episode
mutable state (action queue, temporal-ensemble buffers) which `Policy.reset()`
clears between episodes, so **sequential** reuse is safe. Two wrappers driving
the same checkpoint **concurrently** would share that state; `PersistentPolicy`
serialises inference behind a per-call lock so two threads reusing one handle
never interleave. If you genuinely need two independent live copies of the same
checkpoint, opt out of sharing with `create_policy(..., cache_model=False)`.

This is an in-process worker: the model and cache live in one Python process.
There is no background daemon or cross-process IPC - inference is GIL- and
GPU-serialised, so the per-call lock gives correct reuse without a separate
worker process. Cross-process sharing is a separate concern.

## See also

- [LeRobot Local](lerobot-local.md) - the in-process HF model provider.
- [Overview](overview.md) - the `Policy` ABC and factory.
