---
description: The predicate DSL, stop_when and success_when, rollout observers, declarative benchmark specs and how to read a rollout result.
---

# Predicates and rollouts

By the end of this page you can stop a rollout on a scene condition, score an evaluation with the same vocabulary, watch every applied action through an observer, and register a JSON benchmark.

```python
import json
import tempfile

from strands_robots.simulation import create_simulation
from strands_robots.simulation.predicates import PREDICATE_REGISTRY, make_predicate

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot("so101")
sim.add_object(name="cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.25, 0.0, 0.015])

print(len(PREDICATE_REGISTRY), sorted(PREDICATE_REGISTRY)[:4])
above = make_predicate("body_above_z", body="cube", z=0.1)
near = make_predicate("distance_less_than", body_a="so101/gripper", body_b="cube", threshold=0.15)
print(above(sim), near(sim))

events = []
result = sim.run_policy(
    robot_name="so101", policy_provider="mock", n_steps=200, control_frequency=50.0,
    stop_when={"predicate": "joint_above", "joint": "1", "value": 0.3},
    observer=events.append,
)
print(result["content"][0]["text"].splitlines()[0])
print(type(events[0]).__name__, type(events[-1]).__name__, events[-1].stopped_reason, events[-1].applied_actions)

ev = sim.eval_policy(robot_name="so101", policy_provider="mock", n_episodes=2, max_steps=30,
                     success_when={"predicate": "joint_above", "joint": "1", "value": 0.3})
print(ev["content"][0]["text"].splitlines()[1])

spec = {
    "instruction": "Raise the first joint.",
    "default_robot": "so101",
    "supported_robots": ["so101"],
    "max_steps": 60,
    "success": {"all": [{"predicate": "joint_above", "joint": "1", "value": 0.3}]},
    "failure": {"any": [{"predicate": "body_below_z", "body": "cube", "z": -0.1}]},
    "dense_reward": [{"predicate": "joint_progress", "joint": "1", "target": 0.5, "weight": 1.0}],
}
path = tempfile.mktemp(suffix=".json")
with open(path, "w") as fh:
    json.dump(spec, fh)
sim.register_benchmark_from_file("raise_joint", path)
bench = sim.evaluate_benchmark("raise_joint", policy_provider="mock", n_episodes=2)
print(bench["content"][0]["text"].splitlines()[1])
sim.cleanup()
```

You should see:

```text
30 ['base_ang_vel_xy', 'base_below_z', 'base_beyond_x', 'base_beyond_y']
False False
Policy stopped early (stop_when condition met) on 'so101'
RunPolicyStarted RunPolicyEnded predicate 20
Episodes: 2 | Success: 2/2 (100.0%)
Episodes: 2 | Success: 2 | Failure: 0 (100.0% success)
```

## Predicates

`strands_robots/simulation/predicates.py` is a closed registry of factories: `make_predicate(name, **kwargs)` returns a `(sim) -> bool | float` callable and nothing reaches `eval`, so a clause is safe to accept from an LLM tool call. Each factory checks its keywords and values; a wrong keyword is a `ValueError` naming the accepted ones.

| kind | predicates |
|---|---|
| bool, bodies | `body_above_z`, `body_below_z`, `body_upright`, `body_on`, `body_inside`, `inside_region`, `distance_less_than`, `contact_between`, `contact_any`, `grasped` |
| bool, joints | `joint_above`, `joint_below` |
| bool, particles | `particles_inside`, `particles_spilled` |
| bool, floating base | `base_tipped`, `base_below_z`, `base_beyond_x`, `base_beyond_y`, `base_yaw_beyond` |
| float, rewards | `distance_neg`, `joint_progress`, `particles_inside_fraction`, `base_velocity`, `base_velocity_tracking`, `base_height`, `base_orientation`, `base_lin_vel_z`, `base_ang_vel_xy`, `constant` |
| stateful | `staged_reward` (a phase machine of `{"reward": ..., "advance_when": ..., "bonus": ...}` stages) |

Joint predicates resolve names scene-wide, so a task object loaded with `add_robot(urdf_path=...)` exposes `carton/cap_hinge` to them. `grasped`'s `gripper_prefix` names the gripper body (`so101/gripper`), jaws included; unmatched prefixes are refused. `register_predicate(name, factory)` adds yours.

## stop_when and success_when

`run_policy(stop_when=...)` ends a rollout early on a bool clause: one call `{"predicate": ..., **kwargs}` or an `all` / `any` group. Float-valued terms are rejected (a non-zero float reads as always true), and so is an empty clause. A clause already true at reset is reported as `stop_when_true_at_reset`, not credited to the policy. `eval_policy(success_when=...)` takes the same shape and counts episodes; `success_fn` takes a callable.

## Observers

`run_policy(observer=callable)` receives frozen events: one `RunPolicyStarted` (`run_id`, `policy`, `control_frequency`, `action_horizon`, `total_steps`), one `RunPolicyStep` per applied action (`observation`, `action`, `action_resolution` in `full | partial | none | unknown`, `observation_age_steps`, `sim_time_s`), and exactly one `RunPolicyEnded` (`outcome`, `stopped_reason` in `budget | predicate | cancelled | error`, `applied_actions`, `action_errors`). `observation` and `action` are borrowed; copy what you keep. The observer is telemetry; it cannot remove cancellation or recording.

## Benchmarks

A benchmark spec is a dict or a JSON / YAML file: `name`, `instruction`, `default_robot`, `supported_robots`, `max_steps` (default 300), optional `scene`, `success`, `failure`, `dense_reward`. `register_benchmark_from_file(name, path)` registers it; `register_builtin_benchmarks()` adds the shipped locomotion set `go2_walk_forward`, `go2_strafe_left`, `go2_turn_left`, `g1_walk_forward`, `t1_walk_forward` (`vx`, `vy`, `wz` tracking, the three progress predicates). `list_benchmarks()` lists them; registration is opt-in and importing the package mutates no registry.

`evaluate_benchmark(name, policy_provider, policy_config, n_episodes, seed, ...)` resets to the benchmark's scene each episode, applies its `success` and `failure` clauses, sums `dense_reward` per step, and refuses a robot outside `supported_robots` (`BenchmarkCompatibilityError`).

## Reading a result

The four **posture** flags of `run_policy` and `eval_policy` (`fast_mode`, `reset_between`, `wbc_install_torque_control`, `async_rtc`) select a branch, so a non-boolean is refused rather than read by truthiness: `fast_mode="false"` would otherwise run unpaced. `SimEngine._validate_posture_flags` checks them before any robot is resolved, so a refused call builds no policy and touches no scene; `PolicyRunner.run` does not repeat it.

Every rollout returns `{"status", "content": [{"text"}, {"json"}]}`. The `json` block of `eval_policy` and `evaluate_benchmark` carries `success_rate`, `n_success`, `episodes_completed`, `avg_steps`, `actions_applied`, `stopped_early`, `pass_hat_k` (the probability that `k` consecutive attempts succeed, `k` up to 8, estimated as `C(c, k) / C(n, k)`), inference timing (`avg_inference_ms`, `max_inference_ms`, RTC counters), `policy_load_time_s`, and a per-episode list. Read `pass_hat_k` before deploying: a policy at 60% clears five in a row about 8% of the time.

`seed` reseeds the client RNGs once and derives a per-episode seed from a master RNG, so two evaluations with the same seed replay the same episodes: bit-exact for a state-only policy, to a render tolerance for a camera policy on GPU rendering (`MUJOCO_GL=egl` renders a static scene with 1 LSB differences between frames, so seeded VLA rollouts drift; compare `success_rate`, not frames). `video={"path": "out.mp4", "fps": 30, "camera": "front", "width": 640, "height": 480}` records an MP4 per episode (`fps` capped at control frequency); `video_paths` in the `json` block lists them.
