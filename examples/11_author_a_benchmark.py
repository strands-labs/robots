#!/usr/bin/env python3
"""Author your own benchmark from a spec dict and evaluate a policy on it.

Goal: Show the declarative benchmark DSL - how to define a *new* task (success
condition, failure conditions, dense shaping reward) as a plain dict of named
predicates, with no Python subclass and no ``eval`` of untrusted strings. This
is the same path the built-in benchmarks use, and because the spec is pure data
it is safe to load from JSON/YAML an agent produced.

The task defined here is a harder curriculum variant of the built-in
``go2_walk_forward``: walk the Go2 past 4 m (not 2 m) at a faster 1.5 m/s target,
failing if it tips or its base drops. ``DeclarativeBenchmark.from_dict(spec)``
compiles the dict against the closed predicate registry (an unknown predicate or
a bad argument is rejected at compile time, not at eval time), ``register_benchmark``
adds it to the registry, and ``evaluate_benchmark`` scores it - identical to the
built-ins.

Runs on the ``mock`` policy so it needs no GPU, checkpoint, or hardware; mock
does not actually walk, so success is expected to be false - the point is
authoring and scoring a custom task. Point ``--policy`` at a trained provider to
score it on your task.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: the compiled benchmark's metadata, then its evaluation metrics.
Runtime: ~10 seconds on CPU.
"""

from __future__ import annotations

import argparse

from strands_robots import Robot
from strands_robots.simulation.benchmark import register_benchmark
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark

# The whole task, as data. Every "predicate" name is resolved against the closed
# registry in strands_robots.simulation.predicates - there is no arbitrary code
# here, so a spec like this is safe to load from an untrusted JSON/YAML source.
BENCHMARK_SPEC = {
    "name": "go2_walk_far_fast",
    "instruction": "Walk forward at 1.5 m/s and cover at least 4 meters without tipping.",
    "default_robot": "unitree_go2",
    "supported_robots": ["unitree_go2"],
    "max_steps": 800,
    # SUCCESS: base travels past x = 4 m.
    "success": {"all": [{"predicate": "base_beyond_x", "x": 4.0}]},
    # FAILURE: tips over, or the base collapses below 0.18 m.
    "failure": {
        "any": [
            {"predicate": "base_tipped", "tol": 0.7},
            {"predicate": "base_below_z", "z": 0.18},
        ]
    },
    # DENSE REWARD: track a 1.5 m/s forward twist, hold nominal height + upright.
    "dense_reward": [
        {"predicate": "base_velocity_tracking", "vx": 1.5, "lin_weight": 1.0, "ang_weight": 0.5},
        {"predicate": "base_height", "target": 0.32, "weight": 0.5},
        {"predicate": "base_orientation", "weight": 0.5},
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2, help="number of evaluation episodes")
    parser.add_argument("--policy", default="mock", help="policy provider to score (mock needs no GPU)")
    args = parser.parse_args()

    # Compile the spec: an unknown predicate name or a wrong argument raises here,
    # before anything runs, rather than failing mid-episode.
    benchmark = DeclarativeBenchmark.from_dict(BENCHMARK_SPEC)
    register_benchmark(benchmark.name, benchmark)
    print(f"Registered custom benchmark: {benchmark.name}")
    print(f"  robot       : {benchmark.default_robot}")
    print(f"  instruction : {benchmark.instruction}")

    # Evaluate it exactly like a built-in benchmark.
    sim = Robot(benchmark.default_robot, mesh=False)
    result = sim.evaluate_benchmark(benchmark.name, policy_provider=args.policy, n_episodes=args.episodes)
    if result.get("status") == "error":
        msg = "; ".join(c.get("text", "") for c in result.get("content", []))
        raise RuntimeError(f"evaluate_benchmark failed: {msg}")

    metrics = next((c["json"] for c in result.get("content", []) if "json" in c), {})
    print(f"\n{benchmark.name} with the {args.policy!r} policy:")
    print(f"  success_rate : {metrics.get('success_rate')}")
    print(f"  avg_reward   : {metrics.get('avg_reward')}")
    print(f"  avg_steps    : {metrics.get('avg_steps')}")
    print("\n(mock does not walk, so success_rate is expected to be 0 - the point is that")
    print(" you authored and scored a new task with a dict. Point --policy at a trained")
    print(" locomotion provider to score it for real.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
