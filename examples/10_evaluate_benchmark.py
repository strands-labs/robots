#!/usr/bin/env python3
"""Register the built-in locomotion benchmarks and evaluate a policy on one.

Goal: Show the evaluation loop - how to score a policy on a task and read back
success rate and reward, not just watch it move. ``register_builtin_benchmarks()``
adds five ready-to-run velocity-tracking locomotion tasks (Go2 walk / strafe /
turn, G1 walk, T1 walk); ``list_benchmarks()`` reports what is registered and
which robot each one targets; ``evaluate_benchmark(...)`` runs N episodes and
returns a metrics dict (``success_rate``, ``avg_reward``, ``avg_steps``, plus a
per-episode breakdown).

This uses the ``mock`` policy so it runs anywhere - no GPU, no checkpoint, no
hardware. A mock policy will not actually walk, so expect a low success rate;
the point is the eval harness and the numbers it produces. Swap
``policy_provider`` (and a real checkpoint via ``policy_config``) to score a
trained locomotion policy on the identical benchmark.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: the registered benchmark table, then a metrics line per episode
set. Runtime: ~10 seconds on CPU.
"""

from __future__ import annotations

import argparse

from strands_robots import Robot
from strands_robots.simulation import register_builtin_benchmarks
from strands_robots.simulation.benchmark import list_benchmarks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="go2_walk_forward", help="benchmark to evaluate")
    parser.add_argument("--episodes", type=int, default=2, help="number of evaluation episodes")
    parser.add_argument("--policy", default="mock", help="policy provider to score (mock needs no GPU)")
    args = parser.parse_args()

    # Register the five built-in locomotion benchmarks (idempotent).
    register_builtin_benchmarks()

    # What is available, and which robot each benchmark targets.
    print("Registered benchmarks:")
    registry = list_benchmarks()
    for name, meta in registry.items():
        print(f"  {name:20} robot={meta['default_robot']:14} max_steps={meta['max_steps']}")

    if args.benchmark not in registry:
        raise SystemExit(f"unknown benchmark {args.benchmark!r}; choose one of {sorted(registry)}")

    # Spawn the benchmark's target robot and evaluate. evaluate_benchmark builds
    # the task scene, runs the policy for up to max_steps per episode, scores the
    # success predicate, and accumulates the dense reward.
    robot_name = registry[args.benchmark]["default_robot"]
    sim = Robot(robot_name, mesh=False)

    result = sim.evaluate_benchmark(
        args.benchmark,
        policy_provider=args.policy,
        n_episodes=args.episodes,
    )
    if result.get("status") == "error":
        msg = "; ".join(c.get("text", "") for c in result.get("content", []))
        raise RuntimeError(f"evaluate_benchmark failed: {msg}")

    metrics = next((c["json"] for c in result.get("content", []) if "json" in c), {})
    print(f"\n{args.benchmark} on {robot_name} with the {args.policy!r} policy:")
    print(f"  episodes      : {metrics.get('episodes_completed')}")
    print(f"  success_rate  : {metrics.get('success_rate')}")
    print(f"  avg_reward    : {metrics.get('avg_reward')}")
    print(f"  avg_steps     : {metrics.get('avg_steps')}")
    print("\n(The mock policy does not walk, so success_rate is expected to be low - the")
    print(" point is the eval harness. Point --policy at a trained provider to score it.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
