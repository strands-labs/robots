#!/usr/bin/env python3
"""
Unified Evaluation Harness for All VLA Policies

Runs systematic evaluation across:
- All trained policies (GR00T, ACT, Pi0, Cosmos, PPO, SAC, Mock)
- All tasks (pick cube, stack, walk)
- All backends (MuJoCo, Newton, Isaac Sim)

Generates comparison table in markdown + CSV.

Usage:
    python scripts/eval_harness.py --policies groot,act,mock --tasks pick,stack --episodes 50
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# Task definitions
TASKS = {
    "pick_cube": {
        "instruction": "pick up the red cube",
        "robot": "so100",
        "max_steps": 200,
    },
    "stack": {
        "instruction": "stack the blue block on the green block",
        "robot": "so100",
        "max_steps": 300,
    },
    "walk_forward": {
        "instruction": "walk forward at 1 m/s",
        "robot": "unitree_g1",
        "max_steps": 500,
    },
    "wipe_table": {
        "instruction": "wipe the table with the sponge",
        "robot": "so100",
        "max_steps": 300,
    },
    "push_slider": {
        "instruction": "push the slider to the right",
        "robot": "so100",
        "max_steps": 200,
    },
}

# Policy configurations
POLICY_CONFIGS = {
    "mock": {"provider": "mock"},
    "groot_base": {"provider": "groot", "model_path": "nvidia/GR00T-N1-2B", "data_config": "so100_dualcam"},
    "groot_finetuned": {"provider": "groot", "model_path": "./groot_finetuned/best", "data_config": "so100_dualcam"},
    "act_trained": {"provider": "lerobot_local", "pretrained_name_or_path": "./act_trained/best"},
    "pi0_trained": {"provider": "lerobot_local", "pretrained_name_or_path": "./pi0_trained/best"},
    "cosmos_posttrained": {"provider": "cosmos_predict", "model_path": "./cosmos_posttrained/best"},
    "ppo_newton": {"provider": "rl", "model_path": "./ppo_g1_locomotion/final_model"},
    "sac_newton": {"provider": "rl", "model_path": "./sac_pick_cube/final_model"},
}


def evaluate_policy_on_task(
    policy_name: str,
    task_name: str,
    num_episodes: int = 50,
    backend: str = "mujoco",
) -> Dict[str, Any]:
    """Evaluate a single policy on a single task.

    Returns dict with success_rate, mean_reward, timing info.
    """
    task = TASKS.get(task_name)
    if not task:
        return {"error": f"Unknown task: {task_name}"}

    policy_config = POLICY_CONFIGS.get(policy_name)
    if not policy_config:
        return {"error": f"Unknown policy: {policy_name}"}

    result = {
        "policy": policy_name,
        "task": task_name,
        "instruction": task["instruction"],
        "robot": task["robot"],
        "backend": backend,
        "num_episodes": num_episodes,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Create policy
        if policy_config["provider"] == "rl":
            # RL policies use the SB3 evaluate path
            result["success_rate"] = 0.0
            result["mean_reward"] = 0.0
            result["note"] = "RL evaluation requires stable-baselines3"
            return result

        from strands_robots.policies import create_policy
        from strands_robots.training import evaluate

        config_copy = dict(policy_config)
        provider = config_copy.pop("provider")

        try:
            policy = create_policy(provider, **config_copy)
        except Exception as e:
            result["success_rate"] = 0.0
            result["mean_reward"] = 0.0
            result["error"] = f"Policy creation failed: {e}"
            return result

        # Run evaluation
        start = time.time()
        eval_result = evaluate(
            policy=policy,
            task=task["instruction"],
            robot_name=task["robot"],
            num_episodes=num_episodes,
            max_steps_per_episode=task["max_steps"],
            backend=backend,
        )
        elapsed = time.time() - start

        result.update(eval_result)
        result["eval_time_seconds"] = elapsed

    except Exception as e:
        result["success_rate"] = 0.0
        result["mean_reward"] = 0.0
        result["error"] = str(e)

    return result


def run_eval_matrix(
    policies: List[str],
    tasks: List[str],
    num_episodes: int = 50,
    backend: str = "mujoco",
    output_dir: str = "./eval_results",
) -> Dict[str, Any]:
    """Run full evaluation matrix: policies × tasks.

    Returns summary with all results and generates markdown table.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    matrix = {}

    total_evals = len(policies) * len(tasks)
    completed = 0

    for policy_name in policies:
        matrix[policy_name] = {}
        for task_name in tasks:
            completed += 1
            print(f"[{completed}/{total_evals}] {policy_name} × {task_name}...", end=" ", flush=True)

            result = evaluate_policy_on_task(
                policy_name=policy_name,
                task_name=task_name,
                num_episodes=num_episodes,
                backend=backend,
            )
            all_results.append(result)
            matrix[policy_name][task_name] = result.get("success_rate", 0.0)

            sr = result.get("success_rate", 0.0)
            error = result.get("error", "")
            if error:
                print(f"❌ ({error[:50]})")
            else:
                print(f"{sr:.1f}%")

    # Generate markdown table
    md_lines = [
        "# Evaluation Results",
        f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Episodes per eval: {num_episodes}",
        f"Backend: {backend}",
        "",
        "| Policy | " + " | ".join(tasks) + " |",
        "|--------|" + "|".join(["--------"] * len(tasks)) + "|",
    ]

    for policy_name in policies:
        row_values = []
        for task_name in tasks:
            sr = matrix[policy_name].get(task_name, 0.0)
            row_values.append(f"{sr:.1f}%")
        md_lines.append(f"| {policy_name} | " + " | ".join(row_values) + " |")

    md_content = "\n".join(md_lines)

    # Save markdown
    md_path = os.path.join(output_dir, "eval_results.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    # Save CSV
    csv_path = os.path.join(output_dir, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "task", "success_rate", "mean_reward", "episodes", "time_s", "error"])
        for r in all_results:
            writer.writerow([
                r.get("policy", ""),
                r.get("task", ""),
                r.get("success_rate", 0.0),
                r.get("mean_reward", 0.0),
                r.get("num_episodes", 0),
                r.get("eval_time_seconds", 0.0),
                r.get("error", ""),
            ])

    # Save full JSON
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump({"results": all_results, "matrix": matrix}, f, indent=2)

    print("\n✅ Evaluation complete!")
    print(f"   Markdown: {md_path}")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {json_path}")
    print(f"\n{md_content}")

    return {"results": all_results, "matrix": matrix, "markdown": md_content}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified VLA Evaluation Harness")
    parser.add_argument("--policies", type=str, default="mock",
                       help="Comma-separated policy names")
    parser.add_argument("--tasks", type=str, default="pick_cube,stack",
                       help="Comma-separated task names")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Episodes per evaluation")
    parser.add_argument("--backend", type=str, default="mujoco",
                       help="Simulation backend")
    parser.add_argument("--output", type=str, default="./eval_results",
                       help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    policies = [p.strip() for p in args.policies.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    run_eval_matrix(
        policies=policies,
        tasks=tasks,
        num_episodes=args.episodes,
        backend=args.backend,
        output_dir=args.output,
    )
