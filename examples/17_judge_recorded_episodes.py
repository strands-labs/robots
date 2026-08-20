#!/usr/bin/env python3
"""Label recorded episodes with a judge layered on deterministic predicate verdicts.

Goal: Close the labeling gap in automated episode farming. The deterministic
layer (benchmark predicates over simulator state - examples 10/11) is
authoritative for what it can measure: success / failure per the task
definition. A judge - in production a VLM agent reading the recorded frames -
adds what predicates cannot: a quality grade, a failure-mode tag from a fixed
taxonomy, a free-text note. The judge can never overturn a deterministic
verdict; a disagreement is recorded as a ``disputes_verdict`` annotation for
human review. Labels land in a schema-versioned JSON sidecar
(``episode_labels.json``) next to the dataset, so training can filter episodes
(success + high-quality only) without rewriting the dataset.

Pipeline demonstrated end to end: record (with per-episode predicate stops) ->
deterministic verdicts -> judge annotations -> judge/human agreement on a
holdout -> filtered re-training on the judge-approved subset.

This runs a scripted heuristic judge so it needs no model endpoint - it calls
the exact four tools (``load_episode`` / ``sample_frames`` /
``read_predicate_verdict`` / ``write_label``) a VLM agent drives. For a real
VLM, build the agent instead (any strands model provider; a local
OpenAI-compatible endpoint works - no cloud dependency required)::

    from strands.models import BedrockModel  # or any strands model provider
    from strands_robots.tools.episode_judge import create_judge_agent

    judge = create_judge_agent(model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-5-v1:0"))
    judge(f"Label every episode of the dataset at {root}. Use sample_frames "
          "with include_images=True to look at the recording.")

Dependencies: pip install "strands-robots[sim-mujoco,lerobot]"
Expected output: per-episode verdicts, judge labels (one deliberate dispute
showing the verdict stands), an agreement report, the filtered episode subset,
and a 2-step ACT training run on that subset. Runtime: ~2 minutes on CPU
(software rendering; the training step needs the lerobot extra and is skipped
with a note when it is absent).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys

from strands_robots import Robot
from strands_robots.episode_labels import (
    filter_episodes,
    labels_path,
    measure_agreement,
    read_labels,
    record_deterministic_verdicts,
)
from strands_robots.tools.episode_judge import read_predicate_verdict, sample_frames, write_label

# The task, in the same predicate DSL evaluate_benchmark scores with
# (example 11): sweep the arm base past 0.3 rad. run_policy takes the clause
# as a per-episode stop_when gate, so each recorded episode ends at its own
# predicate hit (deterministic success) or at the step budget (failure).
TASK = "sweep the arm base past 0.3 rad"
SUCCESS_CLAUSE = {"all": [{"predicate": "joint_above", "joint": "Rotation", "value": 0.3}]}
BENCHMARK_NAME = "so100_pan_reach"
# Deliberately tight budget: with the mock policy's phase carrying across
# episodes, episode 0 needs ~24 steps and runs out, episodes 1+ need ~9 and
# succeed - so the dataset contains both verdict classes deterministically.
STEP_BUDGET = 15

# The four tools are wrapped by the Strands @tool decorator; the scripted
# judge calls the raw functions - byte-identical to what the agent invokes.
_sample_frames = getattr(sample_frames, "__wrapped__", None) or sample_frames
_read_verdict = getattr(read_predicate_verdict, "__wrapped__", None) or read_predicate_verdict
_write_label = getattr(write_label, "__wrapped__", None) or write_label


def _json_payload(result: dict) -> dict:
    return next((c["json"] for c in result.get("content", []) if "json" in c), {})


def record_dataset(root: str, episodes: int) -> list[dict]:
    """Record ``episodes`` rollouts and return per-episode deterministic verdicts."""
    sim = Robot("so100", mesh=False)
    try:
        started = sim.start_recording(
            repo_id="local/judge_demo",
            root=root,
            fps=50,  # must equal the rollout's control_frequency
            task=TASK,
            cameras=["default"],
            overwrite=True,
        )
        if started.get("status") == "error":
            raise RuntimeError(f"start_recording failed: {started['content'][0]['text']}")

        result = sim.run_policy(
            robot_name="so100",
            policy_provider="mock",
            instruction=TASK,
            n_episodes=episodes,
            n_steps=STEP_BUDGET,
            control_frequency=50.0,
            stop_when=SUCCESS_CLAUSE,
            seed=0,
        )
        if result.get("status") == "error":
            raise RuntimeError(f"run_policy failed: {result['content'][0]['text']}")
        rollouts = _json_payload(result)["episodes"]

        stopped = sim.stop_recording()
        if stopped.get("status") == "error":
            raise RuntimeError(f"stop_recording failed: {stopped['content'][0]['text']}")
    finally:
        sim.destroy()

    # Stage one of the two-stage verdict: success is the predicate's call
    # ("predicate" = the stop_when clause fired; "budget" = it never did),
    # measured on the same rollout the frames came from.
    return [
        {
            "episode": rollout["episode"],
            "success": rollout["stopped_reason"] == "predicate",
            "failure": rollout["stopped_reason"] == "budget",
            "steps": rollout["steps_used"],
        }
        for rollout in rollouts
    ]


def scripted_judge(root: str, episodes: list[int]) -> None:
    """Stage two: annotate each episode through the judge tools.

    A stand-in heuristic where a real deployment points create_judge_agent at
    a VLM - the tool calls (and the sidecar they write) are identical. The
    grade follows the contract JUDGE_SYSTEM_PROMPT states for the VLM:
    quality grades the EXECUTION visible in the recording (here the
    rms_state_jerk smoothness statistic), never the outcome. The verdict
    already carries success/failure, so a grade that re-derives it says
    nothing filter_episodes does not already know - and an unsteered VLM
    drifts into exactly that (measured on a graded ladder, PR #2486 review).
    """
    # Grade smoothness relative to the episode set: these recordings share
    # one policy and one task, so the smoothest episode anchors "high" and
    # an episode is marked down only for measurably rougher execution.
    jerks: dict[int, float | None] = {}
    for episode in episodes:
        frames = _json_payload(_sample_frames(root, episode, n_frames=3))
        # rms_state_jerk is the smoothness statistic (rms third difference);
        # max_state_delta is a peak per-step magnitude for discontinuities.
        jerks[episode] = frames["rms_state_jerk"]
    baseline = min((j for j in jerks.values() if j is not None), default=None)

    for episode in episodes:
        verdict = _json_payload(_read_verdict(root, episode))
        jerk = jerks[episode]
        if jerk is None or baseline is None or baseline == 0.0:
            quality = "medium"  # no usable smoothness ratio: no execution evidence either way
        elif jerk <= 2.0 * baseline:
            quality = "high"
        elif jerk <= 5.0 * baseline:
            quality = "medium"
        else:
            quality = "low"
        if verdict["success"]:
            failure_mode = None
            note = f"reached the target in {verdict.get('steps')} steps"
        else:
            failure_mode = "incomplete"
            note = f"budget exhausted at {verdict.get('steps')} steps without reaching the target"
        if jerk is not None:
            note += f"; rms state jerk {jerk:.1f}"
        result = _write_label(
            root,
            episode,
            quality=quality,
            failure_mode=failure_mode,
            note=note,
            success_opinion=verdict["success"],
            judge_model="scripted-heuristic",
        )
        if result.get("status") == "error":
            raise RuntimeError(f"write_label failed: {result['content'][0]['text']}")
        print(f"  episode {episode}: quality={quality} failure_mode={failure_mode}")


def demonstrate_precedence(root: str) -> None:
    """Show the contract: a disagreeing judge annotates, the verdict stands."""
    before = read_labels(root)["episodes"]["0"]["deterministic"]
    result = _write_label(
        root,
        0,
        quality="low",
        failure_mode="incomplete",
        note="the arm looked close enough on the video - disagreeing on purpose",
        success_opinion=not before["success"],
        judge_model="scripted-heuristic",
    )
    record = _json_payload(result)
    after = read_labels(root)["episodes"]["0"]["deterministic"]
    print(f"  deterministic verdict before: success={before['success']}")
    print(f"  judge success_opinion       : {record['judge']['success_opinion']}")
    print(f"  recorded as                 : disputes_verdict={record['judge']['disputes_verdict']}")
    print(f"  deterministic verdict after : success={after['success']} (unchanged - the judge annotates)")


def train_on_filtered_subset(root: str, output_dir: str, chosen: list[int]) -> None:
    """Re-train on the judge-approved subset without rewriting the dataset."""
    from strands_robots.training import TrainSpec, create_trainer

    trainer = create_trainer("lerobot_local", device="cpu")
    spec = TrainSpec(
        dataset_root=root,
        base_model="",
        output_dir=output_dir,
        steps=2,
        save_freq=2,
        global_batch_size=2,
        extra={
            "policy_type": "act",
            "num_workers": 0,
            # The filter: lerobot's DatasetConfig.episodes, reached through
            # the typed extra passthrough - the dataset itself is untouched.
            "dataset.episodes": chosen,
        },
    )
    problems = trainer.validate(spec)
    if problems:
        raise RuntimeError(f"TrainSpec rejected: {problems}")
    result = trainer.train(spec)
    print(f"  training status: {result.status} (checkpoint: {result.checkpoint_dir})")
    if result.status == "error":
        raise RuntimeError(result.message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3, help="episodes to record and label")
    parser.add_argument("--root", default="/tmp/judge_demo_dataset", help="dataset directory (wiped and re-recorded)")
    parser.add_argument("--output", default="/tmp/judge_demo_ft", help="output directory for the filtered training run")
    parser.add_argument("--skip-train", action="store_true", help="stop after filtering (no lerobot training run)")
    args = parser.parse_args()

    print(f"[1/6] Recording {args.episodes} episodes of '{TASK}' with per-episode predicate stops...")
    verdicts = record_dataset(args.root, args.episodes)
    record_deterministic_verdicts(args.root, verdicts, benchmark=BENCHMARK_NAME)
    for verdict in verdicts:
        print(f"  episode {verdict['episode']}: success={verdict['success']} steps={verdict['steps']}")

    print("\n[2/6] Judge annotations (scripted heuristic standing in for a VLM):")
    scripted_judge(args.root, [v["episode"] for v in verdicts])

    print("\n[3/6] Precedence check - the judge can never overturn the verdict:")
    demonstrate_precedence(args.root)

    # Calibration before trusting the judge to filter training data: compare
    # against a small human-labeled holdout. Here the "human" labels are
    # inline; in a real pipeline a person reviews a handful of episodes.
    print("\n[4/6] Judge/human agreement on a holdout:")
    human_holdout = {
        1: {"quality": "high", "failure_mode": None},
        2: {"quality": "medium", "failure_mode": None},
    }
    agreement = measure_agreement(args.root, human_holdout)
    print(f"  episodes compared      : {agreement['episodes_compared']}")
    print(f"  quality agreement      : {agreement['quality_agreement']:.0%}")
    print(f"  failure-mode agreement : {agreement['failure_mode_agreement']:.0%}")

    print("\n[5/6] Filtering for training (deterministic success + judge quality >= medium):")
    chosen = filter_episodes(args.root, require_success=True, min_quality="medium")
    print(f"  selected episodes: {chosen} (sidecar: {labels_path(args.root)})")
    print(f"  labels on disk   : {json.dumps({k: v['judge']['quality'] for k, v in read_labels(args.root)['episodes'].items() if 'judge' in v})}")
    if not chosen:
        print("  nothing cleared the bar; skipping the training run.")
        return 0

    if args.skip_train:
        print("\n[6/6] --skip-train: would train ACT on the subset via TrainSpec(extra={'dataset.episodes': ...}).")
        return 0
    if importlib.util.find_spec("lerobot") is None:
        print("\n[6/6] lerobot extra not installed; skipping the training run.")
        print('  Install with: pip install "strands-robots[lerobot]" and re-run.')
        return 0

    print(f"\n[6/6] 2-step ACT training run on the judge-filtered subset {chosen}:")
    shutil.rmtree(args.output, ignore_errors=True)
    train_on_filtered_subset(args.root, args.output, chosen)
    print("\nDone. The sidecar travels with the dataset; downstream loaders filter on it")
    print("without rewriting a single parquet shard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
