#!/usr/bin/env python3
"""End-to-end VLA-on-G1 workflow: record -> fine-tune -> deploy.

Chains the three stages of the humanoid VLA pipeline on the Unitree G1:

1. RECORD  - drive the G1 in sim, capture a LeRobotDataset (teleop data).
2. TUNE    - post-train Isaac-GR00T N1.7 on the recorded data (optional/gated).
3. DEPLOY  - deploy the (fine-tuned or pre-trained) checkpoint with WBC
             (SONIC whole-body control) for locomotion.

Each stage is self-contained and gated:
- By default only stages 1 + 3 run (record + deploy with a mock/pre-trained
  checkpoint). This completes in ~10 seconds on CPU with no external services.
- Pass ``--tune`` to enable stage 2 (requires Docker + a GPU for Isaac-GR00T
  fine-tuning; takes ~hours). The deploy stage then uses the fine-tuned output.
- Pass ``--checkpoint /path/to/GEAR-SONIC`` to skip recording + fine-tuning and
  jump straight to deploy with an existing SONIC checkpoint.

This example proves the three pieces compose - dataset_recorder, GR00T Trainer,
and WBCPolicy - as one coherent pipeline, the deploy stage of issue #471.

Upstream reference:
    https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html

Dependencies:
    pip install "strands-robots[sim-mujoco,lerobot,wbc]"
    # For stage 2 (fine-tuning): Docker + GPU + pip install "strands-robots[groot-service]"

Usage:
    # Quick demo (record + deploy with mock policy, ~10s):
    python examples/vla_g1_workflow.py

    # Full pipeline with real fine-tuning:
    python examples/vla_g1_workflow.py --tune --base-model nvidia/GR00T-N1.7-3B

    # Deploy-only with an existing SONIC checkpoint:
    python examples/vla_g1_workflow.py --checkpoint /path/to/GEAR-SONIC
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl")


# ---------------------------------------------------------------------------
# Stage 1: RECORD - collect locomotion data in sim
# ---------------------------------------------------------------------------


def stage_record(dataset_root: str, n_episodes: int, steps_per_episode: int, checkpoint: str | None = None) -> str:
    """Drive the G1 in sim, capturing a LeRobotDataset.

    Data source, in order of preference:
    - A real SONIC checkpoint (``checkpoint=...``): drive the G1 with WBCPolicy
      so the recorded data is actual locomotion (the realistic workflow). This
      is what you want when collecting demonstrations to fine-tune on.
    - Otherwise: a MockPolicy generates synthetic data that exercises the
      recording pipeline without hardware or weights (the quick-demo path).

    In a true workflow this would be teleop data (LeRobot's teleop recorder with
    a real G1 or a VR controller); WBC-driven locomotion is the autonomous
    stand-in.

    Returns the dataset root path.
    """
    from strands_robots import MockPolicy, Robot

    print("\n=== Stage 1: RECORD ===")
    print(f"  Recording {n_episodes} episode(s), {steps_per_episode} steps each")
    print(f"  Dataset root: {dataset_root}")

    sim = Robot("unitree_g1", mesh=False)

    # Prefer WBC-driven locomotion when a real checkpoint is available, so the
    # recorded dataset contains genuine walking motion rather than random poses.
    record_kwargs: dict = {}
    if checkpoint and os.path.isdir(checkpoint):
        from strands_robots.policies import create_policy
        from strands_robots.policies.wbc import install_wbc_torque_control

        policy = create_policy("wbc", checkpoint=checkpoint, walk=True)
        # WBC emits joint-POSITION targets; the G1 scene's position-servo
        # actuators (uniform kp=500) override SONIC's tuned PD and the robot
        # falls. Installing the torque controller flips those actuators to
        # torque mode and applies the SONIC PD law at the right decimation, so
        # the G1 actually WALKS through run_policy. Pair with control_frequency
        # =50 Hz (the controller's dt=0.005 x decimation 4).
        install_wbc_torque_control(sim, policy, "unitree_g1")
        record_kwargs = {
            "policy_kwargs": {"target_velocity": [0.5, 0.0, 0.0]},
            "action_horizon": 1,
            "control_frequency": 50.0,
        }
        print(f"  Data source: WBCPolicy (real locomotion) from {checkpoint}")
    else:
        policy = MockPolicy()
        print("  Data source: MockPolicy (synthetic - pass --record-checkpoint for real locomotion data)")

    for ep in range(n_episodes):
        start = sim.start_recording(
            repo_id="local/g1_vla_demo",
            root=dataset_root,
            fps=30,
            task="walk forward",
            overwrite=(ep == 0),
        )
        if start.get("status") != "success":
            text = (start.get("content") or [{}])[0].get("text", "unknown error")
            print(f"  ERROR: start_recording failed: {text}", file=sys.stderr)
            print("  Hint: recording requires pip install 'strands-robots[lerobot]'", file=sys.stderr)
            raise SystemExit(1)

        sim.run_policy(
            robot_name="unitree_g1",
            policy_object=policy,
            instruction="walk forward",
            n_steps=steps_per_episode,
            **record_kwargs,
        )
        sim.stop_recording()
        print(f"  Episode {ep + 1}/{n_episodes} recorded.")

    sim.destroy()
    print(f"  Dataset saved -> {dataset_root}")
    return dataset_root


# ---------------------------------------------------------------------------
# Stage 2: FINE-TUNE (optional) - post-train GR00T N1.7 on the recorded data
# ---------------------------------------------------------------------------


def stage_finetune(dataset_root: str, base_model: str, output_dir: str, steps: int) -> str:
    """Post-train Isaac-GR00T N1.7 on the recorded G1 locomotion data.

    Uses the ``Trainer`` abstraction (``create_trainer("groot")``) which wraps
    the ``gr00t_inference`` Docker tool's training pipeline under the hood.
    This is the same interface ``07_post_tune_any_policy.py`` uses for any
    provider - just with ``"groot"`` and a G1 dataset.

    Returns the fine-tuned checkpoint directory.
    """
    from strands_robots.training import TrainSpec, create_trainer

    print("\n=== Stage 2: FINE-TUNE (GR00T N1.7) ===")
    print(f"  Base model:  {base_model}")
    print(f"  Dataset:     {dataset_root}")
    print(f"  Output:      {output_dir}")
    print(f"  Steps:       {steps}")

    trainer = create_trainer("groot")
    spec = TrainSpec(
        dataset_root=dataset_root,
        base_model=base_model,
        output_dir=output_dir,
        steps=steps,
        save_freq=max(1, steps // 4),
        extra={
            "embodiment": "unitree_g1",
            "data_config": "unitree_g1",
        },
    )

    problems = trainer.validate(spec)
    if problems:
        print("  Spec validation failed:", file=sys.stderr)
        for p in problems:
            print(f"    - {p}", file=sys.stderr)
        raise SystemExit(1)

    result = trainer.train(spec)
    print(f"  Train result: {result.status}")
    if result.status != "success":
        print(f"  ERROR: {result.message}", file=sys.stderr)
        raise SystemExit(1)

    exported = trainer.export(spec, result.checkpoint_dir)
    print(f"  Exported checkpoint -> {exported}")
    return str(exported)


# ---------------------------------------------------------------------------
# Stage 3: DEPLOY - run the G1 with WBC (SONIC whole-body control)
# ---------------------------------------------------------------------------


def stage_deploy(checkpoint: str | None, duration: float, vx: float) -> None:
    """Deploy the (fine-tuned or pre-trained) SONIC checkpoint on the G1.

    Uses ``WBCPolicy`` (provider ``"wbc"`` / shorthand ``"sonic"``) to drive
    the 15 leg+waist DOFs of the G1 in MuJoCo simulation. The policy reads
    ``target_velocity`` from the per-call goal kwargs.

    When no real checkpoint is available (the default quick-demo path), this
    stage runs with ``allow_missing_models=True`` and a stub session - proving
    the pipeline wiring without requiring downloaded weights.
    """
    from strands_robots import Robot
    from strands_robots.policies import create_policy

    print("\n=== Stage 3: DEPLOY (WBC / SONIC) ===")
    print(f"  Checkpoint: {checkpoint or '(mock - no real weights)'}")
    print(f"  Command:    vx={vx} m/s for {duration}s")

    sim = Robot("unitree_g1", mesh=False)

    if checkpoint and os.path.isdir(checkpoint):
        policy = create_policy("wbc", checkpoint=checkpoint, walk=True)
        print(f"  Loaded real WBCPolicy from {checkpoint}")
    else:
        # Mock path: prove the deploy wiring without real weights.
        from strands_robots.policies.wbc import WBCPolicy

        policy = WBCPolicy(walk=True, allow_missing_models=True)
        # Inject a stub session so the policy runs (returns zero offsets =
        # hold default pose, safe for a demo).
        import numpy as np

        class _Stub:
            class _I:
                name = "obs"

            def get_inputs(self):  # type: ignore[no-untyped-def]
                return [self._I()]

            def run(self, o, f):  # type: ignore[no-untyped-def]
                return [np.zeros((1, 15), dtype=np.float32)]

        policy.policy_session = _Stub()
        print("  (mock session - deploy wiring demo, no real locomotion)")

    result = sim.run_policy(
        robot_name="unitree_g1",
        policy_object=policy,
        instruction="walk forward",
        policy_kwargs={"target_velocity": [vx, 0.0, 0.0]},
        duration=duration,
        control_frequency=50.0,
        action_horizon=1,
    )
    print(f"  Result: {result['status']}")
    sim.destroy()


# ---------------------------------------------------------------------------
# Main: parse args and chain stages
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end VLA-on-G1 workflow: record -> fine-tune -> deploy.")
    p.add_argument(
        "--tune",
        action="store_true",
        help="Enable stage 2 (fine-tuning). Requires Docker + GPU.",
    )
    p.add_argument(
        "--base-model",
        default="nvidia/GR00T-N1.7-3B",
        help="Base model for fine-tuning (default: nvidia/GR00T-N1.7-3B).",
    )
    p.add_argument(
        "--checkpoint",
        default="",
        help="Skip record+tune; deploy this existing SONIC checkpoint directly.",
    )
    p.add_argument(
        "--record-checkpoint",
        default="",
        help="Drive the RECORD stage with WBC from this SONIC checkpoint "
        "(real locomotion data) instead of a MockPolicy.",
    )
    p.add_argument("--dataset-root", default="/tmp/strands_vla_g1_dataset")
    p.add_argument("--output-dir", default="/tmp/strands_vla_g1_ft")
    p.add_argument("--tune-steps", type=int, default=1000)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--steps-per-episode", type=int, default=100)
    p.add_argument("--deploy-duration", type=float, default=5.0)
    p.add_argument("--vx", type=float, default=0.5, help="Forward velocity for deploy.")
    args = p.parse_args()

    if args.checkpoint:
        # Deploy-only shortcut: skip record + tune.
        stage_deploy(args.checkpoint, args.deploy_duration, args.vx)
    else:
        # Stage 1: Record (WBC-driven if --record-checkpoint given, else mock)
        dataset_root = stage_record(
            args.dataset_root,
            args.episodes,
            args.steps_per_episode,
            checkpoint=args.record_checkpoint or None,
        )

        # Stage 2: Fine-tune (optional, gated behind --tune)
        checkpoint = None
        if args.tune:
            checkpoint = stage_finetune(dataset_root, args.base_model, args.output_dir, args.tune_steps)
        else:
            print("\n  [stage 2 skipped - pass --tune to enable fine-tuning]")

        # Stage 3: Deploy
        stage_deploy(checkpoint, args.deploy_duration, args.vx)

    print("\n=== VLA-on-G1 workflow complete ===")


if __name__ == "__main__":
    main()
