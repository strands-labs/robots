#!/usr/bin/env python3
"""Debug why a MolmoAct2 (or any lerobot_local) policy "runs but does not move".

A common report: the SO-101 arm produces predictions every control tick but does
not visibly move in MuJoCo. The usual root causes are silent until you look:

  1. ACTION-DIM mismatch - the model emits fewer values than the embodiment
     declares actuators, so unmatched joints are zero-filled (frozen).
  2. UNITS - SO-arm checkpoints (MolmoAct2 etc.) emit arm joints in DEGREES and
     the gripper in RANGE_0_100, but the MuJoCo sim joints are RADIANS. Feeding
     raw degrees saturates the radian joint limits and the arm freezes.
  3. A starved obs/rename pipeline -> the model keeps emitting near-zero actions.

This script makes those visible. By default it needs NO model weights: it pushes
a known degree-space action through the ``so101`` embodiment mapping into the
MuJoCo sim and logs per-step joint deltas, proving the mapping yields real
motion. Pass ``--checkpoint <hf_repo_or_dir>`` to instead roll out the real
policy via ``sim.run_policy`` (the lerobot_local diagnostics then warn on the
console if the action stream is degenerate).

Dependencies: pip install "strands-robots[sim-mujoco]"   (+[molmoact2] for --checkpoint)
Run:
    MUJOCO_GL=egl python examples/vla/molmoact2_so101_debug.py
    MUJOCO_GL=egl python examples/vla/molmoact2_so101_debug.py --checkpoint allenai/MolmoAct2-SO100_101
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from strands_robots import create_simulation
from strands_robots.policies.lerobot_local.embodiment import load_embodiment

ROBOT = "so101"


def _read_joints(sim, joints):
    obs = sim.get_observation(ROBOT, skip_images=True)
    return {k: float(np.ravel(obs[k])[0]) for k in joints}


def demo_mapping(sim, joints, emb) -> None:
    """Push a degree-space action through the embodiment mapping and log motion."""
    print(
        f"embodiment '{emb.name}': action_keys={emb.action_keys} "
        f"action_units={emb.action_units} gripper_index={emb.gripper_index} "
        f"gripper_joint_range={emb.gripper_joint_range}"
    )

    # SO-arm checkpoint convention: arm joints DEGREES, gripper 0..100.
    deg_action = [30.0, 30.0, 30.0, 30.0, 30.0, 50.0]
    rad_action = emb.model_action_to_sim(deg_action)
    print(f"model action (deg/0..100): {deg_action}")
    print(f"mapped to sim (rad):       {[round(v, 3) for v in rad_action]}")

    action = {k: rad_action[i] for i, k in enumerate(joints)}
    before = _read_joints(sim, joints)
    for step in range(20):
        sim.send_action(action, robot_name=ROBOT, n_substeps=10)
        if step % 5 == 0:
            now = _read_joints(sim, joints)
            delta = {k: round((now[k] - before[k]) * 180.0 / math.pi, 2) for k in joints}
            print(f"step={step:2d} joint_delta_deg={delta}")

    after = _read_joints(sim, joints)
    max_delta_deg = max(abs(after[k] - before[k]) for k in joints) * 180.0 / math.pi
    print(
        f"max cumulative joint motion: {max_delta_deg:.2f} deg "
        f"({'OK - arm moves' if max_delta_deg > 5.0 else 'FROZEN - check mapping/units'})"
    )

    # Contrast: raw degrees fed straight in saturate the radian joint limits.
    raw = {k: deg_action[i] for i, k in enumerate(joints)}
    b2 = _read_joints(sim, joints)
    for _ in range(20):
        sim.send_action(raw, robot_name=ROBOT, n_substeps=10)
    a2 = _read_joints(sim, joints)
    print(
        "raw-degrees (no mapping) joint positions saturate at limits: "
        f"{[round(a2[k], 2) for k in joints]} (started {[round(b2[k], 2) for k in joints]})"
    )


def rollout_real(sim, checkpoint: str, instruction: str) -> None:
    """Roll out the real policy; lerobot_local logs warnings if actions degenerate."""
    print(f"running real policy {checkpoint!r} - watch for 'lerobot_local:' warnings")
    result = sim.run_policy(
        robot_name=ROBOT,
        policy_provider="lerobot_local",
        policy_config={
            "pretrained_name_or_path": checkpoint,
            "embodiment": ROBOT,
            "inference_action_mode": "continuous",
        },
        instruction=instruction,
        n_steps=30,
        action_horizon=30,
    )
    print(f"run_policy result: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="HF repo id or local dir of a real policy")
    parser.add_argument("--instruction", default="pick up the cube")
    args = parser.parse_args()

    sim = create_simulation("mujoco")
    try:
        sim.create_world()
        sim.add_robot(ROBOT)
        emb = load_embodiment(ROBOT)
        joints = emb.action_keys
        if args.checkpoint:
            rollout_real(sim, args.checkpoint, args.instruction)
        else:
            demo_mapping(sim, joints, emb)
    finally:
        sim.cleanup()


if __name__ == "__main__":
    main()
