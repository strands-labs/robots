"""Productized on-robot deploy of the v2 (82-D, asymmetric-critic) bed-reach policy.

This is the lift of the bed-reach deploy onto the robotics-connect generalized harness
(``lib/policy_deploy.py`` + the G1 ``RobotIO`` binding ``unitree/g1/deploy/g1_robot_io.py``).
It replaces the bespoke ``g1_bedreach_deploy.py`` (kept as the eye-verified as-run record): that
harness hardcoded an 85-D obs concatenation and an inline ``SIM_GAINS`` table, neither of which
survives the v2 change (the actor dropped ``base_lin_vel`` → 82-D). Here EVERYTHING comes from
``deploy_contract_v2.json`` — obs term order, joint order, scales, default offsets, and the PD
gains — so the same code deploys any future re-dump with zero edits, and the obs can never go
out of sync with the trained net.

The de-risk ladder is the shared one (each rung gates the next). **The recommended useful reach is
`--stage arms`** — it is fall-safe and needs no gantry, and the bedside reach does not obviously
need leg/CoM control the vendor balancer can't provide. `--stage whole` is gantry-only and is a
last resort, taken on only if the reach demonstrably needs whole-body balance.
  --stage offline   read-only: build the 82-D obs from live DDS, run the policy, print. No motion.
                    ALWAYS run this first (verifies obs/joint-map/action on the live robot).
  --stage arms      RECOMMENDED. Fall-safe: only the arm joints via rt/arm_sdk; the legs stay on
                    the vendor balance controller (the robot cannot fall). Rate-limited + clamped
                    + motion-blended. No gantry needed.
  --stage whole     LAST RESORT, gantry-only. Full whole-body via rt/lowcmd. The vendor controller
                    is NOT released in software — the OPERATOR puts the robot in low-level Develop
                    mode by hand first (suspend → L2+B → L2+R2); this script VERIFIES that and proves
                    the handheld abort latches before any motion. Then it mirrors the Unitree low-level
                    deploy sequence: move-to-default (legs bear load via a commanded stiff stance, NOT
                    the vendor balancer) → OPERATOR CHECKPOINT (lower the tether STAGE 1 so the feet
                    bear weight, verify it stands, then `touch` the --proceed-file) → policy (lower the
                    tether STAGE 2 once stably balancing) → balance-preserving return to neutral →
                    gentle pose→default + kp→0 release. Default `--whole-mode stand` — prove a stable
                    STAND before ever `--whole-mode reach`. Support = FEET ON THE GROUND with a SLACK
                    gantry (never fully suspended).

SAFETY: NEVER ``kill -9`` this process (a hard kill latches the last high-gain command → runaway;
see robotics-connect SAFETY.md). The handheld controller abort (any button) is the in-loop stop;
SafeStop turns a clean exit / SIGTERM / exception into a compliant damp.

Run on the robot (DDS iface eth0):
  ROBOTICS_CONNECT_ROOT=/home/unitree/robotics-connect \
  /home/unitree/miniconda3/envs/unitree_deploy/bin/python g1_bedreach_deploy_v2.py \
    --stage offline --iface eth0
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)                                   # .../isaac_bed_making/rl

DEFAULT_CONTRACT = os.path.join(RL, "deploy_contract_v2.json")
DEFAULT_POLICY = os.path.join(
    RL, "logs/bed_reach_g1edu/2026-06-13_00-40-39_transfer_v2_critic/exported/policy.pt"
)
# A bedside reach target (base frame +x fwd / +y left / +z up), inside the trained command range.
DEFAULT_TARGET = [0.45, -0.25, -0.05, 1.0, 0.0, 0.0, 0.0]
# A neutral, minimal-reach STAND command (near, centred, low — well inside the command box). The
# first whole-body bring-up uses this so the policy's first action is small; confirm |a|max is small
# at --stage offline with this command before any whole-body STAND.
STAND_TARGET = [0.20, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0]

# The 10 arm joints driven via rt/arm_sdk for the fall-safe --stage arms.
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
]
# Conservative arm clamps (rad) — a backstop; the ladder's rate limit is the primary smoother.
ARM_LIMITS = {
    "left_shoulder_pitch_joint": (-2.6, 2.6), "right_shoulder_pitch_joint": (-2.6, 2.6),
    "left_shoulder_roll_joint": (-1.5, 2.2), "right_shoulder_roll_joint": (-2.2, 1.5),
    "left_shoulder_yaw_joint": (-2.5, 2.5), "right_shoulder_yaw_joint": (-2.5, 2.5),
    "left_elbow_joint": (-1.0, 2.0), "right_elbow_joint": (-1.0, 2.0),
    "left_wrist_roll_joint": (-1.9, 1.9), "right_wrist_roll_joint": (-1.9, 1.9),
}


def _rc_root() -> str:
    """Locate the robotics-connect checkout (the robot path first, then this dev box)."""
    env = os.environ.get("ROBOTICS_CONNECT_ROOT")
    for c in ([env] if env else []) + ["/home/unitree/robotics-connect",
                                        "/home/aifabric/workspaces/git/robotics-connect"]:
        if c and os.path.exists(os.path.join(c, "lib", "policy_deploy.py")):
            return c
    raise FileNotFoundError("robotics-connect not found — set ROBOTICS_CONNECT_ROOT")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["offline", "arms", "whole"], default="offline")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--contract", default=DEFAULT_CONTRACT)
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="exported TorchScript policy.pt")
    ap.add_argument("--target", type=float, nargs=7, default=None,
                    help="hand_target pose: x y z qw qx qy qz (base frame)")
    ap.add_argument("--steps", type=int, default=3, help="offline: inference steps to print")
    ap.add_argument("--seconds", type=float, default=5.0, help="arms/whole: run length")
    ap.add_argument("--vmax", type=float, default=1.0, help="arms: joint rate limit (rad/s)")
    ap.add_argument("--whole-mode", choices=["stand", "reach"], default="stand",
                    help="whole: 'stand' (neutral command — prove this first) or 'reach' (full target)")
    ap.add_argument("--in-develop-mode", action="store_true",
                    help="whole: operator asserts the robot is in low-level Develop mode (only used "
                         "when MotionSwitcher.CheckMode is unreadable because Develop paused it)")
    ap.add_argument("--to-default", type=float, default=2.0, help="whole: scripted move-to-default seconds")
    ap.add_argument("--settle", type=float, default=1.0, help="whole: minimum hold-at-default before proceed")
    ap.add_argument("--blend", type=float, default=2.5, help="whole: blend default→policy seconds")
    ap.add_argument("--cmd-ramp", type=float, default=2.0, help="whole: ramp STAND→reach command seconds")
    ap.add_argument("--return-s", type=float, default=2.5, dest="return_s",
                    help="whole: ramp command back to neutral (policy still balancing) seconds")
    ap.add_argument("--release-s", type=float, default=1.0, dest="release_s",
                    help="whole: gentle pose→default + kp→0 release seconds")
    ap.add_argument("--stand-hold-max", type=float, default=120.0, dest="stand_hold_max",
                    help="whole: max seconds to hold at default waiting for the operator proceed (else damps)")
    ap.add_argument("--proceed-file", default="/tmp/whole_proceed", dest="proceed_file",
                    help="whole: operator-checkpoint sentinel — create this file (touch) to start the policy")
    args = ap.parse_args()

    rc = _rc_root()
    _load("safe_stop", os.path.join(rc, "lib", "safe_stop.py"))
    pd = _load("robotics_connect_policy_deploy", os.path.join(rc, "lib", "policy_deploy.py"))
    g1io = _load("robotics_connect_g1_robot_io", os.path.join(rc, "unitree", "g1", "deploy", "g1_robot_io.py"))

    contract = pd.DeployContract.load(args.contract)
    if not contract.gains:
        raise SystemExit(f"{args.contract} carries no gains — whole-body deploy would be zero-torque. "
                         "Re-dump with dump_deploy_contract.py.")
    target = args.target if args.target is not None else DEFAULT_TARGET
    print(f"[deploy] contract={os.path.basename(args.contract)} "
          f"({contract.n} joints, {contract.obs_total_dim}-D obs)  policy={args.policy}")
    print(f"[deploy] stage={args.stage}  iface={args.iface}  hand_target={target}")

    # Control exactly the 23 action joints (matches the eye-verified reference; never touches the
    # 6 absent EDU joints). G1RobotIO maps these names → SDK motor indices internally.
    io = g1io.G1RobotIO(iface=args.iface, names=contract.action_joint_names,
                        assume_develop_mode=args.in_develop_mode)
    io.connect()
    dep = pd.PolicyDeploy(contract, args.policy, io)

    try:
        if args.stage == "offline":
            dep.run_offline(target, steps=args.steps)
        elif args.stage == "arms":
            # The arm overlay only does anything when the high-level balance controller is RUNNING
            # (it cedes arm authority to rt/arm_sdk at weight=1.0); in low-level Develop mode rt/arm_sdk
            # is a no-op. Probe the loco service so a wrong-mode run fails loud instead of silently.
            if io._high_level_service_alive() is False:
                print("[deploy] REFUSING --stage arms: the high-level balance controller is DOWN "
                      "(loco service not answering) — rt/arm_sdk would no-op. Stand the robot up in "
                      "Regular/AI-Sport mode (the vendor balancer must be running), then retry.")
                return
            # Fall-safe arm overlay; PD comes from the contract (sim-parity kp=40/kd=10 for arms).
            dep.run_partial(target, subset=ARM_JOINTS, seconds=args.seconds,
                            clamp=ARM_LIMITS, vmax_rad_s=args.vmax)
        elif args.stage == "whole":
            # The first whole-body pass STANDS on a neutral command; only --whole-mode reach ramps to
            # the bedside target. NO software vendor release — the harness verifies operator-driven
            # Develop mode (suspend → L2+B → L2+R2) and proves the abort latches before any motion.
            whole_target = STAND_TARGET if args.whole_mode == "stand" else target
            # Operator checkpoint: after move-to-default, run_whole holds the stance and waits for the
            # proceed file before starting the policy (the operator lowers the tether stage 1 first).
            try:
                os.remove(args.proceed_file)            # clear any stale sentinel so we never auto-proceed
            except FileNotFoundError:
                # Sentinel is absent on a fresh run; nothing to clear.
                pass
            print("\n*** STAGE WHOLE — LAST RESORT, GANTRY-ONLY. Support = FEET ON THE GROUND with a "
                  "SLACK gantry (never fully suspended). Hardware e-stop + the battery in reach. ***")
            print(f"    whole-mode={args.whole_mode}  command={whole_target}")
            print("    Operator: be in low-level Develop mode (suspend → L2+B → L2+R2). NEVER kill -9.")
            print("    CHECKPOINT: after move-to-default, lower the tether (stage 1) + verify it stands,")
            print(f"    then START THE POLICY by creating the proceed file:  touch {args.proceed_file}")
            dep.run_whole(whole_target, seconds=args.seconds, neutral_command=STAND_TARGET,
                          to_default_s=args.to_default, settle_s=args.settle, blend_s=args.blend,
                          cmd_ramp_s=args.cmd_ramp, return_s=args.return_s, release_s=args.release_s,
                          stand_hold_max_s=args.stand_hold_max,
                          proceed_fn=lambda: os.path.exists(args.proceed_file))
    finally:
        io.shutdown()


if __name__ == "__main__":
    main()
