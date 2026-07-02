"""Bed-making orchestrator — the Device Connect demo state machine.

Ties together the three pieces built 2026-06-18 into the value story:

  AUTONOMOUS attempt(s)            (g1_bed_pull_v2.py --abort-on-stall + bed_grip_v1.py)
      │  the RL reach + claw grasp + draw, watched by the pull-failure detector
      ▼
  detect FAILURE                   (/tmp/pull_failed — the calibrated anchored-stall verdict)
      │  e.g. it grabbed the fixed mattress cover, not the free sheet → the draw stalled
      ▼
  ASK THE HUMAN for help           (banner + /tmp/robot_needs_help + optional --help-hook)
      │  ← THIS graceful degradation is the Device Connect value: the robot knows it failed
      │    and escalates, instead of silently believing it succeeded
      ▼
  deterministic HANDOFF fallback   (g1_bed_handoff_v1.py + bed_grip_v1.py --present-claw)
      │  present palm-up open claw → human lays the sheet in & signals (creates /tmp/grab_go)
      │  → claw closes → deterministic draw toward the head
      ▼
  DONE

Per the original spec: try autonomously up to --max-attempts (default 2), THEN hand off. (I argued one
honest attempt may demo better than two visible failures — so it is tunable; set --max-attempts 1.)

An autonomous attempt is judged FAILED on EITHER failure signal: an anchored stall (/tmp/pull_failed,
the calibrated draw-resistance verdict) OR an empty grab (/tmp/grab_empty, the grasp-confirm verdict —
the claw closed on nothing). The empty-grab guard closes the old blind spot where a draw on air read as
success. It is ENFORCED only under --require-fabric; default is measure-only (the grip logs a grasp
verdict but still proceeds) until the touch/proximity thresholds are calibrated on hardware — the same
measure-first discipline as the stall detector's --abort-on-stall. (Open follow-up: a stronger
fabric-in-hand cue from vision, for when both touch and proximity are marginal.)

--dry simulates the state transitions with NO robot motion (verify the logic). Live needs the robot
stood in Regular mode at the bedside and a clean GET_FSM_ID 0 (pass --assume-balancer-up).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

DEPLOY = "/home/unitree/bedreach_deploy"
RC = "/home/unitree/robotics-connect-deploy"
SENTINELS = ["/tmp/arm_reached", "/tmp/grab_go", "/tmp/sheet_gripped", "/tmp/draw_done",
             "/tmp/pull_failed", "/tmp/grab_empty", "/tmp/robot_needs_help", "/tmp/human_ack"]


def banner(msg: str) -> None:
    line = "═" * max(40, len(msg) + 4)
    print(f"\n{line}\n  {msg}\n{line}\n", flush=True)


def clear_sentinels(dry: bool) -> None:
    if dry:                               # a pure simulation must not touch real robot-adjacent state
        return
    for f in SENTINELS:
        try:
            os.remove(f)
        except OSError:
            pass


def _env() -> dict:
    e = dict(os.environ)
    e["ROBOTICS_CONNECT_ROOT"] = RC
    return e


def _bg(cmd: list[str], log_path: str, dry: bool):
    """Launch a sub-script in the background, stdout/err → log_path. Returns Popen (or None in dry)."""
    print(f"  $ {' '.join(cmd)}  > {log_path}", flush=True)
    if dry:
        return None
    f = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            cwd=DEPLOY, env=_env())


def _wait(proc, timeout_s: float, dry: bool) -> None:
    if dry or proc is None:
        return
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.terminate()                      # SIGTERM → the scripts SafeStop-damp on exit
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()


def autonomous_attempt(n: int, args) -> bool:
    """One autonomous reach+grasp+draw, watched by the failure detector. Returns True on SUCCESS."""
    banner(f"AUTONOMOUS ATTEMPT {n}/{args.max_attempts} — reach · grasp · draw")
    clear_sentinels(args.dry)
    py = ["python", "-u"]
    grip_cmd = py + ["bed_grip_v1.py", "--empty-file", args.empty_file]
    if args.require_fabric:                          # gate the draw on fabric-in-hand (empty-grab guard)
        grip_cmd.append("--require-fabric")
    grip = _bg(grip_cmd, "/tmp/orch_grip.log", args.dry)
    pull_cmd = py + ["g1_bed_pull_v2.py", "--abort-on-stall", "--empty-file", args.empty_file]
    if args.assume_balancer_up:
        pull_cmd.append("--assume-balancer-up")
    if args.grasp_wrist_roll is not None:
        pull_cmd += ["--grasp-wrist-roll", str(args.grasp_wrist_roll)]
    pull = _bg(pull_cmd, "/tmp/orch_pull.log", args.dry)
    _wait(pull, args.attempt_timeout, args.dry)
    _wait(grip, 10, args.dry)

    if args.dry:
        failed = n <= args.dry_fail_attempts        # simulate: first N attempts fail
        empty = failed and args.dry_fail_reason == "empty"
    else:
        # FAIL if: the grasp closed on NOTHING (empty grab), the detector flagged an anchored stall, OR
        # the attempt crashed/was killed. A non-zero exit must NOT read as success, and neither may an
        # empty grab — a confident false-success is the exact failure this escalation path exists to kill.
        crashed = pull is not None and pull.returncode not in (0, None)
        if crashed:
            print(f"  (autonomous attempt exited {pull.returncode} — treating as failure)", flush=True)
        empty = os.path.exists(args.empty_file)
        failed = crashed or empty or os.path.exists(args.fail_file)
    if failed:
        reason = ("empty grab — the claw closed on nothing (no sheet in hand)" if empty
                  else "anchored/stalled draw (likely grabbed the mattress cover)")
        print(f"  ✗ attempt {n} FAILED — {reason}.", flush=True)
        return False
    print(f"  ✓ attempt {n} succeeded — the draw moved freely.", flush=True)
    return True


def ask_human(args) -> None:
    """The Device Connect moment: the robot knows it failed and escalates to a human."""
    banner("ROBOT NEEDS HELP — asking the human (Device Connect)")
    msg = (f"Bed-making: autonomous grasp failed after {args.max_attempts} attempts. "
           "Please come place the sheet edge in my hand.")
    if not args.dry:
        try:
            with open("/tmp/robot_needs_help", "w") as f:
                f.write(msg + "\n")
        except OSError:
            pass
    print(f"  → {msg}", flush=True)
    if args.help_hook:                              # pluggable: wire to a real Device Connect notify
        print(f"  → help-hook: {args.help_hook}", flush=True)
        if not args.dry:
            try:
                subprocess.run(args.help_hook, shell=True, timeout=20, env=_env())
            except Exception as e:                  # a flaky notifier must not abort the demo
                print(f"  (help-hook error, continuing: {e})", flush=True)
    # Wait for the human to acknowledge they are coming (DC or operator creates /tmp/human_ack),
    # else proceed after a grace period so the demo never deadlocks.
    print(f"  waiting up to {args.ack_timeout:.0f}s for human ack (/tmp/human_ack)…", flush=True)
    if not args.dry:
        t_end = time.time() + args.ack_timeout
        while time.time() < t_end and not os.path.exists("/tmp/human_ack"):
            time.sleep(0.5)
    print("  human engaged — proceeding to the handoff.", flush=True)


def handoff(args) -> bool:
    """Deterministic present → human places the sheet & signals → close → draw. True on success."""
    banner("HANDOFF — present palm-up claw · human places the sheet · draw")
    clear_sentinels(args.dry)
    py = ["python", "-u"]
    grip_cmd = py + ["bed_grip_v1.py", "--present-claw", "--arm-reached-file", "/tmp/grab_go",
                     "--empty-file", args.empty_file]
    if args.require_fabric:
        grip_cmd.append("--require-fabric")
    grip = _bg(grip_cmd, "/tmp/orch_grip.log", args.dry)
    hand_cmd = py + ["g1_bed_handoff_v1.py", "--empty-file", args.empty_file]
    if args.assume_balancer_up:
        hand_cmd.append("--assume-balancer-up")
    hand = _bg(hand_cmd, "/tmp/orch_handoff.log", args.dry)
    print("  → robot is presenting. Human: lay the sheet in the palm, withdraw, then signal "
          "(create /tmp/grab_go — your 'close').", flush=True)
    _wait(hand, args.handoff_timeout, args.dry)
    _wait(grip, 10, args.dry)
    if args.dry:
        ok = not args.dry_handoff_fail
    else:
        # The handoff writes /tmp/draw_done in BOTH its drew-the-sheet and no-grip branches, so require
        # a clean exit, no stall verdict, AND no empty-grab (the human's placement actually took).
        crashed = hand is not None and hand.returncode not in (0, None)
        ok = ((not crashed) and os.path.exists("/tmp/draw_done")
              and not os.path.exists(args.fail_file) and not os.path.exists(args.empty_file))
    print(f"  {'✓ handoff drew the sheet.' if ok else '✗ handoff did not complete.'}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-attempts", type=int, default=2, help="autonomous tries before the handoff")
    ap.add_argument("--assume-balancer-up", action="store_true")
    ap.add_argument("--grasp-wrist-roll", type=float, default=-1.5, help="palm-to-edge roll for the autonomous grasp")
    ap.add_argument("--fail-file", default="/tmp/pull_failed")
    ap.add_argument("--empty-file", default="/tmp/grab_empty",
                    help="empty-grab sentinel (claw closed on nothing) — an autonomous-attempt failure")
    ap.add_argument("--require-fabric", action="store_true",
                    help="enforce fabric-in-hand: an empty grab aborts the draw and escalates (like a "
                         "stall). Default measure-only — enable after the confirm thresholds are calibrated.")
    ap.add_argument("--help-hook", default=None, help="shell command to notify the human (Device Connect)")
    ap.add_argument("--ack-timeout", type=float, default=60.0, help="seconds to wait for human ack")
    ap.add_argument("--attempt-timeout", type=float, default=120.0)
    ap.add_argument("--handoff-timeout", type=float, default=180.0)
    ap.add_argument("--dry", action="store_true", help="simulate the state machine, NO robot motion")
    ap.add_argument("--dry-fail-attempts", type=int, default=99, help="[dry] how many attempts fail")
    ap.add_argument("--dry-fail-reason", choices=["stall", "empty"], default="stall",
                    help="[dry] why the simulated attempts fail — exercises the stall vs empty-grab message")
    ap.add_argument("--dry-handoff-fail", action="store_true", help="[dry] make the handoff fail too")
    args = ap.parse_args()

    banner("BED-MAKING ORCHESTRATOR" + ("  [DRY RUN — no robot motion]" if args.dry else ""))
    succeeded = False
    for n in range(1, args.max_attempts + 1):
        if autonomous_attempt(n, args):
            succeeded = True
            break

    if succeeded:
        banner("DONE — bed made autonomously ✓")
        return

    ask_human(args)
    if handoff(args):
        banner("DONE — bed made WITH HUMAN HELP via handoff ✓  (graceful degradation = Device Connect value)")
    else:
        banner("HANDOFF INCOMPLETE — escalate / retry")


if __name__ == "__main__":
    main()
