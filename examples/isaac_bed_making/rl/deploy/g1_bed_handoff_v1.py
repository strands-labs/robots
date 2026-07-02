"""Human→robot bedsheet HANDOFF + deterministic pull (no RL).

The fallback path for when the autonomous grasp can't get the sheet: the robot PRESENTS an open
claw (palm up, thumb opposed, fingers open) at a human-comfortable height, the human lays the sheet
edge into the hand, the robot closes the claw and DRAWS the sheet — all on the arm-overlay
(vendor balancer holds the legs, arms via rt/arm_sdk). Unlike g1_bed_pull_v2, NOTHING here uses the
RL policy: the present pose, the draw and the retract are all deterministic joint-space ramps, so the
"pull from an intermediate present pose" is in-distribution by construction (there is no distribution).

Why deterministic: the RL draw is trained from the come-in pose; a draw started from a present/handoff
pose would be off-distribution. The draw is just a hand translation, which a rate-limited joint-space
ramp already does collision-free (same trick as the lift in g1_bed_pull_v2). So the policy earns its
keep on the autonomous reach only; the assisted path is fully scripted.

Coordination (sentinel handshake, mirrors the pull):
  present HANDOFF_POSE → hold, waiting for ``--gripped-file`` → deterministic DRAW → ``--draw-done-file``
  → retract (exact reverse) → blend out → damp.
The HAND is driven by bed_grip_v1.py in a second process: run it with
``--arm-reached-file /tmp/grab_go --present-claw`` so it shows the open claw while waiting, then closes
on a ``touch /tmp/grab_go`` once the human has placed the sheet.

Fall-safe throughout: legs on the vendor balancer, arms weight-blended in/out, rate-limited + clamped,
abort polled every tick, SafeStop damps on every exit.

  --present-only   present HANDOFF_POSE, hold for eye-verify, then release (validate the pose first)
  (default)        present → wait-for-grip → deterministic draw → release
"""
from __future__ import annotations

import argparse
import os
import time

from bed_deploy_common import ARM_JOINTS, ARM_LIMITS, LEFT_ARM, RIGHT_ARM, load, rc_root
from bed_fail_detect import DrawResistanceMonitor, read_arm_tau

# APPROACH: clear the low mattress the SAME way the lift does — go up + shoulder-wide FIRST, so the
# arm rises out to the side ABOVE the bed instead of sweeping forward through it. Ramping straight to
# the present pose fouled the mattress side (hardware, 2026-06-18). Reused from g1_bed_pull_v2's lift.
APE_HANGER = {  # raise wide + up, elbow bent — hand hangs high, clear of the mattress
    "left_shoulder_pitch_joint": -0.52, "left_shoulder_roll_joint": 1.65,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 1.50, "left_wrist_roll_joint": 0.0,
}

# Deterministic LEFT-arm HANDOFF_POSE (joint-space, rad): hand presented HIGH and clear of the bed,
# forearm up, PALM UP (positive wrist roll — opposite of the palm-to-edge grasp roll), so a human can
# lay the sheet edge across the open claw WITHOUT the robot fouling the mattress. Reached FROM
# ape_hanger (above the bed), never by sweeping forward+down. Tune via --present-only / --present-wrist-roll.
HANDOFF_POSE = {
    "left_shoulder_pitch_joint": -0.60,   # arm up
    "left_shoulder_roll_joint": 0.95,     # stay somewhat abducted (out to the side, clear of the bed)
    "left_shoulder_yaw_joint": 0.25,      # forearm angled in toward the human
    "left_elbow_joint": 1.45,             # forearm up so the hand presents high, not down over the bed
    "left_wrist_roll_joint": -1.74,       # PALM UP to receive the sheet (eye-verified on hardware 2026-06-18)
}

# Deterministic DRAW after the grab: sweep the held sheet toward the head of the bed (robot's +y/left)
# and up, dragging the cover. A single tunable waypoint; eye-verify + tune the sweep on hardware.
DRAW_POSE = {
    "left_shoulder_pitch_joint": -0.70,   # lift a touch
    "left_shoulder_roll_joint": 1.40,     # abduct further toward the head side — sweep the sheet +y
    "left_shoulder_yaw_joint": 0.25,
    "left_elbow_joint": 1.05,             # draw the hand in/up as it sweeps
    "left_wrist_roll_joint": -1.74,       # keep the palm-up/claw orientation through the draw
}


def run_handoff(dep_dt, io, SafeStop, *, present_only, approach_waypoints, handoff_pose, draw_pose,
                present_s, draw_s, draw_hold_s, blend_s, vmax_rad_s, hold_timeout_s, gains,
                gripped_file, draw_done_file, monitor=None, abort_on_stall=False,
                fail_file="/tmp/pull_failed", empty_file="/tmp/grab_empty", log=print):
    if not io.arm_abort():
        log("[handoff] abort not armed (release all controller buttons) — refusing")
        return False
    dt = dep_dt
    max_step = vmax_rad_s * dt
    n_blend = max(1, int(blend_s / dt))
    cmdq = {n: io.read_state().q[n] for n in ARM_JOINTS}   # all 10 arm joints; right held at start
    traj = [dict(cmdq)]                                     # recorded forward path, replayed to retract

    def _clamp(n, v):
        lo, hi = ARM_LIMITS.get(n, (-3.14, 3.14))
        return max(lo, min(hi, v))

    def ramp_to(joint_goals, seconds, label, record=True, monitor=None, abort_on_stall=False):
        n = max(1, int(seconds / dt))
        for k in range(n):
            if io.abort_tripped():
                log(f"[handoff] abort during {label}")
                return False
            for j, goal in joint_goals.items():
                cmdq[j] += max(-max_step, min(max_step, _clamp(j, goal) - cmdq[j]))
            io.publish_targets(cmdq, gains, weight=1.0)
            if record:
                traj.append(dict(cmdq))
            if monitor is not None:
                m = monitor.update(io.read_state(), cmdq, read_arm_tau(io, LEFT_ARM))
                if k % 20 == 0:
                    log(f"[handoff] {label}: follow={m['follow']:.3f} dq={m['dq']:.3f} "
                        f"tau={m['tau']:.2f} yaw={m['yaw']:.1f}")
                if m["tripped"] and abort_on_stall:
                    log(f"[handoff] STALL detected during {label} — stopping the draw early (anchored load)")
                    break
            time.sleep(dt)
        return True

    def hold(seconds, label):
        n = max(1, int(seconds / dt))
        for _ in range(n):
            if io.abort_tripped():
                log(f"[handoff] abort during {label}")
                return False
            io.publish_targets(cmdq, gains, weight=1.0)
            time.sleep(dt)
        return True

    def retract():
        log("[handoff] retracting along the reverse path")
        for pose in reversed(traj):
            if io.abort_tripped():
                log("[handoff] abort during retract")
                return False
            for j in ARM_JOINTS:
                cmdq[j] = pose[j]
            io.publish_targets(cmdq, gains, weight=1.0)
            time.sleep(dt)
        return True

    for f in (gripped_file, draw_done_file, fail_file, empty_file):
        try:
            os.remove(f)
        except OSError:
            pass

    with SafeStop(io.damp_once, name="bed_handoff"):
        for k in range(n_blend):                                   # blend overlay IN (hold start pose)
            if io.abort_tripped():
                return log("[handoff] abort during blend-in") or False
            io.publish_targets(cmdq, gains, weight=(k + 1) / n_blend)
            time.sleep(dt)

        for name, goals in approach_waypoints:                     # APPROACH up+wide, clear the mattress
            log(f"[handoff] approach → {name}")
            if not ramp_to(goals, present_s, f"approach:{name}"):
                return False
        log("[handoff] presenting HANDOFF_POSE (palm up, open claw — place the sheet)")
        if not ramp_to(handoff_pose, present_s, "present"):        # DETERMINISTIC PRESENT
            return False

        if present_only:
            log("[handoff] --present-only: holding 4 s for eye-verify, then releasing")
            if not hold(4.0, "present-hold"):
                return False
        else:
            log(f"[handoff] presented — waiting for the grip ({gripped_file}). "
                f"Place the sheet, then signal the grip to close.")
            t_end = time.time() + hold_timeout_s
            gripped = False
            while not os.path.exists(gripped_file):                # HOLD for the human + grip
                if io.abort_tripped():
                    return log("[handoff] abort during present-hold") or False
                if os.path.exists(empty_file):                     # grip confirmed EMPTY (sheet not placed right)
                    log("[handoff] EMPTY GRAB signaled — the sheet was not in the hand; skipping the draw")
                    break
                if time.time() > t_end:
                    log("[handoff] handoff timeout — no grip; skipping the draw")
                    break
                if not hold(dt, "present-wait"):
                    return False
            else:
                gripped = True
                log("[handoff] grip confirmed — drawing the sheet (deterministic)")
            if gripped:
                if monitor is not None:
                    monitor.start(io.read_state())
                if not ramp_to(draw_pose, draw_s, "draw", monitor=monitor,    # DETERMINISTIC DRAW
                               abort_on_stall=abort_on_stall):
                    return False
                open(draw_done_file, "w").close()
                log(f"[handoff] draw complete — signaled {draw_done_file}")
                if monitor is not None:
                    log(f"[handoff] DRAW VERDICT: {monitor.summary()}")
                    if monitor.tripped:
                        try:
                            open(fail_file, "w").close()
                        except OSError:
                            pass
                        log(f"[handoff] ⚠ draw FAILED (anchored load). wrote {fail_file}.")
                if not hold(draw_hold_s, "draw-hold"):
                    return False
            else:
                open(draw_done_file, "w").close()

        if not retract():                                          # EXACT REVERSE retraction
            return False
        for k in range(n_blend):                                   # blend overlay OUT
            if io.abort_tripped():
                return log("[handoff] abort during blend-out") or False
            io.publish_targets(cmdq, gains, weight=1.0 - (k + 1) / n_blend)
            time.sleep(dt)
    log("[handoff] complete (overlay blended out, damped)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--contract", default="/home/unitree/bedreach_deploy/deploy_contract_v2.json")
    ap.add_argument("--present-only", action="store_true", help="present HANDOFF_POSE, hold, release")
    ap.add_argument("--present-wrist-roll", type=float, default=None,
                    help="override the HANDOFF_POSE wrist roll (rad, ±1.9) — +ve = palm up. Tune the "
                         "receive orientation here.")
    ap.add_argument("--present-s", type=float, default=4.0, help="seconds to ramp into HANDOFF_POSE")
    ap.add_argument("--draw-s", type=float, default=4.0, help="seconds for the deterministic draw")
    ap.add_argument("--draw-hold", type=float, default=1.0, help="seconds to hold at the drawn pose")
    ap.add_argument("--blend", type=float, default=1.2, help="seconds to blend the arm overlay in/out")
    ap.add_argument("--vmax", type=float, default=1.0, help="arm joint rate limit (rad/s)")
    ap.add_argument("--hold-timeout", type=float, default=120.0, help="max present-hold before giving up")
    ap.add_argument("--assume-balancer-up", action="store_true",
                    help="skip the in-context loco liveness gate; use ONLY after a clean GET_FSM_ID 0.")
    ap.add_argument("--gripped-file", default="/tmp/sheet_gripped")
    ap.add_argument("--draw-done-file", default="/tmp/draw_done")
    ap.add_argument("--stall-follow", type=float, default=0.28,
                    help="following-error threshold (rad) — free pull ~0.18, anchored ~0.35 (cal 2026-06-18)")
    ap.add_argument("--stall-tau", type=float, default=11.0,
                    help="joint-torque threshold — free pull ~7, anchored ~14 (cal 2026-06-18)")
    ap.add_argument("--stall-yaw", type=float, default=6.0, help="torso yaw-drift threshold (deg) — logged only")
    ap.add_argument("--stall-sustain", type=float, default=0.5, help="seconds over-threshold to call a stall")
    ap.add_argument("--abort-on-stall", action="store_true", help="stop the draw early on a detected stall")
    ap.add_argument("--fail-file", default="/tmp/pull_failed", help="sentinel written on a stall verdict")
    ap.add_argument("--empty-file", default="/tmp/grab_empty",
                    help="sentinel the grip writes on an EMPTY grab (sheet not in hand); skips the draw")
    args = ap.parse_args()

    handoff_pose = dict(HANDOFF_POSE)
    if args.present_wrist_roll is not None:
        handoff_pose["left_wrist_roll_joint"] = args.present_wrist_roll
    draw_pose = dict(DRAW_POSE)
    if args.present_wrist_roll is not None:                         # keep the claw orientation through the draw
        draw_pose["left_wrist_roll_joint"] = args.present_wrist_roll

    rc = rc_root()
    ss = load("safe_stop", os.path.join(rc, "lib", "safe_stop.py"))
    pd = load("robotics_connect_policy_deploy", os.path.join(rc, "lib", "policy_deploy.py"))
    g1io = load("robotics_connect_g1_robot_io", os.path.join(rc, "unitree", "g1", "deploy", "g1_robot_io.py"))

    contract = pd.DeployContract.load(args.contract)
    if not contract.gains:
        raise SystemExit(f"{args.contract} carries no gains — overlay would be zero-torque.")
    print(f"[handoff] rc={rc}  contract={os.path.basename(args.contract)} ({contract.n}j)")
    print(f"[handoff] mode={'PRESENT-ONLY' if args.present_only else 'present→wait-grip→draw'}  "
          f"present_wrist_roll={handoff_pose['left_wrist_roll_joint']}")

    io = g1io.G1RobotIO(iface=args.iface, names=contract.action_joint_names)
    io.connect()
    dep = pd.PolicyDeploy(contract, "/home/unitree/bedreach_deploy/policy_v2.pt", io)  # gains/dt only
    monitor = DrawResistanceMonitor(LEFT_ARM, dep.dt, follow_thresh_rad=args.stall_follow,
                                    tau_thresh=args.stall_tau, yaw_thresh_deg=args.stall_yaw,
                                    sustain_s=args.stall_sustain)
    try:
        if not args.assume_balancer_up:
            if io._high_level_service_alive() is False:
                print("[handoff] REFUSING: balance controller DOWN. Stand in Regular mode or pass "
                      "--assume-balancer-up after a clean GET_FSM_ID 0.")
                return
        else:
            print("[handoff] --assume-balancer-up: skipping in-context liveness gate (clean probe confirmed).")
        run_handoff(
            dep.dt, io, ss.SafeStop, present_only=args.present_only,
            approach_waypoints=[("ape_hanger", APE_HANGER)],
            handoff_pose=handoff_pose, draw_pose=draw_pose,
            present_s=args.present_s, draw_s=args.draw_s, draw_hold_s=args.draw_hold,
            blend_s=args.blend, vmax_rad_s=args.vmax, hold_timeout_s=args.hold_timeout, gains=dep.gains(),
            gripped_file=args.gripped_file, draw_done_file=args.draw_done_file,
            monitor=monitor, abort_on_stall=args.abort_on_stall, fail_file=args.fail_file,
            empty_file=args.empty_file,
        )
    finally:
        io.shutdown()


if __name__ == "__main__":
    main()
