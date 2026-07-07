"""Odom-frame walk_to wrapper for the G1, wired to the /tmp/ABORT non-destructive abort.

Drives the vendor LocoClient (via G1Locomotion) to an absolute odometry-frame (x, y, yaw) using the
closed-loop helpers in lib/locomotion.py, then squares heading with turn_to. Registers /tmp/ABORT as
the abort source so an operator ENTER (g1_abort.py) issues a COMMANDED StopMove mid-walk -- never a
process kill (a kill would latch the last velocity and run the robot on).

CAVEAT: odometry on this rig drifts/glitches, so the LiDAR-gated approach (g1_lidar_approach.py) is the
preferred, odom-free path to the bed. This wrapper is kept for the rehearsal/abort-test workflow.
"""
import math, sys, argparse, os
# On-robot deploy script. G1Locomotion is provided by Arm's internal robotics-connect stack
# (private), which runs on the physical G1 and is not part of this repo.
sys.path.insert(0, "/home/unitree/robotics-connect-deploy/unitree/g1/locomotion")
try:
    from g1_locomotion import G1Locomotion
except ImportError as _e:
    raise SystemExit(
        "g1_walk_to.py runs on the physical G1 and requires Arm's internal robotics-connect "
        "stack (g1_locomotion), which is not included in this repository."
    ) from _e

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default="eth0")
ap.add_argument("--tx", type=float, required=True)
ap.add_argument("--ty", type=float, required=True)
ap.add_argument("--tyaw-deg", type=float, required=True)
ap.add_argument("--vmax", type=float, default=0.22)
ap.add_argument("--tol", type=float, default=0.06)
ap.add_argument("--abort-flag", default="/tmp/ABORT")
a = ap.parse_args()

# Start clean: a stale abort flag must not insta-abort this run.
try: os.remove(a.abort_flag)
except FileNotFoundError: pass  # no stale abort flag to clear on a fresh run

loco = G1Locomotion(iface=a.iface); loco.connect()
loco.set_abort_source(lambda: os.path.exists(a.abort_flag))

p0 = loco.pose()
print(f"[walk] start=({p0.x:+.2f},{p0.y:+.2f},{math.degrees(p0.yaw):+.1f}deg) "
      f"target=({a.tx:+.2f},{a.ty:+.2f},{a.tyaw_deg:+.1f}deg) "
      f"dist={math.hypot(a.tx-p0.x,a.ty-p0.y):.2f}m vmax={a.vmax} abort_flag={a.abort_flag}")
r2 = "skipped"   # r1 is always set by walk_to() below before it is read
try:
    r1 = loco.walk_to((a.tx, a.ty), tolerance_m=a.tol, vmax=a.vmax); print("[walk] walk_to ->", r1)
    if r1 is None:
        r2 = loco.turn_to(math.radians(a.tyaw_deg)); print("[walk] turn_to ->", r2)
    else:
        print("[walk] turn_to -> skipped (walk_to did not arrive)")
finally:
    try: loco.stop()            # commanded StopMove; guarded so a hiccup never skips shutdown
    except Exception as e: print("[walk] stop() error:", e)
p1 = loco.pose()
err = math.hypot(a.tx-p1.x, a.ty-p1.y)
print(f"[walk] final=({p1.x:+.2f},{p1.y:+.2f},{math.degrees(p1.yaw):+.1f}deg) pos_err={err:.2f}m")
try: loco.shutdown()
except Exception: pass  # best-effort DDS teardown
# Exit gate uses a looser 0.15 m than --tol (0.06): turn_to can nudge the base a couple cm while
# squaring heading, so the *post-turn* position is judged against the looser bound, by design.
sys.exit(0 if (r1 is None and r2 is None and err <= 0.15) else 2)
