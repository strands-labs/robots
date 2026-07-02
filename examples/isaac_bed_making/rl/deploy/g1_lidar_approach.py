"""LiDAR-range-gated forward approach for the G1 -- ODOM-FREE, ram-safe.

Creeps forward under velocity control, gating on the LiDAR-measured nearest forward distance R.
Stops when R has dropped by --advance (the proven Tier-1 advance) OR R hits the --hard-min-r floor.
Abort: writes/polls /tmp/ABORT every tick -> commanded StopMove (never a process kill -- a kill would
latch the last velocity and run the robot on). Logs R-vs-t telemetry (jsonl) and saves LiDAR cloud
.npy snapshots at start + arrival for the assets.

NOTE on tuning: forward gait below ~0.18 m/s will not initiate stepping on the G1, so --vmax defaults
to 0.20. The ~1 control tick of blind travel between range reads (~0.13 s) is folded into the
EMPIRICAL stop calibration (operator picks --target-r against observed R_final), not modelled here.
"""
import sys, time, os, json, argparse
import numpy as np
sys.path.insert(0, "/home/unitree/robotics-connect-deploy/unitree/g1/locomotion")
sys.path.insert(0, "/home/unitree/robotics-connect-deploy/unitree/g1/lidar_sight")
from g1_locomotion import G1Locomotion
from lidar_sight import LidarSight

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default="eth0")
ap.add_argument("--advance", type=float, default=0.80, help="forward advance (m), measured by LiDAR R drop")
ap.add_argument("--target-r", type=float, default=None, help="absolute target R (overrides --advance)")
ap.add_argument("--hard-min-r", type=float, default=0.28, help="INDEPENDENT runtime safety floor: stop if R below this")
ap.add_argument("--vmax", type=float, default=0.20, help="creep speed (m/s); below ~0.18 the G1 gait won't step")
ap.add_argument("--abort-flag", default="/tmp/ABORT")
ap.add_argument("--telem", default="/tmp/approach_telem.jsonl")
ap.add_argument("--cap-prefix", default="/tmp/cap")
ap.add_argument("--timeout", type=float, default=30.0)
a = ap.parse_args()

YMAX, ZLO, ZHI, XLO, XHI = 0.20, -0.50, 0.20, 0.25, 3.0
def nearest_forward(points):
    m = (np.abs(points[:,1])<YMAX)&(points[:,2]>=ZLO)&(points[:,2]<=ZHI)&(points[:,0]>=XLO)&(points[:,0]<=XHI)
    n = int(m.sum())
    if n < 12: return None, n
    # 3rd percentile, not min(): a robust "nearest" that rejects a few spurious near returns.
    return float(np.percentile(points[m,0],3)), n

try: os.remove(a.abort_flag)
except FileNotFoundError: pass

loco = G1Locomotion(iface=a.iface); loco.connect()          # LocoClient FIRST (DDS ordering)
loco.set_abort_source(lambda: os.path.exists(a.abort_flag))
lidar = LidarSight.instance(topic="rt/utlidar/cloud_livox_mid360")

def read_R(n_med=3):
    rs = []
    for _ in range(n_med):
        c = lidar.latest_cloud()
        if c is not None:
            r,_ = nearest_forward(c.points)
            if r is not None: rs.append(r)
        time.sleep(0.03)
    if not rs: return None
    rs.sort(); return rs[len(rs)//2]    # lower-median by design (small n, robust enough)

def snap(tag):
    c = lidar.latest_cloud()
    if c is not None:
        np.save(f"{a.cap_prefix}_lidar_{tag}.npy", c.points)
        print(f"[approach] saved LiDAR snapshot {a.cap_prefix}_lidar_{tag}.npy ({len(c.points)} pts)", flush=True)

R0 = read_R(7)
if R0 is None:
    loco.stop(); loco.shutdown(); sys.exit("[approach] no LiDAR range at start -- refusing to move")
# target = the commanded stop; hard_min_r is an INDEPENDENT runtime floor enforced in the loop below,
# so a too-aggressive target can never drive the robot closer than hard_min_r to the bed.
target = a.target_r if a.target_r is not None else (R0 - a.advance)
print(f"[approach] R0={R0:.3f} advance={a.advance} -> target_R={target:.3f} hard_min={a.hard_min_r} vmax={a.vmax}", flush=True)
snap("start")

tele = open(a.telem, "w"); t0 = time.time(); result = "?"; miss = 0
try:
    while True:
        if os.path.exists(a.abort_flag): result = "aborted"; break
        if time.time()-t0 > a.timeout: result = "timeout"; break
        R = read_R(1); now = round(time.time()-t0,3)
        if R is None:
            miss += 1; loco.set_velocity(0,0,0)          # hold in place, never creep blind
            tele.write(json.dumps({"t":now,"R":None,"vx":0.0})+"\n"); tele.flush()
            if miss >= 8: result = "lidar_lost"; break
            time.sleep(0.1); continue
        miss = 0
        if R <= target or R <= a.hard_min_r: result = "arrived" if R <= target else "hard_min"; break
        loco.set_velocity(a.vmax,0,0)
        tele.write(json.dumps({"t":now,"R":round(R,3),"vx":a.vmax})+"\n"); tele.flush()
        time.sleep(0.1)
finally:
    # Guard every cleanup step independently: a DDS hiccup in stop() must NOT skip the re-issued stop
    # or shutdown -- the robot being commanded to halt is the safety-critical invariant.
    try: loco.stop()
    except Exception as e: print("[approach] stop() error:", e, flush=True)
    try: Rf = read_R(5)
    except Exception: Rf = None
    try:
        tele.write(json.dumps({"t":round(time.time()-t0,3),"R":Rf if Rf is None else round(Rf,3),"vx":0.0,"event":result})+"\n")
        tele.close()
    except Exception: pass
    print(f"[approach] RESULT={result}  R_final={Rf if Rf is None else round(Rf,3)}", flush=True)
    try: snap("arrival")
    except Exception: pass
    try: loco.stop()                                     # belt-and-suspenders in case the first stop() raised
    except Exception: pass
    try: loco.shutdown()
    except Exception as e: print("[approach] shutdown() error:", e, flush=True)
sys.exit(0 if result == "arrived" else 2)
