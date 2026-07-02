"""Operator-held NON-DESTRUCTIVE abort for the G1 in Regular mode.

Press ENTER -> StopMove (halt walking, return to balance-stand / FSM 500) AND write /tmp/ABORT
(which the walk/pull poll as an abort source). The robot stays STANDING in Regular mode -- it is
NOT damped and does NOT collapse. This is the safe alternative to the controller's L2+B whole-body
damp. Keep L2+B only as the absolute last resort.

Why software (not a controller button): in Regular / AI-Sport mode the handheld buttons fire vendor
*gestures*, not a clean abort latch -- so a commanded StopMove from here is the reliable non-destructive
stop. Run this in its OWN TTY during every autonomous run, finger on ENTER:
  python /tmp/g1_abort.py eth0
"""
import os, sys

iface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.loco.g1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID

ChannelFactoryInitialize(0, iface)
c = LocoClient(); c.SetTimeout(3.0); c.Init()
c._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "{}")  # warm the reply reader so StopMove lands instantly (private API: fragile)

# clear any stale flag from a previous run
try: os.remove("/tmp/ABORT")
except FileNotFoundError: pass

def stop_now():
    # Independent safety layers: write the flag AND StopMove. NEVER let one failing block the other.
    try:
        open("/tmp/ABORT", "w").close()
    except Exception as e:
        print(">>> WARN: could not write /tmp/ABORT:", e, flush=True)
    fired = False
    for _ in range(3):                      # a few StopMoves to be certain it lands
        try:
            c.StopMove(); fired = True
        except Exception as e:
            print(">>> StopMove error:", e, flush=True)
    print(">>> ABORT FIRED: StopMove sent. Robot HALTED in Regular mode (FSM 500), NOT damped." if fired
          else ">>> ABORT: StopMove did NOT confirm -- use L2+B if the robot is still moving.", flush=True)

print("=" * 64)
print(" ABORT READY.  Press ENTER to STOP -> StopMove + balance-stand (FSM 500).")
print(" Robot stays UPRIGHT (no damp, no collapse).  Ctrl-C to quit.")
print("=" * 64, flush=True)

while True:
    try:
        line = sys.stdin.readline()
    except KeyboardInterrupt:
        break
    if line == "":                          # EOF (no TTY / stdin closed) -- do NOT hot-loop StopMove
        print(">>> stdin closed (no TTY); abort watcher exiting. Run it in a real terminal.", flush=True)
        break
    stop_now()
    print("    (re-armed -- press ENTER again to abort again)", flush=True)
