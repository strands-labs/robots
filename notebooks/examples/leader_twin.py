"""Your hand on an SO-101 leader drives a simulated SO-101 in the dashboard (the leader is only read).

    python leader_twin.py /dev/ttyACM0 my_leader    # the port and the id you gave lerobot-calibrate
"""
import sys, threading, time
import numpy as np
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from strands_robots import Robot
from strands_robots.dashboard import Dashboard
from strands_robots.units import lerobot_units

arm = Robot("so101")
units = lerobot_units(arm.description, {"*": "degrees", "gripper": "range_0_100"})
leader = SO101Leader(SO101LeaderConfig(port=sys.argv[1], id=sys.argv[2] if len(sys.argv) > 2 else "my_leader"))
leader.connect(calibrate=False)

def follow():
    n, t0 = 0, time.perf_counter()
    while True:
        a = leader.get_action()
        q = units.to_rad(np.array([a[f"{k}.pos"] for k in arm.action_keys]))
        arm.act(q)
        n += 1
        if n % 150 == 0:
            print(f"{n / (time.perf_counter() - t0):.1f} Hz  " + " ".join(f"{k}={v:+.2f}" for k, v in zip(arm.action_keys, q)), flush=True)
        time.sleep(1 / 30)

threading.Thread(target=follow, daemon=True).start()
print("dashboard: http://127.0.0.1:8095", flush=True)
Dashboard(arm.sim).serve(port=8095)
