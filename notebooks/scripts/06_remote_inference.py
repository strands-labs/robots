# %% [markdown]
# # 06 · Remote inference
#
# The model runs in a `lerobot-policy-server` process; the arm runs here. They speak LeRobot's own remote-inference
# protocol over Zenoh, and `Policy("zenoh://host:port/<deployment>")` is the client.
#
# ![one frame out, one chunk back; sequential holds, splice overlaps](media/06/remote.svg)
#
# Needs `uv pip install -e ".[mujoco,lerobot]"`. Everything below runs on one machine over loopback, port 7450.
# Timings were measured in a background process tree whose timers macOS coalesces; read them as a shape, not a benchmark.

# %%
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import contextlib
import importlib
import io
import logging
import signal
import subprocess
import sys
import tempfile
import threading
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from strands_robots import Description, Policy, Robot

media = Path("media/06") if Path("media").is_dir() else Path("notebooks/media/06")
media.mkdir(parents=True, exist_ok=True)


def quiet_import(module):
    """Import `module` with stderr closed: LeRobot's import chain prints three known-harmless macOS lines (torch's
    "Redirects are not supported" note, and an ObjC notice that opencv and av both bundle libavdevice)."""
    sys.stderr.flush()
    saved, null = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 2)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return importlib.import_module(module)
    finally:
        os.dup2(saved, 2)
        os.close(null), os.close(saved)


quiet_import("lerobot.remote_inference")
# lerobot reads the checkpoint's saved device ("cuda") before device="cpu" applies and warns; the run is on cpu.
logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
print("lerobot", version("lerobot"), "| eclipse-zenoh", version("eclipse-zenoh"))

# %% [markdown]
# ## Write the server config
#
# `server_config` reads the checkpoint's features against the robot: joint names, cameras at their trained size, units.
# It is the same Hub ACT as in 05, trained in degrees, so `norm=` says so.

# %%
from strands_robots.policy.lerobot import server_config, write_server_yaml

CHECKPOINT = "robotfuel/act_so101_t16b"
ENDPOINT = "tcp/127.0.0.1:7450"
work = Path(tempfile.mkdtemp(prefix="nb06-"))

cfg = server_config(
    Description.load("so101"),
    CHECKPOINT,
    deployment="act",
    endpoint=ENDPOINT,
    device="cpu",
    norm={"*": "degrees", "gripper": "range_0_100"},
)
yaml_path = write_server_yaml(cfg, work / "act.yaml")
print(yaml_path.read_text().split("features:")[0].replace(str(work), "<work>"))

# %% [markdown]
# ## Start the server
#
# A separate process, exactly as on a GPU box: `lerobot-policy-server --config_path=act.yaml`. It is ready when its
# log says so.

# %%
SERVER = str(Path(sys.executable).with_name("lerobot-policy-server"))


def start_server():
    log = work / "server.log"
    proc = subprocess.Popen([SERVER, f"--config_path={yaml_path}"], stdout=log.open("w"), stderr=subprocess.STDOUT)
    t0 = time.perf_counter()
    while "Policy server ready" not in log.read_text():
        if proc.poll() is not None or time.perf_counter() - t0 > 180:
            raise RuntimeError(f"server did not start (exit {proc.returncode}):\n{log.read_text()[-2000:]}")
        time.sleep(0.5)
    print(f"server pid {proc.pid} ready in {time.perf_counter() - t0:.1f} s")
    return proc, log


server, log = start_server()

# %% [markdown]
# `describe()` asks the server what it serves, without taking a session.

# %%
URL = "zenoh://127.0.0.1:7450/act"
d = Policy(URL).describe()
for k in ("deployment", "artifact_identity", "semantics", "execution_contracts", "session_status"):
    print(f"{k:>20}: {d[k]}")
print(f"{'limits':>20}: action_deadline_s={d['limits']['action_deadline_s']} idle_timeout_s={d['limits']['idle_timeout_s']}")

# %% [markdown]
# ## Drive the sim arm with it
#
# The policy needs a `wrist` camera at 640x480; `attach` checks that before the session is admitted. This deployment
# advertises no RTC contract, so `execution="auto"` resolves to `"sequential"`: send a still frame, hold until the
# whole chunk arrives, play it, ask again.


# %%
def remote_arm():
    a = Robot("so101")
    a.sim.camera("wrist", position=(0.0, 0.0, 0.05), target=(0.0, -0.1, -0.05), size=(640, 480), attach="so101/gripper")
    a.sim.camera("view", position=[0.45, -0.45, 0.35], target=[0, 0, 0.12], size=(320, 240))
    return a


arm = remote_arm()
remote = Policy(URL)
frames = []
seq = arm.run(remote, seconds=12, stop_when=lambda obs: frames.append(arm.sim.render("view")) or False)
arm.sim.close()
print(remote, "->", remote.execution)
print(f"{seq.steps} steps, stopped by {seq.stopped_by}, fault {seq.fault} | holds {seq.holds} ({seq.held_seconds:.2f} s) | deadline misses {seq.deadline_misses}")

# %% [markdown]
# ![the sim SO-101 driven by a policy in another process](media/06/remote_arm.gif)

# %%
gif = [Image.fromarray(f) for f in frames[::4]]
gif[0].save(media / "remote_arm.gif", save_all=True, append_images=gif[1:], duration=133, loop=0)
print(media / "remote_arm.gif", f"{(media / 'remote_arm.gif').stat().st_size / 1e3:.0f} KB")

# %% [markdown]
# ## Latency, from the policy's own ledger
#
# A sequential session logs every chunk it received: wall time from frame out to chunk in, and the server's own split.

# %%
wall = np.array([c["wall_s"] for c in remote.calls])
model = np.array([c["server"].get("policy", np.nan) for c in remote.calls])
print(f"{len(wall)} calls, {remote.calls[0]['rows']} rows each")
print(f"round trip  p50 {np.percentile(wall, 50) * 1e3:.0f} ms   p95 {np.percentile(wall, 95) * 1e3:.0f} ms")
print(f"model only  p50 {np.nanpercentile(model, 50) * 1e3:.0f} ms   p95 {np.nanpercentile(model, 95) * 1e3:.0f} ms")
print("(measured in a coalesced background process tree, CPU inference, loopback)")

# %% [markdown]
# ## Sequential vs splice
#
# `execution="splice"` is LeRobot's engine unchanged: it asks for the next chunk one turnaround before the queue
# empties, from a moving arm, and merges. A policy that holds returns `None` from `act()`; the loop writes nothing,
# counts a hold, and never books it as a deadline miss. Both runs below render only the policy's own camera (the run
# above also rendered the GIF's camera every tick, which is where its deadline misses came from).

# %%
results = {}
for execution in ("sequential", "splice"):
    arm = remote_arm()
    p = Policy(URL, execution=execution)
    results[execution] = (arm.run(p, seconds=8), len(p.calls))
    arm.sim.close()
print(f"{'execution':<11} {'steps':>5} {'holds':>5} {'held s':>6} {'misses':>6} {'chunks':>6}  stopped_by")
for name, (r, calls) in results.items():
    chunks = calls if name == "sequential" else "-"
    print(f"{name:<11} {r.steps:>5} {r.holds:>5} {r.held_seconds:>6.2f} {r.deadline_misses:>6} {chunks:>6}  {r.stopped_by}")

# %% [markdown]
# The same rule holds for any policy: `None` is a hold.

# %%
arm = Robot("so101")
r = arm.run(Policy(lambda obs: None if int(obs.t * 30) % 3 else {"shoulder_pan": 0.5}), hz=30, seconds=1)
print(f"{r.steps} steps, {r.holds} holds, {r.deadline_misses} misses")
arm.sim.close()

# %% [markdown]
# ## When the server dies mid-run
#
# Two seconds in, the server process is killed (`SIGKILL`, no goodbye). The client sees its liveliness token vanish,
# plays out what it already has, and ends the run as a fault naming the cause. The arm holds its last target.

# %%
arm = remote_arm()
doomed = Policy(URL)
killer = threading.Timer(2.0, server.send_signal, args=(signal.SIGKILL,))
killer.start()
t0 = time.perf_counter()
r = arm.run(doomed, seconds=20)
print(f"after {time.perf_counter() - t0:.2f} s: {r.steps} steps, stopped by {r.stopped_by!r}")
print("fault:", r.fault)
print("server exit code:", server.wait(10))
arm.sim.close()

# %% [markdown]
# The server's side of the story, from its log:

# %%
for line in log.read_text().splitlines():
    if "e/server.py" in line or "cy_server.py" in line:
        print(line.split(" ", 3)[-1][:150])

# %% [markdown]
# ## A hosted endpoint
#
# A server across the internet is the same line with the link it listens on, written as a lease gives it:
#
# ```python
# from strands_robots import Policy, Robot
#
# arm = Robot("so101", port="/dev/tty.usbmodem...", id="my_follower")   # needs a real arm
# policy = Policy("zenoh://tls/<host>:7447/<deployment>")
# arm.run(policy, instruction="put the x in the centre", seconds=30)
# ```
#
# Off the LAN the client changes LeRobot's defaults, and anything you name yourself wins:

# %%
from strands_robots.policy.factory import zenoh_locator
from strands_robots.policy.lerobot.link import wan_defaults

for spec in ("zenoh://127.0.0.1:7450/act", "zenoh://gpu-box:7447/act", "zenoh://tls/zenoh.example.com:7447/act"):
    endpoint, deployment = zenoh_locator(spec)
    print(f"{spec}\n  endpoint {endpoint!r}, deployment {deployment!r}\n  {wan_defaults(endpoint, {}) or 'LeRobot defaults'}")

# %% [markdown]
# Next: **07 · Dashboard** puts a browser on the same world.
