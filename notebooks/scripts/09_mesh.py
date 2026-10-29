import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # nb: script-only
# %% [markdown]
# # 09 · Robots on a mesh
#
# `Mesh("zenoh://", tenant=...)` makes every mounted robot a device other processes can discover, call, watch and
# stop. Every write still lands in the robot's own safety chain. Here two meshes talk over loopback TCP, then a robot
# in a second OS process joins.
#
# ![Two peers on one tenant: a bench mounts robots, a controller discovers, invokes and broadcasts estop](media/09/mesh_peers.svg)

# %%
import json
import os
import subprocess
import sys
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from strands_robots import Robot, Simulation
from strands_robots.mesh import Mesh, RequestRefused, authority

import _nb

MEDIA = _nb.MEDIA / "09"
MEDIA.mkdir(parents=True, exist_ok=True)
PORT = 7450  # explicit endpoints switch multicast scouting off
TENANT = f"nb09-{uuid.uuid4().hex[:6]}"
ME, BENCH = f"zenoh://?listen=tcp/127.0.0.1:{PORT}", f"zenoh://?connect=tcp/127.0.0.1:{PORT}"

me = Mesh(ME, tenant=TENANT, operator="alice@laptop")
bench = Mesh(BENCH, tenant=TENANT, operator="bench@lab")
print(me)
print(bench)

# %% [markdown]
# ## Mount
#
# `bench.mount(robot, labels=...)` announces the robot with a presence token, answers its rpcs and publishes its state.
# `Robot("so101", mesh=bench)` does the same in one line.

# %%
arm_world = Simulation("mujoco", render_size=(480, 360))
arm = Robot("so101", sim=arm_world)
arm_world.camera("cam", position=(0.55, -0.45, 0.4), target=(0.0, 0.0, 0.15), size=(480, 360))
g1 = Robot("g1", sim=Simulation())
print(bench.mount(arm, labels={"role": "arm", "bench": "left"}, state_hz=50))
print(bench.mount(g1, labels={"role": "humanoid", "bench": "left"}))

# %% [markdown]
# ## Discover
#
# A selector is `device(*)` or `device(key:value,...)` over the labels plus `id`, `type` and `tenant`; a bare id means
# `device(id:<id>)`.

# %%
t0 = time.perf_counter()
me.discover(min_count=2, timeout=5.0)
print(f"both devices visible to the controller after {(time.perf_counter() - t0) * 1000:.0f} ms")
for sel in ("device(*)", "device(type:robot_arm)", "device(role:humanoid)", "device(bench:left,type:humanoid)",
            "so101", "device(bench:right)"):
    print(f"{sel:36s} -> {[d.id for d in me.discover(sel)]}")
d = me.discover("so101")[0]
print("record:", d, "| controls:", [c["name"] for c in d.manifest["controls"]][:3], "...")

# %% [markdown]
# ## Invoke
#
# `act` over the mesh runs `confirm_act` on the device. A refusal from the device's chain comes back as the same
# `RequestRefused`, message verbatim.

# %%
obs = me.invoke("so101", "act", action={"shoulder_pan": 0.3}, steps=50)
print("act -> the device reads shoulder_pan =", round(arm.observe().joints["shoulder_pan"], 3))
try:
    me.invoke("so101", "act", action={"shoulder_pan": 99.0})
except RequestRefused as e:
    remote = str(e)
try:
    arm.confirm_act({"shoulder_pan": 99.0})
except RequestRefused as e:
    local = str(e)
print("over the mesh:", remote)
print("verbatim:", remote == local)
for rpc, kw in (("teleport", {"x": 1}), ("act", {"action": {"shoulder_pan": 0.1}, "seconds": 1.0})):
    try:
        me.invoke("so101", rpc, **kw)
    except (LookupError, TypeError) as e:
        print(f"{rpc}{sorted(kw)}: {type(e).__name__}: {e}")
print("rtt over loopback, 50 describe calls:")
ts = []
for _ in range(50):
    t0 = time.perf_counter()
    me.invoke("so101", "describe")
    ts.append((time.perf_counter() - t0) * 1000)
print(f"  p50 {np.percentile(ts, 50):.2f} ms, p90 {np.percentile(ts, 90):.2f} ms (this host, loaded)")

# %% [markdown]
# ## State over the mesh
#
# The payload is the one `arm.subscribe("state", ...)` gives in-process; `seq` gaps are drops.

# %%
def wave(obs):
    t = obs.t
    return {"shoulder_pan": 0.8 * np.sin(1.5 * t), "shoulder_lift": -0.4 + 0.3 * np.sin(t),
            "elbow_flex": 0.6 + 0.4 * np.sin(1.2 * t), "wrist_flex": 0.5 * np.sin(2 * t), "gripper": 0.6}


got = []
unsubscribe = me.subscribe("so101", "state", got.append)
arm.run(wave, hz=30, seconds=1.0, realtime=True)
unsubscribe()
seqs = [g["seq"] for g in got]
print(f"{len(got)} samples; gaps in seq: {sum(b - a - 1 for a, b in zip(seqs, seqs[1:]))}")
print("last:", {k: round(v, 4) for k, v in got[-1].items() if k.startswith(("seq", "shoulder_pan", "elbow_flex.pos"))})

# %% [markdown]
# ## `connect()`: a Robot whose body is on the other peer
#
# Each tick of `run()` is one `observe` rpc and one `act` rpc; every write passes the device's chain.

# %%
arm.act({k: 0.0 for k in arm.action_keys}, seconds=0.5)
remote = me.connect("so101")
print(remote, type(remote.body).__name__)
frames = []


def caption(img, text, color=(0, 204, 96)):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 22], fill=(40, 41, 42))
    dr.text((8, 5), text, fill=color)
    return np.asarray(im)


def remote_wave(obs):
    if round(obs.t * 30) % 3 == 0:
        frames.append(caption(arm_world.render("cam"), f"me.connect('so101').run(wave)  t={obs.t:4.2f}s"))
    return wave(obs)


res = remote.run(remote_wave, hz=30, steps=120)
print(f"steps={res.steps} stopped_by={res.stopped_by!r}")
print("remote reads == device reads:", np.allclose(remote.observe().joints.positions, arm.observe().joints.positions))
Image.fromarray(frames[0]).save(MEDIA / "connect_run.gif", save_all=True, duration=100, loop=0, optimize=True,
                                append_images=[Image.fromarray(f) for f in frames[1:]])
print(f"media/09/connect_run.gif: {len(frames)} frames, {(MEDIA / 'connect_run.gif').stat().st_size / 1e6:.2f} MB")

# %% [markdown]
# ![The SO-101 driven from the controller mesh through connect()](media/09/connect_run.gif)

# %% [markdown]
# ## `broadcast("estop")`
#
# It goes out on the tenant's `@estop` key (every mesh latches its matching devices on hearing it, no discovery
# needed) and as an rpc to each discovered match, whose replies are the acks.
#
# ![broadcast estop: the tenant key latches every device, the per-device rpcs come back as acks](media/09/broadcast.svg)

# %%
task = arm.start(wave, hz=30, seconds=30, realtime=True)
time.sleep(0.5)
acked = me.broadcast("estop")
task.halt(2.0)
print("acked:", acked, "| arm run ended:", task.result.stopped_by)
for r in (arm, g1):
    print(r.id, "estopped:", r.estopped, "| lockout:", {k: v for k, v in me.invoke(r.id, "lockout").items() if k != "at"})

# %% [markdown]
# ## Who may resume
#
# Names are claims. Over zenoh, even on loopback, a device accepts a named resume only with `trust_names=True`; on
# any network you do not control it wants an Ed25519 assertion signed for this stop and this device.

# %%
try:
    me.invoke("so101", "resume", approved_by=me.operator)
except RequestRefused as e:
    print("default authority:", e)
bench.resume_authority = authority.Authority(names=("safety-lead",), trust_names=True)
try:
    me.invoke("so101", "resume", approved_by="mallory")
except RequestRefused as e:
    print("mallory:", e)
me.invoke("g1", "resume", approved_by="safety-lead")
me.invoke("so101", "resume", approved_by=me.operator)  # the operator who stopped it
print("after safety-lead (g1) and alice (arm):", arm.estopped, g1.estopped)

# %%
private, public = authority.keygen()
bench.resume_authority = authority.Authority(public_keys=(public,))
me.broadcast("estop", selector="so101")
try:
    me.invoke("so101", "resume", approved_by=me.operator)
except RequestRefused as e:
    print("unsigned:", e)
me.resume_key = private  # invoke() now fetches the lockout epoch and signs for it
me.invoke("so101", "resume", approved_by=me.operator)
print("signed resume -> latched:", arm.estopped, "| resumed_by:", arm.estop_latch.resumed_by)

# %% [markdown]
# ## A robot in a second OS process
#
# Same package, a new process with its own world, joining the controller's endpoint. It trusts the operator's public
# key and nothing else.

# %%
peer_code = '''
import sys, time
from strands_robots import Robot
from strands_robots.mesh import Mesh, authority
url, tenant, key = sys.argv[1:4]
mesh = Mesh(url, tenant=tenant, operator="bench-b@lab", resume_authority=authority.Authority(public_keys=(key,)))
mesh.mount(Robot("so101", id="so101-b"), labels={"role": "arm", "bench": "right"})
print("ready", flush=True)
time.sleep(120)
'''
t0 = time.perf_counter()
peer = subprocess.Popen([sys.executable, "-c", peer_code, BENCH, TENANT, public], stdout=subprocess.PIPE, text=True)
assert peer.stdout.readline().strip() == "ready"
print(f"pid {peer.pid} ready after {time.perf_counter() - t0:.1f} s (imports + MuJoCo)")
t0 = time.perf_counter()
print(me.discover("device(bench:right)", min_count=1, timeout=5.0),
      f"found {(time.perf_counter() - t0) * 1000:.0f} ms after ready")
print("arms:", [d.id for d in me.discover("device(role:arm)")])

far = me.connect("so101-b")
res = far.run(wave, hz=30, steps=60)
print(f"connect('so101-b').run: steps={res.steps} stopped_by={res.stopped_by!r}, "
      f"shoulder_pan now {far.observe().joints['shoulder_pan']:+.3f}")
print("estop device(bench:right) acked by", me.broadcast("estop", selector="device(bench:right)"))
print("lockout in the other process:", me.invoke("so101-b", "lockout")["engaged"])
me.invoke("so101-b", "resume", approved_by=me.operator)  # signed with me.resume_key
print("after a signed resume:", me.invoke("so101-b", "lockout")["engaged"])

# %%
peer.kill()
t0 = time.perf_counter()
while any(d.id == "so101-b" for d in me.discover()) and time.perf_counter() - t0 < 10:
    time.sleep(0.005)
print(f"SIGKILL -> gone from discover() after {(time.perf_counter() - t0) * 1000:.0f} ms (presence_ttl {me.presence_ttl:g} s)")
print("present:", [d.id for d in me.discover()], "| peer exit code:", peer.wait(5))

# %% [markdown]
# ## `use_mesh`: the whole room as one agent tool
#
# `me.tool` is generated from the manifests present. A refusal goes back to the model as text; the model may stop
# but never resume.

# %%
tool = me.tool
schema = tool.tool_spec["inputSchema"]["json"]
print(tool.tool_spec["description"])
print("actions:", schema["properties"]["action"]["enum"])


def call(**inp):
    r = tool._tool_func({"toolUseId": "nb", "input": inp})
    return r["status"], r["content"][0]["text"][:160]


for inp in ({"action": "list", "selector": "device(role:arm)"},
            {"action": "act", "device_id": "so101", "act": {"shoulder_pan": 99.0}},
            {"action": "resume", "device_id": "so101", "approved_by": "alice@laptop"}):
    print(inp["action"], "->", *call(**inp))

# %% [markdown]
# One real model turn (Amazon Bedrock, `STRANDS_MODEL_ID`).

# %%
from strands import Agent

agent = Agent(model=os.environ["STRANDS_MODEL_ID"], tools=[tool], callback_handler=None)
t0 = time.perf_counter()
reply = agent("List the arms on the mesh, move so101's shoulder_pan to 0.4 rad, then e-stop every device. "
              "Answer in two sentences.")
print(f"({time.perf_counter() - t0:.1f} s)", str(reply).strip())
for m in agent.messages:
    for b in m["content"]:
        if "toolUse" in b:
            print("  call:", json.dumps(b["toolUse"]["input"]))
print("arm estopped:", arm.estopped, "| shoulder_pan:", round(arm.observe().joints["shoulder_pan"], 3))
me.invoke("so101", "resume", approved_by=me.operator)  # the operator, in code
bench.resume_authority = authority.Authority(trust_names=True)
me.invoke("g1", "resume", approved_by=me.operator)
print("operator resumed:", not arm.estopped, not g1.estopped)

# %% [markdown]
# ## The same mesh over AWS IoT
#
# `iot://fake?broker=<name>` is the in-process broker with IoT Core's Direct Messaging semantics; the calls are the
# ones a real `iot://us-west-2` mesh takes.

# %%
fleet = Mesh("iot://fake?broker=nb09&client=bench", tenant=TENANT)
ops = Mesh("iot://fake?broker=nb09&client=ops", tenant=TENANT, operator="ops@cloud")
cloud_arm = Robot("so101", id="arm-1", sim=Simulation())
fleet.mount(cloud_arm, labels={"role": "arm"})
print(ops.discover(min_count=1, timeout=5.0))
ops.invoke("arm-1", "act", action={"shoulder_pan": 0.2}, steps=20)
print("estop acked by", ops.broadcast("estop"), "| latched:", cloud_arm.estopped)
ops.invoke("arm-1", "resume", approved_by="ops@cloud")  # in-process transport: names are trusted
print("resumed:", not cloud_arm.estopped)
fleet.close(); ops.close(); cloud_arm.sim.close()

# %% [markdown]
# The deployment pieces are plain data. `greengrass.recipe()` and `kvs.pipeline()` need no AWS call:

# %%
from strands_robots.mesh.iot import greengrass, kvs

r = greengrass.recipe("1.0.0", "s3://bucket/strands_robots-1.0.0-py3-none-any.whl", tenant="lab",
                      robots=[{"name": "so101", "device_id": "arm-1"}])
print(r["ComponentName"], r["ComponentVersion"], "| run:", r["Manifests"][0]["Lifecycle"]["Run"]["Script"])
print("config:", r["ComponentConfiguration"]["DefaultConfiguration"])
print("kvs stream:", kvs.stream_name("lab", "arm-1", "wrist"))
print(kvs.pipeline(kvs.stream_name("lab", "arm-1", "wrist"), 640, 480, 15, region="us-west-2"))

# %% [markdown]
# ```python
# # needs AWS IoT credentials, device certificates and a Greengrass core
# from strands_robots.mesh.iot import IotOptions, provision
# cloud = Mesh("iot://us-west-2?client=arm-1&cert=~/.strands/iot/arm-1/cert.pem&key=~/.strands/iot/arm-1/key.pem",
#              tenant="lab-a")
# cloud.mount(Robot("so101"), device_id="arm-1", iot=IotOptions(shadow_hz=0.5))
# provision.setup_tenant(iot, "lab-a", region="us-west-2", account="123456789012")
# greengrass.deploy(gg, "arn:aws:iot:us-west-2:123456789012:thinggroup/strands-lab", "1.0.0")
# kvs.publish_cameras(...)   # refuses by name without the GStreamer kvssink plugin
# ```

# %%
me.close()
bench.close()
arm_world.close()
g1.sim.close()
print("closed; peer gone:", peer.poll() is not None)
