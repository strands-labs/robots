import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # nb: script-only
# %% [markdown]
# # 08 · The robot is a device
#
# Every `Robot` carries a generated manifest, a safety chain that every agent or dashboard write passes, and an
# e-stop that latches until a named operator releases it. Everything below runs in MuJoCo or on an in-memory
# driver; nothing touches a serial port.
#
# ![The safety chain: range, safe, interlocks, validators, operator gate; estop bypasses it](media/08/chain.svg)

# %%
import threading
import time
import numpy as np
from PIL import Image, ImageDraw

from strands_robots import Robot, Simulation
from strands_robots.robot.safety import EStopped, RequestRefused, Watchdog

import _nb

MEDIA = _nb.MEDIA / "08"
MEDIA.mkdir(parents=True, exist_ok=True)

sim = Simulation("mujoco", render_size=(480, 360))
arm = Robot("so101", sim=sim)
sim.camera("cam", position=(0.55, -0.45, 0.4), target=(0.0, 0.0, 0.15), size=(480, 360))
g1 = Robot("g1", sim=Simulation())  # a neighbour in its own world, for its manifest
print(arm, "|", g1)

# %% [markdown]
# ## The manifest
#
# `robot.manifest` is generated from the robot's description and where its body lives. The mesh, the agent tools and
# the dashboard all read this one object.

# %%
for r in (arm, g1):
    m = r.manifest
    print(f"{m.device_id:10s} type={m.device_type:10s} real={m.real} hold={m.position_hold} world={m.world} "
          f"sensors={len(m.sensors)} controls={len(m.controls)} rpcs={len(m.rpcs)}")
print("g1 floating base sensor:", g1.manifest.sensors[-1])

# %%
fmt = lambda b: "-" if b is None else f"[{b[0]:+.4f}, {b[1]:+.4f}]"
print(f"{'control':16s} {'unit':4s} {'range':20s} {'safe':20s} on_disconnect")
for c in arm.manifest.controls:
    print(f"{c.name:16s} {c.unit:4s} {fmt(c.range):20s} {fmt(c.safe):20s} {c.on_disconnect}")

# %%
for r in arm.manifest.rpcs:
    params = ", ".join(r.params.get("properties", {})) or "-"
    print(f"{r.name:10s} ({params})  {r.doc}")
print("events:", [(e.name, e.hz) for e in arm.manifest.events])
d = arm.describe()
print("describe(): task =", d["task"], "| estop =", d["estop"], "| keys include", sorted(d)[:6])

# %% [markdown]
# ## A refused write
#
# `confirm_act()` is the call an agent tool or the dashboard makes. Your own `robot.act()` is the operator's and skips
# the chain. A refusal is a `RequestRefused` (a `PermissionError`) naming the joint, the value and the bound.

# %%
try:
    arm.confirm_act({"elbow_flex": 3.1})
except RequestRefused as e:
    print(type(e).__name__, "->", e)
print("RequestRefused is a PermissionError:", issubclass(RequestRefused, PermissionError))

# %% [markdown]
# ## The chain, traced
#
# Append your own checks to `robot.interlocks` or `robot.validators`; each sees the full target vector in order. A
# range refusal stops before either runs.

# %%
trace = []


def interlock(robot, request):
    trace.append("interlock")


def cup_held(robot, request):
    trace.append("validator")
    if abs(dict(request.items())["wrist_roll"]) > 0.1:
        raise RequestRefused("wrist_roll must stay within 0.1 rad of zero while the cup is held")


arm.interlocks.append(interlock)
arm.validators.append(cup_held)
for label, action in [("out of range", {"elbow_flex": 3.1}),
                      ("validator says no", {"wrist_roll": 0.5}),
                      ("all clear", {"wrist_roll": 0.05})]:
    trace.clear()
    try:
        arm.confirm_act(action)
        outcome = "written"
    except RequestRefused as e:
        outcome = f"refused: {e}"
    print(f"{label:18s} ran: {' -> '.join(['range', 'safe', *trace]) if trace else 'range (stopped)':38s} {outcome}")
arm.act({"wrist_roll": 0.5}, seconds=0.5)  # the operator's own code: no chain
print(f"robot.act (operator) wrist_roll = {arm.observe().joints['wrist_roll']:+.3f}")
arm.interlocks.remove(interlock)
arm.validators.remove(cup_held)

# %% [markdown]
# ## A hardware-like body
#
# `Robot(name, driver=<Driver>)` is a real robot as far as the package is concerned (`is_real=True`). The package
# ships no in-memory driver, so here is one: it meets the `Driver` protocol and keeps its joints in a numpy array.

# %%
from strands_robots.description import Description
from strands_robots.types import JointState


class MemoryDriver:
    """A bus with no wire: writes land at once, reads return them."""

    supports_position_hold, calibration, resolution = True, None, None

    def __init__(self, name):
        self.names = Description.load(name).action_keys
        self.q = np.zeros(len(self.names))
        self.writes, self.stops = 0, 0

    def read(self):
        n = len(self.names)
        return JointState(self.names, self.q.copy(), np.zeros(n), np.zeros(n))

    def write(self, targets):
        self.q, self.writes = np.asarray(targets, dtype=float).copy(), self.writes + 1

    def pose(self): return None
    def images(self): return {}
    def extras(self): return {"writes": self.writes}
    def hold(self): pass
    def estop(self): self.stops += 1
    def calibrate(self, *, interactive=True): raise NotImplementedError("no bus")
    def close(self): pass


bus = MemoryDriver("so101")
real = Robot("so101", driver=bus, twin=False)
print(real, "| is_real:", real.is_real, "| manifest.real:", real.manifest.real)
real.act({"shoulder_pan": 0.2})  # your own code is the operator: no gate
print("direct act -> shoulder_pan", round(real.observe().joints["shoulder_pan"], 3), "| bus writes", bus.writes)
try:
    real.confirm_act({"shoulder_pan": 0.3})  # what an agent tool or the dashboard sends
except RequestRefused as e:
    print("agent write, no confirm=:", e)

# %% [markdown]
# `confirm=` is the operator's yes for one command, asked with exactly the targets. `approve=` is the yes for a whole
# agent run, asked once with its `RunPlan`. Inside `from_agent()` the run loop gates every step.

# %%
from strands_robots.robot.run.source import from_agent

asked = []
real = Robot("so101", driver=MemoryDriver("so101"), twin=False,
             confirm=lambda t: asked.append(np.round(t, 3).tolist()) or abs(t[0]) < 0.5,
             approve=lambda plan: print("operator sees:", plan) or plan.steps <= 30)
real.confirm_act({"shoulder_pan": 0.3})
print("confirm asked with", asked[-1], "-> written")
try:
    real.confirm_act({"shoulder_pan": 0.8})
except RequestRefused as e:
    print("operator said no:", e)


def sway(obs):
    return {"shoulder_pan": 0.2 * np.sin(4 * obs.t)}


with from_agent():
    res = real.run(sway, hz=30, steps=15)
    print("approved run:", res.stopped_by, res.steps, "steps")
    try:
        real.run(sway, hz=30, steps=300)
    except RequestRefused as e:
        print("refused run:", e)

# %% [markdown]
# A body with no proven `on_disconnect` (the G1's registry entry has none) cannot be put under a watchdog, so it can
# never be mounted where a mover may vanish.

# %%
g1_real = Robot("g1", driver=MemoryDriver("g1"), twin=False)
print("g1 hardware.on_disconnect:", g1_real.description.hardware.on_disconnect)
try:
    Watchdog(g1_real, timeout_ms=200)
except RequestRefused as e:
    print("Watchdog refused:", e)
g1_real.close()
real.close()

# %% [markdown]
# ## Where a run starts, and how fast a write may move
#
# A policy that declares `start_pose` (in its own units, here radians) is ramped there over `start_seconds` before
# its first tick: a chunked policy trained from one rest pose never sees a jump.

# %%
class FromRest:
    start_pose = {"shoulder_pan": 0.6, "elbow_flex": 0.9}
    start_seconds = 1.0

    def act(self, obs, *, instruction=None):
        return None  # hold: this policy only shows where the run began

    def reset(self):
        pass


arm.act({k: 0.0 for k in arm.action_keys}, seconds=0.5)
events = []
off = arm.subscribe("run.started", events.append)
res = arm.run(FromRest(), steps=5)
off()
q = arm.observe().joints
print("run.started payload seconds:", events[0]["seconds"], "| start_pose:", np.round(events[0]["start_pose"], 3).tolist())
print(f"after the run: shoulder_pan={q['shoulder_pan']:+.3f} elbow_flex={q['elbow_flex']:+.3f} | holds={res.holds}")

# %% [markdown]
# On a Feetech bus every write is capped at `max_speed` (default 2 rad/s): a goal 1 rad away becomes a ramp. `Slew`
# is that cap on its own, on a hand-stepped clock so the numbers are exact.

# %%
from strands_robots.robot.drivers.slew import MAX_SPEED, Slew

clock = [0.0]
slew = Slew(MAX_SPEED, now=lambda: clock[0])
slew.seen(np.zeros(1))
rows = []
for i in range(1, 7):
    clock[0] = i / 30  # one write per 30 Hz tick
    rows.append(float(slew.apply(np.array([1.0]), read=lambda: None)[0]))
print(f"max_speed={MAX_SPEED} rad/s, goal 1.0 rad, written per tick:", [round(r, 4) for r in rows])
print("writes the cap shortened:", slew.count, "| Slew(None) writes the goal as given:",
      float(Slew(None).apply(np.array([1.0]), read=lambda: None)[0]))

# %% [markdown]
# ```python
# # needs a real arm: the same cap on the bus, or opted out
# arm = Robot("so101", port="/dev/tty.usbmodem5A7A0185821", id="blue_follower", max_speed=1.0)
# arm = Robot("so101", port="/dev/tty.usbmodem5A7A0185821", id="blue_follower", max_speed=None)
# ```

# %% [markdown]
# ## `estop()` mid-run
#
# A policy waves the arm in real time; after one second another thread calls `arm.estop()`. It takes no lock and
# skips the chain; the run loop sees the latch on its next tick.
#
# ![estop latches, every write is refused, resume needs a named operator for the stop they saw](media/08/estop_latch.svg)

# %%
arm.act({k: 0.0 for k in arm.action_keys}, seconds=0.5)
frames, ticks = [], {"before": 0, "after": 0}


def caption(img, text, color=(0, 204, 96)):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(40, 41, 42))
    d.text((8, 5), text, fill=color)
    return np.asarray(im)


def wave(obs):
    ticks["after" if arm.estopped else "before"] += 1
    t = obs.t
    return {"shoulder_pan": 0.8 * np.sin(1.5 * t), "shoulder_lift": -0.4 + 0.3 * np.sin(t),
            "elbow_flex": 0.6 + 0.4 * np.sin(1.2 * t), "wrist_flex": 0.5 * np.sin(2 * t), "gripper": 0.6}


def film(obs):  # stop_when runs on the run's own thread, where rendering is safe
    if round(obs.t * 30) % 2 == 0:
        frames.append(caption(sim.render("cam"), f"arm.run(wave)  t={obs.t:4.2f}s  running"))
    return False


def operator():
    time.sleep(1.0)
    arm.estop(by="alice")


threading.Thread(target=operator, name="operator").start()
res = arm.run(wave, hz=30, seconds=10, realtime=True, stop_when=film)
q0 = np.array(arm.observe().joints.positions)
drift = 0.0
for _ in range(20):  # 1.3 s more world time with the latch engaged
    sim.step(int((1 / 15) / sim.physics.timestep))
    drift = max(drift, float(np.max(np.abs(np.array(arm.observe().joints.positions) - q0))))
    frames.append(caption(sim.render("cam"), f"estop(by='alice')  t={sim.time:4.2f}s  holding", (255, 107, 107)))
print(f"stopped_by={res.stopped_by!r} steps={res.steps} | policy calls after the press: {ticks['after']}")
print(f"latch: engaged={arm.estopped} by={arm.estop_latch.stopped_by!r} count={arm.estop_latch.count}")
print(f"max joint drift while latched: {np.rad2deg(drift):.3f} deg")
frames[0].shape and Image.fromarray(frames[0]).save(
    MEDIA / "estop_run.gif", save_all=True, append_images=[Image.fromarray(f) for f in frames[1:]],
    duration=66, loop=0, optimize=True)
print(f"media/08/estop_run.gif: {len(frames)} frames, {(MEDIA / 'estop_run.gif').stat().st_size / 1e6:.2f} MB")

# %% [markdown]
# ![The SO-101 waving, then holding still after estop()](media/08/estop_run.gif)

# %% [markdown]
# ## The latch and `resume()`
#
# While latched every write is refused, the operator's own `act` included. `resume()` takes the approver's name, and
# `stop=` pins it to the stop the operator was shown.

# %%
for label, call in [("act while latched", lambda: arm.act({"shoulder_pan": 0.2})),
                    ("resume()", lambda: arm.resume()),
                    ("resume(approved_by='  ')", lambda: arm.resume(approved_by="  "))]:
    try:
        call()
        print(f"{label}: accepted")
    except RequestRefused as e:
        print(f"{label}: {type(e).__name__} -> {e}")
print("EStopped is a RequestRefused:", issubclass(EStopped, RequestRefused))

seen = arm.estop_latch.count  # what the operator's screen showed
arm.resume(approved_by="alice", stop=seen)
arm.estop(by="bob")  # a second, different stop
try:
    arm.resume(approved_by="alice", stop=seen)  # alice's yes was for the first one
except RequestRefused as e:
    print("stale yes:", e)
arm.resume(approved_by="bob", stop=arm.estop_latch.count)
print("released:", not arm.estopped, "| resumed_by:", arm.estop_latch.resumed_by)

# %% [markdown]
# ## `hold()` and `subscribe()`

# %%
arm.act({"shoulder_pan": 0.6, "elbow_flex": 0.8}, seconds=0.5)
arm.hold()
q0 = np.array(arm.observe().joints.positions)
sim.step(int(1.0 / sim.physics.timestep))
print(f"hold(): max joint change over 1 s of world time {np.rad2deg(np.max(np.abs(arm.observe().joints.positions - q0))):.3f} deg")

got, seen_events = [], []
unsubscribe = arm.subscribe("state", got.append, hz=20)
offs = [arm.subscribe(e, lambda p, e=e: seen_events.append(e)) for e in ("run.started", "run.stopped", "estop")]
arm.run(wave, hz=30, seconds=0.5, realtime=True)
unsubscribe()
for off in offs:
    off()
n = len(got)
print(f"{n} state samples; seq contiguous: {[g['seq'] for g in got] == list(range(n))}; events: {seen_events}")
print("one sample:", {k: round(v, 4) for k, v in list(got[-1].items())[:5]})

# %% [markdown]
# ## The watchdog
#
# `Watchdog(robot, timeout_ms=)` wants a `beat()` per command. When the beats stop past the timeout, each control
# applies its `on_disconnect` once. A simulated body's watchdog runs on world time, so the boundary is exact.

# %%
arm.act({"shoulder_pan": 0.6}, seconds=0.3)
with sim.manual_time():
    dog = Watchdog(arm, timeout_ms=100)
    dog.beat()
    t0 = sim.time
    for dt in (0.05, 0.06, 0.0):
        sim.step(round(dt / sim.physics.timestep))
        print(f"{(sim.time - t0) * 1000:5.0f} ms since beat -> poll() fired: {dog.poll()}")
    dog.beat()
    print("    0 ms after a new beat -> poll() fired:", dog.poll())

# %%
from strands_robots.robot.device import ON_DISCONNECT, Control

print("on_disconnect is one of", ON_DISCONNECT)
for kw in ({"on_disconnect": "apply"}, {"safe": (-3.0, 0.0)}):
    try:
        Control("gripper.pos", "rad", (-0.17, 1.75), kw.get("safe", (-0.17, 1.75)), kw.get("on_disconnect", "hold"))
    except ValueError as e:
        print("refused when built:", e)

# %% [markdown]
# ```python
# # needs a real arm: the same verbs on the bench
# arm = Robot("so101", port="/dev/tty.usbmodem5A7A0185821", id="blue_follower",
#             confirm=lambda targets: input(f"move to {targets}? ") == "y")
# arm.manifest.real, arm.twin          # True, the simulated twin that mirrors every read
# arm.estop(by="alice")                # torque stays on at the measured pose (position_hold)
# arm.resume(approved_by="alice", stop=arm.estop_latch.count)
# ```

# %%
g1.sim.close()
sim.close()
print("closed")
