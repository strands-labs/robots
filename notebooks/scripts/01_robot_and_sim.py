# %% [markdown]
# # 01 · Robot and Simulation
#
# A `Robot` is a body in a world. A `Simulation` is the world: it owns time and holds every robot in it.
# This notebook puts an SO-101 arm and a Unitree G1 in one world, moves them, films them, and runs the one loop.
#
# ![two robots, one world, one clock](media/01/one_world.svg)
#
# Needs `uv pip install -e ".[mujoco,onnx]"` (`onnx` for the G1's standing policy, fetched from the Hugging Face Hub once).

# %%
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from strands_robots import Description, Policy, Robot, Simulation

MEDIA = Path("media/01") if Path("media").is_dir() else Path("notebooks/media/01")
MEDIA.mkdir(parents=True, exist_ok=True)


def gif(frames, name, size, ms):
    """Save frames as an animated GIF in MEDIA; returns its size in KB."""
    imgs = [Image.fromarray(f).resize(size) for f in frames]
    imgs[0].save(MEDIA / name, save_all=True, append_images=imgs[1:], duration=ms, loop=0)
    return round((MEDIA / name).stat().st_size / 1e3)


# %% [markdown]
# ## A description, before any world
#
# `Description.load(name)` resolves the model file from the registry and reads joints, actuators and limits from the
# model itself. Nothing is simulated yet.

# %%
d = Description.load("so101")
print(d.name, "|", d.format, "|", d.category, "|", d.model_path.name)
print("dof", d.dof, "| action_keys", d.action_keys)
print("keyframes", d.keyframes, "| floating base", d.floating)
print("gripper", d.gripper)
print("hardware", d.hardware.driver, d.hardware.lerobot_type, dict(d.hardware.lerobot_norm))
for k, (lo, hi) in d.limits.items():
    print(f"  {k:13s} {math.degrees(lo):7.1f} .. {math.degrees(hi):6.1f} deg")

# %%
g = Description.load("g1")
print(g.name, "| dof", g.dof, "| floating base", g.floating, "| keyframes", g.keyframes)

# %% [markdown]
# ## Two robots, one world
#
# A robot given no `sim=` joins `Simulation.default()`. Both land in the same world: one physics step, one clock.
# Ids are unique per world; a second `so101` would be `so101_2`.

# %%
arm = Robot("so101")
g1 = Robot("g1", position=[0.7, 0.3, 0], keyframe="stand")
sim = arm.sim
assert sim is g1.sim is Simulation.default()
print(sim)
print(sim.physics, "-> control_dt", sim.physics.control_dt, "s")
print("ids", list(sim.robots))

# %% [markdown]
# ## observe, act, set_pose
#
# `observe()` returns an `Observation`: joints in radians, the base pose (a floating base also reports its velocities),
# images. `act()` writes targets for this body and steps the whole world once.

# %%
o = g1.observe(images=False)
print("g1 pelvis at", o.pose.position.round(3), "| moving at", o.pose.linear_velocity.round(3), "m/s")

for _ in range(40):
    o = arm.act({"shoulder_pan": 0.5, "shoulder_lift": -0.4, "elbow_flex": 0.5})
print(f"after 40 acts: t = {sim.time:.2f} s; the G1 reads the same clock: {g1.observe(images=False).t:.2f} s")
print({k: round(v, 3) for k, v in o.joints.as_dict().items()})

# %% [markdown]
# An unknown joint is refused by name; a target past a limit is clipped to it.

# %%
try:
    arm.act({"elbo_flex": 0.1})
except KeyError as e:
    print("KeyError:", e)
targets = arm.coerce({"elbow_flex": 9.0})
print(f"elbow_flex=9.0 becomes the target {targets[2]:.3f} rad, its upper limit")

# %% [markdown]
# `set_pose` teleports a base. It is a scene change (`revision` bumps); time does not move and joint state survives.

# %%
rev, t = sim.revision, sim.time
arm.set_pose(position=[0.0, -0.05, 0.0])
print("arm base", arm.pose.position, "| revision", rev, "->", sim.revision, "| time", round(t, 2), "->", round(sim.time, 2))

# %% [markdown]
# ## Cameras
#
# A world camera sees the scene. A camera attached to a robot (`attach="so101"`, its base) or to one of its links
# (`attach="so101/gripper"`, a wrist view) moves with it and appears in that robot's observations.

# %%
print("links:", sim.describe()["robots"]["so101"]["links"])
sim.camera("front", position=[1.5, -1.6, 1.1], target=[0.35, 0.15, 0.6], size=(480, 320))
sim.camera("wrist", position=[0.035, 0, -0.02], target=[-0.2, 0, -0.09], attach="so101/gripper", size=(480, 320))
o = arm.observe()
print("arm.observe().images:", {k: v.shape for k, v in o.images.items()})
print("g1.observe().images:", list(g1.observe().images))
Image.fromarray(np.hstack([sim.render("front"), o.images["wrist"]]))

# %% [markdown]
# ## Both robots move: `Simulation.run`
#
# `sim.run({robot: policy, ...})` drives many robots in lock-step: every robot observes, every policy acts, then the
# world steps once. The arm runs a plain function. The G1 runs `Policy("holosoma")`, an ONNX whole-body controller
# that keeps it standing (05 covers policies).

# %%
frames = []


def wave(obs):
    if round(obs.t * 50) % 5 == 0:
        frames.append(sim.render("front"))
    return {"shoulder_pan": 0.9 * math.sin(2 * obs.t), "elbow_flex": 0.5 + 0.5 * math.sin(3 * obs.t), "shoulder_lift": -0.4}


results = sim.run({arm: wave, g1: Policy("holosoma")}, hz=50, seconds=6)
for rid, r in results.items():
    print(f"{rid:11s} {r.steps} steps | {r.seconds:.2f} s sim | {r.hz_target:g} Hz | stopped_by={r.stopped_by}")
print(f"G1 pelvis height after 6 s: {g1.pose.position[2]:.3f} m | {len(frames)} frames")
print("gif", gif(frames, "two_robots.gif", (360, 240), 100), "KB")

# %% [markdown]
# ![the arm waves while the G1 stands](media/01/two_robots.gif)
#
# One clock: both results end at the same sim time.

# %%
print({rid: round(r.final.t, 4) for rid, r in results.items()})

# %% [markdown]
# ## The one loop: `Robot.run`
#
# `robot.run(policy, hz=, seconds= | steps= | stop_when=)` is observe -> policy -> act, written once. A bare callable is a
# policy. In a world, `hz` sets how many physics steps each act covers and time is sim time, so it runs as fast as the
# machine allows (`realtime=True` paces it to the wall clock). It returns a `RunResult`.
#
# ![observe, policy, act, until a limit](media/01/run_loop.svg)

# %%
r = arm.run(lambda obs: {"shoulder_pan": -0.6, "elbow_flex": 0.2}, hz=50, seconds=2)
print(f"steps={r.steps} seconds={r.seconds:.2f} hz_target={r.hz_target:g} hz_achieved={r.hz_achieved:.1f}")
print(f"stopped_by={r.stopped_by} recorded={r.recorded} fault={r.fault} deadline_misses={r.deadline_misses}")
print("final", {k: round(v, 3) for k, v in r.final.joints.as_dict().items()})

# %% [markdown]
# The policy named two joints; `shoulder_lift` and `wrist_flex` drifted. On `dev` a joint an action leaves out is
# re-targeted to where it is measured each step, so gravity wins a little every step. Return every joint (or a full
# vector) when it matters.

# %% [markdown]
# `stop_when` ends on a condition instead of a count. A run with no limit at all is refused.

# %%
r = arm.run(lambda obs: {"gripper": 1.2}, hz=30, stop_when=lambda obs: obs["gripper"] > 1.0)
print(f"stop_when: {r.steps} steps, stopped_by={r.stopped_by}, gripper={r.final['gripper']:.3f}")
try:
    arm.run(lambda obs: {})
except ValueError as e:
    print("ValueError:", e)

# %% [markdown]
# `start()` runs the same loop on a background thread; `stop()` ends it and returns its `RunResult`.

# %%
task = arm.start(lambda obs: {"wrist_roll": math.sin(obs.t)}, hz=50, realtime=True, seconds=30)
time.sleep(0.5)
r = arm.stop()
print(f"start/stop: {r.steps} steps, {r.seconds:.2f} s sim, stopped_by={r.stopped_by}")

# %% [markdown]
# ## Manual time
#
# Inside `with sim.manual_time():`, `act()` only writes targets and the caller decides when the world steps.

# %%
with sim.manual_time():
    t0 = sim.time
    arm.act({"shoulder_pan": 0.0})
    arm.act({"shoulder_pan": 0.0})
    print(f"two acts: time {t0:.3f} -> {sim.time:.3f}")
    sim.step(50)
    print(f"sim.step(50): time -> {sim.time:.3f} (50 x {sim.physics.timestep} s)")

# %% [markdown]
# ## The whole world as data
#
# `describe()` is JSON-safe: what an agent tool and the dashboard read.

# %%
w = sim.describe()
print({k: w[k] for k in ("backend", "physics", "time", "revision", "manual_time")})
print({rid: (r["dof"], [round(x, 2) for x in r["pose"]["position"]]) for rid, r in w["robots"].items()})
print(w["cameras"])
sim.close()

# %% [markdown]
# ## Next
#
# [02](02_recording.ipynb): record a run as a LeRobot dataset and read it back.
