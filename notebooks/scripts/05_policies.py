# %% [markdown]
# # 05 · Policies
#
# A policy turns an observation into an action. `Policy(spec)` picks the provider from the form of `spec`.
#
# ![Policy(spec) fans out to five providers by form](media/05/forms.svg)
#
# Needs `uv pip install -e ".[mujoco,onnx,lerobot]"`. The ONNX and LeRobot weights download from the Hugging Face Hub
# on first use.

# %%
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import math
from pathlib import Path

import numpy as np
from PIL import Image

from strands_robots import Policy, Robot, Simulation
from strands_robots.policy.factory import FORMS

media = Path("media/05") if Path("media").is_dir() else Path("notebooks/media/05")
media.mkdir(parents=True, exist_ok=True)


def save_gif(frames, name, *, width=360, fps=25):
    """Frames -> media/05/<name>.gif, scaled to `width` px; returns the path."""
    h = round(frames[0].shape[0] * width / frames[0].shape[1])
    imgs = [Image.fromarray(f).resize((width, h)) for f in frames]
    path = media / f"{name}.gif"
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0)
    print(path, f"{path.stat().st_size / 1e3:.0f} KB")
    return path



def quiet_import(module):
    """Import `module` with fd 2 closed: LeRobot's import chain prints three known-harmless macOS lines (torch's
    "Redirects are not supported" note, and an ObjC notice that opencv and av both bundle libavdevice)."""
    import contextlib
    import importlib
    import io
    import sys

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
print("\n".join(FORMS))

# %% [markdown]
# ## A function is a policy
#
# `Policy(f)` wraps `f(obs) -> action`. The action may name only the joints it moves, in radians; on `dev` the others are re-targeted to their measured pose each step and sag (01).

# %%
arm = Robot("so101")
arm.sim.camera("view", position=[0.45, -0.45, 0.35], target=[0, 0, 0.12], size=(320, 240))

wave = Policy(lambda obs: {"shoulder_pan": 0.8 * math.sin(3 * obs.t), "elbow_flex": 0.5 * math.sin(2 * obs.t)})
frames = []
r = arm.run(wave, hz=30, seconds=3, stop_when=lambda obs: frames.append(arm.sim.render("view")) or False)
print(wave, "|", r.steps, "steps, stopped by", r.stopped_by, "| final pan", round(float(r.final.joints["shoulder_pan"]), 3), "rad")
save_gif(frames[::2], "wave", fps=15)
arm.sim.close()

# %% [markdown]
# ![the SO-101 driven by a function](media/05/wave.gif)
#
# A form it does not know is refused before anything downloads, naming the forms it does know.

# %%
for spec, kw in [("nope", {}), ("microduck?vz=1", {}), ("microduck", {"bogus": 1}), ("hold", {}), (42, {})]:
    try:
        Policy(spec, **kw)
    except (ValueError, TypeError) as e:
        print(f"Policy({spec!r}{', **' + repr(kw) if kw else ''})\n  {type(e).__name__}: {str(e).splitlines()[0]}")

# %% [markdown]
# ## A named ONNX preset: the G1 stands
#
# `Policy("holosoma")` is a whole-body controller for the Unitree G1: a hash-checked ONNX graph run with onnxruntime
# at 50 Hz. It installs the PD gains it was trained with before step one.


# %%
def watch(robot, sim, policy, seconds, every=5):
    """Run `policy`, keep the base height every step and a frame every `every` steps."""
    frames, zs, x0 = [], [], float(robot.pose.position[0])

    def tick(obs):
        zs.append(float(obs.pose.position[2]))
        if len(zs) % every == 1:
            frames.append(sim.render("cam"))
        return False

    res = robot.run(policy, seconds=seconds, stop_when=tick)
    return res, np.array(zs), float(robot.pose.position[0]) - x0, frames


sim = Simulation("mujoco", render_size=(400, 300))
g1 = Robot("g1", sim=sim, keyframe="stand")
sim.camera("cam", position=(2.0, -2.0, 1.4), target=(0.0, 0.0, 0.75), size=(400, 300))
stand = Policy("holosoma")
print(stand, "| hz", stand.hz, "| units", stand.units, "| gains kp[:3]", stand.gains[0][:3])
res, zs, _, frames = watch(g1, sim, stand, seconds=8)
print(f"{res.steps} steps at {res.hz_target:g} Hz, stopped by {res.stopped_by} | pelvis min {zs.min():.3f} m, final {zs[-1]:.3f} m")
save_gif(frames, "g1_stands", fps=10)
sim.close()

# %% [markdown]
# ![the G1 standing under holosoma](media/05/g1_stands.gif)
#
# ## Parameters ride in the spec: the duck walks
#
# `"microduck?vx=0.3"` is `Policy("microduck", vx=0.3)`: Pollen's walking policy with a 0.3 m/s forward command.

# %%
sim = Simulation("mujoco", render_size=(400, 300))
duck = Robot("microduck", sim=sim)
sim.camera("cam", position=(0.65, -1.9, 0.7), target=(0.65, 0.0, 0.05), size=(400, 300))
walk = Policy("microduck?vx=0.3")
res, zs, dx, frames = watch(duck, sim, walk, seconds=8)
print(f"{walk} | {res.steps} steps, stopped by {res.stopped_by} | walked {dx:.2f} m in 8 s | height min {zs.min():.3f} m")
save_gif(frames, "duck_walks", fps=10)
sim.close()

# %% [markdown]
# ![the microduck walking forward](media/05/duck_walks.gif)
#
# ## A LeRobot checkpoint from the Hub, in-process
#
# `"owner/repo"` is a LeRobot checkpoint. [`robotfuel/act_so101_t16b`](https://huggingface.co/robotfuel/act_so101_t16b)
# is a public ACT trained on a real SO-101 with one 640x480 wrist camera. Its config says what it needs:

# %%
import json

from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

repo = "robotfuel/act_so101_t16b"
cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
print({k: v["shape"] for k, v in cfg["input_features"].items()}, "->", cfg["output_features"]["action"]["shape"])
print("chunk_size", cfg["chunk_size"], "| n_action_steps", cfg["n_action_steps"])

# %% [markdown]
# A checkpoint does not say what units its numbers are in. Its saved normalizer stats do: `wrist_roll` reaches -155,
# outside LeRobot's -100..100 range mode, so this one was trained in degrees.

# %%
stats = load_file(hf_hub_download(repo, "policy_preprocessor_step_3_normalizer_processor.safetensors"))
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
for stat in ("min", "max"):
    print(f"state {stat:>3}", {n: round(float(v), 1) for n, v in zip(names, stats[f"observation.state.{stat}"])})

# %% [markdown]
# Name the units with `norm=` (LeRobot's own per-motor norm modes). Left out, the SO-101 registry default
# (range mode) is assumed. The sim needs a camera named `wrist` at the trained size, on the gripper.


# %%
def act_arm():
    a = Robot("so101")
    a.sim.camera("wrist", position=(0.0, 0.0, 0.05), target=(0.0, -0.1, -0.05), size=(640, 480), attach="so101/gripper")
    a.sim.camera("view", position=[0.45, -0.45, 0.35], target=[0, 0, 0.12], size=(320, 240))
    return a


import logging

# lerobot reads the checkpoint's saved device ("cuda") before device="cpu" applies and warns; the run is on cpu.
logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
finals = {}
for label, norm in [("registry default", None), ("degrees", {"*": "degrees", "gripper": "range_0_100"})]:
    arm = act_arm()
    act = Policy(repo, device="cpu", norm=norm)
    frames = []
    r = arm.run(act, seconds=4, stop_when=lambda obs: frames.append(arm.sim.render("view")) or False)
    finals[label] = (act.inner.contract.semantics, r, frames, arm.observe().images["wrist"])
    arm.sim.close()

for label, (sem, r, _, _) in finals.items():
    deg = np.degrees(r.final.joints.positions).round(1)
    print(f"{label:>16}: {sem} | {r.steps} steps, {r.stopped_by} | final (deg) {dict(zip(names, deg.tolist()))}")
print(act, "is a", type(act).__name__, "around", type(act.inner).__name__)

# %% [markdown]
# Under the wrong units `wrist_roll` is driven into its -160° limit; under degrees it stays inside the -156..-72° the
# training data covers. The model never saw this sim, so the motion is not a task; the units are the point.

# %%
_, r, frames, wrist = finals["degrees"]
save_gif(frames[::2], "act_in_process", fps=15)
Image.fromarray(wrist).resize((320, 240)).save(media / "act_wrist.png")
print(media / "act_wrist.png")

# %% [markdown]
# ![the Hub ACT driving the sim SO-101](media/05/act_in_process.gif) ![what the policy sees](media/05/act_wrist.png)
#
# ## Units, at the edge only
#
# Inside strands every joint is radians. A LeRobot deployment names its units with a `semantics` string; strands maps
# that to a conversion with `units/semantics.json`.

# %%
from strands_robots import Description
from strands_robots.units import semantics_for, units_for_semantics

desc = Description.load("so101")
q = np.array([0.5, -0.3, 0.2, 0.0, 0.1, 0.8])  # radians
for norm in (None, {"*": "degrees", "gripper": "range_0_100"}):
    sem = semantics_for(desc, norm)
    u = units_for_semantics(sem, desc)
    print(f"{sem}: {np.round(u.from_rad(q), 1).tolist()}  round trip {np.allclose(u.to_rad(u.from_rad(q)), q)}")

# %% [markdown]
# ## Wrappers
#
# `SequencePolicy` plays a fixed list then holds the last action. `Perturb` makes any policy see the world the way a real
# arm reports it: noisy joints, noisy pixels, late actions.

# %%
from strands_robots.policy import Perturb, SequencePolicy

arm = Robot("so101")
script = SequencePolicy([{"shoulder_pan": 0.1 * i} for i in range(10)])
r = arm.run(Perturb(script, joint_noise=0.01, action_delay=3, seed=0), hz=30, seconds=1)
print(r.steps, "steps | pan", round(float(r.final.joints["shoulder_pan"]), 3), "rad (the last scripted target is 0.9)")
arm.sim.close()

# %% [markdown]
# Next: **06 · Remote inference** serves the same LeRobot checkpoint over Zenoh and drives the arm from another process.
