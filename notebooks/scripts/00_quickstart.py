# %% [markdown]
# # 00 · Quickstart
#
# An SO-101 arm in a simulated world: create it, read it, move it, watch it. Five minutes.
#
# ![observe, act, render](media/00/loop.svg)
#
# ## Install
#
# ```bash
# git clone https://github.com/strands-labs/robots && cd robots && git checkout dev
# uv venv -p 3.12 && source .venv/bin/activate
# uv pip install -e ".[mujoco]"
# ```
#
# That is everything this notebook needs. Agents, recording and policies need more extras; their notebooks say which.

# %%
from strands_robots import Robot

arm = Robot("so101")
arm

# %% [markdown]
# No world was given, so the arm joined `Simulation.default()`, a headless MuJoCo world created for it.

# %%
arm.sim

# %% [markdown]
# `observe()` reads the body: joint positions in radians, the base pose, and images from cameras attached to it.

# %%
obs = arm.observe()
print("t =", obs.t)
print({k: round(v, 3) for k, v in obs.joints.as_dict().items()})

# %% [markdown]
# `act()` writes joint targets and advances the world one control step (20 ms). A joint you leave out is re-targeted to where it is now, so gravity pulls it a little each step (01 shows it).

# %%
obs = arm.act({"shoulder_pan": 0.5})
print(f"t = {obs.t:.2f} s, shoulder_pan = {obs['shoulder_pan']:.3f} rad")

# %% [markdown]
# One step only moves part of the way. `seconds=` ramps to the target and returns the pose reached.

# %%
obs = arm.act({"shoulder_pan": 0.5, "elbow_flex": 0.6}, seconds=1.0)
print(f"t = {obs.t:.2f} s, shoulder_pan = {obs['shoulder_pan']:.3f}, elbow_flex = {obs['elbow_flex']:.3f}")

# %% [markdown]
# A camera, and one frame from it.

# %%
from PIL import Image

arm.sim.camera("front", position=[0.5, -0.6, 0.42], target=[-0.08, -0.05, 0.08], size=(480, 360))
Image.fromarray(arm.sim.render("front"))

# %% [markdown]
# A wave: 90 control steps, one frame every second step.

# %%
import math

frames = []
for i in range(90):
    t = i * 0.02
    arm.act({"shoulder_pan": 0.8 * math.sin(3 * t), "elbow_flex": 0.6 + 0.4 * math.sin(5 * t)})
    if i % 2 == 0:
        frames.append(arm.sim.render("front"))
print(len(frames), "frames,", f"{arm.sim.time:.2f} s of sim time")

# %% [markdown]
# ![the arm waving](media/00/wave.gif)

# %%
from pathlib import Path

media = Path("media/00") if Path("media").is_dir() else Path("notebooks/media/00")
media.mkdir(parents=True, exist_ok=True)
gif = [Image.fromarray(f).resize((360, 270)) for f in frames]
gif[0].save(media / "wave.gif", save_all=True, append_images=gif[1:], duration=40, loop=0)
print(media / "wave.gif", f"{(media / 'wave.gif').stat().st_size / 1e3:.0f} KB")

# %% [markdown]
# ## Hand it to an agent
#
# `arm.tool` and `arm.sim.tool` are [Strands](https://strandsagents.com) agent tools generated from the robot and the
# world. With `uv pip install -e ".[mujoco,tools]"` and Amazon Bedrock credentials:
#
# ```python
# from strands import Agent
#
# agent = Agent(tools=[arm.tool, arm.sim.tool])
# agent("add a red cube in front of the arm, then point the arm at it")
# while True:
#     agent(input("> "))
# ```
#
# [03](03_agent_builds_a_world.ipynb) runs this for real.
#
# ## Next
#
# [01](01_robot_and_sim.ipynb): what a robot is, two robots in one world, cameras, and the run loop.
