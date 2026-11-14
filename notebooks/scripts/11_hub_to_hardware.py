# Paired script for notebooks/11_hub_to_hardware.ipynb.
# Run: .venv/bin/python notebooks/scripts/11_hub_to_hardware.py
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # nb: script-only

# %% [markdown]
# # 11 · From the Hugging Face Hub to robot hardware
#
# The companion to [From the Hugging Face Hub to robot hardware with Strands Agents and
# LeRobot](https://huggingface.co/blog/amazon/strands-lerobot-hub-to-hardware), on this release. One SO-101 and one
# agent, five steps: the agent over the robot's tools, a demonstration recorded as a LeRobot v3 dataset (and pushed
# to the Hub if you set `HF_USER`), a policy pulled from the Hub by name, the same code on a physical arm, and the
# robot on a mesh where an agent reaches it by discovery.
#
# Everything runs in MuJoCo on a laptop. Steps 1, 3 and 5 make real model calls (Amazon Bedrock with AWS
# credentials, model id in `STRANDS_MODEL_ID`); Step 3 downloads a small ACT checkpoint from the Hub.
#
# ![an agent over the robot's tools; the Robot's safety chain in front of a MuJoCo world or a USB arm; a Hub policy, a LeRobot v3 recording pushed to the Hub, and the arm on a mesh](media/11/architecture.svg)

# %%
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import math
import tempfile
import uuid
from pathlib import Path

from strands import Agent

import _nb
from strands_robots import Policy, Robot, Simulation
from strands_robots.mesh import Mesh

_nb.quiet_lerobot()
MODEL_ID = os.environ.setdefault("STRANDS_MODEL_ID", "global.anthropic.claude-opus-4-8")
HF_USER = os.environ.get("HF_USER")  # set it (and `hf auth login`) to push the dataset to your Hub account
TASK = "pick up the red cube"
WORK = Path(tempfile.mkdtemp(prefix="hub2hw-"))
print("model:", MODEL_ID, "| push to Hub:", bool(HF_USER))

# %% [markdown]
# ## Step 1: the agent over the robot's tools
#
# `Robot("so101")` is an SO-101 in a MuJoCo world (no hardware, no GPU). `arm.tool` is the robot's agent tool,
# generated from its manifest; `sim.tool` edits the world. A front camera and a wrist camera ride on the arm, so a
# recording carries both views.

# %%
sim = Simulation()
arm = Robot("so101", sim=sim)
sim.camera("front", position=(0.5, -0.6, 0.42), target=(-0.08, -0.05, 0.08), size=(320, 240), attach=arm)
sim.camera("wrist", position=(0.0, 0.0, 0.05), target=(0.0, -0.1, -0.05), size=(640, 480), attach="so101/gripper")


def wave(obs):  # a placeholder motion, so the workflow runs before you have a trained policy
    return {"shoulder_pan": 0.6 * math.sin(3.0 * obs.t), "elbow_flex": 0.5 + 0.3 * math.sin(2.0 * obs.t)}


agent = Agent(model=MODEL_ID, tools=[arm.tool(policies={"wave": wave}), sim.tool], callback_handler=None)
reply = agent("Add a red cube about 3 cm on a side on the floor in front of the arm (the arm reaches along -y, "
              "about 0.2 m out). Answer in one sentence.")
print(str(reply).strip())
print("calls:", *_nb.tool_calls(agent), sep="\n  ")
_nb.show(_nb.image(sim.render("front"), width=480))

# %% [markdown]
# ## Step 2: record a demonstration as a LeRobot dataset
#
# `run(record=...)` writes LeRobot v3 through LeRobot's own writer, one episode per run. A folder keeps it local;
# `record="hf://<user>/<name>", push=True` uploads it to the Hub when the episode ends. Recording is a code-side
# argument: the agent runs policies, the operator decides what is kept.
#
# The `wave` placeholder is intentional: the episode is structurally complete (joint states, actions, both camera
# streams) but it is not useful training data.

# %%
target = f"hf://{HF_USER}/strands-hub-to-hardware" if HF_USER else WORK / "strands-hub-to-hardware"
r = arm.run(wave, hz=30, seconds=3, record=target, push=bool(HF_USER), instruction=TASK)
print(f"{r.steps} steps at {r.hz_target} Hz, stopped by {r.stopped_by} -> {target}")

# %% [markdown]
# LeRobot's own loader reads it back, with no Strands code in the path.

# %%
from lerobot.datasets.lerobot_dataset import LeRobotDataset

if HF_USER:
    ds = LeRobotDataset(f"{HF_USER}/strands-hub-to-hardware")
else:
    ds = LeRobotDataset("local/strands-hub-to-hardware", root=target)
print("episodes", ds.num_episodes, "| frames", ds.num_frames, "| fps", ds.fps)
print("features:", [k for k in ds.features if not k.endswith("_index") and k != "timestamp"])
print("first frame:", {k: tuple(v.shape) for k, v in ds[0].items() if hasattr(v, "shape") and v.ndim})

# %% [markdown]
# ## Step 3: a policy from the Hub, by name
#
# A Hub repo id is a policy spec: `<owner>/<repo>` loads a LeRobot checkpoint in-process. This ACT was trained on a
# real SO-101 with a wrist camera, so here it moves the simulated arm but does not complete a grasp in this scene; a
# checkpoint trained on your own demonstrations swaps in by changing the string. The agent picks it by name, as it
# picks `wave`.

# %%
HUB_POLICY = "robotfuel/act_so101_t16b"
act = Policy(HUB_POLICY, device="cpu", norm={"*": "degrees", "gripper": "range_0_100"})
print(act)
agent = Agent(model=MODEL_ID, tools=[arm.tool(policies={"wave": wave, "act": act}), sim.tool], callback_handler=None)
before = arm.observe(images=False).joints.positions.copy()
reply = agent(f"Run the act policy for 3 seconds with the instruction '{TASK}', then tell me in one sentence what "
              "the run reported.")
print(str(reply).strip())
print("calls:", *_nb.tool_calls(agent), sep="\n  ")
moved = arm.observe(images=False).joints.positions - before
print("joint change from the policy (rad):", [round(float(x), 3) for x in moved])
_nb.show(_nb.image(sim.render("front"), width=480))

# %% [markdown]
# ## Step 4: the same code on a physical SO-101 (reference)
#
# Calibrate once with LeRobot (`lerobot-calibrate --robot.type=so101_follower --robot.id=my_follower`), then change
# the constructor. A real arm writes nothing without the operator: `operator=` sees a `Move` for every command a
# tool asks for, and a policy run's `RunPlan` once.
#
# ```python
# arm = Robot("so101", port="/dev/ttyACM0", operator=lambda req: input(f"{req}? [y/N] ") == "y")
# agent = Agent(model=MODEL_ID, tools=[arm.tool(policies={"act": act})])
# agent(f"Run the act policy for 10 seconds with the instruction '{TASK}'.")
# ```
#
# Notebook 08 shows the gate and the watchdog on a hardware-like arm.

# %% [markdown]
# ## Step 5: the robot on a mesh
#
# `bench.mount(arm)` announces the robot on a Zenoh mesh with its manifest; another `Mesh` discovers it by labels and
# hands an agent one tool, `use_mesh`, for every device present. Two meshes in one process here, on a loopback
# endpoint; on a LAN they are two machines, and `iot://...` puts them on AWS IoT Core.

# %%
port, tenant = 7460, f"nb11-{uuid.uuid4().hex[:6]}"
me = Mesh(f"zenoh://?listen=tcp/127.0.0.1:{port}", tenant=tenant, operator="you@laptop")
bench = Mesh(f"zenoh://?connect=tcp/127.0.0.1:{port}", tenant=tenant, operator="bench@lab")
print(bench.mount(arm, labels={"role": "arm"}))
me.discover(min_count=1, timeout=5.0)
print("discovered:", [d.id for d in me.discover("device(role:arm)")])

fleet = Agent(model=MODEL_ID, tools=[me.tool], callback_handler=None)
reply = fleet("List every arm on the mesh, then send each one to its home pose (every joint at 0 rad) over one "
              "second. Answer in two sentences.")
print(str(reply).strip())
print("calls:", *_nb.tool_calls(fleet), sep="\n  ")
print("home now:", [round(float(x), 3) for x in arm.observe(images=False).joints.positions])

# %% [markdown]
# ## Clean up

# %%
me.close()
bench.close()
sim.close()
print("closed; the local dataset is under", WORK)
