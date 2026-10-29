# Paired script for notebooks/03_agent_builds_a_world.ipynb.
# Run: .venv/bin/python notebooks/scripts/03_agent_builds_a_world.py          (the scripted turns)
#      .venv/bin/python notebooks/scripts/03_agent_builds_a_world.py --repl   (type your own)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # nb: script-only

# %% [markdown]
# # 03 · An agent builds a world
#
# A Strands agent gets two tools, `sim.tool` and `robot.tool`, and edits a MuJoCo world from plain English.
# Every turn below is a real Claude call on Amazon Bedrock; the renders and tool calls are what that run did, and a
# re-run gets different replies.
#
# ![a sentence becomes tool calls, the world rebuilds, the agent looks](media/03/turn.svg)
#
# Needs `uv pip install -e ".[mujoco]" strands-agents` and Bedrock access: `AWS_BEARER_TOKEN_BEDROCK` (or AWS
# credentials) plus a model id in `STRANDS_MODEL_ID`.

# %%
import io
import json
import os
import sys

import numpy as np
from PIL import Image
from strands import Agent
from strands.tools.executors import SequentialToolExecutor

import _nb
from strands_robots import Robot, Simulation

MEDIA = _nb.MEDIA / "03"
MEDIA.mkdir(parents=True, exist_ok=True)
print("model:", os.environ["STRANDS_MODEL_ID"])

# %% [markdown]
# One world, one SO-101, and a table: a fixed box 4 cm high in front of the arm.

# %%
sim = Simulation()
arm = Robot("so101", sim=sim)
sim.add_object("table", shape="box", size=(0.12, 0.2, 0.02), position=(0.3, 0.0, 0.02), color=(0.55, 0.4, 0.3, 1), free=False)
sim.camera("hero", position=(0.85, -0.7, 0.55), target=(0.15, 0.0, 0.05), size=(480, 360))
_nb.show(_nb.image(sim.render("hero"), width=480))

# %% [markdown]
# Both tools are generated from the code: one `rpc` selector per tool, one argument object per rpc.

# %%
for tool in (sim.tool, arm.tool):
    rpcs = tool.tool_spec["inputSchema"]["json"]["properties"]["rpc"]["enum"]
    print(f"{tool.tool_name:12s} {rpcs}")
print(json.dumps(sim.tool.tool_spec["inputSchema"]["json"]["properties"]["add_object"]["properties"]["shape"]))

# %% [markdown]
# The agent. `SequentialToolExecutor` runs one tool call at a time: with strands' default (concurrent) executor, a
# turn that adds a camera and renders it in the same message can render first and fail.

# %%
agent = Agent(
    model=os.environ["STRANDS_MODEL_ID"],
    tools=[sim.tool, arm.tool],
    tool_executor=SequentialToolExecutor(),
    callback_handler=None,
)


SEEN = set()  # tool-call ids already printed (strands trims old messages, so an index into agent.messages drifts)


def new_calls():
    """The tool calls not printed yet, one line each: rpc, its arguments, ok or the error the model read."""
    results = {
        b["toolResult"]["toolUseId"]: b["toolResult"]
        for m in agent.messages for b in m["content"] if "toolResult" in b
    }
    for m in agent.messages:
        for b in m["content"]:
            if "toolUse" in b and b["toolUse"]["toolUseId"] not in SEEN:
                use = b["toolUse"]
                SEEN.add(use["toolUseId"])
                rpc = use["input"].get("rpc")
                res = results.get(use["toolUseId"], {})
                note = "ok" if res.get("status") == "success" else res.get("content", [{}])[0].get("text", "")[:120]
                print(f"  {use['name']}.{rpc}({json.dumps(use['input'].get(rpc) or {})})  -> {note}")


def turn(ask, name):
    """One request: render before, the agent's turn, the calls it made, render after (saved to media/03)."""
    before = sim.render("hero")
    reply = agent(ask)
    after = sim.render("hero")
    print(f"> {ask}\n")
    new_calls()
    print(f"\n{str(reply).strip()}")
    pair = np.concatenate([before, np.full((before.shape[0], 8, 3), 255, np.uint8), after], axis=1)
    Image.fromarray(pair).save(MEDIA / f"{name}.png")
    _nb.show(_nb.image(pair, width=720))

# %% [markdown]
# ## Turn 1: a red cube on the table
#
# Before on the left, after on the right (the fixed `hero` camera).

# %%
turn("add a red cube on the table", "t1_red_cube")

# %% [markdown]
# `size` is MuJoCo's half-extent (a radius for a sphere), and nothing in the tool schema says so: read the cube's `size`
# in the call above against the size the reply claims.

# %% [markdown]
# ## Turn 2: a blue ball and a light
#
# There is no verb that adds a light (`randomize(lighting=)` only scales the existing one), so watch what the agent
# says about that half of the request.

# %%
turn("add a blue ball and a light", "t2_ball_and_light")

# %% [markdown]
# ## Turn 3: a camera looking at the arm

# %%
turn("put a camera looking at the arm", "t3_camera")

# %% [markdown]
# ## Turn 4: the agent looks through its camera
#
# `render` returns the picture as an image block, so the model sees it. Below is the exact image it was sent.

# %%
reply = agent("render the camera you added and tell me in one sentence what you see")
new_calls()
print(f"\n{str(reply).strip()}")
seen = [
    c["image"]["source"]["bytes"]
    for m in agent.messages for b in m["content"] if "toolResult" in b
    for c in b["toolResult"]["content"] if "image" in c
]
saw = np.asarray(Image.open(io.BytesIO(seen[-1])).convert("RGB"))
Image.fromarray(saw).save(MEDIA / "t4_what_the_agent_saw.png")
_nb.show(_nb.image(saw, width=480))

# %% [markdown]
# ## The world it built
#
# `describe()` is what the agent reads too.

# %%
world = sim.describe()
for o in world["objects"]:
    print(f"{o['name']:10s} {o['shape']:8s} at {np.round(o['position'], 3).tolist()}")
print("cameras:", [c["name"] for c in world["cameras"]])

# %% [markdown]
# An orbit around it. There is no verb that moves a camera, so each frame is `remove` + `camera` again, a rebuild.

# %%
frames = []
for i in range(36):
    a = 2 * np.pi * i / 36
    if i:
        sim.remove("orbit")
    sim.camera("orbit", position=(0.15 + 0.75 * np.cos(a), 0.75 * np.sin(a), 0.5), target=(0.15, 0.0, 0.05), size=(400, 300))
    frames.append(sim.render("orbit"))
_nb.gif(frames, "03/orbit.gif", width=400, fps=12)

# %% [markdown]
# ![orbit of the world the agent built](media/03/orbit.gif)
#
# ## Your turn
#
# The same agent in a loop. In a terminal: `python notebooks/scripts/03_agent_builds_a_world.py --repl`.

# %%
if "--repl" in sys.argv:
    while True:
        print(agent(input("> ")))

# %%
sim.close()
