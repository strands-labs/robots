# Paired script for notebooks/04_agent_drives_the_robot.ipynb.
# Run: .venv/bin/python notebooks/scripts/04_agent_drives_the_robot.py          (the scripted turns)
#      .venv/bin/python notebooks/scripts/04_agent_drives_the_robot.py --repl   (type your own)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # nb: script-only

# %% [markdown]
# # 04 · An agent drives the robot
#
# The same two tools as notebook 03, now used to move an SO-101: poses from plain English, a policy run by name,
# an e-stop the agent can press but not release, and a hardware arm whose operator gate refuses the model.
# Every turn is a real Claude call on Amazon Bedrock; a re-run gets different replies.
#
# ![an agent's act goes through the safety chain; a refusal comes back to the model as text](media/04/gate.svg)

# %%
import json
import math
import os
import sys

import numpy as np
from strands import Agent
from strands.tools.executors import SequentialToolExecutor

import _nb
from strands_robots import Robot, Simulation

MEDIA = _nb.MEDIA / "04"
MEDIA.mkdir(parents=True, exist_ok=True)
print("model:", os.environ["STRANDS_MODEL_ID"])

# %% [markdown]
# A world with the arm and a red cube on the floor in front of it (the arm reaches along -y).

# %%
sim = Simulation()
arm = Robot("so101", sim=sim)
sim.add_object("red_cube", shape="box", size=(0.015, 0.015, 0.015), position=(0.0, -0.22, 0.015), color=(0.85, 0.1, 0.1, 1))
sim.camera("side", position=(0.65, -0.45, 0.35), target=(0.0, -0.12, 0.12), size=(480, 360))
_nb.show(_nb.image(sim.render("side"), width=480))

# %% [markdown]
# A policy is any callable from an `Observation` to joint targets. Registered on the tool as a handle, the model runs it
# by name; it never types a network address or a file path.

# %%
def wave(obs):
    return {"shoulder_pan": 0.6 * math.sin(3.0 * obs.t)}


robot_tool = arm.tool(policies={"wave": wave})
print(json.dumps(robot_tool.tool_spec["inputSchema"]["json"]["properties"]["run"]["properties"]["policy"])[-120:])

agent = Agent(
    model=os.environ["STRANDS_MODEL_ID"],
    tools=[robot_tool, sim.tool],
    tool_executor=SequentialToolExecutor(),  # one call at a time (notebook 03 says why)
    callback_handler=None,
)


SEEN = set()  # tool-call ids already printed (strands trims old messages, so an index into agent.messages drifts)


def new_calls(of):
    """One line per tool call not printed yet: rpc, arguments, ok or the error text the model read."""
    results = {b["toolResult"]["toolUseId"]: b["toolResult"] for m in of.messages for b in m["content"] if "toolResult" in b}
    out = []
    for m in of.messages:
        for b in m["content"]:
            if "toolUse" in b and b["toolUse"]["toolUseId"] not in SEEN:
                use = b["toolUse"]
                SEEN.add(use["toolUseId"])
                rpc = use["input"].get("rpc")
                res = results.get(use["toolUseId"], {})
                ok = res.get("status") == "success"
                note = "ok" if ok else res.get("content", [{}])[0].get("text", "")[:160]
                print(f"  {use['name']}.{rpc}({json.dumps(use['input'].get(rpc) or {})})  -> {note}")
                out.append((rpc, use["input"].get(rpc) or {}, ok))
    return out


def turn(ask, of=None):
    of = of or agent
    reply = of(ask)
    print(f"> {ask}\n")
    calls = new_calls(of)
    print(f"\n{str(reply).strip()}")
    return calls


def joints():
    return {k: round(v, 3) for k, v in arm.observe(images=False).joints.as_dict().items()}

# %% [markdown]
# ## A pose from a sentence
#
# The model has no named poses to pick from; it reads the joint names and limits from `describe` and chooses numbers.

# %%
moves = turn("move the arm to a ready pose: lift the shoulder, bend the elbow, open the gripper, taking one second")
print(joints())
_nb.show(_nb.image(sim.render("side"), width=480))

# %% [markdown]
# ## Toward the cube
#
# The tools can read where the cube is (`sim_world.pose`), but not where the gripper is: `pose` takes a robot base or
# an object, not a link, and there is no inverse kinematics verb. Watch how the model gets there anyway.

# %%
moves += turn("move the gripper so it is just above the red cube")
print(joints())
_nb.show(_nb.image(sim.render("side"), width=480))

# %% [markdown]
# ## A policy, by name

# %%
turn("run the wave policy for 2 seconds")
print(joints())

# %% [markdown]
# ## The motion, replayed
#
# Rendering happens here, after the turns, in a second identical world so the agent's world is left as it was: the
# model's own successful `act` calls in 6 slices each (the straight-line ramp `act(seconds=)` takes), then 2 s of wave.

# %%
replay = Simulation()
arm2 = Robot("so101", sim=replay)
replay.add_object("red_cube", shape="box", size=(0.015, 0.015, 0.015), position=(0.0, -0.22, 0.015), color=(0.85, 0.1, 0.1, 1))
replay.camera("side", position=(0.65, -0.45, 0.35), target=(0.0, -0.12, 0.12), size=(360, 270))
frames = [replay.render("side")]
for rpc, args, ok in moves:
    if rpc != "act" or not ok:
        continue
    start = arm2.observe(images=False).joints.as_dict()
    goal, secs = args["action"], args.get("seconds") or 0.2
    for k in range(1, 7):
        arm2.act({j: start[j] + (v - start[j]) * k / 6 for j, v in goal.items()}, seconds=secs / 6)
        frames.append(replay.render("side"))
for _ in range(16):  # 2 s of wave, one frame per 0.125 s
    arm2.act(wave(arm2.observe(images=False)), seconds=0.125)
    frames.append(replay.render("side"))
_nb.gif(frames, "04/drive.gif", width=360, fps=11, colors=128)
replay.close()

# %% [markdown]
# ![the arm moving through the model's own act calls, then the wave policy](media/04/drive.gif)
#
# ## E-stop: the agent can press it, not release it

# %%
turn("emergency stop the arm, then try to move shoulder_pan to 0 and tell me what happened")
print("estopped:", arm.estopped)

# %%
turn("release the e-stop and resume")
print("estopped:", arm.estopped, "| rpcs the model has:", robot_tool.tool_spec["inputSchema"]["json"]["properties"]["rpc"]["enum"])

# %% [markdown]
# `resume` is the operator's, from Python, with a name on it.

# %%
arm.resume(approved_by="operator")
print("estopped:", arm.estopped)

# %% [markdown]
# ## A hardware arm: the operator gate
#
# An SO-101 on a Feetech bus. There is no arm on this machine, so the bus is the test suite's byte-level fake of six
# STS3215 servos; the `Robot` code path is the real one. Nothing reaches the bus without the operator's say.
#
# With a real arm the only change is the first line: `Robot("so101", port="/dev/tty.usbmodem...", confirm=...)`.

# %%
sys.path.insert(0, str(_nb.ROOT / "tests"))
from test_drivers import CAL, FakeSerial  # noqa: E402

from strands_robots import Description  # noqa: E402
from strands_robots.robot.drivers import feetech as ft  # noqa: E402


def hardware_arm(**gates):
    serial = FakeSerial()
    bus = ft.FeetechBus("/dev/fake", serial=serial)
    driver = ft.FeetechDriver(Description.load("so101"), port="/dev/fake", calibration=CAL, bus=bus)
    return Robot("so101", driver=driver, twin=False, **gates), serial


def goal_writes(serial):
    return sum(1 for f in serial.frames if f[4] == ft.SYNC_WRITE and f[5] == ft.REG_GOAL_POSITION)


def hw_agent(robot):
    return Agent(model=os.environ["STRANDS_MODEL_ID"], tools=[robot.tool(policies={"wave": wave})],
                 tool_executor=SequentialToolExecutor(), callback_handler=None)

# %% [markdown]
# No `confirm=`: the model's `act` is refused, and the refusal is what it reads.

# %%
real, bus = hardware_arm()
turn("move shoulder_pan to 0.3", of=hw_agent(real))
print("goal frames on the bus:", goal_writes(bus))
real.close()

# %% [markdown]
# A `confirm=` callable sees the full target vector (radians, `action_keys` order) and answers yes or no. This one
# allows `shoulder_pan` within ±0.5 rad.

# %%
asked = []


def confirm(targets):
    asked.append(np.round(targets, 3).tolist())
    return abs(targets[0]) <= 0.5


real, bus = hardware_arm(confirm=confirm)
turn("move shoulder_pan to 0.3, then to 1.2", of=hw_agent(real))
print("operator asked", len(asked), "times; goal frames on the bus:", goal_writes(bus))
print("shoulder_pan now", round(real.observe(images=False)["shoulder_pan"], 3))
real.close()

# %% [markdown]
# A policy run from an agent on hardware asks a different gate once, `approve=`, with the whole plan.

# %%
plans = []
real, bus = hardware_arm(approve=lambda plan: plans.append(plan) or False)
turn("run the wave policy for 1 second", of=hw_agent(real))
print("plans the operator saw:", plans, "| goal frames:", goal_writes(bus))
real.close()

# %% [markdown]
# ## Your turn
#
# In a terminal: `python notebooks/scripts/04_agent_drives_the_robot.py --repl` drives the simulated arm.

# %%
if "--repl" in sys.argv:
    while True:
        print(agent(input("> ")))

# %%
sim.close()
