"""An agent builds a scene around an SO-101, moves the arm and looks at the result, through sim.tool and arm.tool.

    python agent_builds_a_scene.py      # Claude on Amazon Bedrock; STRANDS_MODEL_ID picks another model id

Nothing here is a hand-written tool: both are generated from the world's and the robot's verbs.
"""

import json
import os

from strands import Agent

from strands_robots import Robot, Simulation

ASKS = (
    "Place a red, a green and a blue 3 cm cube on the ground within 0.3 m of the arm, "
    "then add a camera named 'hero' at (0.6, -0.5, 0.4) looking at (0, 0, 0.1).",
    "Observe the arm, move shoulder_pan by +0.3 rad and let it settle for 400 control steps, "
    "then observe again and tell me the delta.",
    "Render the hero camera and describe what you see in one sentence.",
)


def trace(agent: Agent) -> list[tuple[str, str, str]]:
    """Every tool call of the conversation as (tool, rpc, status), in order."""
    calls, status = [], {}
    for message in agent.messages:
        for block in message["content"]:
            if "toolUse" in block:
                use = block["toolUse"]
                calls.append((use["toolUseId"], use["name"], (use.get("input") or {}).get("rpc", "?")))
            if "toolResult" in block:
                status[block["toolResult"]["toolUseId"]] = block["toolResult"]["status"]
    return [(name, rpc, status.get(tid, "-")) for tid, name, rpc in calls]


def main(model=None) -> tuple[Simulation, Robot, Agent]:
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    agent = Agent(model=model or os.environ.get("STRANDS_MODEL_ID"), tools=[sim.tool, arm.tool], callback_handler=None)
    for ask in ASKS:
        print(f"> {ask}\n{agent(ask)}\n")
    for name, rpc, status in trace(agent):
        print(f"  {name}.{rpc}: {status}")
    print(
        json.dumps(
            {"objects": [o["name"] for o in sim.describe()["objects"]], "shoulder_pan": arm.observe()["shoulder_pan"]}
        )
    )
    return sim, arm, agent


if __name__ == "__main__":
    main()
