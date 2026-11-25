"""An SO-101 and a G1 in one world, driven by a Strands agent served as an Amazon Bedrock AgentCore Runtime app.

    uv pip install "strands-robots[mujoco,tools]" bedrock-agentcore bedrock-agentcore-starter-toolkit
    python agentcore_fleet.py           # POST /invocations on :8080 (localhost; all interfaces in a container)
    agentcore configure -e agentcore_fleet.py -rf requirements.txt && agentcore launch

The local server has no authentication, so do not publish port 8080 from a container. Deployed, AgentCore Runtime
authenticates callers.

For the deploy, write a requirements.txt with strands-robots[mujoco,tools] and bedrock-agentcore. The robot models are
fetched with git on first use, so add git to the Dockerfile that agentcore configure writes before you launch, for
example RUN apt-get update && apt-get install -y --no-install-recommends git. Rendering a camera on a headless host
also needs MUJOCO_GL=egl and the system EGL library.

AgentCore Runtime runs each session in its own microVM, so one process holds one world, one agent and one
conversation. Send every request for that world with the same runtime session id.

{"prompt": "..."} runs one agent turn; a second prompt while a turn is running is refused (409), not queued.
{"estop": "g1"} (or "all") latches the e-stop directly: no model turn, and it never waits for one. Sent before the
world is built, it is held and latched as soon as the world exists, before any turn can move a robot; if it cannot
latch, the world is not started and later requests get a 503. A latched robot stays stopped for the rest of the
session. A turn cannot be cancelled.
The agent gets each robot's tool and not sim.tool, so it cannot edit the world, and it can e-stop a robot but not
resume one.
STRANDS_MODEL_ID picks the model (Claude on Amazon Bedrock by default).
"""

import os
import threading

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from starlette.exceptions import HTTPException
from strands import Agent

from strands_robots import Robot, Simulation

app = BedrockAgentCoreApp()
FLEET: dict = {}
_BUILD = threading.Lock()
TURN = threading.Lock()
ROBOT_IDS = ("so101", "g1")
PENDING: set = set()  # every requested e-stop; the build latches them, so one arriving mid-build is not lost
HALTED: set = set()  # robots whose requested e-stop failed to latch; prompts are refused while any is listed
USAGE = 'send {"prompt": "<text>"} or {"estop": "<robot id>" | "all"}'


def _halted() -> HTTPException:
    return HTTPException(503, f"the e-stop requested for {sorted(HALTED)} did not latch; restart the session")


def fleet(model=None) -> dict:
    """The world, its two robots and the agent, built once per process."""
    with _BUILD:
        if not FLEET:
            sim = Simulation()
            try:
                with sim.editing():
                    robots = {"so101": Robot("so101", sim=sim), "g1": Robot("g1", sim=sim, position=[1.0, 0.0, 0.0])}
                agent = Agent(
                    model=model or os.environ.get("STRANDS_MODEL_ID") or None,
                    tools=[r.tool for r in robots.values()],
                    callback_handler=None,
                )
            except BaseException:
                sim.close()
                raise
            FLEET.update(sim=sim, robots=robots, agent=agent)
            failed = []
            for t in list(PENDING):  # an e-stop sent during the build holds before any turn can move a robot
                try:
                    robots[t].estop(by="agentcore-caller")
                except Exception:
                    failed.append(t)
            if failed:  # never hand out a world with a requested e-stop unlatched
                HALTED.update(failed)
                FLEET.clear()
                sim.close()
                raise RuntimeError(f"could not latch the e-stop requested for {failed}; the world was not started")
        return FLEET


@app.entrypoint
def invoke(payload: dict) -> dict:
    """{"prompt": "..."} -> {"reply": "..."}; {"estop": "<robot id>" | "all"} -> {"estopped": [...]}.

    Before the world is built, an e-stop returns {"pending": [...]} and is latched as soon as it is.
    """
    if not isinstance(payload, dict) or len(payload) != 1 or not ({"prompt", "estop"} & set(payload)):
        raise HTTPException(400, USAGE)
    if "estop" in payload:
        which = payload["estop"]
        if not isinstance(which, str):
            raise HTTPException(400, USAGE)
        targets = list(ROBOT_IDS) if which == "all" else [which]
        if any(t not in ROBOT_IDS for t in targets):
            raise HTTPException(400, f"unknown robot {which!r}; robots are {sorted(ROBOT_IDS)} or 'all'")
        PENDING.update(targets)  # recorded before FLEET is read, so a build finishing now still latches it
        robots = FLEET.get("robots")  # read without _BUILD: an e-stop never waits on the first build
        if robots is None:
            if HALTED:  # the world was refused, so a held e-stop would never be latched
                raise _halted()
            return {"pending": targets}  # held, not latched yet: latched as soon as the world is built
        failed = {}
        for t in targets:  # try every robot, so one failing latch never leaves the others running
            try:
                robots[t].estop(by="agentcore-caller")
            except Exception as e:
                failed[t] = f"{type(e).__name__}: {e}"
        if failed:
            HALTED.update(failed)  # no further turn runs while a requested e-stop is not latched
            raise HTTPException(500, f"e-stop failed for {failed}; any others latched and prompts are refused")
        return {"estopped": targets}
    prompt = payload["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(400, USAGE)
    if HALTED:
        raise _halted()
    if not TURN.acquire(blocking=False):
        raise HTTPException(409, 'a turn is already running; send {"estop": ...} to stop, or retry when it ends')
    try:
        if HALTED:  # again under TURN: an e-stop may have failed since the check above
            raise _halted()
        return {"reply": str(fleet()["agent"](prompt)).strip()}
    finally:
        TURN.release()


if __name__ == "__main__":
    app.run()
