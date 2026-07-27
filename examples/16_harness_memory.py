#!/usr/bin/env python3
"""Harness memory: save a solution trace, reuse it under spatial perturbation.

Goal: Show the Harness-VLA memory pattern (arXiv:2607.08448) with the
``harness_memory`` tool - the difference between an agent that re-derives
everything per episode and one that reuses HOW a task was solved without
replaying WHERE objects happened to be:

  - Bootstrap run: solve the task once (MockPolicy stands in for a real VLA),
    then commit the *solution skeleton* - primitive ordering, not coordinates -
    with ``save_trace``, plus a cross-task failure model with ``append_rule``.
  - Second run: the cube spawns at a PERTURBED position. The agent loads the
    trace (which arrives with the re-grounding contract prepended: never
    replay literal coordinates), re-localizes the cube from the CURRENT scene
    via ``get_body_state``, and executes the same skeleton against the new
    position.

The pattern is what matters here, not policy skill: MockPolicy produces test
actions, so "solving" is scripted - but the memory round-trip, the perturbed
re-grounding, and the contract text are all real.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: bootstrap saves a 3-step trace; the second run loads it,
re-localizes the cube at its perturbed position, and finishes with
"run_policy status: success". Runtime: ~5 seconds on CPU.
"""

from __future__ import annotations

from strands_robots import MockPolicy, Robot
from strands_robots.tools.harness_memory import harness_memory

TASK = "put_cube_in_zone"


def _json(result: dict) -> dict:
    """First json content block of a tool result."""
    return next((c["json"] for c in result.get("content", []) if "json" in c), {})


def _cube_position(sim) -> list[float]:
    """Re-localize the cube from the CURRENT observation (never from memory)."""
    state = _json(sim._dispatch_action("get_body_state", {"body_name": "cube"}))
    return [round(float(v), 4) for v in state["position"]]


def _build_scene(cube_position: list[float]):
    """Fresh scene: so100 + a cube at the given position."""
    sim = Robot("so100", mesh=False)
    sim.add_object(
        name="cube",
        shape="box",
        size=[0.025, 0.025, 0.025],
        position=cube_position,
        color=[1, 0, 0, 1],
        mass=0.05,
    )
    for _ in range(3):
        sim.step()
    return sim


def bootstrap_run() -> None:
    """First encounter with the task: solve it, then commit the memory."""
    sim = _build_scene(cube_position=[0.2, 0.0, 0.05])
    result = sim.run_policy(
        robot_name="so100",
        policy_object=MockPolicy(),
        instruction="pick up the red cube",
        n_steps=30,
    )
    print(f"bootstrap run_policy status: {result['status']}")
    sim.destroy()

    # Commit the solution SKELETON: primitive ordering + where the policy call
    # sits. get_body_state marks "re-localize the target first"; the xyz that
    # was observed during the run is deliberately NOT stored.
    save = harness_memory(
        action="save_trace",
        task=TASK,
        trace=[
            {"action": "get_body_state", "body_name": "cube"},
            {"action": "run_policy", "instruction": "pick up the red cube", "n_steps": 30},
            {"action": "get_state"},
        ],
        summary={
            "task": "put the red cube in the drop zone",
            "success": True,
            "strategy": "re-localize the cube, use the policy for grasping, verify with sim state",
            "avoid": ["do not reuse reference xyz values", "verify placement from the current scene"],
        },
        backend="mujoco",
        robot="so100",
    )
    print(f"save_trace status: {save['status']}")

    rule = harness_memory(
        action="append_rule",
        kind="failure_model",
        text=(
            "If the gripper closes but the object does not move with the end "
            "effector, treat the attempt as an empty grasp: re-localize the "
            "object and re-stage before retrying."
        ),
    )
    print(f"append_rule status: {rule['status']}")


def memory_run() -> None:
    """Second run: cube is PERTURBED; reuse the skeleton, re-ground everything."""
    # Session start: global rules + the task's trace go into the agent context.
    rules = _json(harness_memory(action="load_rules"))
    print(f"loaded {len(rules['failure_models'])} failure model(s)")

    loaded = harness_memory(action="load_trace", task=TASK)
    payload = _json(loaded)
    # The re-grounding contract is the FIRST text block - it travels with the
    # memory, so any agent that loads a trace also receives the discipline.
    print(f"contract: {loaded['content'][0]['text'][:74]}...")
    trace = payload["trace"]
    print(f"loaded trace: {len(trace)} steps, strategy: {payload['summary']['strategy']}")

    # Perturbed scene: the cube spawns somewhere else entirely.
    sim = _build_scene(cube_position=[0.15, 0.1, 0.05])

    # Execute the skeleton, honoring the contract: step 1 re-localizes the
    # cube from the CURRENT observation instead of any remembered position.
    assert trace[0]["action"] == "get_body_state"
    cube_now = _cube_position(sim)
    print(f"re-localized cube at {cube_now} (perturbed, not the reference scene)")

    assert trace[1]["action"] == "run_policy"
    result = sim.run_policy(
        robot_name="so100",
        policy_object=MockPolicy(),
        instruction=trace[1]["instruction"],
        n_steps=trace[1]["n_steps"],
    )
    print(f"memory run_policy status: {result['status']}")

    assert trace[2]["action"] == "get_state"
    sim._dispatch_action("get_state", {})
    sim.destroy()


def main() -> int:
    bootstrap_run()
    memory_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
