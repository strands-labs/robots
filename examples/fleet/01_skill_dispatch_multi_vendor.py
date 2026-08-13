#!/usr/bin/env python3
"""Capability-based skill dispatch across heterogeneous robots.

Goal: One MuJoCo world hosts three robots from three vendors (an SO-101 arm,
a LeKiwi wheeled base, a Unitree Go2 quadruped - repeated ``add_robot`` on the
same world). A skills table maps each skill name to capability requirements
over REGISTRY METADATA (category, joint count, gripper) and to an execution
binding (a ``create_policy`` provider or a motion primitive). Each robot's
capability manifest is derived from that metadata alone, so the dispatch path
contains no per-embodiment branching anywhere: swapping a vendor is a one-line
fleet edit, and the dispatcher never learns robot names beyond the manifests.
Matching runs through the suite's shared ``capabilities.py`` hard-constraint
filter; a task no robot can serve is rejected with a machine-readable,
per-robot reason - never silently dropped. Every dispatch passes a
human-in-the-loop gate before anything executes.

Dependencies: pip install "strands-robots[sim-mujoco]"
              (--dry-run needs only the base package: no simulator)
Expected output: two tasks with distinct capability requirements land on the
                 correct robots (the staging task on the arm, the transport on
                 a mobile base), one infeasible task is rejected naming each
                 robot's failing constraint, and the approved work executes -
                 concurrently via run_multi_policy plus a move_to primitive on
                 MuJoCo; sequentially via per-robot run_policy elsewhere.
Runtime: ~2 seconds with --dry-run; under ~90 seconds live (cached assets).

Note: Set STRANDS_MESH_HITL_ACTIONS=none to auto-approve dispatches (CI /
      smoke mode; logged loudly) - the default is an interactive prompt per
      dispatch. --view opens the passive MuJoCo viewer (needs a display).
      --backend isaac runs the identical dispatch layer on Isaac Sim.

Part of the fleet suite (epic #2179); the shared manifest schema lives in
``capabilities.py`` (also consumed by example 05, #2185).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

_FLEET_DIR = Path(__file__).resolve().parent
if str(_FLEET_DIR) not in sys.path:
    # examples/ is not an installed package; the shared schema module lives
    # next to this script.
    sys.path.insert(0, str(_FLEET_DIR))

from capabilities import (  # noqa: E402
    CapabilityManifest,
    Skill,
    StepRequirement,
    feasible_robots,
)

from strands_robots.registry import get_robot  # noqa: E402

SITE = "cell-a"

# The fleet: instance name -> registry embodiment, maximally heterogeneous on
# purpose (three vendors, three morphologies). Swapping a vendor is this one
# line - nothing below reads these names except to address the sim.
FLEET: dict[str, str] = {"so101": "so101", "lekiwi": "lekiwi", "go2": "unitree_go2"}
FLEET_POSITION: dict[str, list[float]] = {"so101": [0.0, 0.0, 0.0], "lekiwi": [1.0, 0.0, 0.0], "go2": [0.0, 1.2, 0.0]}

# The skills table: skill name -> capability requirements (over registry
# metadata: category, joint count, gripper) -> execution binding (a
# create_policy provider, or a motion primitive). A robot OFFERS a skill iff
# its registry metadata satisfies the requirements - so dispatch is a
# capability match by construction, never an embodiment switch.
SKILLS: dict[str, dict[str, Any]] = {
    "stage_part": {
        "requires": {"category": "arm", "min_joints": 5, "gripper": True},
        "offer": {"payload_kg": 0.5, "fixture": "smif_pod", "zones": ("bench",)},
        "binding": {"execute": "move_to", "position": [0.2, 0.0, 0.15]},
    },
    "transport_tote": {
        "requires": {"category": "mobile", "min_joints": 3, "gripper": False},
        "offer": {"payload_kg": 5.0, "fixture": "tote_clamp", "zones": ("bench", "stock")},
        "binding": {"execute": "policy", "policy_provider": "mock"},
    },
}

# The work: two tasks with distinct capability requirements plus one no robot
# can serve (40 kg exceeds every offer), so the explicit-rejection path is
# exercised on every run.
TASKS: list[dict[str, Any]] = [
    {"task_id": "T-01", "skill": "stage_part", "payload_kg": 0.2, "zones": ("bench",)},
    {"task_id": "T-02", "skill": "transport_tote", "payload_kg": 4.0, "zones": ("bench", "stock")},
    {"task_id": "T-03", "skill": "transport_tote", "payload_kg": 40.0, "zones": ("bench", "stock")},
]


def robot_capability_metadata(embodiment: str) -> dict[str, Any]:
    """Read one embodiment's capability-relevant metadata from the registry.

    Raises ValueError for an unknown embodiment - a robot whose capabilities
    cannot be read must never be dispatched to.
    """
    meta = get_robot(embodiment)
    if meta is None:
        raise ValueError(f"unknown embodiment {embodiment!r}: not in the robot registry")
    return {"category": meta.get("category"), "joints": meta.get("joints", 0), "gripper": "gripper" in meta}


def _meets(meta: dict[str, Any], requires: dict[str, Any]) -> bool:
    return (
        meta["category"] == requires["category"]
        and meta["joints"] >= requires["min_joints"]
        and (meta["gripper"] or not requires["gripper"])
    )


def build_manifests(fleet: dict[str, str]) -> list[CapabilityManifest]:
    """Derive one capability manifest per robot from registry metadata alone."""
    manifests = []
    for name, embodiment in fleet.items():
        meta = robot_capability_metadata(embodiment)
        offered = tuple(
            Skill(name=skill, **spec["offer"]) for skill, spec in SKILLS.items() if _meets(meta, spec["requires"])
        )
        manifests.append(CapabilityManifest(robot=name, site=SITE, skills=offered))
    return manifests


def _task_requirement(task: dict[str, Any]) -> StepRequirement:
    return StepRequirement(
        skill=task["skill"],
        site=SITE,
        payload_kg=task["payload_kg"],
        fixture=SKILLS[task["skill"]]["offer"]["fixture"],
        zones=tuple(task["zones"]),
    )


def _task_instruction(task: dict[str, Any]) -> str:
    return f"task {task['task_id']}: {task['skill']} {task['payload_kg']} kg in {'/'.join((SITE, *task['zones']))}"


def match_skill_for_task(
    manifests: Sequence[CapabilityManifest], task: dict[str, Any]
) -> tuple[list[CapabilityManifest], list[dict[str, Any]]]:
    """Run the shared hard-constraint filter for one task.

    Returns ``(feasible, rejections)`` - the same shape as
    :func:`capabilities.feasible_robots`: feasible sorted by robot name, one
    machine-readable rejection per excluded robot.
    """
    return feasible_robots(list(manifests), _task_requirement(task))


def dispatch_tasks(
    tasks: Sequence[dict[str, Any]],
    manifests: Sequence[CapabilityManifest],
    approve: Callable[[str, str, str], bool],
    execute: Callable[[list[dict[str, Any]]], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Match every task, gate each dispatch on approval, then execute the batch.

    ``approve(action, robot, instruction)`` is the HITL seam;
    ``execute(assignments)`` is the execution seam (run_multi_policy +
    move_to on MuJoCo, sequential run_policy elsewhere, a loopback in
    --dry-run and the smoke test). Every task ends in exactly one of
    ``dispatched`` / ``rejected`` / ``declined`` - no outcome is silent.
    """
    summary: dict[str, Any] = {"dispatched": [], "rejected": [], "declined": []}
    assignments: list[dict[str, Any]] = []
    for task in tasks:
        feasible, rejections = match_skill_for_task(manifests, task)
        if not feasible:
            reason = {"task_id": task["task_id"], "code": "no_capable_robot", "rejections": rejections}
            summary["rejected"].append(reason)
            print(f"  REJECT {task['task_id']}: no capable robot")
            for r in rejections:
                print(f"    {r['robot']}: fails {r['constraint']} (required {r['required']!r}, has {r['actual']!r})")
            continue
        robot = feasible[0].robot  # deterministic: feasible is sorted by name
        instruction = _task_instruction(task)
        if not approve("dispatch", robot, instruction):
            summary["declined"].append({"task_id": task["task_id"], "robot": robot, "code": "hitl_declined"})
            print(f"  DECLINED {task['task_id']}: operator declined dispatch to {robot}")
            continue
        assignments.append(
            {
                "task_id": task["task_id"],
                "robot": robot,
                "skill": task["skill"],
                "instruction": instruction,
                "binding": SKILLS[task["skill"]]["binding"],
            }
        )
        print(f"  MATCH {task['task_id']}: {task['skill']} -> {robot} (of {[m.robot for m in feasible]})")
    results = execute(assignments)
    for assignment in assignments:
        result = results.get(assignment["task_id"], {"status": "error", "detail": "executor returned no result"})
        if result.get("status") != "success":
            raise RuntimeError(f"execution failed for {assignment['task_id']} on {assignment['robot']}: {result}")
        summary["dispatched"].append({k: assignment[k] for k in ("task_id", "robot", "skill")})
        print(f"  DONE {assignment['task_id']} ({assignment['robot']})")
    return summary


def make_hitl_gate() -> Callable[[str, str, str], bool]:
    """Human-in-the-loop gate applied BEFORE any command reaches a robot.

    Interactive by default; ``STRANDS_MESH_HITL_ACTIONS=none`` opts out for
    CI/smoke runs (same env contract as the robot_mesh tool, and just as loud
    about it). A decline is a normal outcome, not an error: the task is
    recorded as declined and nothing executes.
    """
    if os.environ.get("STRANDS_MESH_HITL_ACTIONS", "").strip().lower() == "none":

        def auto_approve(action: str, robot: str, instruction: str) -> bool:
            print(f"  [HITL] auto-approved {action} -> {robot} (STRANDS_MESH_HITL_ACTIONS=none): {instruction!r}")
            return True

        return auto_approve

    def prompt_operator(action: str, robot: str, instruction: str) -> bool:
        answer = input(f"  [HITL] approve {action} -> {robot}: {instruction!r} [y/N] ")
        return answer.strip().lower() in {"y", "yes", "approve", "approved"}

    return prompt_operator


def make_loopback_executor() -> Callable[[list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Dry-run execution seam: prints each assignment and reports success."""

    def execute(assignments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        results = {}
        for a in assignments:
            print(f"  [dry-run] {a['robot']} <- {a['instruction']!r} via {a['binding']['execute']}")
            results[a["task_id"]] = {"status": "success", "detail": "dry-run loopback"}
        return results

    return execute


def make_synchronized_executor(sim: Any, n_steps: int) -> Callable[[list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """MuJoCo execution seam: move_to primitives, then ONE run_multi_policy.

    Motion-primitive bindings run first (move_to is a synchronous IK servo);
    every policy binding then executes in a single synchronized multi-robot
    loop - one physics step per tick for all robots, per the
    :meth:`run_multi_policy` contract.
    """
    from strands_robots.policies import create_policy

    def execute(assignments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        policy_batch = [a for a in assignments if a["binding"]["execute"] == "policy"]
        for a in assignments:
            if a["binding"]["execute"] == "move_to":
                results[a["task_id"]] = sim.move_to(robot_name=a["robot"], position=a["binding"]["position"])
        if policy_batch:
            reply = sim.run_multi_policy(
                policies={a["robot"]: create_policy(a["binding"]["policy_provider"]) for a in policy_batch},
                instructions={a["robot"]: a["instruction"] for a in policy_batch},
                n_steps=n_steps,
            )
            for a in policy_batch:
                results[a["task_id"]] = reply
        return results

    return execute


def make_sequential_executor(
    sim: Any, n_steps: int, seed: int
) -> Callable[[list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Portability-fallback execution seam for non-MuJoCo backends (epic D8).

    ``run_multi_policy`` and the motion primitives are MuJoCo-only today
    (#2122 tracks run_multi_policy parity on Isaac, #2123 the motion
    primitives), so every binding here - including move_to-bound skills -
    executes as a sequential per-robot ``run_policy``, the base-ABC contract
    (strands_robots.simulation.base) every backend implements. The dispatch
    layer above runs unchanged; this executor is the entire portability
    boundary.
    """

    def execute(assignments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        results = {}
        for a in assignments:
            results[a["task_id"]] = sim.run_policy(
                robot_name=a["robot"],
                policy_provider=a["binding"].get("policy_provider", "mock"),
                instruction=a["instruction"],
                n_steps=n_steps,
                seed=seed,
            )
        return results

    return execute


def make_dispatcher_tools(
    manifests: Sequence[CapabilityManifest],
    approve: Callable[[str, str, str], bool],
    execute: Callable[[list[dict[str, Any]]], dict[str, dict[str, Any]]],
) -> list[Any]:
    """The dispatcher agent's toolbox: list_robots, match_skill, dispatch.

    Each tool wraps the same deterministic functions the scripted path calls,
    so an LLM dispatcher can choose WHEN to match and dispatch but can never
    bypass the capability filter or the HITL gate.
    """
    from strands import tool

    tasks_by_id = {t["task_id"]: t for t in TASKS}

    @tool
    def list_robots() -> dict[str, Any]:
        """List every robot in the fleet with its site and offered skills."""
        return {
            m.robot: {"site": m.site, "skills": [{"name": s.name, "zones": list(s.zones)} for s in m.skills]}
            for m in manifests
        }

    @tool
    def match_skill(task_id: str) -> dict[str, Any]:
        """Run the capability filter for one task and report the verdict.

        Args:
            task_id: The id of the task to match (e.g. ``"T-01"``).
        """
        task = tasks_by_id.get(task_id)
        if task is None:
            return {"error": f"unknown task_id {task_id!r}; known: {sorted(tasks_by_id)}"}
        feasible, rejections = match_skill_for_task(manifests, task)
        return {"feasible": [m.robot for m in feasible], "rejections": rejections}

    @tool
    def dispatch(task_id: str) -> dict[str, Any]:
        """Dispatch one task through the HITL gate to its matched robot.

        Args:
            task_id: The id of the task to dispatch (e.g. ``"T-01"``).
        """
        task = tasks_by_id.get(task_id)
        if task is None:
            return {"error": f"unknown task_id {task_id!r}; known: {sorted(tasks_by_id)}"}
        return dispatch_tasks([task], manifests, approve, execute)

    return [list_robots, match_skill, dispatch]


def _build_sim(backend: str, seed: int, view: bool) -> Any:
    """One world, three heterogeneous robots via repeated add_robot."""
    from strands_robots.simulation import create_simulation

    random.seed(seed)
    sim = create_simulation(backend)
    result = sim.create_world()
    if result.get("status") != "success":
        sim.destroy()
        raise RuntimeError(f"create_world failed: {result}")
    try:
        for name, embodiment in FLEET.items():
            result = sim.add_robot(name=name, data_config=embodiment, position=FLEET_POSITION[name])
            if result.get("status") != "success":
                raise RuntimeError(f"add_robot({name}) failed: {result}")
        if view:
            result = sim.open_viewer()  # passive viewer; needs a local display
            if result.get("status") != "success":
                raise RuntimeError(f"--view requested but the viewer failed: {result}")
    except BaseException:
        sim.destroy()
        raise
    return sim


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="mujoco", help="simulation backend (mujoco is the acceptance target)")
    parser.add_argument("--seed", type=int, default=42, help="seed for the run")
    parser.add_argument("--n-steps", type=int, default=50, help="policy steps per dispatched skill")
    parser.add_argument("--view", action="store_true", help="open the passive viewer (needs a local display)")
    parser.add_argument("--dry-run", action="store_true", help="no simulator: loopback execution seam")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="drive dispatch through a Strands Agent holding [list_robots, match_skill, dispatch]",
    )
    args = parser.parse_args(argv)

    manifests = build_manifests(FLEET)
    for m in manifests:
        print(f"fleet: {m.robot}@{m.site} offers {[s.name for s in m.skills]}")

    sim = None
    if args.dry_run:
        execute = make_loopback_executor()
    else:
        sim = _build_sim(args.backend, args.seed, args.view)
        if args.backend == "mujoco":
            execute = make_synchronized_executor(sim, args.n_steps)
        else:
            execute = make_sequential_executor(sim, args.n_steps, args.seed)

    approve = make_hitl_gate()
    try:
        if args.agent:
            from strands import Agent

            agent = Agent(
                tools=make_dispatcher_tools(manifests, approve, execute),
                system_prompt=(
                    "You dispatch tasks to a robot fleet. For every task id, call match_skill, "
                    "then dispatch it only if a feasible robot exists. Report rejected tasks verbatim."
                ),
            )
            agent(f"Dispatch these tasks in order: {[t['task_id'] for t in TASKS]}")
            return 0
        summary = dispatch_tasks(TASKS, manifests, approve, execute)
    finally:
        if sim is not None:
            sim.destroy()

    print(f"\nsummary: {summary}")
    if not summary["rejected"]:
        raise RuntimeError("expected the infeasible task to be rejected with an explicit reason")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
