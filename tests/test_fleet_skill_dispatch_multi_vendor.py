"""Smoke test for examples/fleet/01_skill_dispatch_multi_vendor.py (issue #2180).

The dispatch layer is exercised end to end with no simulator and no network:
capability manifests are derived from the real robot registry's metadata
(category, joint count, gripper), matching runs through the suite's shared
``capabilities.py`` hard-constraint filter, the HITL gate is driven in both
its CI (auto-approve) and declined postures, and execution goes through stub
seams that record what would have run. The executor-construction tests pin
the two execution shapes the example ships: one synchronized
``run_multi_policy`` batch plus ``move_to`` primitives on MuJoCo, and the
sequential per-robot ``run_policy`` portability fallback (epic D8) elsewhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_FLEET_DIR = Path(__file__).resolve().parent.parent / "examples" / "fleet"
_EXAMPLE_PATH = _FLEET_DIR / "01_skill_dispatch_multi_vendor.py"

# Loaded under a distinctive module name: "capabilities" (its sibling import)
# is generic enough to collide, so both are evicted between loads.
_MODULE_NAME = "fleet_skill_dispatch_multi_vendor_example"


@pytest.fixture
def example(monkeypatch):
    """Load the example module fresh, with its sibling schema importable."""
    monkeypatch.syspath_prepend(str(_FLEET_DIR))
    for name in (_MODULE_NAME, "capabilities"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _EXAMPLE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    yield mod
    for name in (_MODULE_NAME, "capabilities"):
        sys.modules.pop(name, None)


def _approve_all(action: str, robot: str, instruction: str) -> bool:
    return True


class _RecordingExecutor:
    """Execution seam stub: records assignments, reports success."""

    def __init__(self):
        self.assignments: list[dict[str, Any]] = []

    def __call__(self, assignments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        self.assignments.extend(assignments)
        return {a["task_id"]: {"status": "success"} for a in assignments}


def test_manifests_derive_from_registry_metadata_not_names(example):
    """Offered skills follow category/joints/gripper metadata, not robot names.

    The arm (and only the arm) offers the staging skill because only its
    registry entry is category ``arm`` with a gripper; both mobile robots -
    a wheeled base and a quadruped from different vendors - offer transport
    because their metadata satisfies that skill's requirements.
    """
    manifests = {m.robot: m for m in example.build_manifests(example.FLEET)}
    assert [s.name for s in manifests["so101"].skills] == ["stage_part"]
    assert [s.name for s in manifests["lekiwi"].skills] == ["transport_tote"]
    assert [s.name for s in manifests["go2"].skills] == ["transport_tote"]


def test_unknown_embodiment_is_refused_not_dispatched(example):
    """A robot whose capabilities cannot be read is never given a manifest."""
    with pytest.raises(ValueError, match="not in the robot registry"):
        example.build_manifests({"ghost": "no_such_embodiment_zz"})


def test_dispatch_selects_the_correct_robot_for_two_distinct_tasks(example):
    """Two tasks with distinct capability requirements land on distinct robots.

    T-01 (stage_part: arm + gripper) can only go to the arm; T-02
    (transport_tote: mobile) goes to a mobile robot chosen deterministically
    from the sorted feasible set. No per-embodiment branching exists to make
    this happen - the verdicts are the capability filter's alone.
    """
    manifests = example.build_manifests(example.FLEET)
    executor = _RecordingExecutor()

    summary = example.dispatch_tasks(example.TASKS, manifests, _approve_all, executor)

    dispatched = {d["task_id"]: d["robot"] for d in summary["dispatched"]}
    assert dispatched["T-01"] == "so101"
    assert dispatched["T-02"] == "go2"  # first of the sorted feasible mobiles
    assert [a["task_id"] for a in executor.assignments] == ["T-01", "T-02"]


def test_infeasible_task_is_rejected_with_a_per_robot_reason(example):
    """A task no robot can serve is rejected explicitly, never silently.

    T-03's 40 kg payload exceeds every transport offer; the arm does not
    offer the skill at all. The rejection carries one machine-readable
    reason per robot naming the failing constraint, and the executor never
    sees the task.
    """
    manifests = example.build_manifests(example.FLEET)
    executor = _RecordingExecutor()

    summary = example.dispatch_tasks(example.TASKS, manifests, _approve_all, executor)

    assert [r["task_id"] for r in summary["rejected"]] == ["T-03"]
    reasons = {r["robot"]: r for r in summary["rejected"][0]["rejections"]}
    assert reasons["lekiwi"]["constraint"] == "payload_kg"
    assert reasons["go2"]["constraint"] == "payload_kg"
    assert reasons["so101"]["constraint"] == "skill"
    assert all(a["task_id"] != "T-03" for a in executor.assignments)


def test_hitl_decline_is_a_structured_outcome_and_nothing_executes(example):
    """A declined approval records the task as declined; no work reaches a robot."""
    manifests = example.build_manifests(example.FLEET)
    executor = _RecordingExecutor()

    summary = example.dispatch_tasks(example.TASKS, manifests, lambda action, robot, instruction: False, executor)

    assert summary["dispatched"] == []
    assert executor.assignments == []
    assert {d["task_id"] for d in summary["declined"]} == {"T-01", "T-02"}
    assert all(d["code"] == "hitl_declined" for d in summary["declined"])


def test_hitl_gate_auto_approves_only_in_ci_mode(example, monkeypatch):
    """STRANDS_MESH_HITL_ACTIONS=none is the CI posture (epic D4)."""
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "none")
    assert example.make_hitl_gate()("dispatch", "so101", "task T-01") is True

    monkeypatch.delenv("STRANDS_MESH_HITL_ACTIONS")
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert example.make_hitl_gate()("dispatch", "so101", "task T-01") is False
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert example.make_hitl_gate()("dispatch", "so101", "task T-01") is True


class _StubSim:
    """Sim stub recording the execution calls the two executors make."""

    def __init__(self):
        self.move_to_calls: list[dict[str, Any]] = []
        self.multi_policy_calls: list[dict[str, Any]] = []
        self.run_policy_calls: list[dict[str, Any]] = []

    def move_to(self, robot_name, position):
        self.move_to_calls.append({"robot_name": robot_name, "position": position})
        return {"status": "success"}

    def run_multi_policy(self, policies, instructions, n_steps):
        self.multi_policy_calls.append({"policies": policies, "instructions": instructions, "n_steps": n_steps})
        return {"status": "success"}

    def run_policy(self, **kwargs):
        self.run_policy_calls.append(kwargs)
        return {"status": "success"}


def _matched_assignments(example):
    manifests = example.build_manifests(example.FLEET)
    assignments = []
    for task in example.TASKS:
        feasible, _rejections = example.match_skill_for_task(manifests, task)
        if feasible:
            assignments.append(
                {
                    "task_id": task["task_id"],
                    "robot": feasible[0].robot,
                    "skill": task["skill"],
                    "instruction": example._task_instruction(task),
                    "binding": example.SKILLS[task["skill"]]["binding"],
                }
            )
    return assignments


def test_synchronized_executor_batches_policies_into_one_run_multi_policy(example):
    """MuJoCo execution: move_to for the primitive binding, ONE multi-policy batch.

    The policy-bound dispatches share a single synchronized ``run_multi_policy``
    call (one physics step per tick for all robots), and the primitive-bound
    staging skill goes through ``move_to`` - never a per-robot policy loop.
    """
    sim = _StubSim()
    execute = example.make_synchronized_executor(sim, n_steps=7)

    results = execute(_matched_assignments(example))

    assert [c["robot_name"] for c in sim.move_to_calls] == ["so101"]
    assert len(sim.multi_policy_calls) == 1
    batch = sim.multi_policy_calls[0]
    assert set(batch["policies"]) == {"go2"}
    assert batch["n_steps"] == 7
    assert sim.run_policy_calls == []
    assert results["T-01"]["status"] == "success"
    assert results["T-02"]["status"] == "success"


def test_sequential_fallback_executes_every_binding_via_run_policy(example):
    """The D8 portability fallback: per-robot run_policy, dispatch unchanged.

    On a backend without run_multi_policy / motion primitives (#2122, #2123),
    every assignment - including the move_to-bound one - executes through the
    base-ABC ``run_policy``, seeded, one robot at a time.
    """
    sim = _StubSim()
    execute = example.make_sequential_executor(sim, n_steps=7, seed=42)

    results = execute(_matched_assignments(example))

    assert sim.multi_policy_calls == [] and sim.move_to_calls == []
    assert [c["robot_name"] for c in sim.run_policy_calls] == ["so101", "go2"]
    assert all(c["n_steps"] == 7 and c["seed"] == 42 for c in sim.run_policy_calls)
    assert all(results[t]["status"] == "success" for t in ("T-01", "T-02"))


def test_executor_failure_raises_instead_of_reporting_success(example):
    """A failed execution is fatal - the summary never absorbs an error."""
    manifests = example.build_manifests(example.FLEET)

    def broken_executor(assignments):
        return {a["task_id"]: {"status": "error", "detail": "actuator fault"} for a in assignments}

    with pytest.raises(RuntimeError, match="execution failed for T-01"):
        example.dispatch_tasks(example.TASKS, manifests, _approve_all, broken_executor)
