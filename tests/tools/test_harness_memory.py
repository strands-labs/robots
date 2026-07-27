"""Behavior tests for the ``harness_memory`` agent tool.

Pins the issue's acceptance criteria (Harness VLA style memory):

- Round-trip: ``save_trace`` then, with a fresh store instance (simulating a
  new process), ``load_trace`` returns identical trace + summary content plus
  the re-grounding contract prepended.
- Trace entries referencing unknown actions are rejected with a clear error.
- Unsafe task names are rejected (regression tests for path traversal and
  shell metacharacters).
- Global rules append/load round-trips, kind allowlist, and size caps.
- ``STRANDS_MEMORY_DIR`` relocates the store.
- Every action returns the ``{"status", "content"}`` tool-result contract with
  ASCII-only text, and the tool never raises.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from typing import Any

import pytest

from strands_robots.tools.harness_memory import (
    REGROUNDING_CONTRACT,
    HarnessMemory,
    get_valid_actions,
    harness_memory,
)
from tests.tool_result_contract import assert_strands_tool_result, tool_json


def _texts(result: dict[str, Any]) -> str:
    """Concatenate all content ``text`` fields from a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def _assert_ascii(result: dict[str, Any]) -> None:
    """Every user-facing text must be plain ASCII (no emojis)."""
    text = _texts(result)
    assert text.isascii(), f"non-ASCII characters in tool output: {text!r}"


def _check(result: dict[str, Any]) -> dict[str, Any]:
    """Assert the tool-result contract + ASCII rule, return the result."""
    assert_strands_tool_result(result)
    _assert_ascii(result)
    return result


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    """Point STRANDS_MEMORY_DIR at a temp dir so the store is isolated."""
    d = tmp_path / "memory"
    monkeypatch.setenv("STRANDS_MEMORY_DIR", str(d))
    return d


VALID_TRACE = [
    {"action": "run_policy", "instruction": "grasp the black bowl", "n_steps": 50},
    {"action": "move_object", "name": "bowl", "position": [0.12, -0.08, 0.92]},
    {"action": "get_state"},
]

VALID_SUMMARY = {
    "task": "put the black bowl on the wooden tray",
    "success": True,
    "strategy": "use the policy for grasping, then analytic transport and release",
    "avoid": ["do not reuse reference xyz values"],
}


# --------------------------------------------------------------------------- #
# Action vocabulary
# --------------------------------------------------------------------------- #
def test_valid_actions_include_sim_enum_and_registered_tools():
    actions = get_valid_actions()
    # Sim tool enum members
    assert "run_policy" in actions
    assert "add_object" in actions
    assert "get_state" in actions
    # Registered strands_robots tools
    assert "pose_tool" in actions
    assert "harness_memory" in actions


# --------------------------------------------------------------------------- #
# Round-trip (acceptance: save -> new process -> load, identical + contract)
# --------------------------------------------------------------------------- #
def test_save_load_round_trip_with_regrounding_contract(memory_dir):
    save = _check(
        harness_memory(
            action="save_trace",
            task="put_bowl_on_tray",
            trace=VALID_TRACE,
            summary=VALID_SUMMARY,
            backend="mujoco",
            robot="so100",
        )
    )
    assert save["status"] == "success"
    provenance = tool_json(save)["provenance"]
    assert provenance["backend"] == "mujoco"
    assert provenance["robot"] == "so100"
    assert provenance["strands_robots_version"]

    load = _check(harness_memory(action="load_trace", task="put_bowl_on_tray"))
    assert load["status"] == "success"
    payload = tool_json(load)

    # Identical content back
    assert payload["trace"] == VALID_TRACE
    for key, value in VALID_SUMMARY.items():
        assert payload["summary"][key] == value
    assert payload["summary"]["provenance"] == provenance

    # The re-grounding contract travels with the memory: first text block AND
    # in the structured payload.
    assert load["content"][0]["text"] == REGROUNDING_CONTRACT
    assert payload["regrounding_contract"] == REGROUNDING_CONTRACT
    assert "never replay literal coordinates" in REGROUNDING_CONTRACT.lower()


def test_round_trip_survives_fresh_store_instance(memory_dir):
    """Persistence across store instances stands in for a new process."""
    store = HarnessMemory()
    store.save_trace("stack_cubes", VALID_TRACE, VALID_SUMMARY)
    del store

    fresh = HarnessMemory()
    loaded = fresh.load_trace("stack_cubes")
    assert loaded is not None
    trace, summary = loaded
    assert trace == VALID_TRACE
    assert summary["strategy"] == VALID_SUMMARY["strategy"]


def test_round_trip_survives_new_process(memory_dir):
    """Acceptance: save_trace -> NEW PROCESS -> load_trace returns identical
    content plus the re-grounding contract."""
    harness_memory(action="save_trace", task="stack_cubes", trace=VALID_TRACE, summary=VALID_SUMMARY)

    script = (
        "import json, sys\n"
        "from strands_robots.tools.harness_memory import REGROUNDING_CONTRACT, harness_memory\n"
        "result = harness_memory(action='load_trace', task='stack_cubes')\n"
        "assert result['status'] == 'success', result\n"
        "assert result['content'][0]['text'] == REGROUNDING_CONTRACT\n"
        "payload = next(c['json'] for c in result['content'] if 'json' in c)\n"
        "json.dump(payload, sys.stdout)\n"
    )
    env = dict(os.environ, STRANDS_MEMORY_DIR=str(memory_dir))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trace"] == VALID_TRACE
    assert payload["regrounding_contract"]


def test_save_trace_replaces_existing(memory_dir):
    harness_memory(action="save_trace", task="t1", trace=VALID_TRACE, summary=VALID_SUMMARY)
    shorter = [{"action": "run_policy", "instruction": "grasp"}]
    _check(harness_memory(action="save_trace", task="t1", trace=shorter, summary=VALID_SUMMARY))
    payload = tool_json(harness_memory(action="load_trace", task="t1"))
    assert payload["trace"] == shorter


def test_trace_file_is_jsonl_one_entry_per_line(memory_dir):
    harness_memory(action="save_trace", task="t1", trace=VALID_TRACE, summary=VALID_SUMMARY)
    trace_file = memory_dir / "tasks" / "t1.trace.jsonl"
    lines = [ln for ln in trace_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == len(VALID_TRACE)
    assert all(isinstance(json.loads(ln), dict) for ln in lines)


# --------------------------------------------------------------------------- #
# Trace validation (acceptance: unknown actions rejected with clear error)
# --------------------------------------------------------------------------- #
def test_unknown_action_in_trace_rejected(memory_dir):
    bad = [{"action": "run_policy"}, {"action": "teleport_object"}]
    result = _check(harness_memory(action="save_trace", task="t1", trace=bad, summary=VALID_SUMMARY))
    assert result["status"] == "error"
    text = _texts(result)
    assert "teleport_object" in text
    assert "unknown action" in text.lower()
    # Nothing persisted
    assert tool_json(harness_memory(action="list_tasks"))["tasks"] == []


def test_trace_entry_without_action_rejected(memory_dir):
    result = _check(harness_memory(action="save_trace", task="t1", trace=[{"xyz": [0, 0, 0]}], summary=VALID_SUMMARY))
    assert result["status"] == "error"
    assert "action" in _texts(result)


def test_trace_entry_non_dict_rejected(memory_dir):
    result = _check(
        harness_memory(action="save_trace", task="t1", trace=["run_policy"], summary=VALID_SUMMARY)  # type: ignore[list-item]
    )
    assert result["status"] == "error"


def test_empty_trace_rejected(memory_dir):
    result = _check(harness_memory(action="save_trace", task="t1", trace=[], summary=VALID_SUMMARY))
    assert result["status"] == "error"


def test_free_form_code_action_rejected(memory_dir):
    bad = [{"action": "__import__('os').system('rm -rf /')"}]
    result = _check(harness_memory(action="save_trace", task="t1", trace=bad, summary=VALID_SUMMARY))
    assert result["status"] == "error"


def test_summary_must_be_object(memory_dir):
    result = _check(
        harness_memory(action="save_trace", task="t1", trace=VALID_TRACE, summary="it worked")  # type: ignore[arg-type]
    )
    assert result["status"] == "error"


def test_trace_entry_count_cap(memory_dir, monkeypatch):
    monkeypatch.setattr("strands_robots.tools.harness_memory._MAX_TRACE_ENTRIES", 2)
    result = _check(harness_memory(action="save_trace", task="t1", trace=VALID_TRACE, summary=VALID_SUMMARY))
    assert result["status"] == "error"
    assert "too long" in _texts(result)


# --------------------------------------------------------------------------- #
# Task name safety (acceptance: traversal + metacharacters regressions)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_task",
    [
        "../escape",
        "..",
        "a/../../etc/passwd",
        "tasks/../../secrets",
        "..\\windows\\escape",
        "/etc/passwd",
        "a;rm -rf /",
        "a|b",
        "a$(whoami)",
        "a`id`",
        "a b",
        "a\nb",
        "a\x00b",
        "task.name",
        "",
    ],
)
def test_unsafe_task_names_rejected(memory_dir, bad_task):
    for act in ("save_trace", "load_trace", "delete_trace"):
        kwargs: dict[str, Any] = {"action": act, "task": bad_task}
        if act == "save_trace":
            kwargs.update(trace=VALID_TRACE, summary=VALID_SUMMARY)
        result = _check(harness_memory(**kwargs))
        assert result["status"] == "error", f"{act} accepted unsafe task {bad_task!r}"
    # Nothing escaped the store or landed inside it
    assert tool_json(harness_memory(action="list_tasks"))["tasks"] == []


def test_overlong_task_name_rejected(memory_dir):
    result = _check(harness_memory(action="save_trace", task="x" * 129, trace=VALID_TRACE, summary=VALID_SUMMARY))
    assert result["status"] == "error"
    assert "too long" in _texts(result)


def test_writes_stay_under_memory_dir(memory_dir, tmp_path):
    harness_memory(action="save_trace", task="safe_task", trace=VALID_TRACE, summary=VALID_SUMMARY)
    outside = [
        p
        for p in tmp_path.rglob("*")
        if p.is_file() and memory_dir not in p.parents and not p.is_relative_to(memory_dir)
    ]
    assert outside == []
    assert (memory_dir / "tasks" / "safe_task.trace.jsonl").exists()


# --------------------------------------------------------------------------- #
# list_tasks / delete_trace
# --------------------------------------------------------------------------- #
def test_list_tasks_empty_and_populated(memory_dir):
    result = _check(harness_memory(action="list_tasks"))
    assert result["status"] == "success"
    assert tool_json(result)["tasks"] == []

    harness_memory(action="save_trace", task="beta", trace=VALID_TRACE, summary=VALID_SUMMARY)
    harness_memory(action="save_trace", task="alpha", trace=VALID_TRACE, summary=VALID_SUMMARY)
    result = _check(harness_memory(action="list_tasks"))
    assert tool_json(result)["tasks"] == ["alpha", "beta"]


def test_delete_trace_removes_both_files(memory_dir):
    harness_memory(action="save_trace", task="t1", trace=VALID_TRACE, summary=VALID_SUMMARY)
    result = _check(harness_memory(action="delete_trace", task="t1"))
    assert result["status"] == "success"
    assert not (memory_dir / "tasks" / "t1.trace.jsonl").exists()
    assert not (memory_dir / "tasks" / "t1.summary.json").exists()
    assert tool_json(harness_memory(action="list_tasks"))["tasks"] == []


def test_delete_missing_trace_errors(memory_dir):
    result = _check(harness_memory(action="delete_trace", task="never_saved"))
    assert result["status"] == "error"


def test_load_missing_trace_errors(memory_dir):
    result = _check(harness_memory(action="load_trace", task="never_saved"))
    assert result["status"] == "error"
    assert "never_saved" in _texts(result)


# --------------------------------------------------------------------------- #
# Global memory (rules)
# --------------------------------------------------------------------------- #
def test_rules_append_and_load_round_trip(memory_dir):
    failure = (
        "If the gripper closes but the object does not move with the end "
        "effector, treat the attempt as an empty grasp and re-localize."
    )
    success = "Verify placement with the benchmark success signal before declaring done."

    r1 = _check(harness_memory(action="append_rule", kind="failure_model", text=failure))
    assert r1["status"] == "success"
    r2 = _check(harness_memory(action="append_rule", kind="success_rule", text=success))
    assert r2["status"] == "success"

    loaded = _check(harness_memory(action="load_rules"))
    payload = tool_json(loaded)
    assert payload["failure_models"] == [failure]
    assert payload["success_rules"] == [success]


def test_append_rule_rejects_bad_kind(memory_dir):
    result = _check(harness_memory(action="append_rule", kind="vibe", text="whatever"))
    assert result["status"] == "error"
    assert "success_rule" in _texts(result)


def test_append_rule_rejects_multiline_and_empty(memory_dir):
    assert _check(harness_memory(action="append_rule", kind="success_rule", text="a\nb"))["status"] == "error"
    assert _check(harness_memory(action="append_rule", kind="success_rule", text="   "))["status"] == "error"
    assert _check(harness_memory(action="append_rule", kind="success_rule", text=None))["status"] == "error"


def test_append_rule_size_cap(memory_dir):
    result = _check(harness_memory(action="append_rule", kind="success_rule", text="x" * 2001))
    assert result["status"] == "error"
    assert "too long" in _texts(result)


def test_rule_count_cap(memory_dir, monkeypatch):
    monkeypatch.setattr("strands_robots.tools.harness_memory._MAX_RULES_PER_KIND", 2)
    assert harness_memory(action="append_rule", kind="success_rule", text="one")["status"] == "success"
    assert harness_memory(action="append_rule", kind="success_rule", text="two")["status"] == "success"
    result = _check(harness_memory(action="append_rule", kind="success_rule", text="three"))
    assert result["status"] == "error"
    assert "full" in _texts(result)


def test_load_rules_empty(memory_dir):
    payload = tool_json(_check(harness_memory(action="load_rules")))
    assert payload == {"success_rules": [], "failure_models": []}


# --------------------------------------------------------------------------- #
# Storage location
# --------------------------------------------------------------------------- #
def test_strands_memory_dir_env_var_relocates_store(tmp_path, monkeypatch):
    custom = tmp_path / "elsewhere"
    monkeypatch.setenv("STRANDS_MEMORY_DIR", str(custom))
    harness_memory(action="save_trace", task="t1", trace=VALID_TRACE, summary=VALID_SUMMARY)
    assert (custom / "tasks" / "t1.trace.jsonl").exists()


def test_default_store_under_base_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("STRANDS_MEMORY_DIR", raising=False)
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path / "base"))
    harness_memory(action="save_trace", task="t1", trace=VALID_TRACE, summary=VALID_SUMMARY)
    assert (tmp_path / "base" / "memory" / "tasks" / "t1.trace.jsonl").exists()


# --------------------------------------------------------------------------- #
# Tool contract
# --------------------------------------------------------------------------- #
def test_unknown_tool_action_errors(memory_dir):
    result = _check(harness_memory(action="defragment"))
    assert result["status"] == "error"
    assert "save_trace" in _texts(result)


def test_missing_required_params_error_not_raise(memory_dir):
    for kwargs in (
        {"action": "save_trace"},
        {"action": "save_trace", "task": "t1"},
        {"action": "save_trace", "task": "t1", "trace": VALID_TRACE},
        {"action": "load_trace"},
        {"action": "delete_trace"},
        {"action": "append_rule"},
        {"action": "append_rule", "kind": "success_rule"},
    ):
        result = _check(harness_memory(**kwargs))
        assert result["status"] == "error", f"expected error for {kwargs}"


def test_module_source_is_ascii():
    source = inspect.getfile(HarnessMemory)
    with open(source, encoding="utf-8") as f:
        content = f.read()
    assert content.isascii(), "harness_memory.py must contain only ASCII characters"
