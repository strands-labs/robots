#!/usr/bin/env python3
"""Harness memory: task-specific solution traces + global success rules / failure models.

Persistent memory layer for agentic robot sessions, modeled on Harness VLA
(arXiv:2607.08448). Two memory kinds:

- **Task-Specific Memory**: per-task procedural JSONL trace (one primitive
  invocation per line - the solution *skeleton*) plus a semantic JSON summary
  (why it worked, what to avoid). Spatial arguments in a trace are
  reference-scene bindings, NOT replayable coordinates: the re-grounding
  contract travels with every ``load_trace`` result so any agent that loads a
  trace also receives the "never replay literal coordinates" discipline.
- **Global Memory**: cross-task plain-text success rules and failure models,
  one per line.

Storage is dumb and auditable: JSONL + JSON + text files under
``~/.strands_robots/memory/`` (override with ``STRANDS_MEMORY_DIR``). No
embeddings, no retrieval model - task keying is exact-name, as in the paper.

Why a dedicated tool rather than a generic memory/journal tool (e.g.
``strands_tools`` ``journal``/``memory``): the value here is not storage -
any tool can persist bytes - it is the validation and retrieval *contract*
around the memory, which a free-text store cannot provide:

- **Traces are schema-enforced, not free text.** Every entry must name an
  action from the simulation tool enum or a registered ``strands_robots``
  tool (:func:`get_valid_actions`); free-form code is rejected at save time
  AND re-validated at load time, because the store is a long-lived,
  user-editable directory whose content is later injected into planner
  context (LLM-input-safety baseline, AGENTS.md / PR #92). A journal entry
  read back into the prompt is an unvalidated injection channel.
- **The re-grounding contract travels with the memory.** ``load_trace``
  prepends the "never replay literal coordinates - re-localize from the
  current observation" contract to every result. This retrieval discipline
  is what the paper's +56-point perturbation delta rests on; a generic
  loader returns bytes without it.
- **Robotics provenance is stamped structurally** (backend, robot, library
  version, timestamp) so stale traces are identifiable, and task names are
  path-safety validated before any file is touched.


Session integration pattern (the agent decides what to commit - no automatic
writes):

1. At session start, include ``load_rules()`` output in the agent prompt, plus
   ``load_trace(task)`` when the current task matches a stored key.
2. The agent executes the task, re-localizing every object from the current
   observation (the trace supplies primitive ordering, not coordinates).
3. After a successful rollout, the agent calls ``save_trace`` with the
   primitive sequence it actually used; later attempts may replace a trace
   with a shorter one.
4. Recoverable failures worth remembering become ``append_rule``
   failure-model lines.
"""

import json
import logging
import os
import re
import time
from importlib import metadata as _importlib_metadata
from pathlib import Path
from typing import Any

from strands import tool

from strands_robots.utils import get_base_dir, safe_join

logger = logging.getLogger(__name__)

# Task names become file names: strict allowlist, no path separators, no
# metacharacters. LLM-provided strings are untrusted (see AGENTS.md, PR #92).
_TASK_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_MAX_TASK_NAME_LEN = 128

# Global-memory rule kinds map to fixed file names (never user-derived).
_RULE_FILES = {
    "success_rule": "success_rules.txt",
    "failure_model": "failure_models.txt",
}

# Size caps: rules are opaque text but bounded; traces are bounded so a single
# save cannot balloon the memory dir.
_MAX_RULE_CHARS = 2000
_MAX_RULES_PER_KIND = 1000
_MAX_TRACE_ENTRIES = 2000
_MAX_TRACE_BYTES = 5 * 1024 * 1024  # serialized JSONL budget per task
_MAX_SUMMARY_BYTES = 64 * 1024

_TRACE_SUFFIX = ".trace.jsonl"
_SUMMARY_SUFFIX = ".summary.json"

# The retrieval contract from Harness VLA (Appendix E.3/E.4): memory is for
# the LLM planner, not an executable macro. It is prepended to every
# load_trace result so the discipline travels WITH the memory.
REGROUNDING_CONTRACT = (
    "RE-GROUNDING CONTRACT: This trace is a solution skeleton from a "
    "reference scene, not an executable macro. Reuse the procedural "
    "structure (primitive ordering, where policy calls sit, transition "
    "points), but NEVER replay literal coordinates or spatial arguments - "
    "they are reference-scene bindings. Re-localize every object from the "
    "current observation before acting, and verify each step's outcome "
    "against the current scene, not against the trace."
)

# Valid trace-entry action vocabulary, resolved lazily (module-level cache).
_valid_actions_cache: frozenset[str] | None = None


def _sim_tool_spec_path() -> Path:
    """Path to the MuJoCo simulation tool spec (the sim action enum source)."""
    return Path(__file__).resolve().parent.parent / "simulation" / "mujoco" / "tool_spec.json"


def get_valid_actions() -> frozenset[str]:
    """Return the set of action names a trace entry may reference.

    The vocabulary is the simulation tool's action enum
    (``strands_robots/simulation/mujoco/tool_spec.json``) plus every
    registered tool name in :mod:`strands_robots.tools`. No free-form code:
    a trace is a sequence of invocations of this vocabulary only.

    Returns:
        Frozen set of valid action names (cached after first load).

    Raises:
        OSError: If the tool spec file cannot be read.
        ValueError: If the tool spec JSON is malformed.
    """
    global _valid_actions_cache
    if _valid_actions_cache is None:
        spec_path = _sim_tool_spec_path()
        with open(spec_path, encoding="utf-8") as f:
            spec = json.load(f)
        try:
            sim_actions = spec["properties"]["action"]["enum"]
        except (KeyError, TypeError) as e:
            raise ValueError(f"malformed simulation tool spec at {spec_path}: {e}") from e
        # Lazy import: strands_robots.tools.__init__ is thin (a name map),
        # but importing it here avoids a circular import at module load.
        import strands_robots.tools as _tools

        _valid_actions_cache = frozenset(sim_actions) | frozenset(_tools.__all__)
    return _valid_actions_cache


def _validate_task_name(task: str | None) -> str:
    """Validate a task key before any path construction.

    Args:
        task: Untrusted task name from the agent.

    Returns:
        The validated task name.

    Raises:
        ValueError: If the name is missing, too long, or contains characters
            outside ``[a-zA-Z0-9_-]``.
    """
    if not task:
        raise ValueError("task required")
    if len(task) > _MAX_TASK_NAME_LEN:
        raise ValueError(f"task name too long ({len(task)} > {_MAX_TASK_NAME_LEN} chars)")
    if not _TASK_NAME_RE.match(task):
        raise ValueError(f"invalid task name {task!r}: must match ^[a-zA-Z0-9_-]+$ (no paths, no metacharacters)")
    return task


def _validate_trace(trace: Any) -> list[dict[str, Any]]:
    """Validate a trace: a bounded list of action dicts from the known vocabulary.

    Args:
        trace: Untrusted trace payload from the agent.

    Returns:
        The validated trace (list of dicts, each with a known ``action``).

    Raises:
        ValueError: If the trace is not a list of dicts, exceeds size caps,
            an entry lacks an ``action`` string, an entry references an
            unknown action, or an entry is not JSON-serializable.
    """
    if not isinstance(trace, list) or not trace:
        raise ValueError("trace must be a non-empty list of action dicts")
    if len(trace) > _MAX_TRACE_ENTRIES:
        raise ValueError(f"trace too long ({len(trace)} > {_MAX_TRACE_ENTRIES} entries)")
    valid_actions = get_valid_actions()
    unknown: list[str] = []
    total_bytes = 0
    for i, entry in enumerate(trace):
        if not isinstance(entry, dict):
            raise ValueError(f"trace[{i}] must be a dict, got {type(entry).__name__}")
        action_name = entry.get("action")
        if not isinstance(action_name, str) or not action_name:
            raise ValueError(f"trace[{i}] missing 'action' (string naming a sim action or registered tool)")
        if action_name not in valid_actions:
            unknown.append(f"trace[{i}]: {action_name!r}")
        try:
            total_bytes += len(json.dumps(entry, sort_keys=True))
        except (TypeError, ValueError) as e:
            raise ValueError(f"trace[{i}] is not JSON-serializable: {e}") from e
    if unknown:
        raise ValueError(
            "trace references unknown actions (must name a simulation action or a registered "
            f"strands_robots tool): {'; '.join(unknown)}. Valid actions: {', '.join(sorted(valid_actions))}"
        )
    if total_bytes > _MAX_TRACE_BYTES:
        raise ValueError(f"trace too large ({total_bytes} > {_MAX_TRACE_BYTES} bytes)")
    return trace


def _validate_summary(summary: Any) -> dict[str, Any]:
    """Validate a summary: a JSON object within the size budget.

    Args:
        summary: Untrusted summary payload from the agent.

    Returns:
        The validated summary dict.

    Raises:
        ValueError: If the summary is not a dict, not JSON-serializable, or
            exceeds the size budget.
    """
    if not isinstance(summary, dict) or not summary:
        raise ValueError("summary must be a non-empty JSON object (dict)")
    try:
        size = len(json.dumps(summary, sort_keys=True))
    except (TypeError, ValueError) as e:
        raise ValueError(f"summary is not JSON-serializable: {e}") from e
    if size > _MAX_SUMMARY_BYTES:
        raise ValueError(f"summary too large ({size} > {_MAX_SUMMARY_BYTES} bytes)")
    return summary


def _validate_rule_text(text: str | None) -> str:
    """Validate a global-memory rule line.

    Rules are opaque text but size-capped and line-oriented: control
    characters (including newlines) are rejected so one appended rule is
    always exactly one line in the store.

    Args:
        text: Untrusted rule text from the agent.

    Returns:
        The stripped rule text.

    Raises:
        ValueError: If the text is missing, too long, or contains control
            characters.
    """
    if not text or not text.strip():
        raise ValueError("text required")
    stripped = text.strip()
    if len(stripped) > _MAX_RULE_CHARS:
        raise ValueError(f"rule too long ({len(stripped)} > {_MAX_RULE_CHARS} chars)")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in stripped):
        raise ValueError("rule must be a single line of printable text (no control characters)")
    return stripped


def _version_string() -> str:
    """Best-effort strands-robots version for trace provenance."""
    try:
        return _importlib_metadata.version("strands-robots")
    except _importlib_metadata.PackageNotFoundError:
        return "unknown"


class HarnessMemory:
    """File-backed harness memory store (task traces + global rules).

    Layout under *storage_dir* (default ``get_base_dir()/memory``, i.e.
    ``~/.strands_robots/memory/``; both ``STRANDS_MEMORY_DIR`` and
    ``STRANDS_BASE_DIR`` relocate it):

    - ``tasks/<task>.trace.jsonl`` - one primitive invocation per line
    - ``tasks/<task>.summary.json`` - semantic summary + provenance
    - ``global/success_rules.txt`` / ``global/failure_models.txt`` - one
      rule per line

    All file names derived from agent input go through
    :func:`_validate_task_name` + :func:`strands_robots.utils.safe_join`.
    Store content is re-validated on load: the memory directory is long-lived
    and user-editable, so it is not a trust boundary. Not safe for concurrent
    writers to the same store; the intended use is one agent session per
    store. All files are read and written as UTF-8 regardless of locale.
    """

    def __init__(self, storage_dir: Path | None = None):
        if storage_dir is None:
            custom = os.getenv("STRANDS_MEMORY_DIR")
            storage_dir = Path(custom) if custom else get_base_dir() / "memory"
        self.storage_dir = storage_dir
        self.tasks_dir = storage_dir / "tasks"
        self.global_dir = storage_dir / "global"

    def _ensure_dirs(self) -> None:
        """Create the store layout. Called by write paths only, so read-only
        actions on a fresh box do not create directories as a side effect."""
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.global_dir.mkdir(parents=True, exist_ok=True)

    def _trace_path(self, task: str) -> Path:
        return safe_join(self.tasks_dir, task + _TRACE_SUFFIX)

    def _summary_path(self, task: str) -> Path:
        return safe_join(self.tasks_dir, task + _SUMMARY_SUFFIX)

    def save_trace(
        self,
        task: str,
        trace: list[dict[str, Any]],
        summary: dict[str, Any],
        *,
        backend: str | None = None,
        robot: str | None = None,
    ) -> dict[str, Any]:
        """Write (or replace) a task's trace + summary. Returns the provenance block.

        The write is atomic per store: both payloads are fully written to
        temp files first, then moved into place with :func:`os.replace`, so a
        failed save never leaves a torn trace/summary pair or an orphaned
        trace (a summary-less trace would be listed by ``list_tasks`` but
        never loadable by ``load_trace``).
        """
        self._ensure_dirs()
        provenance = {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "strands_robots_version": _version_string(),
            "backend": backend,
            "robot": robot,
        }
        stored_summary = dict(summary)
        stored_summary["provenance"] = provenance
        trace_path = self._trace_path(task)
        summary_path = self._summary_path(task)
        trace_lines = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in trace)
        summary_text = json.dumps(stored_summary, indent=2, sort_keys=True) + "\n"
        # Two-phase commit: write both temp files completely, then rename both.
        # An error while writing leaves the store untouched; the unavoidable
        # residual window is the instant between the two os.replace calls.
        trace_tmp = trace_path.with_suffix(trace_path.suffix + ".tmp")
        summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
        try:
            trace_tmp.write_text(trace_lines, encoding="utf-8")
            summary_tmp.write_text(summary_text, encoding="utf-8")
            os.replace(trace_tmp, trace_path)
            os.replace(summary_tmp, summary_path)
        except OSError:
            trace_tmp.unlink(missing_ok=True)
            summary_tmp.unlink(missing_ok=True)
            raise
        return provenance

    def load_trace(self, task: str) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        """Return (trace, summary) for *task*, or None if not stored.

        Store content is re-validated on every load (same checks as the save
        path): the memory directory is a long-lived plain-text store that may
        have been edited outside this tool, so nothing is handed to the
        planner without passing the trace/summary validators again.

        Raises:
            ValueError: If the stored trace or summary is corrupt or fails
                re-validation.
        """
        trace_path = self._trace_path(task)
        summary_path = self._summary_path(task)
        if not trace_path.exists() or not summary_path.exists():
            return None
        try:
            trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(
                f"stored memory for task {task!r} is corrupt ({e}); delete it with delete_trace and re-save"
            ) from e
        # Re-validate before anything reaches planner context: the store is
        # not a trust boundary (hand-edited or truncated files must not be
        # echoed back as a "success" payload).
        try:
            _validate_trace(trace)
            _validate_summary(summary)
        except ValueError as e:
            raise ValueError(
                f"stored memory for task {task!r} failed re-validation ({e}); delete it with delete_trace and re-save"
            ) from e
        return trace, summary

    def delete_trace(self, task: str) -> bool:
        """Delete a task's trace + summary. Returns True if anything was removed."""
        removed = False
        for path in (self._trace_path(task), self._summary_path(task)):
            if path.exists():
                path.unlink()
                removed = True
        return removed

    def list_tasks(self) -> list[str]:
        """Sorted task keys that are actually loadable.

        A key is listed only when BOTH the trace and the summary file exist
        (matching ``load_trace``'s requirement) and the stem passes the same
        task-name validation as the write path - stray files planted in the
        store never advertise keys the tool would refuse to load.
        """
        tasks = []
        for p in self.tasks_dir.glob("*" + _TRACE_SUFFIX):
            name = p.name[: -len(_TRACE_SUFFIX)]
            if not _TASK_NAME_RE.match(name) or len(name) > _MAX_TASK_NAME_LEN:
                continue
            if not (self.tasks_dir / (name + _SUMMARY_SUFFIX)).exists():
                continue
            tasks.append(name)
        return sorted(tasks)

    def append_rule(self, kind: str, text: str) -> int:
        """Append one rule line under *kind*; returns the new rule count."""
        self._ensure_dirs()
        path = self.global_dir / _RULE_FILES[kind]
        existing = self._read_rules(path)
        if len(existing) >= _MAX_RULES_PER_KIND:
            raise ValueError(f"{kind} store full ({_MAX_RULES_PER_KIND} rules); delete or consolidate first")
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        return len(existing) + 1

    def load_rules(self) -> dict[str, list[str]]:
        """Return all global rules keyed by kind."""
        return {kind: self._read_rules(self.global_dir / fname) for kind, fname in _RULE_FILES.items()}

    @staticmethod
    def _read_rules(path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"rule store at {path.name} is not valid UTF-8 ({e})") from e
        return [line for line in content.splitlines() if line.strip()]


@tool
def harness_memory(
    action: str,
    task: str | None = None,
    trace: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
    kind: str | None = None,
    text: str | None = None,
    backend: str | None = None,
    robot: str | None = None,
) -> dict[str, Any]:
    """Persist and retrieve harness memory: task solution traces + global rules.

    Task-Specific Memory stores HOW a task was solved (primitive ordering,
    where policy calls sit) - never WHERE objects happened to be. Loaded
    traces carry a re-grounding contract: reuse the procedural structure, but
    never replay literal coordinates; re-localize every object from the
    current observation. Global Memory stores cross-task success rules and
    failure models as plain text.

    Actions:
        Task-Specific Memory:
        - "save_trace": Store (or replace) a task's solution trace + summary.
          Requires task, trace (list of action dicts - each entry must name a
          simulation action or registered tool in its "action" field), and
          summary (JSON object: strategy, what to avoid). Optional backend /
          robot are recorded as provenance alongside the library version.
        - "load_trace": Return a task's trace + summary with the re-grounding
          contract prepended. Requires task.
        - "list_tasks": List all stored task keys.
        - "delete_trace": Remove a task's trace + summary. Requires task.

        Global Memory:
        - "append_rule": Append one plain-text rule. Requires kind
          ("success_rule" or "failure_model") and text (single line).
        - "load_rules": Return all success rules and failure models.

    Storage: ``~/.strands_robots/memory/`` (override with
    ``STRANDS_MEMORY_DIR``). Plain JSONL / JSON / text files - auditable,
    no embeddings; task keying is exact-name. The store assumes one agent
    session per store (no concurrent-writer coordination), and everything
    read back from disk is re-validated before it reaches the response.

    Args:
        action: Action to perform.
        task: Task key, matched exactly on later runs. Must match
            ^[a-zA-Z0-9_-]+$ (max 128 chars; no dots, so use "task_v2"
            rather than "task.v2").
        trace: Solution skeleton: list of primitive-invocation dicts, one per
            step, e.g. {"action": "run_policy", "instruction": "grasp the
            bowl"}. Spatial values are reference bindings, not replay targets.
        summary: Semantic summary: task description, strategy, pitfalls to
            avoid ("avoid" list), success flag.
        kind: Rule kind for append_rule: "success_rule" or "failure_model".
        text: Rule text for append_rule (single line, max 2000 chars).
        backend: Optional provenance: simulation backend the trace was
            collected on (e.g. "mujoco").
        robot: Optional provenance: robot the trace was collected with
            (e.g. "so100").

    Returns:
        Dict containing status and response content.
    """
    try:
        memory = HarnessMemory()

        if action == "save_trace":
            validated_task = _validate_task_name(task)
            validated_trace = _validate_trace(trace)
            validated_summary = _validate_summary(summary)
            provenance = memory.save_trace(
                validated_task, validated_trace, validated_summary, backend=backend, robot=robot
            )
            return {
                "status": "success",
                "content": [
                    {"text": f"Saved trace for task '{validated_task}' ({len(validated_trace)} steps)"},
                    {"json": {"task": validated_task, "steps": len(validated_trace), "provenance": provenance}},
                ],
            }

        if action == "load_trace":
            validated_task = _validate_task_name(task)
            loaded = memory.load_trace(validated_task)
            if loaded is None:
                return {
                    "status": "error",
                    "content": [{"text": f"No trace stored for task '{validated_task}'"}],
                }
            loaded_trace, loaded_summary = loaded
            return {
                "status": "success",
                "content": [
                    {"text": REGROUNDING_CONTRACT},
                    {"text": f"Loaded trace for task '{validated_task}' ({len(loaded_trace)} steps)"},
                    {
                        "json": {
                            "regrounding_contract": REGROUNDING_CONTRACT,
                            "task": validated_task,
                            "trace": loaded_trace,
                            "summary": loaded_summary,
                        }
                    },
                ],
            }

        if action == "list_tasks":
            tasks = memory.list_tasks()
            listing = "\n".join(f"- {t}" for t in tasks) if tasks else "(no traces stored)"
            return {
                "status": "success",
                "content": [
                    {"text": f"Stored tasks ({len(tasks)}):\n{listing}"},
                    {"json": {"tasks": tasks}},
                ],
            }

        if action == "delete_trace":
            validated_task = _validate_task_name(task)
            removed = memory.delete_trace(validated_task)
            if not removed:
                return {
                    "status": "error",
                    "content": [{"text": f"No trace stored for task '{validated_task}'"}],
                }
            return {
                "status": "success",
                "content": [
                    {"text": f"Deleted trace for task '{validated_task}'"},
                    {"json": {"task": validated_task, "deleted": True}},
                ],
            }

        if action == "append_rule":
            if kind not in _RULE_FILES:
                return {
                    "status": "error",
                    "content": [{"text": f"kind must be one of: {', '.join(sorted(_RULE_FILES))}"}],
                }
            validated_text = _validate_rule_text(text)
            count = memory.append_rule(kind, validated_text)
            return {
                "status": "success",
                "content": [
                    {"text": f"Appended {kind} ({count} total)"},
                    {"json": {"kind": kind, "count": count}},
                ],
            }

        if action == "load_rules":
            rules = memory.load_rules()
            n_success = len(rules["success_rule"])
            n_failure = len(rules["failure_model"])
            sections = []
            if rules["success_rule"]:
                sections.append("Success rules:\n" + "\n".join(f"- {r}" for r in rules["success_rule"]))
            if rules["failure_model"]:
                sections.append("Failure models:\n" + "\n".join(f"- {r}" for r in rules["failure_model"]))
            body = "\n\n".join(sections) if sections else "(no rules stored)"
            return {
                "status": "success",
                "content": [
                    {"text": f"Global memory ({n_success} success rules, {n_failure} failure models):\n{body}"},
                    {"json": {"success_rules": rules["success_rule"], "failure_models": rules["failure_model"]}},
                ],
            }

        valid = "save_trace, load_trace, list_tasks, delete_trace, append_rule, load_rules"
        return {
            "status": "error",
            "content": [{"text": f"Unknown action: {action}. Available actions: {valid}"}],
        }

    except (ValueError, TypeError, OSError) as e:
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
