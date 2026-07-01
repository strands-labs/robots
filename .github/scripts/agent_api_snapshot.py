#!/usr/bin/env python3
"""Snapshot and diff the AgentTool action contract for breaking-change detection.

The public API most users of this SDK build against is not only the Python
surface (that is covered by the griffe breaking-change check) - it is the set of
AgentTool ACTIONS an agent dispatches and the parameters each action accepts.
That contract lives in each simulation backend's dispatcher: an action string is
resolved to a method (via the backend's ``_ACTION_ALIASES``) and its arguments
are validated against that method's signature. A silent rename of an action or of
an action parameter (for example ``add_camera(name=...)`` becoming
``add_camera(camera_name=...)``) breaks agent code without touching any Python
signature griffe would flag.

This script builds a snapshot of ``{action: [sorted params]}`` for a backend by:
  1. reading the declared action enum from the backend's ``tool_spec.json``,
  2. resolving each action to its method (honoring ``_ACTION_ALIASES``),
  3. recording the method's parameter names (minus ``self``).

Modes:
  snapshot <out.json>        write the current contract to a file
  diff <old.json> <new.json> compare two snapshots, print markdown, exit non-zero
                             if a breaking change (removed action or removed/
                             renamed parameter) is found

Backward-compatible additions (new actions, new parameters) are NOT breaking and
are reported separately as informational.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

# The backend whose dispatch contract we snapshot. MuJoCo is the default,
# CPU-only backend shipped in strands-robots; it is the surface agents call in
# every getting-started path.
TOOL_SPEC = Path("strands_robots/simulation/mujoco/tool_spec.json")


def build_snapshot() -> dict[str, list[str]]:
    """Return {action: [sorted param names]} for the declared action enum."""
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimulation as Sim

    spec = json.loads(TOOL_SPEC.read_text())
    actions = spec["properties"]["action"]["enum"]
    aliases = getattr(Sim, "_ACTION_ALIASES", {}) or {}

    snapshot: dict[str, list[str]] = {}
    for action in actions:
        method = getattr(Sim, aliases.get(action, action), None)
        if method is None:
            # Declared in the enum but not resolvable - record explicitly so a
            # diff surfaces it rather than silently skipping.
            snapshot[action] = ["<unresolved>"]
            continue
        try:
            params = [p for p in inspect.signature(method).parameters if p != "self"]
            snapshot[action] = sorted(params)
        except (ValueError, TypeError):
            snapshot[action] = ["<no-signature>"]
    return snapshot


def diff(old: dict[str, list[str]], new: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """Return (breaking, informational) change lists."""
    breaking: list[str] = []
    info: list[str] = []

    for action in sorted(old):
        if action not in new:
            breaking.append(f"Removed action: `{action}`")
            continue
        old_params, new_params = set(old[action]), set(new[action])
        for p in sorted(old_params - new_params):
            breaking.append(f"`{action}`: removed or renamed parameter `{p}`")
        for p in sorted(new_params - old_params):
            info.append(f"`{action}`: added parameter `{p}`")

    for action in sorted(set(new) - set(old)):
        info.append(f"Added action: `{action}`")

    return breaking, info


def render_markdown(breaking: list[str], info: list[str]) -> str:
    if not breaking and not info:
        return (
            "## No AgentTool API Changes Detected\n\n"
            + "No changes to the dispatched action contract in this PR."
        )
    lines: list[str] = []
    if breaking:
        # Built as a single explicit expression (not two adjacent literals in a
        # list) so it does not read as a missing-comma bug.
        summary_line = (
            f"Found **{len(breaking)}** potential breaking change(s) in the "
            + "dispatched action contract (the actions and parameters agents call):"
        )
        lines += [
            "## AgentTool Breaking Change Warning",
            "",
            summary_line,
            "",
        ]
        lines += [f"- {b}" for b in breaking]
    else:
        lines += ["## AgentTool API Changes (non-breaking)", ""]
    if info:
        lines += ["", "<details><summary>Backward-compatible additions</summary>", ""]
        lines += [f"- {i}" for i in info]
        lines += ["", "</details>"]
    footer = (
        "> Automated static check of the AgentTool action contract. Removed or "
        + "renamed actions/parameters break agent code that calls them. If a change "
        + "is intentional, add a `CHANGELOG.md` entry or a deprecation notice."
    )
    lines += ["", "---", footer]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "snapshot":
        Path(argv[1]).write_text(json.dumps(build_snapshot(), indent=2, sort_keys=True))
        print(f"wrote snapshot -> {argv[1]}")
        return 0

    if len(argv) >= 3 and argv[0] == "diff":
        old = json.loads(Path(argv[1]).read_text())
        new = json.loads(Path(argv[2]).read_text())
        breaking, info = diff(old, new)
        md = render_markdown(breaking, info)
        Path("/tmp/agent-api-changes.md").write_text(md)
        print(md)
        return 1 if breaking else 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
