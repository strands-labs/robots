#!/usr/bin/env python3
"""
Workflow Analyzer for strands-robots GitHub Actions infrastructure.

Parses all `.github/workflows/*.yml` files and produces:
  - Trigger → Runner → Action mapping
  - Dispatch dependency graph
  - Automation coverage (% of events with agent response)
  - Runner fleet inventory

Usage:
    python workflow_analyzer.py                       # Text output
    python workflow_analyzer.py --json                # Save JSON
    python workflow_analyzer.py --workflows /path     # Custom path

Requirements:
    - pyyaml (pip install pyyaml)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def _require_yaml():
    """Raise a clear error if pyyaml is not installed."""
    if yaml is None:
        raise ImportError(
            "pyyaml is required for workflow analysis: pip install pyyaml"
        )


# All possible GitHub event triggers
ALL_GITHUB_EVENTS = frozenset(
    {
        "push",
        "pull_request",
        "pull_request_review",
        "pull_request_review_comment",
        "issues",
        "issue_comment",
        "discussion",
        "discussion_comment",
        "schedule",
        "workflow_dispatch",
        "workflow_call",
        "release",
        "create",
        "delete",
        "fork",
        "watch",
        "label",
    }
)


@dataclass
class WorkflowInfo:
    """Parsed information about a single workflow."""

    filename: str
    name: str
    triggers: list[str] = field(default_factory=list)
    trigger_details: dict[str, str] = field(default_factory=dict)
    runners: list[str] = field(default_factory=list)
    jobs: list[str] = field(default_factory=list)
    dispatches_to: list[str] = field(default_factory=list)
    is_self_hosted: bool = False
    has_schedule: bool = False
    has_agent: bool = False


@dataclass
class AnalysisResult:
    """Full workflow analysis result."""

    workflows: list[WorkflowInfo] = field(default_factory=list)
    total_workflows: int = 0
    triggers_covered: set[str] = field(default_factory=set)
    automation_coverage: float = 0.0
    runner_fleet: dict[str, list[str]] = field(default_factory=dict)
    dispatch_graph: dict[str, list[str]] = field(default_factory=dict)
    scheduled_workflows: list[str] = field(default_factory=list)


def parse_workflow(filepath: Path) -> WorkflowInfo:
    """Parse a single workflow YAML file."""
    _require_yaml()

    with open(filepath) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"  ⚠ Failed to parse {filepath.name}: {e}")
            return WorkflowInfo(filename=filepath.name, name=filepath.name)

    if not isinstance(data, dict):
        return WorkflowInfo(filename=filepath.name, name=filepath.name)

    info = WorkflowInfo(
        filename=filepath.name,
        name=data.get("name", filepath.name),
    )

    # Parse triggers
    on_trigger = data.get("on", data.get(True, {}))
    if isinstance(on_trigger, str):
        info.triggers = [on_trigger]
    elif isinstance(on_trigger, list):
        info.triggers = on_trigger
    elif isinstance(on_trigger, dict):
        info.triggers = list(on_trigger.keys())
        # Extract schedule details
        if "schedule" in on_trigger:
            schedules = on_trigger["schedule"]
            if isinstance(schedules, list):
                crons = [s.get("cron", "") for s in schedules if isinstance(s, dict)]
                info.trigger_details["schedule"] = ", ".join(crons)
                info.has_schedule = True

    # Parse jobs
    jobs = data.get("jobs", {})
    if isinstance(jobs, dict):
        for job_name, job_data in jobs.items():
            info.jobs.append(job_name)
            if not isinstance(job_data, dict):
                continue

            # Runner detection
            runs_on = job_data.get("runs-on", "")
            if isinstance(runs_on, str):
                info.runners.append(runs_on)
                if "self-hosted" in runs_on.lower():
                    info.is_self_hosted = True
            elif isinstance(runs_on, list):
                info.runners.extend(runs_on)
                if any("self-hosted" in str(r).lower() for r in runs_on):
                    info.is_self_hosted = True

            # Check job-level env for agent markers
            job_env = job_data.get("env", {})
            if isinstance(job_env, dict):
                if "STRANDS_PROMPT" in job_env or "SYSTEM_PROMPT" in job_env:
                    info.has_agent = True

            # Check full YAML text for strands references
            job_str = str(job_data)
            if "strands-action" in job_str or "strands_coder" in job_str or "STRANDS_PROMPT" in job_str:
                info.has_agent = True

            # Detect workflow dispatches in steps
            steps = job_data.get("steps", [])
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    # Check for workflow_dispatch calls
                    run_cmd = step.get("run", "")
                    if "workflow_dispatch" in str(run_cmd):
                        # Try to extract target workflow
                        info.dispatches_to.append("(via API)")

                    # Check if this runs the agent
                    uses = step.get("uses", "")
                    if "strands" in str(uses).lower() or "agent" in str(uses).lower():
                        info.has_agent = True
                    if "strands" in str(run_cmd).lower() or "agent" in str(run_cmd).lower():
                        info.has_agent = True

                    # Check env for agent markers
                    env = step.get("env", {})
                    if isinstance(env, dict):
                        if "STRANDS_PROMPT" in env or "SYSTEM_PROMPT" in env:
                            info.has_agent = True

    return info


def analyze_workflows(workflows_dir: str | None = None) -> AnalysisResult:
    """Analyze all workflow files."""
    if workflows_dir:
        wf_path = Path(workflows_dir)
    else:
        wf_path = Path(".github/workflows")
        if not wf_path.exists():
            # Try from repo root
            wf_path = Path(__file__).parent.parent.parent / ".github" / "workflows"

    if not wf_path.exists():
        print(f"Workflows directory not found: {wf_path}")
        sys.exit(1)

    result = AnalysisResult()

    yml_files = sorted(wf_path.glob("*.yml")) + sorted(wf_path.glob("*.yaml"))
    for filepath in yml_files:
        info = parse_workflow(filepath)
        result.workflows.append(info)

    result.total_workflows = len(result.workflows)

    # Compute coverage
    for wf in result.workflows:
        for trigger in wf.triggers:
            result.triggers_covered.add(trigger)

        # Build runner fleet
        for runner in wf.runners:
            if runner not in result.runner_fleet:
                result.runner_fleet[runner] = []
            result.runner_fleet[runner].append(wf.filename)

        # Build dispatch graph
        if wf.dispatches_to:
            result.dispatch_graph[wf.filename] = wf.dispatches_to

        if wf.has_schedule:
            result.scheduled_workflows.append(wf.filename)

    # Automation coverage: what % of common GitHub events have a workflow response
    common_events = {"push", "pull_request", "issues", "issue_comment", "schedule", "workflow_dispatch"}
    covered = result.triggers_covered & common_events
    result.automation_coverage = len(covered) / len(common_events) * 100

    return result


def print_results(result: AnalysisResult) -> None:
    """Print workflow analysis."""
    print("=" * 64)
    print("  WORKFLOW INFRASTRUCTURE ANALYSIS")
    print("=" * 64)
    print()

    # Overview
    print(f"Total workflows:     {result.total_workflows}")
    print(f"Event types covered: {len(result.triggers_covered)} ({', '.join(sorted(result.triggers_covered))})")
    print(f"Automation coverage: {result.automation_coverage:.0f}% of common events")
    print(f"Scheduled workflows: {len(result.scheduled_workflows)}")
    print()

    # Workflow table
    print("📋 Workflow Inventory:")
    print("-" * 80)
    print(f"  {'Filename':<25s} {'Name':<30s} {'Triggers':<20s} {'Agent?':>6s}")
    print("-" * 80)
    for wf in result.workflows:
        triggers_str = ", ".join(wf.triggers[:3])
        if len(wf.triggers) > 3:
            triggers_str += f" +{len(wf.triggers) - 3}"
        agent_str = "🤖" if wf.has_agent else "  "
        self_hosted = "🏠" if wf.is_self_hosted else "  "
        print(f"  {wf.filename:<25s} {wf.name[:29]:<30s} {triggers_str:<20s} {agent_str} {self_hosted}")
    print()

    # Runner fleet
    print("🖥️  Runner Fleet:")
    print("-" * 50)
    for runner, workflows in result.runner_fleet.items():
        runner_type = "☁️ " if "ubuntu" in runner.lower() else "🏠" if "self-hosted" in runner.lower() else "🔧"
        print(f"  {runner_type} {runner}")
        for wf in workflows:
            print(f"      ← {wf}")
    print()

    # Scheduled workflows
    if result.scheduled_workflows:
        print("⏰ Scheduled Workflows:")
        print("-" * 50)
        for wf_name in result.scheduled_workflows:
            wf = next((w for w in result.workflows if w.filename == wf_name), None)
            if wf:
                cron = wf.trigger_details.get("schedule", "N/A")
                print(f"  {wf.filename}: {cron}")
        print()

    # Dispatch graph (text-based)
    print("🔗 Dispatch Architecture:")
    print("-" * 50)
    print()
    print("  ┌─────────────────────────┐")
    print("  │   GitHub Events          │")
    print("  │  (push, issues, PR, ...) │")
    print("  └───────────┬─────────────┘")
    print("              │")
    print("              ▼")

    agent_workflows = [wf for wf in result.workflows if wf.has_agent]
    ci_workflows = [wf for wf in result.workflows if not wf.has_agent]

    if agent_workflows:
        print("  ┌─────────────────────────────────────────┐")
        print("  │  AGENT WORKFLOWS                        │")
        for wf in agent_workflows:
            triggers = ", ".join(wf.triggers[:3])
            hosted = "🏠" if wf.is_self_hosted else "☁️ "
            print(f"  │  {hosted} {wf.filename:<22s} [{triggers}]  │")
        print("  └─────────────────────────────────────────┘")
    print()

    if ci_workflows:
        print("  ┌─────────────────────────────────────────┐")
        print("  │  CI/CD WORKFLOWS                        │")
        for wf in ci_workflows:
            triggers = ", ".join(wf.triggers[:3])
            print(f"  │  ☁️  {wf.filename:<22s} [{triggers}]  │")
        print("  └─────────────────────────────────────────┘")
    print()

    # Summary stats
    print("📊 Summary:")
    print("-" * 50)
    print(f"  Agent workflows:     {len(agent_workflows)}")
    print(f"  CI/CD workflows:     {len(ci_workflows)}")
    print(f"  Self-hosted runners: {sum(1 for wf in result.workflows if wf.is_self_hosted)}")
    print(f"  Cloud runners:       {sum(1 for wf in result.workflows if not wf.is_self_hosted)}")
    print(f"  Unique runners:      {len(result.runner_fleet)}")
    total_jobs = sum(len(wf.jobs) for wf in result.workflows)
    print(f"  Total jobs defined:  {total_jobs}")


def save_json(result: AnalysisResult, output_path: str) -> None:
    """Save results as JSON."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "total_workflows": result.total_workflows,
        "triggers_covered": sorted(result.triggers_covered),
        "automation_coverage_pct": round(result.automation_coverage, 1),
        "runner_fleet": result.runner_fleet,
        "scheduled_workflows": result.scheduled_workflows,
        "dispatch_graph": result.dispatch_graph,
        "workflows": [
            {
                "filename": wf.filename,
                "name": wf.name,
                "triggers": wf.triggers,
                "runners": wf.runners,
                "jobs": wf.jobs,
                "is_self_hosted": wf.is_self_hosted,
                "has_schedule": wf.has_schedule,
                "has_agent": wf.has_agent,
            }
            for wf in result.workflows
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nJSON saved to {output_path}")


def main() -> None:
    _require_yaml()

    parser = argparse.ArgumentParser(
        description="Analyze GitHub Actions workflow infrastructure",
    )
    parser.add_argument("--workflows", help="Path to .github/workflows/ directory")
    parser.add_argument("--json", action="store_true", help="Save results as JSON")
    args = parser.parse_args()

    result = analyze_workflows(workflows_dir=args.workflows)
    print_results(result)

    if args.json:
        save_json(result, "data/workflow_analysis.json")


if __name__ == "__main__":
    main()
