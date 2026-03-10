#!/usr/bin/env python3
"""
Agent Behavior Pattern Analyzer for strands-robots.

Detects and categorizes autonomous agent behavior patterns:
  - Issue → PR → Merge chains
  - Self-healing cycles (CI fail → fix → pass)
  - Longest uninterrupted agent work sessions
  - Scheduled vs event-driven execution ratio
  - Cross-device dispatch patterns

Usage:
    python agent_patterns.py                     # Text output
    python agent_patterns.py --json              # Save JSON
    python agent_patterns.py --repo /path/to     # Custom repo path

Requirements:
    - git (in a git repository)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Agent author identities
AGENT_AUTHORS = frozenset(
    {
        "strands-agent",
        "github-actions[bot]",
        "strands-agent[bot]",
        "strands-bot",
        "Strands Agent",
    }
)


@dataclass
class IssuePRChain:
    """An issue → PR → merge chain."""

    issue_number: int | None
    pr_number: int | None
    branch: str
    commit_count: int
    messages: list[str] = field(default_factory=list)


@dataclass
class SelfHealCycle:
    """A CI failure → fix → pass cycle."""

    fix_sha: str
    fix_message: str
    date: str


@dataclass
class AgentSession:
    """A continuous agent work session (no human commits)."""

    start_sha: str
    end_sha: str
    start_date: str
    end_date: str
    commit_count: int
    messages: list[str] = field(default_factory=list)


@dataclass
class PatternResult:
    """Full pattern analysis result."""

    issue_pr_chains: list[IssuePRChain] = field(default_factory=list)
    self_heal_cycles: list[SelfHealCycle] = field(default_factory=list)
    agent_sessions: list[AgentSession] = field(default_factory=list)
    longest_session: AgentSession | None = None
    scheduled_commits: int = 0
    event_driven_commits: int = 0
    cross_device_dispatches: int = 0
    commit_type_counts: dict[str, int] = field(default_factory=dict)


def run_git(args: list[str], cwd: str | None = None) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def get_commits(repo_path: str | None = None) -> list[dict[str, str]]:
    """Retrieve all commits."""
    log_output = run_git(
        ["log", "--all", "--format=%H|%aN|%aI|%s"],
        cwd=repo_path,
    )
    commits = []
    for line in log_output.split("\n"):
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        commits.append(
            {
                "sha": parts[0],
                "author": parts[1],
                "date": parts[2],
                "message": parts[3],
            }
        )
    return commits


def detect_issue_pr_chains(commits: list[dict[str, str]]) -> list[IssuePRChain]:
    """Detect issue → branch → PR → merge chains."""
    chains: list[IssuePRChain] = []

    # Look for merge commits and branch patterns
    branch_commits: dict[str, list[dict[str, str]]] = defaultdict(list)

    for commit in commits:
        msg = commit["message"]

        # Detect PR merges
        pr_match = re.search(r"Merge pull request #(\d+)", msg)
        if pr_match:
            pr_number = int(pr_match.group(1))
            # Look for issue reference in branch name
            branch_match = re.search(r"from \S+/(\S+)", msg)
            branch = branch_match.group(1) if branch_match else ""
            issue_match = re.search(r"issue-?(\d+)", branch, re.IGNORECASE)
            issue_number = int(issue_match.group(1)) if issue_match else None

            chains.append(
                IssuePRChain(
                    issue_number=issue_number,
                    pr_number=pr_number,
                    branch=branch,
                    commit_count=1,
                    messages=[msg],
                )
            )
            continue

        # Detect branch-based merges (non-PR)
        branch_match = re.search(r"Merge (\S+):", msg)
        if branch_match:
            branch = branch_match.group(1)
            issue_match = re.search(r"issue-?(\d+)", branch, re.IGNORECASE)
            issue_number = int(issue_match.group(1)) if issue_match else None
            chains.append(
                IssuePRChain(
                    issue_number=issue_number,
                    pr_number=None,
                    branch=branch,
                    commit_count=1,
                    messages=[msg],
                )
            )

        # Detect issue references in commit messages
        issue_refs = re.findall(r"#(\d+)", msg)
        for ref in issue_refs:
            branch_commits[f"issue-{ref}"].append(commit)

    return chains


def detect_self_healing(commits: list[dict[str, str]]) -> list[SelfHealCycle]:
    """Detect CI failure → fix cycles."""
    cycles: list[SelfHealCycle] = []

    fix_patterns = [
        r"^fix:",
        r"^fix\(",
        r"fix test",
        r"fix ci",
        r"fix failure",
        r"fix broken",
        r"resolve.*failure",
        r"fix.*error",
    ]

    for commit in commits:
        if commit["author"] not in AGENT_AUTHORS:
            continue

        msg = commit["message"].lower()
        for pattern in fix_patterns:
            if re.search(pattern, msg):
                cycles.append(
                    SelfHealCycle(
                        fix_sha=commit["sha"][:7],
                        fix_message=commit["message"],
                        date=commit["date"][:10],
                    )
                )
                break

    return cycles


def detect_agent_sessions(commits: list[dict[str, str]]) -> list[AgentSession]:
    """Find continuous agent work sessions (no human commits)."""
    sessions: list[AgentSession] = []
    current_session: list[dict[str, str]] = []

    # Reverse for chronological order
    for commit in reversed(commits):
        if commit["author"] in AGENT_AUTHORS:
            current_session.append(commit)
        else:
            if len(current_session) >= 3:  # Minimum 3 commits = a "session"
                sessions.append(
                    AgentSession(
                        start_sha=current_session[0]["sha"][:7],
                        end_sha=current_session[-1]["sha"][:7],
                        start_date=current_session[0]["date"],
                        end_date=current_session[-1]["date"],
                        commit_count=len(current_session),
                        messages=[c["message"][:80] for c in current_session[:5]],
                    )
                )
            current_session = []

    # Don't forget final session
    if len(current_session) >= 3:
        sessions.append(
            AgentSession(
                start_sha=current_session[0]["sha"][:7],
                end_sha=current_session[-1]["sha"][:7],
                start_date=current_session[0]["date"],
                end_date=current_session[-1]["date"],
                commit_count=len(current_session),
                messages=[c["message"][:80] for c in current_session[:5]],
            )
        )

    return sessions


def classify_commit_types(commits: list[dict[str, str]]) -> dict[str, int]:
    """Classify commits by conventional commit type."""
    types: dict[str, int] = defaultdict(int)

    for commit in commits:
        msg = commit["message"]
        # Match conventional commit prefix
        match = re.match(r"^(feat|fix|docs|test|refactor|chore|ci|style|perf|build)[\(:]", msg)
        if match:
            types[match.group(1)] += 1
        elif msg.startswith("Merge"):
            types["merge"] += 1
        else:
            types["other"] += 1

    return dict(sorted(types.items(), key=lambda x: -x[1]))


def detect_cross_device(commits: list[dict[str, str]]) -> int:
    """Count commits that reference cross-device dispatch."""
    patterns = [
        r"thor",
        r"isaac.?sim",
        r"mac.?mini",
        r"gpu",
        r"ec2",
        r"jetson",
        r"newton.*backend",
        r"cosmos.*transfer",
    ]
    count = 0
    for commit in commits:
        msg = commit["message"].lower()
        if commit["author"] in AGENT_AUTHORS:
            for pattern in patterns:
                if re.search(pattern, msg):
                    count += 1
                    break
    return count


def analyze_patterns(repo_path: str | None = None) -> PatternResult:
    """Run full pattern analysis."""
    commits = get_commits(repo_path)
    result = PatternResult()

    result.issue_pr_chains = detect_issue_pr_chains(commits)
    result.self_heal_cycles = detect_self_healing(commits)
    result.agent_sessions = detect_agent_sessions(commits)
    result.commit_type_counts = classify_commit_types(commits)
    result.cross_device_dispatches = detect_cross_device(commits)

    if result.agent_sessions:
        result.longest_session = max(result.agent_sessions, key=lambda s: s.commit_count)

    # Estimate scheduled vs event-driven
    # Scheduled: commits at regular intervals (roughly hourly patterns)
    # Event-driven: commits that reference issues/PRs
    for commit in commits:
        if commit["author"] in AGENT_AUTHORS:
            msg = commit["message"]
            if re.search(r"#\d+", msg) or "issue" in msg.lower() or "pr" in msg.lower():
                result.event_driven_commits += 1
            else:
                result.scheduled_commits += 1

    return result


def print_results(result: PatternResult) -> None:
    """Print analysis results."""
    print("=" * 64)
    print("  AGENT BEHAVIOR PATTERN ANALYSIS")
    print("=" * 64)
    print()

    # Issue → PR chains
    print(f"📋 Issue → PR → Merge Chains: {len(result.issue_pr_chains)}")
    print("-" * 50)
    for chain in result.issue_pr_chains[:15]:
        issue_str = f"#{chain.issue_number}" if chain.issue_number else "N/A"
        pr_str = f"PR #{chain.pr_number}" if chain.pr_number else "branch merge"
        print(f"  Issue {issue_str} → {pr_str} ({chain.branch})")
    if len(result.issue_pr_chains) > 15:
        print(f"  ... and {len(result.issue_pr_chains) - 15} more")
    print()

    # Self-healing
    print(f"🔧 Self-Healing Cycles: {len(result.self_heal_cycles)}")
    print("-" * 50)
    for cycle in result.self_heal_cycles[:10]:
        print(f"  [{cycle.date}] {cycle.fix_sha}: {cycle.fix_message[:70]}")
    if len(result.self_heal_cycles) > 10:
        print(f"  ... and {len(result.self_heal_cycles) - 10} more")
    print()

    # Agent sessions
    print(f"🤖 Agent Work Sessions (≥3 commits): {len(result.agent_sessions)}")
    print("-" * 50)
    for session in sorted(result.agent_sessions, key=lambda s: -s.commit_count)[:10]:
        print(
            f"  {session.start_date[:10]} | {session.commit_count:3d} commits | "
            f"{session.start_sha}..{session.end_sha}"
        )
        for msg in session.messages[:2]:
            print(f"    → {msg}")
    print()

    # Longest session
    if result.longest_session:
        ls = result.longest_session
        print(f"🏆 Longest Agent-Only Session: {ls.commit_count} commits")
        print(f"   From {ls.start_date[:16]} to {ls.end_date[:16]}")
        print(f"   SHA range: {ls.start_sha}..{ls.end_sha}")
        print("   Sample commits:")
        for msg in ls.messages:
            print(f"     → {msg}")
        print()

    # Scheduled vs event-driven
    total_agent = result.scheduled_commits + result.event_driven_commits
    if total_agent > 0:
        print("⏰ Execution Trigger Analysis:")
        print("-" * 50)
        print(f"  Event-driven (references issues/PRs): {result.event_driven_commits:4d}  ({result.event_driven_commits / total_agent * 100:.0f}%)")
        print(f"  Scheduled (autonomous work):          {result.scheduled_commits:4d}  ({result.scheduled_commits / total_agent * 100:.0f}%)")
        print()

    # Cross-device
    print(f"🖥️  Cross-Device Dispatches: {result.cross_device_dispatches} commits")
    print("   (Commits referencing Thor, Isaac Sim, Mac Mini, GPU, EC2)")
    print()

    # Commit types
    print("📊 Commit Type Distribution:")
    print("-" * 50)
    for ctype, count in result.commit_type_counts.items():
        bar = "█" * min(count, 40)
        print(f"  {ctype:12s}  {count:4d}  {bar}")
    print()


def save_json(result: PatternResult, output_path: str) -> None:
    """Save results as JSON."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "issue_pr_chains": len(result.issue_pr_chains),
        "self_heal_cycles": len(result.self_heal_cycles),
        "agent_sessions": len(result.agent_sessions),
        "longest_session_commits": result.longest_session.commit_count if result.longest_session else 0,
        "scheduled_commits": result.scheduled_commits,
        "event_driven_commits": result.event_driven_commits,
        "cross_device_dispatches": result.cross_device_dispatches,
        "commit_type_distribution": result.commit_type_counts,
        "chains": [
            {
                "issue": chain.issue_number,
                "pr": chain.pr_number,
                "branch": chain.branch,
            }
            for chain in result.issue_pr_chains
        ],
        "self_heal_details": [
            {
                "sha": cycle.fix_sha,
                "message": cycle.fix_message,
                "date": cycle.date,
            }
            for cycle in result.self_heal_cycles
        ],
        "sessions": [
            {
                "start": session.start_date,
                "end": session.end_date,
                "commits": session.commit_count,
                "first_message": session.messages[0] if session.messages else "",
            }
            for session in result.agent_sessions
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nJSON saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze agent behavior patterns in git history",
    )
    parser.add_argument("--repo", help="Path to git repository")
    parser.add_argument("--json", action="store_true", help="Save results as JSON")
    args = parser.parse_args()

    result = analyze_patterns(repo_path=args.repo)
    print_results(result)

    if args.json:
        save_json(result, "data/agent_patterns.json")


if __name__ == "__main__":
    main()
