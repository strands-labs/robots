#!/usr/bin/env python3
"""
Commit Attribution Analyzer for strands-robots.

Analyzes git history to determine the ratio of AI agent vs human contributions.
Produces statistics and optional charts showing:
  - Commit counts by author category (agent vs human)
  - Daily commit frequency
  - Longest agent-only commit chains
  - Lines added/removed by category

Usage:
    python analyze_commits.py                    # Basic text output
    python analyze_commits.py --chart            # With matplotlib charts
    python analyze_commits.py --repo /path/to   # Custom repo path
    python analyze_commits.py --json             # JSON output for data/

Requirements:
    - git (must be in a git repository or provide --repo)
    - matplotlib (optional, for --chart)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Authors known to be AI agents
AGENT_AUTHORS = frozenset(
    {
        "strands-agent",
        "github-actions[bot]",
        "strands-agent[bot]",
        "strands-bot",
        "Strands Agent",
    }
)

# Authors known to be human
HUMAN_AUTHORS = frozenset(
    {
        "cagataycali",
        "./c²",
        "Arron Bailiss",
        "Alexander Tsyplikhin",
    }
)


@dataclass
class Commit:
    """A single git commit."""

    sha: str
    author: str
    date: str
    message: str
    insertions: int = 0
    deletions: int = 0

    @property
    def is_agent(self) -> bool:
        return self.author in AGENT_AUTHORS

    @property
    def is_human(self) -> bool:
        return self.author in HUMAN_AUTHORS

    @property
    def category(self) -> str:
        if self.is_agent:
            return "agent"
        if self.is_human:
            return "human"
        return "unknown"


@dataclass
class AnalysisResult:
    """Results of commit analysis."""

    total_commits: int = 0
    agent_commits: int = 0
    human_commits: int = 0
    unknown_commits: int = 0
    commits_by_author: dict[str, int] = field(default_factory=dict)
    commits_by_date: dict[str, dict[str, int]] = field(default_factory=dict)
    longest_agent_chain: int = 0
    longest_agent_chain_commits: list[str] = field(default_factory=list)
    agent_lines_added: int = 0
    agent_lines_removed: int = 0
    human_lines_added: int = 0
    human_lines_removed: int = 0


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
        print(f"Error running git {' '.join(args)}: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def get_commits(repo_path: str | None = None) -> list[Commit]:
    """Retrieve all commits from the git log."""
    # Format: hash|author|date|message
    log_output = run_git(
        [
            "log",
            "--all",
            "--format=%H|%aN|%aI|%s",
        ],
        cwd=repo_path,
    )

    commits = []
    for line in log_output.split("\n"):
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        sha, author, date, message = parts
        commits.append(
            Commit(sha=sha, author=author, date=date, message=message)
        )

    return commits


def get_commit_stats(sha: str, repo_path: str | None = None) -> tuple[int, int]:
    """Get insertions and deletions for a commit."""
    try:
        stat_output = run_git(
            ["show", "--stat", "--format=", sha],
            cwd=repo_path,
        )
        insertions = 0
        deletions = 0
        for line in stat_output.split("\n"):
            if "insertion" in line or "deletion" in line:
                parts = line.strip().split(",")
                for part in parts:
                    part = part.strip()
                    if "insertion" in part:
                        insertions = int(part.split()[0])
                    elif "deletion" in part:
                        deletions = int(part.split()[0])
        return insertions, deletions
    except Exception:
        return 0, 0


def find_longest_agent_chain(commits: list[Commit]) -> tuple[int, list[Commit]]:
    """Find the longest consecutive streak of agent-only commits."""
    best_length = 0
    best_chain: list[Commit] = []
    current_chain: list[Commit] = []

    # Commits are newest-first from git log, reverse for chronological
    for commit in reversed(commits):
        if commit.is_agent:
            current_chain.append(commit)
        else:
            if len(current_chain) > best_length:
                best_length = len(current_chain)
                best_chain = current_chain.copy()
            current_chain = []

    # Check final chain
    if len(current_chain) > best_length:
        best_length = len(current_chain)
        best_chain = current_chain.copy()

    return best_length, best_chain


def analyze(repo_path: str | None = None, include_stats: bool = False) -> AnalysisResult:
    """Run the full commit analysis."""
    result = AnalysisResult()
    commits = get_commits(repo_path)
    result.total_commits = len(commits)

    author_counter: Counter[str] = Counter()
    date_category: dict[str, dict[str, int]] = defaultdict(lambda: {"agent": 0, "human": 0, "unknown": 0})

    for commit in commits:
        author_counter[commit.author] += 1
        category = commit.category
        date_str = commit.date[:10]  # YYYY-MM-DD
        date_category[date_str][category] += 1

        if category == "agent":
            result.agent_commits += 1
        elif category == "human":
            result.human_commits += 1
        else:
            result.unknown_commits += 1

    result.commits_by_author = dict(author_counter.most_common())
    result.commits_by_date = dict(sorted(date_category.items()))

    # Find longest agent chain
    chain_length, chain_commits = find_longest_agent_chain(commits)
    result.longest_agent_chain = chain_length
    result.longest_agent_chain_commits = [
        f"{c.sha[:7]} {c.message[:60]}" for c in chain_commits[:10]
    ]

    # Optional: gather line stats (slower)
    if include_stats:
        print("Gathering line statistics (this may take a moment)...")
        for i, commit in enumerate(commits):
            if i % 50 == 0:
                print(f"  Processing commit {i + 1}/{len(commits)}...", end="\r")
            ins, dels = get_commit_stats(commit.sha, repo_path)
            if commit.is_agent:
                result.agent_lines_added += ins
                result.agent_lines_removed += dels
            elif commit.is_human:
                result.human_lines_added += ins
                result.human_lines_removed += dels
        print()

    return result


def print_results(result: AnalysisResult) -> None:
    """Print analysis results to stdout."""
    print("=" * 64)
    print("  COMMIT ATTRIBUTION ANALYSIS — strands-robots")
    print("=" * 64)
    print()

    # Summary
    agent_pct = result.agent_commits / max(result.total_commits, 1) * 100
    human_pct = result.human_commits / max(result.total_commits, 1) * 100
    print(f"Total commits: {result.total_commits}")
    print(f"  AI Agent:    {result.agent_commits:4d}  ({agent_pct:.1f}%)")
    print(f"  Human:       {result.human_commits:4d}  ({human_pct:.1f}%)")
    if result.unknown_commits:
        unknown_pct = result.unknown_commits / max(result.total_commits, 1) * 100
        print(f"  Unknown:     {result.unknown_commits:4d}  ({unknown_pct:.1f}%)")
    print()

    # Bar chart (text)
    bar_width = 50
    agent_bars = int(agent_pct / 100 * bar_width)
    human_bars = int(human_pct / 100 * bar_width)
    print("  Agent: " + "█" * agent_bars + f" {agent_pct:.0f}%")
    print("  Human: " + "█" * human_bars + f" {human_pct:.0f}%")
    print()

    # By author
    print("By Author:")
    print("-" * 50)
    for author, count in result.commits_by_author.items():
        category = "🤖" if author in AGENT_AUTHORS else "👤" if author in HUMAN_AUTHORS else "❓"
        pct = count / max(result.total_commits, 1) * 100
        print(f"  {category} {author:30s}  {count:4d}  ({pct:.1f}%)")
    print()

    # Longest agent chain
    print(f"Longest agent-only commit chain: {result.longest_agent_chain} commits")
    if result.longest_agent_chain_commits:
        print("  First commits in chain:")
        for commit_str in result.longest_agent_chain_commits[:5]:
            print(f"    {commit_str}")
    print()

    # Daily activity
    print("Daily Commit Frequency:")
    print("-" * 50)
    for date, counts in result.commits_by_date.items():
        total = sum(counts.values())
        agent = counts.get("agent", 0)
        human = counts.get("human", 0)
        bar = "▓" * agent + "░" * human
        print(f"  {date}  {bar:30s}  total={total:3d}  agent={agent}  human={human}")
    print()

    # Line stats
    if result.agent_lines_added or result.human_lines_added:
        print("Lines of Code:")
        print("-" * 50)
        total_added = result.agent_lines_added + result.human_lines_added

        if total_added > 0:
            print(f"  Agent: +{result.agent_lines_added:,}  -{result.agent_lines_removed:,}  ({result.agent_lines_added / total_added * 100:.0f}% of additions)")
            print(f"  Human: +{result.human_lines_added:,}  -{result.human_lines_removed:,}  ({result.human_lines_added / total_added * 100:.0f}% of additions)")


def save_json(result: AnalysisResult, output_path: str) -> None:
    """Save analysis results as JSON."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "total_commits": result.total_commits,
        "agent_commits": result.agent_commits,
        "human_commits": result.human_commits,
        "unknown_commits": result.unknown_commits,
        "agent_percentage": round(result.agent_commits / max(result.total_commits, 1) * 100, 1),
        "human_percentage": round(result.human_commits / max(result.total_commits, 1) * 100, 1),
        "commits_by_author": result.commits_by_author,
        "commits_by_date": result.commits_by_date,
        "longest_agent_chain": result.longest_agent_chain,
        "longest_agent_chain_commits": result.longest_agent_chain_commits,
        "agent_authors": sorted(AGENT_AUTHORS),
        "human_authors": sorted(HUMAN_AUTHORS),
    }
    if result.agent_lines_added or result.human_lines_added:
        data["lines"] = {
            "agent_added": result.agent_lines_added,
            "agent_removed": result.agent_lines_removed,
            "human_added": result.human_lines_added,
            "human_removed": result.human_lines_removed,
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nJSON saved to {output_path}")


def generate_chart(result: AnalysisResult, output_path: str = "data/commit_chart.png") -> None:
    """Generate matplotlib charts (optional)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Pie chart — agent vs human
    ax1 = axes[0]
    sizes = [result.agent_commits, result.human_commits]
    labels = [f"AI Agent\n({result.agent_commits})", f"Human\n({result.human_commits})"]
    colors = ["#4FC3F7", "#FFA726"]
    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    ax1.set_title("Commit Attribution", fontsize=14, fontweight="bold")

    # 2. Bar chart — daily frequency
    ax2 = axes[1]
    dates = list(result.commits_by_date.keys())
    agent_daily = [result.commits_by_date[d].get("agent", 0) for d in dates]
    human_daily = [result.commits_by_date[d].get("human", 0) for d in dates]

    x_labels = [d[5:] for d in dates]  # MM-DD
    x_pos = range(len(dates))
    ax2.bar(x_pos, agent_daily, color="#4FC3F7", label="Agent", alpha=0.8)
    ax2.bar(x_pos, human_daily, bottom=agent_daily, color="#FFA726", label="Human", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Commits")
    ax2.set_title("Daily Commit Frequency", fontsize=14, fontweight="bold")
    ax2.legend()

    # 3. By author
    ax3 = axes[2]
    authors = list(result.commits_by_author.keys())
    counts = list(result.commits_by_author.values())
    bar_colors = ["#4FC3F7" if a in AGENT_AUTHORS else "#FFA726" if a in HUMAN_AUTHORS else "#90A4AE" for a in authors]
    y_pos = range(len(authors))
    ax3.barh(y_pos, counts, color=bar_colors, alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(authors, fontsize=9)
    ax3.set_xlabel("Commits")
    ax3.set_title("Commits by Author", fontsize=14, fontweight="bold")
    ax3.invert_yaxis()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze git commit attribution (agent vs human)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_commits.py                 # Basic analysis
  python analyze_commits.py --chart         # With charts
  python analyze_commits.py --json          # Save JSON to data/
  python analyze_commits.py --stats         # Include line statistics (slow)
  python analyze_commits.py --repo /path    # Analyze different repo
        """,
    )
    parser.add_argument("--repo", help="Path to git repository (default: current directory)")
    parser.add_argument("--chart", action="store_true", help="Generate matplotlib chart")
    parser.add_argument("--json", action="store_true", help="Save results as JSON")
    parser.add_argument("--stats", action="store_true", help="Include line addition/deletion stats (slower)")
    args = parser.parse_args()

    result = analyze(repo_path=args.repo, include_stats=args.stats)
    print_results(result)

    if args.json:
        save_json(result, "data/commit_analysis.json")

    if args.chart:
        generate_chart(result)


if __name__ == "__main__":
    main()
