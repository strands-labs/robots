#!/usr/bin/env python3
"""Tests for Sample 10: Autonomous Repository Case Study.

All 6 scripts use subprocess.run for git commands — we mock subprocess
to test the analysis logic without needing a real git repo.

Uses importlib.util to import sample modules. Registers module in sys.modules
before exec to support @dataclass (which calls sys.modules[cls.__module__]).
"""

import importlib.util

# Skip if samples/ directory not present (requires PR #13)
import os as _os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

if not _os.path.isdir(_os.path.join(_os.path.dirname(__file__), "..", "samples")):
    __import__("pytest").skip("Requires PR #13 (samples)", allow_module_level=True)


# ──────────────────────────────────────────────────────────────────
# Helper: load a sample module by path
# ──────────────────────────────────────────────────────────────────

SAMPLE_DIR = Path(__file__).parent.parent / "samples" / "10_autonomous_repo_casestudy"


def _load_module(filename: str):
    """Import a sample module by filename, registering in sys.modules for @dataclass support."""
    filepath = SAMPLE_DIR / filename
    if not filepath.exists():
        pytest.skip(f"{filepath} not found")
    mod_name = f"_sample10_{filename.replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so @dataclass can find the module namespace
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _completed(stdout: str = "", returncode: int = 0):
    """Return a CompletedProcess with the given stdout."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


# ======================================================================
# analyze_repo.py
# ======================================================================


class TestAnalyzeRepo:
    """Tests for analyze_repo.py functions."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_module("analyze_repo.py")

    @patch("subprocess.run")
    def test_run_git_command(self, mock_run):
        mock_run.return_value = _completed("hello world")
        assert self.mod.run_git_command(["git", "log"]) == "hello world"

    @patch("subprocess.run")
    def test_analyze_commits_per_author(self, mock_run):
        mock_run.return_value = _completed("   150\tgithub-actions[bot]\n    50\tcagataycali")
        authors = self.mod.analyze_commits_per_author()
        assert authors == {"github-actions[bot]": 150, "cagataycali": 50}

    @patch("subprocess.run")
    def test_analyze_commits_per_author_empty(self, mock_run):
        mock_run.return_value = _completed("")
        assert self.mod.analyze_commits_per_author() == {}

    @patch("subprocess.run")
    def test_analyze_agent_vs_human(self, mock_run):
        mock_run.return_value = _completed("   80\tgithub-actions[bot]\n    20\tcagataycali")
        result = self.mod.analyze_agent_vs_human()
        assert result["agent"] == 80
        assert result["human"] == 20
        assert result["total"] == 100
        assert result["agent_percentage"] == 80.0

    @patch("subprocess.run")
    def test_analyze_agent_vs_human_zero_commits(self, mock_run):
        mock_run.return_value = _completed("")
        result = self.mod.analyze_agent_vs_human()
        assert result["total"] == 0
        assert result["agent_percentage"] == 0

    @patch("subprocess.run")
    def test_analyze_commits_per_week(self, mock_run):
        mock_run.return_value = _completed("2026-03-01\n2026-03-01\n2026-03-07")
        weeks = self.mod.analyze_commits_per_week()
        assert isinstance(weeks, dict)
        assert sum(weeks.values()) == 3

    @patch("subprocess.run")
    def test_analyze_commit_messages(self, mock_run):
        mock_run.return_value = _completed("feat: add X\nfix: bug Y\ntest: add Z\nsome other commit\nrefactor: cleanup")
        patterns = self.mod.analyze_commit_messages()
        assert patterns["feat"] == 1
        assert patterns["fix"] == 1
        assert patterns["test"] == 1
        assert patterns["refactor"] == 1
        assert patterns["other"] == 1

    @patch("subprocess.run")
    def test_analyze_commit_messages_parenthetical(self, mock_run):
        """Commits like feat(scope): should match."""
        mock_run.return_value = _completed("feat(sim): add mujoco backend")
        patterns = self.mod.analyze_commit_messages()
        assert patterns["feat"] == 1

    @patch("subprocess.run")
    def test_analyze_file_types_changed(self, mock_run):
        mock_run.return_value = _completed("src/main.py\nsrc/utils.py\nREADME.md\nDockerfile\n")
        types = self.mod.analyze_file_types_changed()
        assert ".py" in types
        assert types[".py"] == 2
        assert ".md" in types
        assert "(no extension)" in types  # Dockerfile

    @patch("subprocess.run")
    def test_analyze_longest_agent_streak(self, mock_run):
        log = "\n".join(
            [
                "github-actions[bot]|feat: A|2026-03-01",
                "github-actions[bot]|fix: B|2026-03-01",
                "github-actions[bot]|test: C|2026-03-01",
                "cagataycali|docs: D|2026-03-02",
            ]
        )
        mock_run.return_value = _completed(log)
        result = self.mod.analyze_longest_agent_streak()
        assert result["streak_length"] == 3
        assert len(result["commits"]) == 3

    @patch("subprocess.run")
    def test_analyze_longest_agent_streak_all_human(self, mock_run):
        mock_run.return_value = _completed("cagataycali|fix: A|2026-03-01")
        result = self.mod.analyze_longest_agent_streak()
        assert result["streak_length"] == 0
        assert result["commits"] == []

    @patch("subprocess.run")
    def test_analyze_longest_agent_streak_strands_agent(self, mock_run):
        """Strands Agent identity is also recognized."""
        mock_run.return_value = _completed("Strands Agent|feat: auto|2026-03-01")
        result = self.mod.analyze_longest_agent_streak()
        assert result["streak_length"] == 1

    @patch("subprocess.run")
    def test_get_first_commit_date(self, mock_run):
        mock_run.return_value = _completed("2025-01-01\n2025-01-02")
        assert self.mod.get_first_commit_date() == "2025-01-01"

    @patch("subprocess.run")
    def test_get_current_stats(self, mock_run):
        mock_run.side_effect = [
            _completed("./src/main.py\n./src/utils.py\n"),
            _completed("  100 ./src/main.py\n  200 ./src/utils.py\n  300 total"),
        ]
        stats = self.mod.get_current_stats()
        assert stats["python_files"] == 2
        assert stats["total_loc"] == 300

    @patch("subprocess.run")
    def test_get_current_stats_no_files(self, mock_run):
        mock_run.return_value = _completed("")
        stats = self.mod.get_current_stats()
        assert stats["python_files"] == 0
        assert stats["total_loc"] == 0

    @patch("subprocess.run")
    def test_get_current_stats_single_file(self, mock_run):
        """Single file: wc -l doesn't produce 'total' line."""
        mock_run.side_effect = [
            _completed("./src/main.py\n"),
            _completed("  42 ./src/main.py"),
        ]
        stats = self.mod.get_current_stats()
        assert stats["python_files"] == 1
        # No 'total' line → total_loc = 0 (documented behavior)
        assert stats["total_loc"] == 0


# ======================================================================
# analyze_commits.py
# ======================================================================


class TestAnalyzeCommits:
    """Tests for analyze_commits.py functions."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_module("analyze_commits.py")

    def test_agent_authors_frozenset(self):
        assert isinstance(self.mod.AGENT_AUTHORS, frozenset)
        assert "github-actions[bot]" in self.mod.AGENT_AUTHORS

    def test_human_authors_frozenset(self):
        assert isinstance(self.mod.HUMAN_AUTHORS, frozenset)
        assert "cagataycali" in self.mod.HUMAN_AUTHORS

    def test_agent_and_human_disjoint(self):
        """Agent and human author sets should not overlap."""
        overlap = self.mod.AGENT_AUTHORS & self.mod.HUMAN_AUTHORS
        assert len(overlap) == 0, f"Overlapping authors: {overlap}"

    @patch("subprocess.run")
    def test_run_git_returns_stdout(self, mock_run):
        mock_run.return_value = _completed("test output")
        if hasattr(self.mod, "run_git"):
            assert self.mod.run_git(["git", "log"]) == "test output"
        elif hasattr(self.mod, "run_git_command"):
            assert self.mod.run_git_command(["git", "log"]) == "test output"
        else:
            pytest.skip("No run_git or run_git_command found")

    @patch("subprocess.run")
    def test_parse_commits_returns_list(self, mock_run):
        if not hasattr(self.mod, "parse_commits"):
            pytest.skip("parse_commits not present")
        mock_run.return_value = _completed("abc1234|2026-03-01|github-actions[bot]|feat: add X|5|3")
        commits = self.mod.parse_commits()
        assert isinstance(commits, list)

    def test_commit_info_dataclass(self):
        """CommitInfo dataclass should exist and be instantiable."""
        if not hasattr(self.mod, "CommitInfo"):
            pytest.skip("CommitInfo not present")
        info = self.mod.CommitInfo(
            hash="abc1234",
            date="2026-03-01",
            author="test",
            message="fix: test",
            additions=5,
            deletions=3,
        )
        assert info.hash == "abc1234"


# ======================================================================
# agent_patterns.py
# ======================================================================


class TestAgentPatterns:
    """Tests for agent_patterns.py functions."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_module("agent_patterns.py")

    def test_agent_authors_defined(self):
        assert isinstance(self.mod.AGENT_AUTHORS, frozenset)
        assert len(self.mod.AGENT_AUTHORS) >= 3

    def test_issue_pr_chain_dataclass(self):
        chain = self.mod.IssuePRChain(issue_number=1, pr_number=2, branch="fix/issue-1", commit_count=3)
        assert chain.issue_number == 1
        assert chain.pr_number == 2
        assert chain.branch == "fix/issue-1"

    def test_issue_pr_chain_optional_fields(self):
        """IssuePRChain supports None values."""
        chain = self.mod.IssuePRChain(issue_number=None, pr_number=None, branch="main", commit_count=0)
        assert chain.issue_number is None

    @patch("subprocess.run")
    def test_detect_issue_pr_chains(self, mock_run):
        if not hasattr(self.mod, "detect_issue_pr_chains"):
            pytest.skip("detect_issue_pr_chains not available")
        mock_run.return_value = _completed("abc1234|2026-03-01|github-actions[bot]|fix: resolve issue #42|fix/issue-42")
        chains = self.mod.detect_issue_pr_chains(
            [
                {
                    "hash": "abc",
                    "date": "2026-03-01",
                    "author": "github-actions[bot]",
                    "message": "fix: resolve issue #42",
                    "branch": "fix/issue-42",
                }
            ]
        )
        assert isinstance(chains, list)


# ======================================================================
# evolution_timeline.py
# ======================================================================


class TestEvolutionTimeline:
    """Tests for evolution_timeline.py functions."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_module("evolution_timeline.py")

    @patch("subprocess.run")
    def test_extract_major_milestones(self, mock_run):
        log = "abc1234|2026-03-01|feat: add groot model\ndef5678|2026-03-02|fix: typo"
        mock_run.return_value = _completed(log)
        milestones = self.mod.extract_major_milestones()
        assert len(milestones) >= 1
        assert milestones[0]["date"] == "2026-03-01"
        assert "groot" in milestones[0]["message"].lower()

    @patch("subprocess.run")
    def test_extract_milestones_no_matches(self, mock_run):
        mock_run.return_value = _completed("abc|2026-03-01|bump version to 0.2.0")
        assert self.mod.extract_major_milestones() == []

    @patch("subprocess.run")
    def test_analyze_feature_additions(self, mock_run):
        log = "abc|2026-03-01|feat: add groot inference\ndef|2026-03-02|feat: add mujoco sim"
        mock_run.return_value = _completed(log)
        features = self.mod.analyze_feature_additions()
        assert isinstance(features, dict)
        found_groot = any("groot" in str(v) for v in features.values())
        assert found_groot

    @patch("subprocess.run")
    def test_analyze_feature_additions_empty(self, mock_run):
        mock_run.return_value = _completed("")
        features = self.mod.analyze_feature_additions()
        assert isinstance(features, dict)

    @patch("subprocess.run")
    def test_analyze_test_growth(self, mock_run):
        log = "abc|2026-03-01|test: add tests\ndef|2026-03-01|test: more\nghi|2026-03-02|test: fix"
        mock_run.return_value = _completed(log)
        growth = self.mod.analyze_test_growth()
        assert isinstance(growth, list)
        assert len(growth) >= 1
        assert growth[-1]["cumulative"] >= 3

    @patch("subprocess.run")
    def test_analyze_test_growth_empty(self, mock_run):
        mock_run.return_value = _completed("")
        assert self.mod.analyze_test_growth() == []

    @patch("subprocess.run")
    def test_milestones_curriculum_keyword(self, mock_run):
        mock_run.return_value = _completed("abc|2026-03-01|feat: add k12 curriculum sample 01")
        milestones = self.mod.extract_major_milestones()
        assert len(milestones) >= 1

    @patch("subprocess.run")
    def test_milestones_initial_commit(self, mock_run):
        mock_run.return_value = _completed("abc|2025-01-01|Initial commit")
        milestones = self.mod.extract_major_milestones()
        assert len(milestones) == 1

    @patch("subprocess.run")
    def test_milestones_sorted_by_date(self, mock_run):
        log = "\n".join(
            [
                "ghi|2026-03-01|feat: multi-robot fleet demo",
                "abc|2025-01-01|initial commit",
                "def|2026-02-01|feat: add isaac sim backend",
            ]
        )
        mock_run.return_value = _completed(log)
        milestones = self.mod.extract_major_milestones()
        dates = [m["date"] for m in milestones]
        assert dates == sorted(dates)

    @patch("subprocess.run")
    def test_feature_additions_simulation_category(self, mock_run):
        mock_run.return_value = _completed("abc|2026-01-01|feat: add mujoco physics simulation")
        features = self.mod.analyze_feature_additions()
        assert "Simulation Environments" in features
        assert len(features["Simulation Environments"]) >= 1


# ======================================================================
# architecture_diagram.py
# ======================================================================


class TestArchitectureDiagram:
    """Tests for architecture_diagram.py (gracefully handles missing pyyaml)."""

    def test_module_importable(self):
        """Module should import even without pyyaml (yaml used at call-time only)."""
        mod = _load_module("architecture_diagram.py")
        public_funcs = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n))]
        assert len(public_funcs) >= 1

    def test_generate_feedback_loop_diagram(self):
        """Diagram generation does not require pyyaml."""
        mod = _load_module("architecture_diagram.py")
        diagram = mod.generate_feedback_loop_diagram()
        assert "AUTONOMOUS DEVELOPMENT CYCLE" in diagram
        assert "Strands Agent" in diagram

    def test_generate_tool_diagram(self):
        """Tool diagram generation does not require pyyaml."""
        mod = _load_module("architecture_diagram.py")
        diagram = mod.generate_tool_diagram()
        assert "STRANDS AGENT TOOLS" in diagram
        assert "shell" in diagram


# ======================================================================
# workflow_analyzer.py
# ======================================================================


class TestWorkflowAnalyzer:
    """Tests for workflow_analyzer.py (gracefully handles missing pyyaml)."""

    def test_module_importable(self):
        """Module should import even without pyyaml (yaml used at call-time only)."""
        mod = _load_module("workflow_analyzer.py")
        public_funcs = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n))]
        assert len(public_funcs) >= 1

    def test_dataclasses_available(self):
        """WorkflowInfo and AnalysisResult should be importable without pyyaml."""
        mod = _load_module("workflow_analyzer.py")
        assert hasattr(mod, "WorkflowInfo")
        assert hasattr(mod, "AnalysisResult")
        # Verify they're instantiable
        wf = mod.WorkflowInfo(filename="test.yml", name="Test")
        assert wf.filename == "test.yml"
        result = mod.AnalysisResult()
        assert result.total_workflows == 0

    def test_all_github_events_frozenset(self):
        """ALL_GITHUB_EVENTS constant should be available without pyyaml."""
        mod = _load_module("workflow_analyzer.py")
        assert hasattr(mod, "ALL_GITHUB_EVENTS")
        assert isinstance(mod.ALL_GITHUB_EVENTS, frozenset)
        assert "push" in mod.ALL_GITHUB_EVENTS
