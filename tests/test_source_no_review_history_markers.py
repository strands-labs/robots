"""Shipped source must not carry review-history markers.

A docstring or comment reaching a downstream reader should describe the code,
not the process that produced it.  ``reviewer caught X``, ``pre-#164
behaviour``, ``requested by @someone during review`` and ``(post-PR #101)``
are all archaeology: they name a conversation the reader cannot open and a
sequence they were never part of, and they crowd out the sentence that would
have told them what the value means.

Every rule below is measured at zero occurrences across shipped source, so
none of them carries an exemption list.  The one rule that admits exceptions
is ``PR #NNN``, and it admits them on a *followable* basis rather than a
per-file one:

* a pull request of a project this package declares as a dependency
  (``merged in lerobot PR #3604`` tells a reader which upstream release ships
  a policy), or
* a citation of this repository's own ``AGENTS.md`` **whose section heading is
  verified to exist** -- so deleting the section fails this guard instead of
  leaving a dangling pointer behind.

The scan reads whole file text rather than only parsed docstrings, matching
the convention of the sibling guard that forbids dangling doc references: a
marker in a user-facing message string is as visible as one in a docstring.
"""

from __future__ import annotations

import inspect
import pathlib
import re
import tomllib

import pytest

import strands_robots

#: Rules with no legitimate use in shipped source.  Each maps a compiled
#: pattern to the label reported when it matches.
_FORBIDDEN: dict[str, str] = {
    r"\breviewer\b": "reviewer",
    r"\breview caught\b": "review caught",
    r"\breview-flagged\b": "review-flagged",
    r"\bduring review\b": "during review",
    r"\bas requested by\b": "as requested by",
    r"\bvariant-[BCD]\b": "variant-B/C/D",
    r"\bpre-#\d+": "pre-#NNN",
    r"\bpost-#\d+": "post-#NNN",
    r"\bpre-PR\b": "pre-PR",
    r"\bpost-PR\b": "post-PR",
    r"\bPR#\d+": "PR#NNN",
    r"\bpull request #\d+": "pull request #NNN",
    # Attribution *shape* rather than handle shape: a bare ``@name`` cannot be
    # forbidden because ``@classmethod``, ``@tool`` and ``@rpath`` are all
    # legitimate, so the rule keys on the crediting phrase that precedes it.
    r"\b(?:by|from|per|thanks(?:\s+to)?|credits?\s+to|asked\s+by|requested\s+by)"
    r"\s+@[A-Za-z][A-Za-z0-9-]{2,}": "attribution to @handle",
}

#: ``PR #NNN`` is matched separately: it is allowed on the two followable
#: bases described in the module docstring and forbidden otherwise.
_PR_REFERENCE = re.compile(r"\bPR #(\d+)")


def _source_root() -> pathlib.Path:
    """Return the shipped package directory, derived from the package itself."""
    return pathlib.Path(inspect.getfile(strands_robots)).parent


def _repo_root() -> pathlib.Path:
    """Return the repository root that owns ``pyproject.toml``."""
    root = _source_root().parent
    assert (root / "pyproject.toml").is_file(), f"no pyproject.toml at {root}"
    return root


def _shipped_modules() -> list[pathlib.Path]:
    return sorted(_source_root().rglob("*.py"))


def _declared_dependency_names() -> frozenset[str]:
    """Return every distribution name this package declares as a dependency.

    Deriving the set from ``pyproject.toml`` keeps the upstream-provenance
    allowance self-maintaining: a project only earns the right to be cited by
    pull request number once it is a declared dependency.
    """
    data = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    specs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)
    names = set()
    for spec in specs:
        name = re.split(r"[\[<>=!~;\s]", spec, maxsplit=1)[0].strip()
        if name:
            names.add(name.lower().replace("_", "-"))
    return frozenset(names)


def _agents_md_pr_headings() -> frozenset[str]:
    """Return the pull request numbers that name a section heading in AGENTS.md."""
    agents = _repo_root() / "AGENTS.md"
    if not agents.is_file():
        return frozenset()
    text = agents.read_text(encoding="utf-8")
    return frozenset(re.findall(r"(?m)^#+ .*\bPR #(\d+)\b", text))


def review_history_markers(
    text: str,
    *,
    dependencies: frozenset[str],
    agents_pr_headings: frozenset[str],
) -> list[tuple[int, str, str]]:
    """Return ``(line number, marker label, line)`` for every marker in ``text``."""
    found: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, label in _FORBIDDEN.items():
            if re.search(pattern, line):
                found.append((lineno, label, line.strip()))
        for match in _PR_REFERENCE.finditer(line):
            number = match.group(1)
            lowered = line.lower()
            names_dependency = any(re.search(rf"\b{re.escape(dep)}\b", lowered) for dep in dependencies)
            cites_verified_section = "AGENTS.md" in line and number in agents_pr_headings
            if not (names_dependency or cites_verified_section):
                found.append((lineno, f"PR #{number}", line.strip()))
    return found


@pytest.fixture(scope="module")
def scan_inputs() -> tuple[frozenset[str], frozenset[str]]:
    return _declared_dependency_names(), _agents_md_pr_headings()


class TestShippedSourceCarriesNoReviewHistory:
    """The markers this guard forbids are absent from every shipped module."""

    def test_no_module_carries_a_review_history_marker(
        self, scan_inputs: tuple[frozenset[str], frozenset[str]]
    ) -> None:
        dependencies, agents_pr_headings = scan_inputs
        offenders: list[str] = []
        for module in _shipped_modules():
            text = module.read_text(encoding="utf-8")
            for lineno, label, line in review_history_markers(
                text, dependencies=dependencies, agents_pr_headings=agents_pr_headings
            ):
                rel = module.relative_to(_source_root())
                offenders.append(f"{rel}:{lineno} [{label}] {line}")
        assert offenders == [], (
            "shipped source carries review-history markers; describe the code "
            "rather than the review that produced it:\n  " + "\n  ".join(offenders)
        )

    @pytest.mark.parametrize("pattern", sorted(_FORBIDDEN), ids=sorted(_FORBIDDEN.values()))
    def test_each_forbidden_marker_is_absent(self, pattern: str) -> None:
        hits = [
            f"{module.relative_to(_source_root())}:{lineno}"
            for module in _shipped_modules()
            for lineno, line in enumerate(module.read_text(encoding="utf-8").splitlines(), start=1)
            if re.search(pattern, line)
        ]
        assert hits == [], f"{pattern!r} still present at: {hits}"


class TestTheScanIsNotVacuous:
    """A scan that reads nothing, or flags nothing by construction, proves nothing."""

    def test_the_scan_reads_the_whole_shipped_package(self) -> None:
        modules = _shipped_modules()
        assert len(modules) > 150, f"only {len(modules)} modules found under {_source_root()}"
        assert any(m.name == "robot.py" for m in modules)

    @pytest.mark.parametrize(
        ("planted", "label"),
        [
            ("# the reviewer asked for this", "reviewer"),
            ("# review caught the missing guard", "review caught"),
            ("# and review-flagged 5x", "review-flagged"),
            ("# requested during review", "during review"),
            ("# as requested by someone", "as requested by"),
            ("# the variant-B repro", "variant-B/C/D"),
            ("# pre-#164 behaviour", "pre-#NNN"),
            ("# post-#85 behaviour", "post-#NNN"),
            ("# pre-PR shape", "pre-PR"),
            ("# post-PR shape", "post-PR"),
            ("# shipped in PR#92", "PR#NNN"),
            ("# shipped in pull request #92", "pull request #NNN"),
            ("# requested by @someone1986", "attribution to @handle"),
            ("# shipped in PR #4242", "PR #4242"),
        ],
    )
    def test_the_scanner_detects_a_planted_marker(
        self,
        planted: str,
        label: str,
        scan_inputs: tuple[frozenset[str], frozenset[str]],
    ) -> None:
        dependencies, agents_pr_headings = scan_inputs
        found = review_history_markers(planted, dependencies=dependencies, agents_pr_headings=agents_pr_headings)
        assert [entry[1] for entry in found] == [label], f"{planted!r} -> {found}"

    @pytest.mark.parametrize(
        "legitimate",
        [
            "    the preview's frame period is 1/fps",
            "    @classmethod",
            "    resolves torchcodec's @rpath lookups on macOS",
            "    a @tool wrapper around the loader",
        ],
    )
    def test_the_scanner_leaves_legitimate_text_alone(
        self, legitimate: str, scan_inputs: tuple[frozenset[str], frozenset[str]]
    ) -> None:
        dependencies, agents_pr_headings = scan_inputs
        assert (
            review_history_markers(legitimate, dependencies=dependencies, agents_pr_headings=agents_pr_headings) == []
        )


class TestThePullRequestAllowanceIsFollowable:
    """``PR #NNN`` survives only where a reader can follow the reference."""

    def test_an_upstream_dependency_may_be_cited_by_pull_request(
        self, scan_inputs: tuple[frozenset[str], frozenset[str]]
    ) -> None:
        dependencies, agents_pr_headings = scan_inputs
        assert "lerobot" in dependencies, "lerobot is expected to be a declared dependency"
        assert (
            review_history_markers(
                "    (e.g. ``molmoact2``, merged in lerobot PR #3604) becomes",
                dependencies=dependencies,
                agents_pr_headings=agents_pr_headings,
            )
            == []
        )

    def test_an_unqualified_pull_request_number_is_still_refused(
        self, scan_inputs: tuple[frozenset[str], frozenset[str]]
    ) -> None:
        dependencies, agents_pr_headings = scan_inputs
        found = review_history_markers(
            "    # kept as a top-level field in PR #85",
            dependencies=dependencies,
            agents_pr_headings=agents_pr_headings,
        )
        assert [entry[1] for entry in found] == ["PR #85"]

    def test_every_agents_md_citation_names_a_section_that_exists(
        self, scan_inputs: tuple[frozenset[str], frozenset[str]]
    ) -> None:
        _, agents_pr_headings = scan_inputs
        cited: set[str] = set()
        for module in _shipped_modules():
            for line in module.read_text(encoding="utf-8").splitlines():
                if "AGENTS.md" in line:
                    cited.update(_PR_REFERENCE.findall(line))
        assert cited, "expected at least one AGENTS.md pull-request citation to verify"
        dangling = sorted(cited - agents_pr_headings)
        assert dangling == [], f"shipped source cites AGENTS.md sections that no longer exist: {dangling}"

    def test_a_citation_of_a_missing_section_is_refused(
        self, scan_inputs: tuple[frozenset[str], frozenset[str]]
    ) -> None:
        dependencies, _ = scan_inputs
        found = review_history_markers(
            "    # see AGENTS.md, PR #99999",
            dependencies=dependencies,
            agents_pr_headings=frozenset({"92"}),
        )
        assert [entry[1] for entry in found] == ["PR #99999"]
