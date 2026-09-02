#!/usr/bin/env python3
"""Refuse a ``uv.lock`` whose transcription of the manifest no longer matches it.

Why this exists
---------------
#2039 added ``.github/workflows/lockfile-parity.yml``, which runs ``uv lock
--check``. That is the authoritative comparison and it catches everything below.
It is also **advisory**: the ``default`` ruleset lists exactly one required check
(``call-test-lint / Test and Lint``), so a drifted lock reaches a reviewer as a
red advisory context beside a green required one.

The offline half #2039 shipped -- ``tests/test_lockfile_parity_gate.py`` -- runs
*inside* the required check, and it asserts two properties of the lock rather
than comparing it: every declared distribution is present somewhere, and no
locked version falls below its declared floor. Against ``main``'s stale lock at
``ad8696b`` that reports **2 of the 5 rows that had actually drifted**::

    lerobot pinned 0.6.0 vs declared >=0.6.1        reported  (floor violation)
    roslibpy absent from the lock                   reported  (distribution absent)
    mink/qpsolvers unrecorded under [sim-mujoco]    MISSED
    mink/qpsolvers unrecorded under [sim-newton]    MISSED
    huggingface-hub recorded >=1.0 vs declared >=1.5 MISSED

The three misses share one cause: a presence-and-floor check samples two
properties of the recorded set, and the drift was in the set's *shape*. ``mink``
and ``qpsolvers`` are both locked -- reachable via ``[cosmos3-sim]`` -- so a
presence check passes while ``[sim-mujoco]`` is recorded as
``['imageio', 'imageio-ffmpeg', 'mujoco', 'robot-descriptions']``: a locked sim
install with no IK stack behind ``move_to``. ``[rosbridge]`` was recorded
**empty**. And ``huggingface-hub`` is pinned at ``1.20.1``, which satisfies the
``>=1.5`` floor, so only the recorded *specifier* was stale -- nothing about the
pin is wrong, and no property of the pin can see it.

What this compares instead
--------------------------
The whole recorded set, against the whole declared set. ``uv.lock`` already
carries uv's own transcription of the manifest in the root package's
``[package.metadata] requires-dist``, so the declared set can be reconstructed
from ``pyproject.toml`` and compared as a set -- offline, no resolver, no
network, no ``uv`` binary, in the required check.

The reconstruction reproduces the relocked file **exactly**: 111 declared rows
against 111 recorded, zero differences in either direction. Against the stale
lock the two sets differ by 32 rows, reported as **24 findings** -- 10
declared-but-not-recorded, 5 recorded-but-not-declared, 9 specifier-drift --
which cover all five rows above and four the issue had not enumerated
(``peft`` and ``transformers`` recorded under extras that no longer declare them,
``scipy``, and ``lerobot[molmoact2]`` unrecorded entirely).

It is 24 rather than 32 because a changed version range appears in *both*
directions of the symmetric difference -- once as declared-but-not-recorded and
once as recorded-but-not-declared -- and reported raw that reads as two
independent problems with two different remedies. Rows are therefore grouped by
``(extra, name, extras)`` before classification, so a changed range is one
finding naming both sides.

Two encodings have to be mirrored rather than approximated, and getting either
wrong produces a guard that false-fails on a correct lock:

- **Self-references are expanded.** 20 of the 88 requirement lines in the
  manifest are ``strands-robots[<other-extra>]``; uv resolves those away and
  records the transitive closure, which is why the recorded set is 111 and not
  68. Comparing the literal declared lines reports **43** spurious "recorded but
  not declared" rows -- so the expansion is not a refinement, it is the
  difference between a working check and one that fails every branch.
- **One key can carry several specifiers, so neither side may be flattened into
  a dictionary.** This is not defensive: uv's own lock at ``ad8696b`` recorded
  ``scipy`` **twice** under ``[all]``, at ``>=1.10.0`` and at ``>=1.14.0,<2.0.0``.
  Keying a dictionary by ``(extra, name, extras)`` drops one of those silently,
  and the dropped row then reads as drift against the other side -- a finding on
  a file that is correct. ``(extra, name, extras)`` therefore maps to a *set* of
  specifiers on both sides. The live pair exercises this as of #3012:
  ``[sim-newton]`` narrows ``mujoco`` to the series its pinned ``newton``
  requires, while still entering ``[sim-mujoco]`` through a self-reference, so
  uv records that one key at **two** specifiers -- ``>=3.5.0,<4.0.0`` from the
  expansion and ``>=3.11.0,<3.12.0`` from the narrowing. Both sides carry both,
  the grouping compares them as a set, and the pair agrees. The ``scipy`` shape
  above stays pinned on a planted pair, which is what keeps the observed-input
  case graded independently of whichever narrowing the manifest happens to
  carry.

One fact worth recording because it makes the comparison exact rather than
approximate: every marker in the lock's ``requires-dist`` is ``extra == '...'``
and nothing else (107 of 111 rows; the other 4 are unconditional), and no
requirement in ``pyproject.toml`` carries an environment marker at all. A
residual-marker comparison is therefore inert today. It is still written as a
**finding** rather than a silent drop, so that if either side ever grows a real
marker this reports it instead of quietly comparing fewer rows.

What it deliberately does not do
--------------------------------
- **It does not re-check that locked versions satisfy declared floors.**
  ``tests/test_lockfile_parity_gate.py`` does that, and duplicating it here would
  create two rulebooks for one rule that can drift apart -- including the
  non-obvious "every locked version, not any" phrasing that a forked resolution
  forces on it. This script compares the two transcriptions of
  the *manifest*; that module compares the lock against the *resolution*.
- **It does not re-resolve.** Whether the recorded set is *achievable* is
  ``uv lock --check``'s question, and it needs the index to answer it. A stale
  transcription is detectable without one, which is the only reason this can run
  in the required check.
- **It does not read ``[tool.uv]``.** Sources, overrides and conflicts change
  what uv resolves, not what the manifest declares, so they cannot make the two
  transcriptions disagree.

Usage
-----
``--repo-root``  directory holding ``pyproject.toml`` and ``uv.lock`` (default:
                 the repository this script lives in).
``--pyproject``  / ``--lock``  override either path individually, which is what
                 makes the planted-pair tests possible.

Exit status is ``1`` when there is at least one finding, else ``0``. The remedy
is always the same and is self-clearing: run ``uv lock`` and commit the result.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

#: This project. It appears in its own manifest as ``strands-robots[<extra>]``
#: recursive extras, which uv expands away rather than recording.
PROJECT_NAME = "strands-robots"

#: The only marker shape either side uses. Matched exactly rather than parsed
#: with ``packaging.markers`` so that anything else becomes a finding instead of
#: evaluating to something plausible under an assumed environment.
_EXTRA_MARKER = re.compile(r"extra == '([^']+)'")

MISSING_FROM_LOCK = "declared-but-not-recorded"
ABSENT_FROM_MANIFEST = "recorded-but-not-declared"
SPECIFIER_DRIFT = "specifier-drift"
UNREADABLE_MARKER = "unreadable-marker"


def canonical_name(name: str) -> str:
    """Normalise a distribution name per PEP 503, so ``zope.interface`` matches ``zope-interface``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _normalise_specifier(specifier: str) -> str:
    """Render a specifier set in a stable, comparable form.

    ``SpecifierSet`` sorts and deduplicates, so ``>=1.0,<2`` and ``<2,>=1.0``
    compare equal. Without this the check would report ordering as drift, which
    is the same class of false finding as the two encodings in the module
    docstring.
    """
    return str(SpecifierSet(specifier))


@dataclass(frozen=True, order=True)
class Row:
    """One requirement, as declared under one extra (or unconditionally).

    ``extra`` of ``None`` is a ``project.dependencies`` entry, which uv records
    with no marker.
    """

    extra: str | None
    name: str
    extras: tuple[str, ...]
    specifier: str

    @property
    def key(self) -> tuple[str | None, str, tuple[str, ...]]:
        """What identifies this requirement independently of its version range."""
        return (self.extra, self.name, self.extras)

    def render(self) -> str:
        extras = f"[{','.join(self.extras)}]" if self.extras else ""
        where = f"[{self.extra}]" if self.extra else "project.dependencies"
        return f"{self.name}{extras}{self.specifier or ' (any version)'} under {where}"


@dataclass(frozen=True)
class Finding:
    """One disagreement between the manifest and the lock's transcription of it."""

    kind: str
    key: tuple[str | None, str, tuple[str, ...]]
    declared: tuple[str, ...] = ()
    recorded: tuple[str, ...] = ()
    detail: str = ""

    def render(self) -> str:
        extra, name, extras = self.key
        extras_text = f"[{','.join(extras)}]" if extras else ""
        where = f"[{extra}]" if extra else "project.dependencies"
        subject = f"{name}{extras_text} under {where}"
        if self.kind == MISSING_FROM_LOCK:
            return f"{subject}  -- declared {self._join(self.declared)}, not recorded in uv.lock at all"
        if self.kind == ABSENT_FROM_MANIFEST:
            return f"{subject}  -- recorded {self._join(self.recorded)} in uv.lock, not declared in pyproject.toml"
        if self.kind == SPECIFIER_DRIFT:
            return f"{subject}  -- declared {self._join(self.declared)}, recorded {self._join(self.recorded)}"
        return f"{subject}  -- {self.detail}"

    @staticmethod
    def _join(specifiers: Sequence[str]) -> str:
        rendered = [s or "(any version)" for s in specifiers]
        return " / ".join(rendered) if rendered else "(nothing)"


def declared_rows(pyproject: dict[str, object]) -> set[Row]:
    """Reconstruct every requirement the manifest declares, with self-references expanded.

    ``strands-robots[sim-mujoco]`` declared under ``[all]`` contributes
    ``[sim-mujoco]``'s requirements to ``[all]``, transitively, which is what uv
    records. Real extra names are used in this example rather than placeholders
    because ``tests/test_dependency_audit.py`` refuses any ``strands-robots[...]``
    mention naming an extra that does not exist -- pip exits 0 on an unknown
    extra and installs nothing, so a placeholder here would read as a broken
    install hint. The recursion carries the set
    of extras already entered so a cycle terminates rather than recursing until
    the interpreter stops it; the manifest has no cycle today and this makes that
    a property of the reader rather than of the input.
    """
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml has no [project] table.")
    optional_raw = project.get("optional-dependencies") or {}
    optional: dict[str, list[str]] = {
        str(extra): [str(spec) for spec in specs] for extra, specs in optional_raw.items()
    }

    def expand(extra: str, entered: frozenset[str]) -> list[Requirement]:
        collected: list[Requirement] = []
        for spec in optional.get(extra, []):
            requirement = Requirement(spec)
            if canonical_name(requirement.name) == PROJECT_NAME:
                for inner in sorted(requirement.extras):
                    if inner not in entered:
                        collected.extend(expand(inner, entered | {inner}))
            else:
                collected.append(requirement)
        return collected

    rows: set[Row] = set()
    for spec in project.get("dependencies") or []:
        requirement = Requirement(str(spec))
        rows.add(
            Row(
                None,
                canonical_name(requirement.name),
                tuple(sorted(requirement.extras)),
                _normalise_specifier(str(requirement.specifier)),
            )
        )
    for extra in optional:
        for requirement in expand(extra, frozenset({extra})):
            rows.add(
                Row(
                    extra,
                    canonical_name(requirement.name),
                    tuple(sorted(requirement.extras)),
                    _normalise_specifier(str(requirement.specifier)),
                )
            )
    return rows


def recorded_rows(lock: dict[str, object]) -> tuple[set[Row], list[Finding]]:
    """Read the root package's ``requires-dist``, and report any marker that is not ``extra == '...'``.

    An unreadable marker is returned as a finding rather than dropped, so a
    manifest that grows an environment marker makes this check say so instead of
    silently comparing fewer rows than it claims to.
    """
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock has no [[package]] array.")
    roots = [
        package
        for package in packages
        if isinstance(package, dict) and canonical_name(str(package.get("name", ""))) == PROJECT_NAME
    ]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one {PROJECT_NAME} package in uv.lock, found {len(roots)}.")
    metadata = roots[0].get("metadata")
    if not isinstance(metadata, dict) or "requires-dist" not in metadata:
        raise ValueError("uv.lock's root package carries no [package.metadata] requires-dist.")

    rows: set[Row] = set()
    findings: list[Finding] = []
    for entry in metadata["requires-dist"]:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"unreadable requires-dist entry: {entry!r}")
        name = canonical_name(str(entry["name"]))
        marker = entry.get("marker")
        extra: str | None = None
        if marker is not None:
            matched = _EXTRA_MARKER.fullmatch(str(marker).strip())
            if matched is None:
                findings.append(
                    Finding(
                        UNREADABLE_MARKER,
                        (None, name, ()),
                        detail=(
                            f"uv.lock records it under the marker {str(marker)!r}, which is not a plain "
                            "extra marker. This checker compares extras only, so the row is reported "
                            "rather than compared."
                        ),
                    )
                )
                continue
            extra = matched.group(1)
        rows.add(
            Row(
                extra,
                name,
                tuple(sorted(str(value) for value in entry.get("extras", ()))),
                _normalise_specifier(str(entry.get("specifier", ""))),
            )
        )
    return rows, findings


def compare(declared: set[Row], recorded: set[Row]) -> list[Finding]:
    """Classify every disagreement between the two sets.

    The two sets' symmetric difference is the complete answer, but reported raw
    it names a changed version range twice -- once as declared-but-not-recorded
    and once as recorded-but-not-declared -- which reads as two independent
    problems. So rows are grouped by ``key`` first, and a key present on both
    sides with differing specifiers becomes one ``specifier-drift`` finding
    naming both sides. A key present on one side only keeps its own class,
    because those want different remedies: a missing row means the lock never
    saw the requirement, an extra row means the manifest dropped it.
    """
    declared_by_key: dict[tuple[str | None, str, tuple[str, ...]], set[str]] = defaultdict(set)
    recorded_by_key: dict[tuple[str | None, str, tuple[str, ...]], set[str]] = defaultdict(set)
    for row in declared:
        declared_by_key[row.key].add(row.specifier)
    for row in recorded:
        recorded_by_key[row.key].add(row.specifier)

    findings: list[Finding] = []
    for key in sorted(set(declared_by_key) | set(recorded_by_key), key=lambda k: (k[1], k[0] or "", k[2])):
        declared_specifiers = declared_by_key.get(key, set())
        recorded_specifiers = recorded_by_key.get(key, set())
        if declared_specifiers == recorded_specifiers:
            continue
        if not recorded_specifiers:
            kind = MISSING_FROM_LOCK
        elif not declared_specifiers:
            kind = ABSENT_FROM_MANIFEST
        else:
            kind = SPECIFIER_DRIFT
        findings.append(
            Finding(
                kind,
                key,
                declared=tuple(sorted(declared_specifiers)),
                recorded=tuple(sorted(recorded_specifiers)),
            )
        )
    return findings


def check(pyproject_path: Path, lock_path: Path) -> list[Finding]:
    """Return every finding for one manifest/lock pair."""
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    recorded, marker_findings = recorded_rows(lock)
    return marker_findings + compare(declared_rows(pyproject), recorded)


def render(findings: Sequence[Finding], pyproject_path: Path, lock_path: Path, declared: int, recorded: int) -> str:
    """Render the job-summary report for one run."""
    lines = [
        "## Lockfile requires-dist parity",
        "",
        "| field | value |",
        "|---|---|",
        f"| manifest | `{pyproject_path}` |",
        f"| lock | `{lock_path}` |",
        f"| declared requirements (self-references expanded) | {declared} |",
        f"| recorded in `[package.metadata] requires-dist` | {recorded} |",
        f"| findings | **{len(findings)}** |",
        "",
    ]
    if not findings:
        lines += [
            "`uv.lock`'s transcription of `pyproject.toml` matches it exactly, in both directions.",
        ]
        return "\n".join(lines)

    by_kind: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        by_kind[finding.kind].append(finding)
    for kind in (MISSING_FROM_LOCK, ABSENT_FROM_MANIFEST, SPECIFIER_DRIFT, UNREADABLE_MARKER):
        group = by_kind.get(kind)
        if not group:
            continue
        lines += [f"### {kind} ({len(group)})", ""]
        lines += [f"- {finding.render()}" for finding in group]
        lines += [""]
    lines += [
        "### What clears this",
        "",
        "Run `uv lock` and commit the result. A dependency change and its relock belong in the",
        "same commit: the lock is a dependency-graph manifest, so GitHub's security scanning",
        "reads it, and a stale one is a stale security surface rather than only an inconvenience",
        "to whoever runs a locked install.",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", default=str(default_root))
    parser.add_argument("--pyproject", default="")
    parser.add_argument("--lock", default="")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    pyproject_path = Path(args.pyproject) if args.pyproject else repo_root / "pyproject.toml"
    lock_path = Path(args.lock) if args.lock else repo_root / "uv.lock"

    for path in (pyproject_path, lock_path):
        if not path.is_file():
            print(f"check_lockfile_parity: no such file: {path}", file=sys.stderr)
            return 2

    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        recorded, marker_findings = recorded_rows(lock)
        declared = declared_rows(pyproject)
        findings = marker_findings + compare(declared, recorded)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"check_lockfile_parity: {exc}", file=sys.stderr)
        return 2

    report = render(findings, pyproject_path, lock_path, len(declared), len(recorded))
    print(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")

    if findings:
        print(
            f"::error title=uv.lock no longer transcribes pyproject.toml::"
            f"{len(findings)} finding(s) comparing the declared requirement set against "
            f"[package.metadata] requires-dist. Run `uv lock` and commit the result."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
