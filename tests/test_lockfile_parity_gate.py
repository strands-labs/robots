"""Contract pins for the lockfile-parity gate and for the lock it guards.

``uv.lock`` is checked in, so it reads as authoritative, and nothing compared it
against ``pyproject.toml``: no workflow ran ``uv lock --check``, ``uv sync
--locked`` or ``uv sync --frozen``.  The lock was last written 2026-07-25
(``4a42ad4``, #1606) and the manifest was edited three times after that
(``c038f9f``, ``5221459``, ``04dcd31``) with no relock, so ``uv lock --check``
failed on ``main`` at ``9d3ad2f`` while CI stayed green throughout.  See #2038.

The consequence is not confined to whoever runs a locked install, which is what
makes the staleness worth a gate rather than a note.  ``uv.lock`` is a
**dependency-graph manifest** -- ``dependencyGraphManifests`` lists it with
``parseable: true`` -- so it is one of the inputs GitHub's security scanning
reads, which is also how #631 cleared 51 Dependabot alerts "via uv lock
upgrade".  Read off the repository's own SBOM while the lock was stale::

    lerobot   0.6.0     the version pyproject.toml forbids (floor >=0.6.1)
    roslibpy  ABSENT    no roslibpy advisory could be reported at all

Both rows are pinned below, and the first one is why this module exists at all
rather than leaving the whole contract to the workflow.  #1930 existed *only* to
buy that lerobot floor, and
``tests/test_lerobot_floor_guarantees_bucket_streaming.py`` pins it -- on the
manifest side.  The artifact that pins the build said ``0.6.0``, one test
asserted the floor and passed, and no test read the other side.  A guarantee
pinned on one side of two files is not pinned.

Two halves, deliberately split by what they can afford to do:

* ``uv lock --check``, in ``.github/workflows/lockfile-parity.yml``, is the
  complete test.  It re-resolves against the index, so it needs the network and
  belongs in a workflow.
* The tests here are **offline and structural**.  They cover the two drift
  classes actually measured above -- a locked version below a declared floor, and
  a declared dependency missing from the lock entirely -- so the required check
  reports them without a network round trip, and they name the drift in the
  manifest's own terms rather than as "the lockfile needs to be updated".

Neither subsumes the other: ``--check`` catches drift these cannot see (a
transitive pin, a marker change), and these fail in the required check, which
``--check`` cannot do while the gate is advisory.

**The floor pin refuses a distribution only when every locked version is below
the floor, not when any is.**  That is measured, not cautious.  ``uv`` forks the
resolution (``[tool.uv] conflicts``), so one distribution can legitimately appear
at several versions: ``gymnasium``, ``torchcodec`` and ``transformers`` each do
today.  The measurement that forced the weaker phrasing was ``robosuite``, locked
at both ``1.4.0`` and ``1.4.1`` against a ``>=1.4.1`` floor from ``[vera-sim]`` -
the ``1.4.0`` fork arrived through the LIBERO adapter's extra and left the lock
with it, so no distribution carries a below-floor version today.  Forking has not
gone away, so an "any version below the floor" rule would still fail a correct
lock the next time one does.  The weaker rule
still names the row that mattered: ``lerobot`` was locked at exactly one version,
``0.6.0``, below its floor from three separate extras.
"""

from __future__ import annotations

import re
import tomllib
from collections import defaultdict
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_LOCK = _REPO_ROOT / "uv.lock"
_GATE = _REPO_ROOT / ".github" / "workflows" / "lockfile-parity.yml"

#: This project, which appears in its own manifest as ``strands-robots[<extra>]``
#: recursive extras. Those resolve to the other extras rather than to a
#: distribution the lock would carry an entry for.
_PROJECT_NAME = "strands-robots"

#: ``[[package]]`` entries carry the distribution name and version on the two
#: lines after the table header, in that order, for every entry uv writes.
_LOCK_PACKAGE_RE = re.compile(
    r'^\[\[package\]\]\nname = "(?P<name>[^"]+)"\nversion = "(?P<version>[^"]+)"',
    re.MULTILINE,
)


def _canonical(name: str) -> str:
    """Normalise a distribution name per PEP 503, so ``zope.interface`` matches ``zope-interface``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_versions() -> dict[str, list[str]]:
    """Map every locked distribution to the versions the lock carries for it.

    A list rather than a scalar because uv forks the resolution, so one
    distribution can appear at several versions - which is the whole reason the
    floor pin below is phrased over "every" version.
    """
    found: dict[str, list[str]] = defaultdict(list)
    for match in _LOCK_PACKAGE_RE.finditer(_LOCK.read_text(encoding="utf-8")):
        found[_canonical(match.group("name"))].append(match.group("version"))
    return dict(found)


def _declared_requirements() -> list[tuple[str, Requirement]]:
    """Return every direct requirement in the manifest, tagged with where it is declared.

    ``project.dependencies`` and ``project.optional-dependencies`` only: those are
    what an installer resolves for ``pip install 'strands-robots[rosbridge]'``, which
    is the population the two measured rows came from. Recursive
    ``strands-robots[...]`` extras and URL requirements are dropped - neither
    names a distribution the lock carries an entry for under that name.
    """
    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]

    rows: list[tuple[str, Requirement]] = []
    for spec in project.get("dependencies", []):
        rows.append(("project.dependencies", Requirement(spec)))
    for extra, specs in project.get("optional-dependencies", {}).items():
        for spec in specs:
            rows.append((extra, Requirement(spec)))

    return [(where, req) for where, req in rows if _canonical(req.name) != _PROJECT_NAME and not req.url]


def _declared_floor(req: Requirement) -> Version | None:
    """Return the lowest version the requirement admits, or None if it sets no floor.

    ``max`` over the floor-setting operators: a requirement carrying more than
    one (``>=0.6.1`` alongside a ``~=`` in a different marker branch) is bounded
    below by the highest of them.
    """
    floors = [Version(spec.version) for spec in req.specifier if spec.operator in (">=", "==", "~=")]
    return max(floors) if floors else None


# --------------------------------------------------------------------------- #
# Non-vacuity. Every assertion below is a statement about what the two parsers
# found, so a parser that quietly matches nothing would satisfy all of them.
# --------------------------------------------------------------------------- #


def test_the_manifest_parser_finds_the_declared_requirements() -> None:
    """The manifest scan must find the requirements the pins below range over."""
    rows = _declared_requirements()
    distributions = {_canonical(req.name) for _, req in rows}
    # 68 requirements over 52 distributions when this was written. Floors, not
    # equalities: declaring a dependency is routine and must not fail this pin.
    assert len(rows) >= 60, f"only {len(rows)} direct requirements parsed: {rows}"
    assert len(distributions) >= 45, f"only {len(distributions)} distributions: {sorted(distributions)}"


def test_the_lock_parser_finds_the_locked_packages() -> None:
    """The lock scan must find the packages the pins below range over."""
    locked = _locked_versions()
    # 264 distributions when this was written; 269 packages counting fork
    # duplicates. A floor, for the same reason as above.
    assert len(locked) >= 200, f"only {len(locked)} distributions parsed from {_LOCK}"
    # A distribution known to be required directly, as the cheapest proof the
    # regex is reading real `[[package]]` tables rather than matching by luck.
    #
    # Deliberately not this project's own entry: uv writes the editable root
    # with a `source = { editable = "." }` line and *no* `version`, so it is
    # absent from this map by construction. That is also why the pins below drop
    # self-referential `strands-robots[<extra>]` requirements rather than looking
    # them up - there is no version there to compare a floor against.
    assert locked.get("numpy"), sorted(locked)[:20]


# --------------------------------------------------------------------------- #
# The two measured drift classes.
# --------------------------------------------------------------------------- #


def test_every_declared_dependency_is_present_in_the_lock() -> None:
    """A dependency the manifest requires must appear in the lock.

    ``roslibpy`` did not.  #1110 (2026-07-31) added the whole rosbridge transport
    and the ``[rosbridge]`` extra that requires it, and ``[all]`` includes that
    extra, but the lock gained no entry - so a resolution from the lock produced
    an install where ``use_rosbridge``'s import cannot succeed, and no advisory
    against ``roslibpy`` could ever be reported for this repository.

    Absence is unambiguous here because a uv lock is *universal*: it carries the
    packages for every platform and marker branch rather than for the resolving
    host, so a missing entry is a missing dependency and not a filtered one.
    """
    locked = _locked_versions()
    missing = sorted(
        {
            f"{_canonical(req.name)} (declared in [{where}])"
            for where, req in _declared_requirements()
            if _canonical(req.name) not in locked
        }
    )
    assert not missing, (
        "these dependencies are declared in pyproject.toml and absent from "
        f"uv.lock, so a resolution from the lock omits them entirely: {missing}. "
        "Run `uv lock` and commit the result."
    )


def test_no_locked_version_falls_below_its_declared_floor() -> None:
    """A locked version must satisfy the floor the manifest declares for it.

    ``lerobot`` was pinned at ``0.6.0`` against a ``>=0.6.1`` floor declared by
    three extras, so the lock **violated its own manifest** - and #1930 exists
    only to buy that floor, which makes the lock's answer the exact opposite of
    the guarantee that pull request paid for.

    Phrased over *every* locked version rather than any: see this module's
    docstring for the measurement that rules out the stronger form.
    """
    locked = _locked_versions()
    violations: list[str] = []
    for where, req in _declared_requirements():
        name = _canonical(req.name)
        floor = _declared_floor(req)
        if floor is None or name not in locked:
            continue
        versions = sorted(Version(v) for v in locked[name])
        if all(version < floor for version in versions):
            violations.append(
                f"{name} locked at {[str(v) for v in versions]} but [{where}] declares a floor of {floor}"
            )

    assert not violations, (
        "the lock pins a version its own manifest forbids, so a resolution from "
        f"the lock cannot satisfy pyproject.toml: {sorted(set(violations))}. "
        "Run `uv lock` and commit the result."
    )


# --------------------------------------------------------------------------- #
# The gate that catches everything these two cannot.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gate_text() -> str:
    assert _GATE.is_file(), (
        f"{_GATE.relative_to(_REPO_ROOT)} is missing. The offline pins in this "
        "module cover two drift classes; the workflow is what compares the lock "
        "against the manifest in full."
    )
    return _GATE.read_text(encoding="utf-8")


def test_the_gate_runs_the_parity_check(gate_text: str) -> None:
    """The workflow must actually run ``uv lock --check``.

    That one command *is* the contract - it re-resolves the manifest and exits
    non-zero when the lock would have to change - so a gate that installs uv and
    checks nothing would pass every pin here while enforcing nothing.
    """
    assert re.search(r"^\s*run:\s*uv lock --check\s*$", gate_text, re.MULTILINE), (
        f"{_GATE.name} does not run `uv lock --check`. Neither `uv lock` (which "
        "rewrites the file and exits 0) nor `uv sync` (which relocks silently) "
        "reports drift."
    )


def test_the_gate_reads_the_merge_tree(gate_text: str) -> None:
    """The checkout must not name the head sha.

    This is the one design decision in the workflow that is invisible from its
    behaviour on a green branch, so it is pinned rather than commented.

    "Does the lock resolve the manifest" is a property of the tree that would
    land, and for a ``pull_request`` event that tree is ``actions/checkout``'s
    default ref, the merge commit.  Naming ``head.sha`` instead would accuse
    every branch that forked before a relock while changing no dependency file -
    #1722, #1035 and #1087 were all open when this landed, none of them touch
    ``pyproject.toml`` or ``uv.lock``, and all three pass on the merge tree and
    would have failed on their own heads.

    It is the opposite choice from ``changelog-fragment.yml``, which must name
    both base and head because it asks what a branch *added*; an addition only
    exists as a diff, while a parity check reads two files and needs one tree.
    """
    head_sha_ref = re.search(r"^\s*ref:\s*\$\{\{\s*github\.event\.pull_request\.head\.sha", gate_text, re.MULTILINE)
    assert head_sha_ref is None, (
        f"{_GATE.name} checks out the pull request head sha. Take the default "
        "merge-commit ref instead: on the head sha, a branch that forked before "
        "a relock fails this gate while having changed no dependency file."
    )


def test_every_action_the_gate_uses_is_sha_pinned(gate_text: str) -> None:
    """``uses:`` references pin a 40-character commit SHA, per AGENTS.md > Action Pinning.

    A moving tag on a third-party action is the ``tj-actions/changed-files``
    supply-chain pattern, and this gate runs on ``pull_request``, so it is
    reachable from a fork.
    """
    references = re.findall(r"^\s*uses:\s*(?P<ref>\S+)", gate_text, re.MULTILINE)
    assert references, f"no `uses:` found in {_GATE.name}"
    unpinned = [ref for ref in references if not re.search(r"@[0-9a-f]{40}$", ref)]
    assert not unpinned, f"these action references in {_GATE.name} do not pin a 40-character commit SHA: {unpinned}"
