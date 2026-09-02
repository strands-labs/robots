#!/usr/bin/env python3
"""Report the files a pull request edits that its base also changed since it branched.

Why this exists
---------------
``main`` went red at ``0e636f8`` from two pull requests that were each
individually green and textually non-conflicting. #1766 and #1763 both edit
``_recompile_preserving_state`` in
``strands_robots/simulation/mujoco/scene_ops.py``, for unrelated reasons. #1766
landed first. Every signal the merge gate offers then read green on #1763 --
``reviewDecision: APPROVED``, ``statusCheckRollup: SUCCESS``,
``mergeable: MERGEABLE``, ``mergeStateStatus: CLEAN`` -- and the squash still
broke the suite, because #1763 carried a *premise* test asserting the exact
defect #1766 had just fixed::

    FAILED test_a_tendon_driven_actuator_is_outside_the_joint_matched_id_scope
            AssertionError: assert 2 not in [2]

None of those four signals could have caught it. They are all computed against
the base the branch was tested on: #1763's checks ran against ``32dc3f5b``,
which predates #1766, so the first evaluation of the two changes *together* was
``main`` itself. ``mergeStateStatus: CLEAN`` in particular is a statement about
**text** -- git had no conflicting hunks to report, and it is not git's job to
know that one branch's assertion describes the other branch's bug.

What one branch cannot see
--------------------------
``M..base`` is empty **by construction** whenever ``M`` is the base the branch
was evaluated against, which is every run: the check clears itself not after
wall-clock time but when the pull request is *re-evaluated*. Two consequences,
both invisible to every per-branch signal:

- For two pull requests that are both still **open**, ``M..base`` contains
  neither of them, so the intersection is empty and the check reports clean on
  both. The first evaluation of the two changes together is ``main``. That is the
  #1763/#1766 topology arriving from the open set rather than from the merged
  base, and it is a property of the *set*, which is the same reason the
  duplicate-claim check had to exist (#2017) -- every other check here reads one
  pull request at a time. This one keys on the file rather than the issue, so two
  pull requests claiming different issues are silent to that check by design.
- Once the sibling merges, nothing invalidates the second pull request's green.
  Stale *approvals* are dismissed on push; a stale *pass* has no equivalent, and
  a pull request idle in review never re-runs. So the exposure is not the interval
  between two merges: it is the interval until the second one's next push, which
  is unbounded for anything sitting in a review queue.

``--all-open`` is the caller for both. It reads the open set from the API and
computes the same intersection twice per pull request -- once against each
sibling's ``M..head``, once against what has landed on the base since its own
``M`` -- reusing the path-set helpers the single-branch mode uses, so the two
modes cannot disagree about what counts as an overlap or as prose. That parity
includes renames: git reports one as its delete plus its add under
``--no-renames``, and the API side takes ``previous_filename`` alongside
``filename`` to reach the same set, because a branch that renames a file and a
branch that edits its old name compose without a conflict marker.

The two sides come from different endpoints, and so have different ceilings. The
head side -- the input to the pairwise mode -- is read from the paginated
pull-request files endpoint, which carries ten times what the compare endpoint's
``files`` does. The base side has no paginated equivalent and keeps the compare
cap, reported as unevaluated for that mode alone.

A file carrying a ``strict=True`` xfail is the highest-value overlap candidate
there is: its whole purpose is to fail when a sibling change lands, so it breaks
a composition that git merges without a single conflict marker. #2233 pinned a
defect that way and #2235 fixed it; composed, the tree was red with no conflict
to resolve, which is why ``mergeStateStatus: CLEAN`` is not merely unhelpful
here but actively reassuring.

What this computes
------------------
The overlap between two path sets, both taken from the branch's merge base ``M``
with its base branch:

- ``M..head`` -- what the pull request edits.
- ``M..base`` -- what landed on the base branch after the pull request branched.

A non-empty intersection does not prove the combination is broken. It proves
something weaker and still worth blocking on: **the combination has never been
compiled**, so every green check on the pull request is evidence about a tree
that is not the tree being merged. For the pair above the intersection is
exactly one entry, ``strands_robots/simulation/mujoco/scene_ops.py``, and one
``pytest`` invocation over the two touched test files would have caught it.

Why the remedy is cheap, and self-clearing
------------------------------------------
Merging the base branch into the pull request advances the merge base to the
base tip. The ``M..base`` set becomes empty, so the intersection does too, and
the checks then re-run against a base that *contains* the newly-landed commits.
The check needs no override to clear: doing the thing it asks for makes it pass.

This is the targeted form of branch protection's "Require branches to be up to
date before merging". That setting demands an update plus a full re-run before
*every* merge, which serialises merges and costs a ~14.7k-test suite each time.
This demands one only when the branch and its base actually edited the same
file, which is the only case where the base moving can invalidate a result.

What the path relation cannot see
---------------------------------
Every relation here intersects changed **paths**, and a test that resolves its
population from a filesystem walk is coupled to files it never names. Its
intersection with the sibling it grades is empty, so the sweep reports clean
while the composition is in fact untested. #2557 added
``tests/test_log_strings_are_ascii.py``, whose population is a walk of the
package, and merged in a batch with #2559 and #2560; both siblings added
tool-result prose, which is exactly the surface that grader scores, and the
pairwise path intersection with each was ``[]``. The batch was safe, but only
because it was checked by hand.

Widening the path set to the *walked root* is not the remedy. Measured on this
repository's open set of 9 pull requests, that relation selects 11 of the 36
pairs -- 9 of them not already reported by the path intersection, and none of
them a defect. The reason is structural: of the 125 walk-population tests in
the tree, the ones that reach furthest are rooted at ``strands_robots`` entire,
so they intersect nearly every open branch, while a narrowly-rooted grader
(``strands_robots/mesh/``) selects only the pair the path intersection already
reports. A relation that fires on a third of all pairs and names no defect is
the ``awaiting-first-review`` failure mode: a finding attached to that much of
the queue reads as boilerplate, and the one batch where it mattered is not
distinguishable from the rest.

What does separate them is composing the two branches and running the grader --
it needs no model of what the grader reads, because the grader reads it. Run by
hand over the open set for #2562's whole-tree grader, that cost about 10 s per
composition and correctly left alone three siblings whose new ``except`` tuples
a naive path-or-keyword heuristic would have flagged. It cannot live in this
mode: the sweep reads the open set from the API and no checkout at all, which
is what lets a caller reporting repository health run it without a clone (pinned
by ``tests/test_merge_base_overlap.py``, whose every sweep test runs from a
directory that is not a repository). So the honest change here is scope, not a
new heuristic: this mode reports what it measured -- shared paths -- and names
the composition class it cannot describe. See #2561.

A module a test names is not a coupling it declined to write down
-----------------------------------------------------------------
The paragraph above is about a population the grader never names. A test that
reaches into a module *by name* is the opposite case, and it produced the same
outcome: ``main`` went red at ``828f80eb`` from #2762 and #2774 with the pairwise
path intersection not merely unhelpful but ``[]``. #2762 added
``tests/drivers/test_reachy_transport_guards_are_reachable.py``, whose fixture
evicts ``"strands_robots.device_connect"`` from ``sys.modules`` and patches
``"strands_robots.device_connect.reachy_transport.api"`` by string; #2774 rewrote
``strands_robots/device_connect/__init__.py`` to resolve its names lazily. Six of
that file's nine tests failed on the composition, five of them standalone, and
neither branch changed a single path the other did (#2791, #2795).

So there is a second relation here, over the same two path sets and one further
input: the dotted module names a pull request's *test* diff writes as string
literals, resolved to the module files they name -- every prefix, because
importing ``a.b.c`` executes ``a/b/__init__.py``, which is where the #2774 edit
was. Measured over 2676 co-open pairs drawn from 400 pull requests
(#2309-#2792):

- the path relation selects 104 pairs;
- this one selects 25, of which 9 are not already reported;
- the walked-root widening rejected above selected 11 of 36.

That is 0.34% of the population against the rejected relation's 31%, and unlike
it these nine name couplings a reader can act on: #2774 + #2762 is the
composition above; #2767 + #2762 and #2767 + #2750 are three driver branches
contending for ``strands_robots/drivers/__init__.py``'s registration table, which
is the file behind the *other* test ``main`` was red in; #2546 + #2545 pairs a
branch removing an export from ``strands_robots/mesh/__init__.py`` with one whose
test names that package. The difference from the rejected widening is not the
threshold but the input: a literal is written by the test author and says exactly
which module is reached, where a walked root is inferred from a glob and says
only that some file under it might be.

What stays out of reach is unchanged, and is still the walk. A population
resolved by ``rglob`` names nothing, so it shares neither a path nor a literal
with the siblings it grades, and the report says so unconditionally.

A role naming a module the composition deletes
----------------------------------------------
The two relations above both read a *change*. This one reads a **removal**, and
it is the direction that put ``main`` red at ``8d0298345``. #3037 removed 96 g1
lookup modules; eight verb pull requests, each already merged, carried docstring
roles citing them (``:mod:`strands_robots.tools.g1.g1_fsm_targets```). #3037
branched before those verbs landed, so no tree ever held both the role and the
deletion until the squash, where
``tests/test_docstring_xref_roles_resolve.py`` -- which resolves every role in
the tree -- reported 44 offending docstrings. git merged the two sides without a
conflict, and the path intersection was empty: the branch deletes files no verb
touches, and the verbs cite them from files the branch does not open.

A role is a coupling stated by name, like the literals above, so the same
resolver serves it: ``named_module_paths`` turns a dotted target into the module
files it needs, and a target is dead when one of them is deleted by the other
side. Both directions are read, because either side can land last:

- the branch deletes the module and the base cites it, read from the base tip;
- the branch cites the module and the base deleted it, read from the branch head.

Each side is read from the tree rather than from a diff, which is what keeps the
population identical to the grader's: a role only counts when it sits in a
docstring, and a hunk cannot be parsed for that. Only *contiguous fully-qualified*
targets are resolved -- a wrapped or short-form role has no decidable module path
here, and the grader reports both on their own account.

Two restrictions keep this quiet where it cannot matter. A base-side role in a
file the branch itself changes is skipped: the branch's own tree holds both
halves, so its own run of the grader already sees it. A base-side deletion of a
file the branch also changes is skipped for the stronger reason that git reports
that one as a conflict.

This relation is absent from ``--all-open`` because it cannot be computed there.
The sweep reads patches and no checkout, and a patch hunk does not say whether
its role is inside a docstring -- so the sweep would have to either guess or
grade a different population than the gate. It names the gap instead.

Prose is reported but does not block
------------------------------------
An overlap confined to ``.md`` / ``.rst`` / ``.txt`` cannot change what the test
suite or the built package does; if two branches edit the same prose region, git
reports a conflict and the merge gate already stops it. Those paths are listed
in the report -- suppressing them entirely would hide a signal a reader may
want -- but they do not set the exit status, so a docs PR that happens to share
a file with a landed docs PR is not asked to re-run a full test suite for a
result that cannot change.

Usage
-----
``--base-ref``  the branch being merged into (default ``main``). Resolved as
                ``origin/<ref>`` when that exists, else as ``<ref>``.
``--head``      the commit under test (default ``HEAD``). In CI this must be the
                pull request's *head* commit, never the
                ``refs/pull/<n>/merge`` commit ``actions/checkout`` produces by
                default -- that commit already contains the base tip, which
                drives the merge base to the base tip and the overlap to the
                empty set, so the check would pass unconditionally. CI *names*
                that commit rather than checking it out, and runs this script
                from the base branch instead: a branch that forked before a gate
                landed does not carry that gate's script (issue #1791). Sound
                because every input below is read from the object database and
                never from the working tree.
``--repo``      repository root (default: the current working directory). One
                branch only: the sweep reads no checkout, so ``--all-open``
                refuses it rather than ignoring it. The sibling gate scripts
                spell ``owner/name`` ``--repo``, so a caller who reaches for it
                here names a path, the sweep reads ``$GITHUB_REPOSITORY``
                instead, and the report describes a repository nobody asked
                about (issue #2569).

``--all-open``  Sweep the open set instead of one branch (see above). Mutually
                exclusive with ``--head`` and ``--repo``, which name a local
                commit and a local checkout.
``--github-repo``
                ``owner/name`` for ``--all-open`` (default:
                ``$GITHUB_REPOSITORY``). Deliberately not ``--repo``: in this
                script ``--repo`` is already a local checkout path, and one flag
                that means a filesystem path in one mode and a slug in the other
                is the kind of ambiguity a caller discovers by getting a wrong
                answer.
``--token``     API token for ``--all-open`` (default: ``$GITHUB_TOKEN``). Needs
                ``pull-requests: read``.

Exit status is ``1`` when a behaviour-bearing path overlaps, else ``0``. That is a
blocking result for one branch, and a report for ``--all-open``: the sweep's
remedy is a decision about merge order plus possibly one composition run, which
no push by either author turns green, so it is deliberately absent from the
required set. A gate a branch cannot clear by doing anything is a report,
whatever it is wired to.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import itertools
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path

#: Suffixes whose overlap cannot change the outcome of the test suite or the
#: contents of the built package, and so is reported without blocking.
PROSE_SUFFIXES = frozenset({".md", ".rst", ".txt"})


class GitError(RuntimeError):
    """A git invocation this script depends on did not succeed.

    Raised rather than returning a sentinel: every caller here needs the real
    commit or path set to say anything true, and a check that silently reports
    "no overlap" because it could not reach the base branch is worse than one
    that fails loudly. A missing base ref in CI is a workflow bug, not a
    property of the pull request.
    """


def _git(*args: str, repo: Path | None = None) -> str:
    """Run one git command and return its stdout, raising ``GitError`` on failure."""
    command = ["git"]
    if repo is not None:
        command += ["-C", str(repo)]
    command += list(args)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise GitError(f"{' '.join(command)} exited {completed.returncode}: {completed.stderr.strip()}")
    return completed.stdout


def resolve_base_ref(base_ref: str, repo: Path | None = None) -> str:
    """Return the revision to treat as the base branch tip.

    Prefers the remote-tracking ref, because a CI checkout of a pull request
    head usually has no local branch for the base: ``actions/checkout`` fetches
    the base as ``refs/remotes/origin/<ref>`` and never creates ``<ref>``. Falls
    back to the bare name so the script is runnable in a normal local clone.
    """
    for candidate in (f"origin/{base_ref}", base_ref):
        try:
            _git("rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}", repo=repo)
        except GitError:
            continue
        return candidate
    raise GitError(f"cannot resolve base ref {base_ref!r} as either 'origin/{base_ref}' or '{base_ref}'")


def merge_base(base: str, head: str, repo: Path | None = None) -> str:
    """Return the commit where ``head`` diverged from ``base``."""
    revision = _git("merge-base", base, head, repo=repo).strip()
    if not revision:
        raise GitError(f"no merge base between {base!r} and {head!r} - is the history shallow?")
    return revision


def changed_paths(start: str, end: str, repo: Path | None = None) -> frozenset[str]:
    """Return the paths that differ between two commits.

    ``--no-renames`` is deliberate. With rename detection a file the base
    renamed appears only under its new name, so a pull request still editing the
    old name would not intersect it. Reporting a rename as its delete plus its
    add puts both names in the set, which is the conservative direction for a
    check whose failure mode is a missed overlap.
    """
    output = _git("diff", "--name-only", "--no-renames", f"{start}..{end}", repo=repo)
    return frozenset(line for line in output.splitlines() if line)


def deleted_paths(start: str, end: str, repo: Path | None = None) -> frozenset[str]:
    """Return the paths that exist at ``start`` and not at ``end``.

    The input to the role relation, which is the one relation here that reads a
    removal rather than a change. ``--no-renames`` for the reason
    :func:`changed_paths` gives, and it matters more here: with rename detection a
    module moved to a new path is not reported as deleted at all, while a role
    citing its old dotted name is dead in the merged tree exactly as if it had
    been removed.
    """
    output = _git("diff", "--name-only", "--no-renames", "--diff-filter=D", f"{start}..{end}", repo=repo)
    return frozenset(line for line in output.splitlines() if line)


def diff_entries(start: str, end: str, repo: Path | None = None) -> tuple[tuple[str, str], ...]:
    """Return ``(path, patch)`` for every test module changed between two commits.

    The single-branch mode's source for :func:`module_literals`, giving it the
    same ``(path, patch)`` shape the sweep builds from the API payload, so one
    extractor serves both. Scoped to :data:`TEST_ROOTS` in the pathspec rather
    than filtered afterwards: the diff of a whole tree is large and none of the
    rest is read.

    ``--no-renames`` for the reason :func:`changed_paths` gives, and because a
    rename reported as a delete plus an add puts the added side's lines in the
    patch, which is where a literal is.
    """
    output = _git("diff", "--no-renames", f"{start}..{end}", "--", *TEST_ROOTS, repo=repo)
    collected: dict[str, list[str]] = {}
    current: str | None = None
    for line in output.splitlines():
        if line.startswith("diff --git "):
            # 'diff --git a/<path> b/<path>'. The b-side is the post-image, which
            # is the side an added line belongs to.
            _, _, remainder = line.partition(" b/")
            current = remainder or None
            if current is not None:
                collected.setdefault(current, [])
            continue
        if current is not None:
            collected[current].append(line)
    return tuple((path, "\n".join(body)) for path, body in sorted(collected.items()))


def is_prose(path: str) -> bool:
    """Whether a path is documentation, and so reported without blocking."""
    return Path(path).suffix.lower() in PROSE_SUFFIXES


def partition_overlap(paths: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split overlapping paths into ``(behaviour_bearing, prose)``, each sorted.

    Sorting is what makes the report and the annotations reproducible across
    runs: ``git diff`` order follows the tree, and a set has no order at all.
    """
    ordered = sorted(set(paths))
    return (
        tuple(path for path in ordered if not is_prose(path)),
        tuple(path for path in ordered if is_prose(path)),
    )


def overlapping_paths(pr_paths: Iterable[str], base_paths: Iterable[str]) -> tuple[str, ...]:
    """Return the sorted paths edited both by the pull request and by its base."""
    return tuple(sorted(frozenset(pr_paths) & frozenset(base_paths)))


#: The import root a dotted module literal must start with to be resolved here.
#: Repo-specific on purpose, like the incident this file is named for: a relation
#: that resolved *any* dotted string would report a stdlib or third-party name
#: that no pull request in this repository can change.
PACKAGE_ROOT = "strands_robots"

#: Roots whose ``.py`` files are read for module literals. A test is the only
#: population that reaches into a module by name for a reason the path relation
#: cannot see -- production code that needs another module imports it, and an
#: import is a path the diff already carries.
TEST_ROOTS = ("tests/", "tests_integ/")

#: How many dotted segments a literal must resolve to before its prefix counts.
#: Two, so ``strands_robots.mesh`` resolves and the bare root does not. Measured
#: over 2676 co-open pairs from 400 pull requests (#2309-#2792): resolving the
#: bare root as well adds four findings, every one of them a pair with #2486,
#: whose edit to ``strands_robots/__init__.py`` every literal in the tree names
#: by its first segment. That is the shallowest coupling expressible and it is
#: the one a reader can least act on, so the root is excluded.
MIN_NAMED_MODULE_SEGMENTS = 2

#: A dotted module path that is the *entire* contents of a string literal. Whole
#: contents rather than a substring: ``"strands_robots.mesh.core"`` is a name
#: handed to ``importlib`` or ``monkeypatch.setattr``, while the same characters
#: inside a sentence are prose about a module, not a reach into one.
_MODULE_LITERAL = re.compile(
    r"""(?P<quote>['"])(?P<name>""" + PACKAGE_ROOT + r"""(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?P=quote)"""
)


def is_test_module(path: str) -> bool:
    """Whether a path is a test module, and so read for module literals."""
    return path.endswith(".py") and path.startswith(TEST_ROOTS)


def added_lines(patch: str) -> tuple[str, ...]:
    """Return the added lines of a unified diff, without their ``+`` marker.

    Added lines only, and so the same ``M..head`` framing every other relation
    here uses: the question is what this pull request introduces, not what the
    file already said. ``+++`` is a header rather than content, and dropping it
    also keeps a path out of the literal set.
    """
    return tuple(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))


def module_literals(entries: Iterable[tuple[str, str]]) -> frozenset[str]:
    """Return the dotted module names a pull request's test diff names as strings.

    ``entries`` is ``(path, patch)`` for every file the pull request touches, in
    whichever form the calling mode can reach: the API's ``patch`` field for the
    sweep, ``git diff`` for the single-branch check. One extractor over one input
    shape is what keeps the two modes from disagreeing about what a literal is,
    the same reason both already share the path-set helpers.
    """
    found: set[str] = set()
    for path, patch in entries:
        if not is_test_module(path):
            continue
        for line in added_lines(patch):
            found.update(match.group("name") for match in _MODULE_LITERAL.finditer(line))
    return frozenset(found)


def named_module_paths(names: Iterable[str]) -> frozenset[str]:
    """Return the module files a set of dotted literals names, prefixes included.

    Every prefix, because importing ``a.b.c`` executes ``a/b/__init__.py`` as
    well: the coupling that broke ``main`` at ``828f80eb`` was #2762's test
    naming ``strands_robots.device_connect.reachy_transport.api`` against #2774's
    edit to ``strands_robots/device_connect/__init__.py``, a prefix two segments
    shorter than the literal. Both spellings of a module are emitted -- a
    dotted name does not say whether it is a module or a package, and a path
    neither branch touches cannot match anything.
    """
    found: set[str] = set()
    for name in names:
        segments = name.split(".")
        for depth in range(MIN_NAMED_MODULE_SEGMENTS, len(segments) + 1):
            stem = "/".join(segments[:depth])
            found.add(f"{stem}.py")
            found.add(f"{stem}/__init__.py")
    return frozenset(found)


def named_module_overlaps(literals: Iterable[str], paths: Iterable[str]) -> tuple[str, ...]:
    """Return the sorted module files named by ``literals`` and changed in ``paths``."""
    return tuple(sorted(named_module_paths(literals) & frozenset(paths)))


#: Sphinx cross-reference roles that name a dotted Python target, restricted to a
#: contiguous fully-qualified :data:`PACKAGE_ROOT` path. The grader this predicts
#: (``tests/test_docstring_xref_roles_resolve.py``) deliberately admits any
#: non-backtick character, so it can report a role whose long path wrapped over a
#: line break; that role has no decidable module path, and reporting it here would
#: name a file this branch may not have touched for a defect the grader already
#: describes better. Short-form targets (``Cls.method``) are out for the same
#: reason: their head resolves against the citing module, which is not a path.
_DOCSTRING_ROLE = re.compile(
    r":(?:mod|class|func|meth|attr|data|obj|exc):`~?(?P<target>" + PACKAGE_ROOT + r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`"
)


def dotted_module_names(paths: Iterable[str]) -> frozenset[str]:
    """Return the dotted names under which a set of deleted module files were imported.

    The narrowing key for :func:`orphaned_roles`: a role can only be orphaned by a
    deletion whose dotted name the citing file spells, so this turns the deleted
    paths into the strings to search for. Non-package and non-module paths drop
    out, because no role can name them.

    Names shallower than :data:`MIN_NAMED_MODULE_SEGMENTS` drop out too. That is
    the bare package root, which :func:`named_module_paths` cannot resolve either,
    so it could never produce a finding -- and as a search key it matches every
    citing file in the tree, which is the one input that would make this relation
    expensive.
    """
    found: set[str] = set()
    for path in paths:
        if not path.endswith(".py") or not path.startswith(f"{PACKAGE_ROOT}/"):
            continue
        stem = path[: -len("/__init__.py")] if path.endswith("/__init__.py") else path[: -len(".py")]
        name = stem.replace("/", ".")
        if len(name.split(".")) >= MIN_NAMED_MODULE_SEGMENTS:
            found.add(name)
    return frozenset(found)


def role_targets(source: str) -> frozenset[str]:
    """Return the qualified role targets named by the docstrings in one module's source.

    Docstrings only, which is the population the grader reads. A role in a comment
    or in a runtime string is a dead pointer on its own account and is not what
    makes the suite red, and this check exists to predict the suite.

    A file that does not parse yields nothing rather than raising: the tree side
    this reads is a committed one, and refusing to report the other 43 findings
    because one unrelated file is mid-refactor would trade a whole result for a
    file nobody asked about.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        text = ast.get_docstring(node, clean=False)
        if text:
            found.update(match.group("target") for match in _DOCSTRING_ROLE.finditer(text))
    return frozenset(found)


def files_naming(rev: str, names: Iterable[str], pathspecs: Sequence[str], repo: Path | None = None) -> tuple[str, ...]:
    """Return the ``.py`` files at ``rev`` whose text contains one of ``names``.

    A narrowing step, not the verdict: ``git grep`` finds the candidates cheaply so
    only those files are parsed. An empty name set searches for nothing and returns
    nothing without invoking git, which is the common case -- most branches delete
    no module at all.

    ``git grep`` exits 1 to mean "no match", which is a result and not a failure,
    so that status is read rather than raised. Any other non-zero status is a
    ``GitError`` for the reason :class:`GitError` gives: a relation that reports
    nothing because it could not look is the failure mode this file is written
    against.
    """
    keys = sorted(names)
    if not keys or not pathspecs:
        return ()
    command = ["git"] + (["-C", str(repo)] if repo is not None else [])
    command += ["grep", "--files-with-matches", "-I", "--fixed-strings"]
    for key in keys:
        command += ["-e", key]
    command += [rev, "--", *pathspecs]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode == 1 and not completed.stdout:
        return ()
    if completed.returncode != 0:
        raise GitError(f"{' '.join(command)} exited {completed.returncode}: {completed.stderr.strip()}")
    prefix = f"{rev}:"
    return tuple(
        sorted(
            line[len(prefix) :]
            for line in completed.stdout.splitlines()
            if line.startswith(prefix) and line.endswith(".py")
        )
    )


def orphaned_roles(
    rev: str,
    deleted: Iterable[str],
    pathspecs: Sequence[str],
    repo: Path | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return ``(citing file, target)`` for every docstring role at ``rev`` that ``deleted`` breaks.

    One side of the role relation. ``rev`` is the tree the roles are read from and
    ``deleted`` the paths the *other* side removes, so both directions are this
    function with its two inputs swapped.

    The verdict is :func:`named_module_paths`, the same resolver the literal
    relation uses: a target names a module file, every prefix included, and the
    role is dead when the other side deletes one of them.
    """
    removed = frozenset(deleted)
    found: set[tuple[str, str]] = set()
    for path in files_naming(rev, dotted_module_names(removed), pathspecs, repo=repo):
        source = _git("show", f"{rev}:{path}", repo=repo)
        for target in role_targets(source):
            if named_module_paths({target}) & removed:
                found.add((path, target))
    return tuple(sorted(found))


def orphaned_role_overlaps(
    *,
    base: str,
    head: str,
    branch_deletions: Iterable[str],
    base_deletions: Iterable[str],
    branch_paths: Iterable[str],
    repo: Path | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return every ``(citing file, target)`` the merged tree would leave unresolvable.

    Both directions, because either side can land last, and both restricted to
    what the merge would actually keep -- see the module docstring. Sorted and
    de-duplicated so a role reachable from both directions is one row.
    """
    edits = frozenset(branch_paths)
    from_base = tuple(
        row for row in orphaned_roles(base, branch_deletions, ("*.py",), repo=repo) if row[0] not in edits
    )
    from_head = orphaned_roles(
        head,
        frozenset(base_deletions) - edits,
        tuple(sorted(path for path in edits if path.endswith(".py"))),
        repo=repo,
    )
    return tuple(sorted(frozenset(from_base) | frozenset(from_head)))


API_ROOT = "https://api.github.com"

#: Pagination ceiling for the open-pull-request listing, mirroring the sibling
#: sweep in ``scripts/check_last_push_approval.py``. A repository holding more
#: open pull requests than this at once has a different problem, and an unbounded
#: loop over a paginated endpoint is how a transient API shape change becomes a
#: hang rather than an error.
_MAX_PAGES = 20

#: GitHub's compare endpoint returns at most this many entries in ``files``. A
#: truncated list is indistinguishable from a complete one in the payload, so a
#: path set that reaches the cap is reported as unevaluated rather than as not
#: overlapping. This check's failure mode is a *missed* overlap, and quietly
#: intersecting a truncated set is exactly how one goes missing. Only the
#: base-side set is read from this endpoint; the head side has a paginated one.
_COMPARE_FILE_CAP = 300

#: The pull-request files endpoint is paginated and stops at this many entries,
#: ten times the compare cap. Reaching it carries the same ambiguity a capped
#: compare does and is reported the same way -- but it is ten times further
#: away, and the head-side set is the input to the pairwise mode, which is the
#: mode that finds defects. The largest pull request in this repository's history
#: is 153 files (#1667, closed unmerged); the largest open one today is 10.
_PULL_FILE_CAP = 3000

#: Page size for that endpoint. ``_PULL_FILE_CAP`` must stay a whole multiple of
#: it, because the loop bound below is the quotient.
_PULL_FILE_PAGE = 100


class ApiError(RuntimeError):
    """A GitHub API call the sweep depends on did not succeed.

    Separate from ``GitError`` because the two modes have disjoint inputs: the
    single-branch check reads the object database and never the network, and the
    sweep reads the network and never a checkout. One shared exception would let
    a report offer a remedy that cannot apply to the mode that produced it.
    """


@dataclasses.dataclass(frozen=True)
class OpenPullRequest:
    """One open pull request's path sets, both taken from its own merge base.

    ``edits`` comes from the paginated pull-request files endpoint and
    ``landed_since`` from the compare endpoint, so the two sides have different
    ceilings: the base side has no paginated equivalent and keeps
    ``_COMPARE_FILE_CAP``.

    ``literals`` comes from the ``patch`` field of the same entries ``edits`` is
    built from, so it costs no additional request: the endpoint already carries
    it. A pull request touching no test module has an empty set, which excludes
    it from the named-module relation and from nothing else.

    ``landed_since`` is ``None`` when that side could not be read. It is an input
    to the stale-base mode only, so an unreadable one excludes the pull request
    from that mode while leaving it in the pairwise comparison -- which matters:
    the base-side set is the one that grows without bound and so the one that
    hits the file cap, and dropping the whole pull request for it would discard a
    pairwise finding this check exists to make.
    """

    number: int
    head_sha: str
    merge_base: str
    behind_by: int
    edits: frozenset[str]
    literals: frozenset[str]
    landed_since: frozenset[str] | None


def _get(url: str, token: str) -> object:
    """Fetch and decode one API response."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "check-merge-base-overlap",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed API host
        return json.load(response)


def resolve_open_pull_requests(repo: str, token: str) -> list[tuple[int, str]]:
    """Return ``(number, head_sha)`` for every open non-draft pull request, sorted.

    Drafts are excluded for the reason the sibling sweep gives: a draft cannot
    merge whatever else is true of it, so a finding on one does not mean what a
    finding here means. Sorting keeps a diff of two reports about changed
    verdicts rather than reordered rows.
    """
    found: list[tuple[int, str]] = []
    for page in range(1, _MAX_PAGES + 1):
        payload = _get(f"{API_ROOT}/repos/{repo}/pulls?state=open&per_page=100&page={page}", token)
        rows = payload if isinstance(payload, list) else []
        for row in rows:
            if not isinstance(row, dict) or row.get("draft"):
                continue
            number = row.get("number")
            head = ((row.get("head") or {}).get("sha")) or ""
            if isinstance(number, int) and head:
                found.append((number, str(head)))
        if len(rows) < 100:
            break
    return sorted(found)


def paths_from_entries(entries: Iterable[object]) -> frozenset[str]:
    """Return every path a file-list entry names, a rename's old name included.

    Both file-carrying endpoints report a rename as a single entry whose
    ``filename`` is the new path and whose ``previous_filename`` is the old one.
    Reading only the first is a false negative in the direction that matters: a
    branch renaming ``foo.py`` and a branch editing ``foo.py`` then share no
    path, while git -- which does detect the rename -- applies the second
    branch's edit to the new name with no conflict to report. That is exactly
    the composition this file exists to find, arriving invisibly.

    This is the API counterpart of ``changed_paths``'s ``--no-renames``, and it
    is what keeps the two modes agreeing about what an overlap is. Taking both
    names can only widen the reported set, which is the safe direction for a
    check whose failure mode is a missed overlap.
    """
    return frozenset(
        str(value)
        for entry in entries
        if isinstance(entry, dict)
        for key in ("filename", "previous_filename")
        if (value := entry.get(key))
    )


def _compare_payload(repo: str, base: str, head: str, token: str) -> tuple[list[object], str, int]:
    """Fetch a three-dot ``base...head`` and return ``(file entries, merge_base_sha, behind_by)``.

    The three-dot form is what makes this the same question the single-branch
    mode asks with git: the ``files`` it reports are the diff from
    ``merge_base(base, head)`` to ``head``, i.e. ``M..head``. Swapping the
    operands therefore yields ``M..base`` from the same endpoint, which is how
    the sweep obtains the base side without resolving a merge base itself or
    fetching a single commit.

    ``_COMPARE_FILE_CAP`` is deliberately not enforced here. Whether a truncated
    ``files`` list is a problem depends on whether the caller reads it, and the
    two callers differ: one wants the paths, the other wants only the two fields
    the paginated endpoint does not carry.
    """
    url = f"{API_ROOT}/repos/{repo}/compare/{base}...{head}"
    payload = _get(url, token)
    if not isinstance(payload, dict):
        raise ApiError(f"{url}: expected an object, got {type(payload).__name__}")
    entries = payload.get("files")
    entries = entries if isinstance(entries, list) else []
    commit = payload.get("merge_base_commit")
    merge_base_sha = str((commit or {}).get("sha") or "") if isinstance(commit, dict) else ""
    behind_by = payload.get("behind_by")
    return entries, merge_base_sha, behind_by if isinstance(behind_by, int) else 0


def compare_paths(repo: str, base: str, head: str, token: str) -> tuple[frozenset[str], str, int]:
    """Return ``(paths, merge_base_sha, behind_by)`` for a three-dot ``base...head``.

    Enforces the compare endpoint's file cap: a path set that reached it is
    incomplete, and intersecting it would report "no overlap" while meaning "did
    not look". Used for the base side, which has no paginated equivalent.
    """
    entries, merge_base_sha, behind_by = _compare_payload(repo, base, head, token)
    if len(entries) >= _COMPARE_FILE_CAP:
        raise ApiError(
            f"{API_ROOT}/repos/{repo}/compare/{base}...{head}: the file list reached the "
            + f"{_COMPARE_FILE_CAP}-entry cap, so the path set is incomplete and an overlap "
            + "computed from it could be a false negative"
        )
    return paths_from_entries(entries), merge_base_sha, behind_by


def compare_fork_point(repo: str, base: str, head: str, token: str) -> tuple[str, int]:
    """Return ``(merge_base_sha, behind_by)`` for ``base...head``, ignoring ``files``.

    The head side takes its paths from ``pull_request_paths``, so this call is
    needed only for the two fields that endpoint does not carry -- and it must
    not fail on a capped ``files`` list it never reads. It once did, via
    ``compare_paths``, which removed a large pull request from the *pairwise*
    comparison as well: the one mode where a 300-file branch is the most likely
    thing on the queue to collide with something.
    """
    _, merge_base_sha, behind_by = _compare_payload(repo, base, head, token)
    return merge_base_sha, behind_by


def literals_from_entries(entries: Iterable[object]) -> frozenset[str]:
    """Return the module literals a file-list payload's test diffs name.

    Reads ``patch`` beside ``filename`` from the entries ``paths_from_entries``
    already consumes. An entry carries no ``patch`` when the endpoint suppressed
    it -- a binary file, or one past the per-file diff cap -- and a missing diff
    contributes nothing rather than raising: this relation is additive, and a
    pull request whose patches are unreadable is still fully evaluated by the
    path relation, so failing here would cost a finding to gain nothing.
    """
    pairs: list[tuple[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("filename")
        patch = entry.get("patch")
        if isinstance(name, str) and isinstance(patch, str):
            pairs.append((name, patch))
    return module_literals(pairs)


def pull_request_file_entries(repo: str, number: int, token: str) -> list[object]:
    """Return every file-list entry for pull request ``number``, paginated.

    Below the compare cap this is the same set ``base...head`` reports -- measured
    on this repository at 7, 10 and 153 files (#1035, #1722, #1667), identical
    both ways, and byte-identical across the whole open queue -- so no verdict
    changes. What changes is the ceiling: paginating raises the head side's from
    ``_COMPARE_FILE_CAP`` to ``_PULL_FILE_CAP``.

    Reaching that ceiling is reported like a capped compare, for the same reason:
    the endpoint stops there without saying so, and a silently short path set is
    how a missed overlap is manufactured.

    Returned whole rather than reduced to paths, because two relations read them:
    ``paths_from_entries`` takes ``filename``, ``literals_from_entries`` takes
    ``patch``. One fetch feeding both is what keeps the named-module relation free
    of additional requests.
    """
    collected: list[object] = []
    for page in range(1, _PULL_FILE_CAP // _PULL_FILE_PAGE + 1):
        url = f"{API_ROOT}/repos/{repo}/pulls/{number}/files?per_page={_PULL_FILE_PAGE}&page={page}"
        payload = _get(url, token)
        rows = payload if isinstance(payload, list) else []
        collected.extend(rows)
        if len(rows) < _PULL_FILE_PAGE:
            return collected
    raise ApiError(
        f"{API_ROOT}/repos/{repo}/pulls/{number}/files: the file list reached the "
        + f"{_PULL_FILE_CAP}-entry ceiling, so the path set is incomplete and an overlap "
        + "computed from it could be a false negative"
    )


def pull_request_paths(repo: str, number: int, token: str) -> frozenset[str]:
    """Return the paths pull request ``number`` edits.

    Kept as the path-only accessor over :func:`pull_request_file_entries` so a
    caller that needs one set is not handed the raw payload.
    """
    return paths_from_entries(pull_request_file_entries(repo, number, token))


def collect_open_pull_requests(
    repo: str, base_ref: str, token: str
) -> tuple[list[OpenPullRequest], list[tuple[int, str]]]:
    """Read both path sets for every open pull request.

    Returns the rows it could evaluate and ``(number, reason)`` for every side it
    could not. A failure on one pull request is named and skipped rather than
    raised: the sweep exists to surface findings across the standing population,
    and one unreachable pull request must not take the rest of the report with
    it. Naming the skips is the same requirement -- an unevaluated pull request
    that says nothing is the failure mode this whole file is written against.
    """
    rows: list[OpenPullRequest] = []
    unevaluated: list[tuple[int, str]] = []
    lookup_failures = (ApiError, urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError)
    for number, head_sha in resolve_open_pull_requests(repo, token):
        try:
            fork_point, behind_by = compare_fork_point(repo, base_ref, head_sha, token)
            entries = pull_request_file_entries(repo, number, token)
            edits = paths_from_entries(entries)
            literals = literals_from_entries(entries)
        except lookup_failures as error:
            unevaluated.append((number, f"not evaluated in either mode - own path set unreadable: {error}"))
            continue
        landed_since: frozenset[str] | None
        try:
            landed_since, _, _ = compare_paths(repo, head_sha, base_ref, token)
        except lookup_failures as error:
            landed_since = None
            unevaluated.append((number, f"stale-base mode only - base-side path set unreadable: {error}"))
        rows.append(
            OpenPullRequest(
                number=number,
                head_sha=head_sha,
                merge_base=fork_point,
                behind_by=behind_by,
                edits=edits,
                literals=literals,
                landed_since=landed_since,
            )
        )
    return rows, unevaluated


def pair_overlaps(
    pull_requests: Iterable[OpenPullRequest],
) -> list[tuple[int, int, tuple[str, ...], tuple[str, ...]]]:
    """Return ``(left, right, blocking, prose)`` for every pair sharing a path."""
    ordered = sorted(pull_requests, key=lambda row: row.number)
    found: list[tuple[int, int, tuple[str, ...], tuple[str, ...]]] = []
    for left, right in itertools.combinations(ordered, 2):
        blocking, prose = partition_overlap(overlapping_paths(left.edits, right.edits))
        if blocking or prose:
            found.append((left.number, right.number, blocking, prose))
    return found


def named_module_pair_overlaps(
    pull_requests: Iterable[OpenPullRequest],
) -> list[tuple[int, int, tuple[str, ...]]]:
    """Return ``(left, right, modules)`` for every pair coupled by a named module.

    The path relation asks whether two branches edited the same file. This one
    asks whether one branch's test *names* a module the other changed, which is a
    coupling with no shared path at all: ``main`` went red at ``828f80eb`` from
    #2762 and #2774, whose changed-path sets intersect to nothing.

    Both directions, because the relation is not symmetric -- one side supplies
    the literal and the other the edit -- and a pair is a finding whichever way
    round that falls.
    """
    ordered = sorted(pull_requests, key=lambda row: row.number)
    found: list[tuple[int, int, tuple[str, ...]]] = []
    for left, right in itertools.combinations(ordered, 2):
        modules = sorted(
            frozenset(named_module_overlaps(left.literals, right.edits))
            | frozenset(named_module_overlaps(right.literals, left.edits))
        )
        if modules:
            found.append((left.number, right.number, tuple(modules)))
    return found


def named_module_stale_overlaps(
    pull_requests: Iterable[OpenPullRequest],
) -> list[tuple[int, int, tuple[str, ...]]]:
    """Return ``(number, behind_by, modules)`` for every named module the base moved.

    The named-module relation with the base branch substituted for a sibling's
    head, mirroring :func:`stale_base_overlaps`. Only one direction is available
    and only one is wanted: the base's own tests are already in the tree the
    branch will be merged into.
    """
    found: list[tuple[int, int, tuple[str, ...]]] = []
    for row in sorted(pull_requests, key=lambda row: row.number):
        if row.landed_since is None or row.behind_by == 0:
            continue
        modules = named_module_overlaps(row.literals, row.landed_since)
        if modules:
            found.append((row.number, row.behind_by, modules))
    return found


def stale_base_overlaps(
    pull_requests: Iterable[OpenPullRequest],
) -> list[tuple[int, int, tuple[str, ...], tuple[str, ...]]]:
    """Return ``(number, behind_by, blocking, prose)`` for every stale-base overlap.

    This is the single-branch computation applied to the population: the same
    intersection, with the base branch substituted for a sibling's head. A pull
    request level with its base has nothing to compare, so it is skipped rather
    than reported as clean.
    """
    found: list[tuple[int, int, tuple[str, ...], tuple[str, ...]]] = []
    for row in sorted(pull_requests, key=lambda row: row.number):
        if row.landed_since is None or row.behind_by == 0:
            continue
        blocking, prose = partition_overlap(overlapping_paths(row.edits, row.landed_since))
        if blocking or prose:
            found.append((row.number, row.behind_by, blocking, prose))
    return found


def render_sweep(
    *,
    repo: str,
    base_ref: str,
    pull_requests: Sequence[OpenPullRequest],
    pairs: Sequence[tuple[int, int, tuple[str, ...], tuple[str, ...]]],
    stale: Sequence[tuple[int, int, tuple[str, ...], tuple[str, ...]]],
    named_pairs: Sequence[tuple[int, int, tuple[str, ...]]],
    named_stale: Sequence[tuple[int, int, tuple[str, ...]]],
    unevaluated: Sequence[tuple[int, str]],
) -> str:
    """Render the sweep report.

    Every multi-line paragraph is a named local joined with explicit ``+``, for
    the reason ``render_report`` gives: implicit concatenation inside a list of
    report lines is indistinguishable from a forgotten comma.
    """
    pair_count = len(pull_requests) * (len(pull_requests) - 1) // 2
    lines = ["## Merge-base overlap check - open set", ""]
    header = (
        f"`{repo}`, base `{base_ref}`: {len(pull_requests)} open non-draft pull request(s), "
        + f"{pair_count} pair(s) compared."
    )
    lines.append(header)
    lines.append("")

    blocking_pairs = [row for row in pairs if row[2]]
    prose_pairs = [row for row in pairs if not row[2]]
    blocking_stale = [row for row in stale if row[2]]

    if not blocking_pairs and not blocking_stale and not named_pairs and not named_stale:
        clean = (
            "No pair in the open set shares a changed path, no pull request shares one "
            + "with what has landed on its base, and no test names a module a sibling or the "
            + "base changed. Nothing here needs a merge-order decision."
        )
        lines.append(clean)
        lines.append("")
    if blocking_pairs:
        pair_heading = f"### Pairs editing the same behaviour-bearing path ({len(blocking_pairs)})"
        pair_why = (
            "Neither pull request's checks have compiled the other's changes, and neither can: "
            + "each ran against a base that contains neither. Whichever merges second inherits a "
            + "green result about a tree that no longer exists."
        )
        lines.append(pair_heading)
        lines.append("")
        lines.append(pair_why)
        lines.append("")
        lines.append("| pull requests | shared path(s) |")
        lines.append("|---|---|")
        for left, right, paths, _ in blocking_pairs:
            rendered = ", ".join(f"`{path}`" for path in paths)
            lines.append(f"| #{left} + #{right} | {rendered} |")
        lines.append("")
    if blocking_stale:
        stale_heading = f"### Pull requests whose base moved under a path they edit ({len(blocking_stale)})"
        stale_why = (
            "The single-branch check would report each of these today, and reported none of them "
            + "when it ran: it reads the base as of that run, and the commits below landed after. "
            + "Nothing re-runs a pull request idle in review, so the green stands until its next push."
        )
        lines.append(stale_heading)
        lines.append("")
        lines.append(stale_why)
        lines.append("")
        lines.append(f"| pull request | behind `{base_ref}` by | path(s) also changed on `{base_ref}` |")
        lines.append("|---|---|---|")
        for number, behind_by, paths, _ in blocking_stale:
            rendered = ", ".join(f"`{path}`" for path in paths)
            lines.append(f"| #{number} | {behind_by} | {rendered} |")
        lines.append("")
    if named_pairs:
        named_heading = f"### Pairs coupled by a module one of them names ({len(named_pairs)})"
        named_why = (
            "Neither pull request edits a path the other does, so the section above cannot "
            + "report them. One of them has a test that reaches into the module below by name - "
            + "through `importlib`, a `sys.modules` key or a string `monkeypatch` target - and "
            + "the other changed that module's file. The name is the coupling, and a name is not "
            + "a path. This is the relation `main` needed at `828f80eb`."
        )
        lines.append(named_heading)
        lines.append("")
        lines.append(named_why)
        lines.append("")
        lines.append("| pull requests | module(s) one names and the other changed |")
        lines.append("|---|---|")
        for left, right, modules in named_pairs:
            rendered = ", ".join(f"`{module}`" for module in modules)
            lines.append(f"| #{left} + #{right} | {rendered} |")
        lines.append("")
    if named_stale:
        named_stale_heading = f"### Pull requests whose base changed a module their tests name ({len(named_stale)})"
        named_stale_why = (
            "Same relation, with the base branch in place of a sibling. The branch edits none "
            + "of these files, so no path-based check reports it, and its own tests reach into "
            + "them by name."
        )
        lines.append(named_stale_heading)
        lines.append("")
        lines.append(named_stale_why)
        lines.append("")
        lines.append(f"| pull request | behind `{base_ref}` by | module(s) its tests name |")
        lines.append("|---|---|---|")
        for number, behind_by, modules in named_stale:
            rendered = ", ".join(f"`{module}`" for module in modules)
            lines.append(f"| #{number} | {behind_by} | {rendered} |")
        lines.append("")
    if prose_pairs:
        prose_heading = (
            f"Also sharing a path, not reported ({len(prose_pairs)} prose-only pair(s)) - prose "
            + "cannot change what the suite or the package does, and a genuine collision inside one "
            + "surfaces as a merge conflict:"
        )
        lines.append(prose_heading)
        lines.append("")
        for left, right, _, paths in prose_pairs:
            rendered = ", ".join(f"`{path}`" for path in paths)
            lines.append(f"- #{left} + #{right}: {rendered}")
        lines.append("")
    if unevaluated:
        unevaluated_heading = (
            f"Unevaluated ({len(unevaluated)}) - named rather than counted as clean, because a "
            + "pull request this check could not read is not a pull request it cleared:"
        )
        lines.append(unevaluated_heading)
        lines.append("")
        for number, reason in unevaluated:
            lines.append(f"- #{number}: {reason}")
        lines.append("")

    remedy = (
        "**To clear a row above:** decide the merge order, run the tests covering the shared "
        + "paths against the composition once, and merge. This is a report, not a required check: "
        + "no push by either author makes a *pair* green."
    )
    lines.append(remedy)
    lines.append("")
    # Unconditional, and last: it qualifies a clean report at least as much as a
    # populated one. A reader who sees no rows is entitled to know which
    # compositions the relations above were never able to describe.
    limits = (
        "**What no relation above covers:** a test that resolves its population from a "
        + "filesystem walk grades files it never names. The named-module relation reaches a "
        + "coupling a test writes down; a walk writes nothing down, so it still shares no path "
        + "and no name with the siblings it grades, and no row above can describe that "
        + "composition. Widening the path set to the walked root was measured and rejected as "
        + "unselective; only composing the branches and running the grader settles those, and "
        + "that needs a checkout this mode does not have. See #2561. The role relation the "
        + "single-branch mode carries - a docstring citing a module the other side deletes - is "
        + "absent here for the same reason: it reads two trees, and this mode reads patches."
    )
    lines.append(limits)
    return "\n".join(lines) + "\n"


def _emit(report: str) -> None:
    """Print the report and append it to the CI job summary when there is one."""
    print(report, end="")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report)


def _run_sweep(repo: str, base_ref: str, token: str) -> int:
    """Sweep the open set. Exit 1 when a behaviour-bearing composition is untested."""
    try:
        pull_requests, unevaluated = collect_open_pull_requests(repo, base_ref, token)
    except (ApiError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as error:
        # Listing the pull requests is the one lookup with no partial result to
        # fall back on: without it there is no population to sweep.
        print(f"::error::merge-base overlap sweep could not list open pull requests: {error}", file=sys.stderr)
        return 1

    pairs = pair_overlaps(pull_requests)
    stale = stale_base_overlaps(pull_requests)
    named_pairs = named_module_pair_overlaps(pull_requests)
    named_stale = named_module_stale_overlaps(pull_requests)
    _emit(
        render_sweep(
            repo=repo,
            base_ref=base_ref,
            pull_requests=pull_requests,
            pairs=pairs,
            named_pairs=named_pairs,
            named_stale=named_stale,
            stale=stale,
            unevaluated=unevaluated,
        )
    )
    # A named-module finding blocks on the same footing as a shared path: both say
    # the composition has never been compiled, and this one resolves only to '.py'
    # module files, so there is no prose half to suppress.
    blocking = any(row[2] for row in pairs) or any(row[2] for row in stale) or bool(named_pairs) or bool(named_stale)
    return 1 if blocking else 0


def render_report(
    *,
    base_ref: str,
    merge_base_sha: str,
    blocking: Sequence[str],
    prose: Sequence[str],
    named: Sequence[str],
    orphaned: Sequence[tuple[str, str]],
    base_change_count: int,
) -> str:
    """Render the Markdown report written to stdout and the CI job summary.

    Every multi-line paragraph below is built as a named local with explicit
    ``+`` rather than from adjacent literals inside the ``lines`` list. Implicit
    concatenation there is indistinguishable from a forgotten comma: a paragraph
    split across two elements silently becomes two report lines, and two
    paragraphs missing their separator silently become one. The join that
    produces the report cannot tell the difference, and neither can a reader.
    """
    lines = ["## Merge-base overlap check", ""]

    if not blocking and not prose and not named and not orphaned:
        no_overlap = (
            f"No overlap. This branch edits nothing that `{base_ref}` has changed since "
            + f"the two diverged at `{merge_base_sha[:8]}` "
            + f"({base_change_count} path(s) changed on `{base_ref}` in that span)."
        )
        lines.append(no_overlap)
        lines.append("")
        lines.append("The checks on this branch were computed against a base that cannot have invalidated them.")
        return "\n".join(lines) + "\n"

    if blocking:
        heading = (
            f"This branch and `{base_ref}` have both changed **{len(blocking)}** "
            + f"behaviour-bearing path(s) since they diverged at `{merge_base_sha[:8]}`:"
        )
        why = (
            "Every check on this branch ran against a base that predates those commits, "
            + "so the combination has not been compiled. A green result here is evidence "
            + "about a different tree than the one that would be merged."
        )
        remedy = (
            "**To clear this:** merge "
            + f"`{base_ref}` into this branch and push. That advances the merge base, "
            + "re-runs the checks against a base containing the landed commits, and makes "
            + "this check pass. Run the tests covering the paths above first - that is "
            + "the cheap part, and it is what the check exists to prompt."
        )
        lines.append(heading)
        lines.append("")
        lines += [f"- `{path}`" for path in blocking]
        lines.append("")
        lines.append(why)
        lines.append("")
        lines.append(remedy)

    if named:
        if blocking:
            lines.append("")
        named_heading = (
            f"This branch's tests name **{len(named)}** module(s) that `{base_ref}` has "
            + f"changed since the two diverged at `{merge_base_sha[:8]}`, and that this branch "
            + "does not edit:"
        )
        named_why = (
            "A path-based comparison cannot see this: the coupling is the name, written into a "
            + "test as an `importlib` argument, a `sys.modules` key or a string `monkeypatch` "
            + "target. The checks on this branch resolved those names against the older module."
        )
        named_remedy = (
            f"**To clear this:** merge `{base_ref}` into this branch and push, having first run "
            + "the tests that name the modules above. This is the relation `main` needed at "
            + "`828f80eb`, where two branches with no shared path composed red."
        )
        lines.append(named_heading)
        lines.append("")
        lines += [f"- `{module}`" for module in named]
        lines.append("")
        lines.append(named_why)
        lines.append("")
        lines.append(named_remedy)

    if orphaned:
        if blocking or named:
            lines.append("")
        orphaned_heading = (
            f"This branch and `{base_ref}` compose to a tree carrying **{len(orphaned)}** "
            + "docstring cross-reference role(s) that name a module the composition does not "
            + "contain:"
        )
        orphaned_why = (
            "One side removes the module and the other cites it, so neither branch is wrong on "
            + "its own and neither one's checks can see it: the role and the deletion are never in "
            + "the same tree until the merge. `tests/test_docstring_xref_roles_resolve.py` "
            + "resolves every role in the tree, so the composition is red - with no conflict for "
            + "git to report, because no file was written by both sides."
        )
        orphaned_remedy = (
            f"**To clear this:** merge `{base_ref}` into this branch, then drop or repoint the "
            + "role(s) above. This is the relation `main` needed at `8d0298345`, where a branch "
            + "removing 96 lookup modules and eight branches citing them were each green."
        )
        lines.append(orphaned_heading)
        lines.append("")
        lines += [f"- `{path}` cites `{target}`" for path, target in orphaned]
        lines.append("")
        lines.append(orphaned_why)
        lines.append("")
        lines.append(orphaned_remedy)

    if prose:
        if blocking or named or orphaned:
            lines.append("")
        prose_heading = (
            f"Also overlapping, not blocking ({len(prose)} documentation path(s)) - "
            + "prose cannot change what the suite or the package does, and a genuine "
            + "collision inside one would surface as a merge conflict:"
        )
        lines.append(prose_heading)
        lines.append("")
        lines += [f"- `{path}`" for path in prose]

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Compute the overlap and return the process exit status."""
    parser = argparse.ArgumentParser(
        prog="check_merge_base_overlap.py",
        description="Report files a pull request edits that its base also changed since it branched.",
    )
    parser.add_argument("--base-ref", default="main", help="branch being merged into (default: main)")
    parser.add_argument(
        "--head",
        default=None,
        help="commit under test (default: HEAD); one branch only, not --all-open",
    )
    parser.add_argument("--repo", default=None, help="repository root (default: current directory)")
    parser.add_argument(
        "--all-open",
        action="store_true",
        help="sweep the open set from the API instead of checking one branch",
    )
    parser.add_argument(
        "--github-repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name for --all-open (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="API token for --all-open (default: $GITHUB_TOKEN)",
    )
    args = parser.parse_args(argv)

    # Mutually exclusive rather than ignored: --head names one commit and --repo
    # one checkout, and the sweep reads neither, so honouring either would answer
    # a question the caller did not ask while looking like it had. --repo is the
    # one a caller is most likely to pass by mistake, because the sibling gate
    # scripts spell owner/name --repo: accepted as a path, it leaves the sweep
    # reading $GITHUB_REPOSITORY and reporting on a repository nobody asked about
    # (issue #2569). Every value-bearing flag the sweep does read is passed to
    # _run_sweep below, which is what pins this partition rather than a list.
    if args.all_open and args.head is not None:
        parser.error("--all-open sweeps the open set and reads no local commit; --head names one branch")
    if args.all_open and args.repo is not None:
        parser.error(
            "--all-open sweeps the open set from the API and reads no local checkout; --repo names one. "
            "To name the repository the sweep reads, pass --github-repo owner/name."
        )
    if args.all_open:
        if not args.github_repo:
            parser.error("--all-open needs --github-repo owner/name (or $GITHUB_REPOSITORY)")
        if not args.token:
            parser.error("--all-open needs --token (or $GITHUB_TOKEN)")
        return _run_sweep(args.github_repo, args.base_ref, args.token)

    head = args.head if args.head is not None else "HEAD"
    repo = Path(args.repo) if args.repo is not None else None

    try:
        base = resolve_base_ref(args.base_ref, repo=repo)
        fork_point = merge_base(base, head, repo=repo)
        pr_paths = changed_paths(fork_point, head, repo=repo)
        base_paths = changed_paths(fork_point, base, repo=repo)
        literals = module_literals(diff_entries(fork_point, head, repo=repo))
        orphaned = orphaned_role_overlaps(
            base=base,
            head=head,
            branch_deletions=deleted_paths(fork_point, head, repo=repo),
            base_deletions=deleted_paths(fork_point, base, repo=repo),
            branch_paths=pr_paths,
            repo=repo,
        )
    except GitError as error:
        # Loud and non-zero: a check that cannot compute its answer must not
        # report the reassuring one.
        print(f"::error::merge-base overlap check could not run: {error}", file=sys.stderr)
        return 1

    blocking, prose = partition_overlap(overlapping_paths(pr_paths, base_paths))
    # Reported separately from 'blocking' rather than folded into it: the two are
    # different relations with different remedial reading, and a module named but
    # not edited is not something the branch "also changed".
    named = tuple(module for module in named_module_overlaps(literals, base_paths) if module not in blocking)
    report = render_report(
        base_ref=args.base_ref,
        merge_base_sha=fork_point,
        blocking=blocking,
        prose=prose,
        named=named,
        orphaned=orphaned,
        base_change_count=len(base_paths),
    )

    print(report, end="")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report)

    for path in blocking:
        annotation = (
            f"::error file={path}::{path} was also changed on {args.base_ref} after this "
            + "branch diverged; the checks on this branch never compiled the two together."
        )
        print(annotation)
    for module in named:
        annotation = (
            f"::error file={module}::{module} was changed on {args.base_ref} after this branch "
            + "diverged, and a test on this branch names it as a string; the checks here "
            + "resolved that name against the older module."
        )
        print(annotation)

    for path, target in orphaned:
        annotation = (
            f"::error file={path}::{path} cites {target} in a docstring, and this branch and "
            + f"{args.base_ref} compose to a tree without that module; every role in the tree is "
            + "resolved by the suite, so the composition is red with no conflict to resolve."
        )
        print(annotation)

    return 1 if blocking or named or orphaned else 0


if __name__ == "__main__":
    sys.exit(main())
