"""``_load_acl_cached``'s docstring describes the cache that ships.

The ACL file is read through one cache, and that cache's docstring is where a
reader auditing the ACL path learns three things: who reads the file through
it, what two callers of it share, and what closes the TOCTOU window between the
``Mesh.start`` shape gate and the wire-config builder. All three had drifted
away from the code, and the last two in the direction that overstates the
guarantee:

* **The caller census was stale.** It said "two callers", naming the gate check
  and the config builder - the pre-#218 ``is_default_acl_in_use`` +
  ``resolve_acl`` pair. Three functions call it now, and the two the docstring
  named are reached from nowhere outside this module: the caller
  ``Mesh.start`` actually goes through is :func:`~strands_robots.mesh._acl_config.snapshot_acl`.
  The module comment above the cache says that pair was superseded ("in an
  earlier revision"), so the docstring was the one place still presenting it in
  the present tense.

* **Callers were said to "get the same dict object".** Every return path
  deep-copies - deliberately, and one of the deep-copy comments records that
  returning the parsed dict directly is what it replaced. Callers share the
  file's contents; identity is precisely what they do not share.

* **It claimed to close "the prior TOCTOU surface".**
  :func:`~strands_robots.mesh._acl_config.snapshot_acl` documents this
  identity-keyed cache as its *second* tier, whose reload on a rewritten file is
  a "by-design refresh window", and attributes the defence to its thread-local
  snapshot. Two docstrings in one module disagreed about which mechanism holds
  the line, and the weaker one claimed it.

These tests pin the docstring against the code rather than against a copy of
it: the caller census and the caller names are derived from the module's own
AST, so a fourth reader of the cache is graded the moment it is added, and the
identity claim is checked by calling the cache twice.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from strands_robots.mesh import _acl_config

#: Count words a caller census can be written with, mapped to the number.
_NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_CENSUS_RE = re.compile(rf"\b({'|'.join(_NUMBER_WORDS)}|\d+)\s+callers?\b", re.IGNORECASE)

#: The function whose docstring is under test.
_SUBJECT = "_load_acl_cached"


def _module_tree() -> ast.Module:
    """Parse the ACL-config module the cache lives in."""
    return ast.parse(Path(inspect.getfile(_acl_config)).read_text(encoding="utf-8"))


def _docstring_of(name: str) -> str:
    """The whitespace-normalized docstring of ``name`` in that module."""
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return " ".join((ast.get_docstring(node) or "").split())
    raise AssertionError(f"{name} is not a function in {_acl_config.__name__}")


def _callers_of(name: str) -> set[str]:
    """Names of the functions in that module that call ``name``."""
    tree = _module_tree()
    defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    def owner(lineno: int) -> str | None:
        """Innermost function containing ``lineno``."""
        best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        for fn in defs:
            if fn.lineno <= lineno <= (fn.end_lineno or fn.lineno):
                if best is None or fn.lineno > best.lineno:
                    best = fn
        return best.name if best else None

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == name:
            enclosing = owner(node.lineno)
            if enclosing is not None and enclosing != name:
                found.add(enclosing)
    return found


def _documented_census(doc: str) -> int | None:
    """The caller count the docstring states, or ``None`` when it states none."""
    match = _CENSUS_RE.search(doc)
    if match is None:
        return None
    token = match.group(1).lower()
    return _NUMBER_WORDS.get(token) or int(token)


def _external_call_sites(name: str) -> set[str]:
    """Modules outside the ACL-config module that call ``name``."""
    own = Path(inspect.getfile(_acl_config))
    package = own.parent.parent
    hits: set[str] = set()
    for path in sorted(package.rglob("*.py")):
        if path == own:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - the package parses
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == name:
                hits.add(path.name)
    return hits


@pytest.fixture
def acl_file(tmp_path: Path) -> Iterator[Path]:
    """A minimal loadable ACL file, with the cache cleared around it."""
    path = tmp_path / "acl.json5"
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "default_permission": "deny",
                "rules": [],
                "subjects": [],
                "policies": [],
            }
        ),
        encoding="utf-8",
    )
    _acl_config._clear_acl_cache_for_test()
    yield path
    _acl_config._clear_acl_cache_for_test()


class TestWhatTwoCallersOfTheCacheShare:
    """Contents, not identity - and the docstring must say which."""

    def test_two_reads_of_one_file_return_distinct_objects(self, acl_file: Path) -> None:
        """The deep-copy contract the body documents, stated as behaviour.

        Passes on both sides of the docstring fix: it is the measurement the
        docstring has to agree with, not a change in what the cache does.
        """
        first = _acl_config._load_acl_cached(acl_file)
        second = _acl_config._load_acl_cached(acl_file)
        assert first == second, (first, second)
        assert first is not second, "callers share the cache's contents, not one object"
        assert first["rules"] is not second["rules"], "the copy must reach nested containers too"

    def test_mutating_what_one_caller_got_does_not_reach_the_next(self, acl_file: Path) -> None:
        """Why identity is withheld: a caller cannot poison the cache."""
        first = _acl_config._load_acl_cached(acl_file)
        pristine = copy.deepcopy(first)
        first["rules"].append({"id": "injected"})
        # Hoisted out of the assert: the read populates the module-level cache, so
        # under ``python -O`` an assert-embedded call would be stripped with it.
        after = _acl_config._load_acl_cached(acl_file)
        assert after == pristine

    def test_the_docstring_does_not_promise_one_shared_object(self) -> None:
        """A promise of shared identity is a promise the deep-copy breaks."""
        doc = _docstring_of(_SUBJECT)
        promised = [claim for claim in ("same dict object", "the same dict", "same object") if claim in doc]
        assert promised == [], (
            f"{_SUBJECT}'s docstring promises callers {promised}, but every return path deep-copies, "
            f"so two reads of one file return distinct objects: {doc!r}"
        )


class TestTheCallerCensusIsDerivedFromTheTree:
    """A reader is told who goes through the cache, and the count is real."""

    def test_the_documented_count_matches_the_callers(self) -> None:
        doc = _docstring_of(_SUBJECT)
        stated = _documented_census(doc)
        actual = _callers_of(_SUBJECT)
        assert stated is not None, f"{_SUBJECT}'s docstring states no caller census: {doc!r}"
        assert stated == len(actual), (
            f"{_SUBJECT}'s docstring says {stated} caller(s); {len(actual)} function(s) call it: {sorted(actual)}"
        )

    def test_every_caller_is_named(self) -> None:
        doc = _docstring_of(_SUBJECT)
        unnamed = sorted(name for name in _callers_of(_SUBJECT) if name not in doc)
        assert unnamed == [], (
            f"{_SUBJECT}'s docstring enumerates its callers but does not name {unnamed}; "
            f"a reader auditing the ACL read path is handed an incomplete set"
        )

    def test_the_caller_the_mesh_start_flow_reaches_is_the_one_named_as_such(self) -> None:
        """Only ``snapshot_acl`` is wired from outside this module.

        The pair the docstring used to name - the pre-#218 shape gate and
        config builder - is reached from nowhere else, which is why the census
        had to be re-derived rather than re-worded.
        """
        assert _external_call_sites("snapshot_acl"), "premise: snapshot_acl is called from outside the module"
        for superseded in ("is_default_acl_in_use", "resolve_acl"):
            assert _external_call_sites(superseded) == set(), (
                f"{superseded} now has callers outside {Path(inspect.getfile(_acl_config)).name}; "
                f"{_SUBJECT}'s docstring says it does not"
            )


class TestTheToctouDefenceIsAttributedToWhatHoldsIt:
    """The weaker tier must not claim the guarantee the stronger one gives."""

    def test_the_docstring_points_at_the_thread_local_snapshot(self) -> None:
        sibling = _docstring_of("snapshot_acl")
        assert "by-design refresh window" in sibling, (
            "premise: snapshot_acl documents this identity-keyed tier as a by-design refresh window"
        )
        doc = _docstring_of(_SUBJECT)
        assert "_set_thread_snapshot" in doc, (
            f"{_SUBJECT} is the tier that reloads on a rewritten file, so its docstring must name what does "
            f"close the window inside one Mesh.start flow: {doc!r}"
        )
        assert "refresh" in doc, (
            f"{_SUBJECT}'s docstring must state that a rewritten file re-loads, which is the window "
            f"snapshot_acl calls by design: {doc!r}"
        )


class TestTheGradersWouldNoticeADriftedDocstring:
    """Non-vacuity: a scan that matched nothing would look like a clean tree."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Four callers read the file through here.", 4),
            ("Two callers in the same flow.", 2),
            ("11 callers reach this.", 11),
            ("Callers share the contents.", None),
        ],
    )
    def test_the_census_parser_reads_a_planted_count(self, text: str, expected: int | None) -> None:
        assert _documented_census(text) == expected

    def test_the_caller_scan_finds_the_real_readers(self) -> None:
        callers = _callers_of(_SUBJECT)
        assert len(callers) >= 2, callers
        assert "snapshot_acl" in callers, callers

    def test_the_scan_is_looking_at_the_shipped_module(self) -> None:
        subject_names = {
            node.name for node in ast.walk(_module_tree()) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert _SUBJECT in subject_names
        assert "snapshot_acl" in subject_names
