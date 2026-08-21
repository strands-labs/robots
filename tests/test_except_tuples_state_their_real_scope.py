"""An ``except`` tuple must not name a class another member already covers.

``except (FileNotFoundError, OSError)`` does not catch two things. A
``FileNotFoundError`` *is* an ``OSError``, so the tuple collapses to
``except OSError`` -- and with it every other ``OSError`` subclass:
``PermissionError``, ``IsADirectoryError``, ``TimeoutError``, the whole
``ConnectionError`` family. The narrow member contributes no scope; what it
contributes is the impression of a small superset over a handler that has a
large one. AGENTS.md > "Exception Clauses Must Be Narrow" asks for "the
smallest superset of expected exception types" and names this exact shape --
``except (ImportError, Exception)`` -- as a bug.

The prose on such a handler is where the impression turns into a claim. Before
this rule, :meth:`~strands_robots.mesh.transport.bridge_transport.BridgeTransport.put`
carried ``except (RuntimeError, ConnectionError, OSError)`` under a docstring
reading "(RuntimeError from closed session, ConnectionError from broker drop,
OSError from socket-level write) are absorbed; everything else propagates". It
named three classes and caught twenty-one. Driven through the real ``put``, a
leg raising ``NotImplementedError``, ``PermissionError``, ``RecursionError`` or
``TimeoutError`` was absorbed and the publish returned normally; only a class
outside both trees propagated. That mattered on this path in particular:
:mod:`strands_robots.mesh.sensors` re-raises ``NotImplementedError`` ahead of
its own broad handler at every sensor loop, to "surface immediately rather than
silently dropping every sensor tick", and the bridge is the transport those
publishes traverse.

Nothing refuses the shape. ``ruff`` has ``B014``, but it reports only a
*duplicate* member -- ``except (ValueError, ValueError)`` -- and says nothing
about a subclass beside its superclass; running the whole ruff catalogue
(``--select ALL``) over ``except (FileNotFoundError, OSError)`` and
``except (ImportError, Exception)`` reports neither. So AGENTS.md's "Lint/review
will catch this" described no mechanism, and twenty-five sites accumulated. They
are refused here instead, in the gate that blocks a merge.

Scope is the hierarchies the language, the standard library and this package
own. A third-party exception tree is excluded on purpose: a dependency can
re-parent its classes between releases -- ``huggingface_hub``'s
``HfHubHTTPError`` is an ``OSError`` only by way of ``httpx.HTTPError`` -- so a
handler naming both it and ``OSError`` is hedging against a hierarchy this
repository does not control, and that hedge is not this rule's business.

Scope is also redundancy only. Whether a handler's *effective* surface should be
narrower -- whether ``except OSError`` around a socket write ought to be
``except ConnectionError`` -- is a behaviour change, one per site, and this rule
says nothing about it. Removing a member another member already covers changes
no behaviour at all.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

import strands_robots

#: Trees scanned for the rule. Reached through the imported package so a layout
#: change cannot silently narrow the scan to nothing.
_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_TREES = ("strands_robots", "tests", "tests_integ", "examples")

#: A tuple below this count means the scan stopped reaching the trees, and a
#: clean result would prove nothing.
_MINIMUM_TUPLES = 200


def _owned_root(root: str) -> bool:
    """Whether ``root``'s exception hierarchy is one this repository can rely on.

    Args:
        root: The first segment of a dotted module path.

    Returns:
        ``True`` for builtins, the standard library and this package. A
        third-party root returns ``False`` and its classes are left unresolved,
        which keeps a hedge against a dependency's own re-parenting out of the
        rule.
    """
    return root == "builtins" or root == "strands_robots" or root in sys.stdlib_module_names


def _import_map(tree: ast.Module) -> dict[str, str]:
    """Map every name the module imports to its dotted path.

    Function-scope imports are included: an exception class imported inside the
    function that handles it is as much a tuple member as a module-scope one.

    Args:
        tree: A parsed module.

    Returns:
        Local name -> dotted path. Relative imports are skipped; the classes
        this rule grades are reached absolutely.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                out[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname or alias.name] = alias.name
    return out


def _resolve(expr: str, imports: dict[str, str]) -> type[BaseException] | None:
    """Resolve a tuple member to its exception class, or ``None``.

    Args:
        expr: The member as written, e.g. ``"OSError"`` or ``"json.JSONDecodeError"``.
        imports: The owning module's import map from :func:`_import_map`.

    Returns:
        The class, or ``None`` when it names a hierarchy outside
        :func:`_owned_root`, is not importable here, or is not an exception.
    """
    direct = getattr(builtins, expr, None)
    if isinstance(direct, type) and issubclass(direct, BaseException):
        return direct
    parts = expr.split(".")
    dotted = ".".join([imports.get(parts[0], parts[0]), *parts[1:]]).split(".")
    if not _owned_root(dotted[0]):
        return None
    for cut in range(len(dotted), 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(dotted[:cut]))
        except ImportError:
            continue
        for attr in dotted[cut:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, type) and issubclass(obj, BaseException):
            return obj
        break
    return None


def redundant_tuple_members(source: str) -> list[tuple[int, str]]:
    """Members of an ``except`` tuple that another member of it already covers.

    Args:
        source: Python source text.

    Returns:
        ``(lineno, "Narrow < Broad")`` per redundancy, sorted. A member this
        rule cannot resolve is skipped rather than reported.
    """
    found: list[tuple[int, str]] = []
    tree = ast.parse(source)
    imports = _import_map(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or not isinstance(node.type, ast.Tuple):
            continue
        members = [(ast.unparse(e), _resolve(ast.unparse(e), imports)) for e in node.type.elts]
        for narrow_name, narrow in members:
            for broad_name, broad in members:
                if narrow is None or broad is None or narrow is broad or narrow_name == broad_name:
                    continue
                if issubclass(narrow, broad):
                    found.append((node.lineno, f"{narrow_name} < {broad_name}"))
    return sorted(set(found))


def _scanned_files() -> list[Path]:
    """Every Python file the rule grades."""
    return sorted(p for tree in _TREES for p in (_REPO_ROOT / tree).rglob("*.py"))


def _tuple_count() -> int:
    """How many ``except`` tuples the scan reaches, for the vacuity floor."""
    total = 0
    for path in _scanned_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        total += sum(1 for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and isinstance(n.type, ast.Tuple))
    return total


class TestNoExceptTupleOverstatesItsNarrowness:
    """The headline rule, plus the vacuity floor that keeps it meaningful."""

    def test_no_tuple_names_a_class_a_sibling_already_covers(self) -> None:
        offenders: list[str] = []
        for path in _scanned_files():
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - unreadable file
                continue
            try:
                hits = redundant_tuple_members(source)
            except SyntaxError:  # pragma: no cover - unparseable file
                continue
            rel = path.relative_to(_REPO_ROOT)
            offenders.extend(f"{rel}:{line}  {pair}" for line, pair in hits)
        assert offenders == [], (
            "An `except` tuple names a class another member of the same tuple already "
            "covers, so the narrow name contributes no scope and the tuple reads as a "
            "smaller superset than the handler has. Drop the covered member - it changes "
            "no behaviour - and if the prose on the handler enumerates the tuple, correct "
            "that too. To narrow the handler's real surface instead, that is a behaviour "
            "change and needs its own justification per site. Offenders:\n  " + "\n  ".join(offenders)
        )

    def test_the_scan_reaches_the_trees_it_claims_to_grade(self) -> None:
        names = {p.relative_to(_REPO_ROOT).parts[0] for p in _scanned_files()}
        assert set(_TREES) <= names, f"the scan reached only {sorted(names)}, not {sorted(_TREES)}"
        count = _tuple_count()
        assert count >= _MINIMUM_TUPLES, (
            f"the scan graded only {count} `except` tuples; below {_MINIMUM_TUPLES} a clean "
            "result says nothing about the trees"
        )


class TestTheRuleReportsRedundancyAndNothingElse:
    """Controls: what the detector must report, and what it must leave alone."""

    @pytest.mark.parametrize(
        ("member_list", "expected"),
        [
            ("FileNotFoundError, OSError", "FileNotFoundError < OSError"),
            ("OSError, FileNotFoundError", "FileNotFoundError < OSError"),
            ("ImportError, Exception", "ImportError < Exception"),
            ("RuntimeError, ConnectionError, OSError", "ConnectionError < OSError"),
        ],
    )
    def test_a_covered_member_is_reported_whichever_order_it_is_written_in(
        self, member_list: str, expected: str
    ) -> None:
        planted = f"def f():\n    try:\n        pass\n    except ({member_list}):\n        pass\n"
        assert redundant_tuple_members(planted) == [(4, expected)]

    @pytest.mark.parametrize(
        "member_list",
        [
            "ValueError, OSError",
            "UnicodeDecodeError, json.JSONDecodeError",
            "RuntimeError, AttributeError, OSError",
            "ImportError, OSError",
        ],
    )
    def test_a_tuple_of_siblings_is_left_alone(self, member_list: str) -> None:
        planted = f"import json\ndef f():\n    try:\n        pass\n    except ({member_list}):\n        pass\n"
        assert redundant_tuple_members(planted) == []

    def test_a_stdlib_member_is_graded(self) -> None:
        planted = "import json\ndef f():\n    try:\n        pass\n    except (json.JSONDecodeError, ValueError):\n        pass\n"
        assert redundant_tuple_members(planted) == [(5, "json.JSONDecodeError < ValueError")]

    def test_a_third_party_hierarchy_is_left_to_hedge(self) -> None:
        planted = (
            "from huggingface_hub.errors import HfHubHTTPError\n"
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except (HfHubHTTPError, OSError):\n"
            "        pass\n"
        )
        assert redundant_tuple_members(planted) == [], (
            "HfHubHTTPError is an OSError only by way of httpx.HTTPError, a hierarchy "
            "this repository does not control, so naming both is a hedge the rule must "
            "not refuse"
        )

    def test_an_unresolvable_member_is_skipped_not_reported(self) -> None:
        planted = "def f():\n    try:\n        pass\n    except (NoSuchThing, OSError):\n        pass\n"
        assert redundant_tuple_members(planted) == []

    def test_a_single_class_handler_is_not_a_tuple(self) -> None:
        planted = "def f():\n    try:\n        pass\n    except OSError:\n        pass\n"
        assert redundant_tuple_members(planted) == []
