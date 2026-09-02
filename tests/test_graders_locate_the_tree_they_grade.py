# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A whole-tree guard locates the tree it grades, not the directory it started in.

Many tests here assert a property over the whole repository - no host paths, no
emoji in a shipped string, no ``emergency_stop`` without an explicit port - by
walking the tree and collecting offenders. Such a sweep has one failure mode
worse than any offender it looks for: rooted at a path that does not exist, it
collects nothing, and "no offenders" is exactly what a clean tree looks like.

``Path("tests")`` and ``Path("docs")`` resolve against the process working
directory, not against the repository, so four guards graded whichever tree the
run happened to start in. Two of them then reported that tree clean. Measured by
planting a real offender in the tree under test and running the same guard twice,
changing only the working directory:

=========================================== ================== ==================
guard                                        cwd = repo root    cwd = elsewhere
=========================================== ================== ==================
``emergency_stop`` on the default port       FAILED (caught it)  **1 passed**
a doc page naming a dataset home            FAILED (caught it)  **1 passed**
=========================================== ================== ==================

The first of those is a safety guard: ``pose_tool``'s ``port`` defaults to
``/dev/ttyACM0``, so the hazard it exists to refuse is a test that de-energizes
whatever arm is plugged into the machine running the suite. It passed with that
hazard present.

Two spellings of the same mistake are graded here, because both were in the tree
and only one is silent:

* the sweep root - ``Path("docs").rglob(...)`` - which yields nothing and reads
  as compliant;
* the individual read - ``Path("strands_robots/.../recording.py").read_text()`` -
  which raises ``FileNotFoundError``. Loud, so it costs a confusing failure
  rather than a false pass, and it is still wrong: the guard names a file it did
  not read.

The rule is therefore about the *locator*, not about the failure it happens to
produce: a relative literal must not reach the filesystem. Roots derived from
``Path(__file__)`` or from a symbol the package defines
(``Path(inspect.getfile(...)).parent``) are the convention the rest of the suite
already keeps, and they answer the same whatever the working directory is.

A relative ``Path`` literal that never touches the disk is not in scope, and the
tree carries three - ``Path("out")`` compared for equality, ``Path("bare_id")``
returned by a resolver, ``Path("test_x.py").stem`` read for its name. Keying on
the filesystem call rather than on the literal is what distinguishes them, so
they need no exemption entry that could later go stale.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

#: The test tree, located from this file - the rule this module states, applied
#: to the module that states it.
_TEST_TREE = pathlib.Path(__file__).resolve().parent

#: Attribute calls that take a path to the filesystem. ``resolve`` and ``stem``
#: are deliberately absent: neither reads anything, so neither turns a relative
#: literal into a claim about a tree.
_FILESYSTEM_CALLS = frozenset(
    {
        "glob",
        "rglob",
        "iterdir",
        "walk",
        "read_text",
        "read_bytes",
        "open",
        "exists",
        "is_file",
        "is_dir",
        "stat",
    }
)


def _is_relative_path_literal(node: ast.AST) -> str | None:
    """The literal of a ``Path("relative/thing")`` call, or ``None``.

    An absolute literal names one tree wherever it runs from, so it is a
    different question and not this one.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path"):
        return None
    if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
        return None
    value = node.args[0].value
    if not isinstance(value, str) or not value or value.startswith(("/", "~")):
        return None
    return value


def _cwd_relative_locators(source: str) -> list[str]:
    """Every relative path literal in ``source`` that reaches the filesystem.

    Both spellings are resolved: the literal used directly as the receiver of a
    filesystem call, and one bound to a name that is then used as that receiver.

    Args:
        source: Python source text.

    Returns:
        ``"<line>: <literal>.<call>"`` per locator, in source order.
    """
    tree = ast.parse(source)
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        literal = _is_relative_path_literal(node.value)
        if isinstance(target, ast.Name) and literal is not None:
            bound[target.id] = literal

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in _FILESYSTEM_CALLS:
            continue
        receiver = node.func.value
        literal = _is_relative_path_literal(receiver)
        if literal is None and isinstance(receiver, ast.Name):
            literal = bound.get(receiver.id)
        if literal is not None:
            found.append((node.lineno, f"{literal!r}.{node.func.attr}()"))
    return [f"{line}: {what}" for line, what in sorted(found)]


class TestNoGuardIsRootedAtTheWorkingDirectory:
    """The inventory, over the test tree rather than over a list of files."""

    def test_no_test_module_reaches_the_filesystem_through_a_relative_literal(self) -> None:
        offenders: dict[str, list[str]] = {}
        scanned = 0
        for path in sorted(_TEST_TREE.rglob("*.py")):
            scanned += 1
            locators = _cwd_relative_locators(path.read_text(encoding="utf-8"))
            if locators:
                offenders[path.relative_to(_TEST_TREE).as_posix()] = locators
        assert scanned > 100, (
            f"the sweep read {scanned} modules, so it graded almost nothing - it is rooted at "
            f"{_TEST_TREE}, which is not this repository's test tree"
        )
        assert not offenders, (
            "these locate a tree by a path relative to the working directory, so they grade "
            "whichever tree the run started in - and a sweep that finds nothing reports the tree "
            f"clean: {offenders}. Derive the root from Path(__file__) or from a package symbol."
        )


class TestTheScannerIsCalibrated:
    """Both spellings caught, and a literal that never reads is not one."""

    @pytest.mark.parametrize(
        ("label", "source"),
        [
            ("sweep root", 'for p in sorted(Path("docs").rglob("*.md")):\n    pass\n'),
            ("single read", 'source = Path("strands_robots/x.py").read_text()\n'),
            ("bound to a name", '_DOCS = Path("docs")\n\n\ndef f():\n    return list(_DOCS.glob("*.md"))\n'),
            ("existence probe", 'if Path("examples/demo.py").exists():\n    pass\n'),
        ],
    )
    def test_a_relative_locator_is_reported(self, label: str, source: str) -> None:
        assert _cwd_relative_locators(source), f"the {label} spelling is no longer caught"

    @pytest.mark.parametrize(
        ("label", "source"),
        [
            (
                "derived from this file",
                'ROOT = Path(__file__).resolve().parents[2]\nlist((ROOT / "docs").rglob("*.md"))\n',
            ),
            ("derived from a symbol", 'p = Path(inspect.getfile(ik)).parent\nlist(p.rglob("*.py"))\n'),
            ("absolute literal", 'Path("/etc/hosts").read_text()\n'),
            ("compared, never read", 'assert cfg(result_dir=Path("out")) == cfg(result_dir="out")\n'),
            ("returned, never read", 'assert resolve("bare_id") == Path("bare_id")\n'),
            ("read for its name", 'assert "/" not in Path("test_x.py").stem\n'),
        ],
    )
    def test_a_path_that_grades_nothing_is_not_reported(self, label: str, source: str) -> None:
        """Over-reach control: the rule is about reaching the disk, not about the literal."""
        assert _cwd_relative_locators(source) == [], f"the {label} case is wrongly flagged"
