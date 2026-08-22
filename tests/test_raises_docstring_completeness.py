"""A ``Raises:`` block its own refusals outgrew licenses a handler that does not fire.

A ``Raises:`` block is a contract, not a courtesy: it is the only place a caller
learns which ``except`` clause to write. ``build_lerobot_command`` documented
``ValueError`` alone, and 28 days later a preflight was added that raises
``RuntimeError`` when the installed lerobot has no DAgger rollout entry point
(#1614, against a block written in #732). A caller who handled exactly what the
docstring licensed - ``except ValueError`` - therefore took an *escaped*
``RuntimeError``, and the only remedy was to read the source.

These tests pin the comparison for every function and method in
:mod:`strands_robots` whose docstring *already* has a ``Raises:`` section: adding
a refusal without documenting it fails here. A docstring with no ``Raises:``
section at all is out of scope, exactly as its ``Args:`` sibling in
``tests/test_args_docstring_completeness.py`` checks a block that exists rather
than demanding one.

The scope is *module-level functions as well as methods*, which matters: the
``Args:`` guard grades public methods of public classes, and the only surface
this comparison finds is a module-level function - it would have been invisible
to a population borrowed unchanged from that guard.

The check is deliberately one-directional. A documented class raised by a helper
the function delegates to is entirely legitimate and common here: 80 surfaces
document a class their own body never constructs, because a validator, a
``subprocess.run(check=True)`` or an optional-dependency shim raises it one frame
down. So only the *forward* direction is graded - a class raised in the
function's own scope must be covered by the function's own block - and the
converse is left alone rather than exempted case by case.

Three shapes are not refusals this function chose, and each is filtered with a
control that fails if the filter is dropped:

* ``raise error_holder[0]`` re-raises an exception captured elsewhere. There is
  no class name at the raise site to compare, so it is skipped.
* ``raise _factory_import_error(exc)`` calls a same-module factory that *returns*
  the exception. The class is the factory's return annotation
  (``-> ImportError``), which is what the block should - and does - name.
* ``run_multi_policy`` raises ``CooperativeStop`` and catches it in the same
  function. Documenting a class the caller can never see would be the opposite
  error, so a class caught in scope is not required.

Evidence is read permissively and claims strictly: any exception-looking token
anywhere in the block credits that class, so the scan can only ever
under-report. :class:`TestTheScanIsNonVacuous` fails if a re-narrowed scan
reports a clean sweep over less than the package.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import re
from pathlib import Path

import strands_robots as package

# Derived from an imported symbol rather than a path literal, so a moved package
# cannot leave this scanning an empty tree while reporting success.
_PACKAGE_ROOT = Path(inspect.getfile(package)).parent

# Any Google-style section header, so the Raises: block ends where the next
# section begins rather than swallowing it.
_SECTION_HEADER = re.compile(
    r"^\s*(?:Args|Arguments|Parameters|Returns|Raises|Attributes|Yields|Example|Examples"
    r"|Note|Notes|Warning|Warnings|Todo):\s*$"
)

# ``ValueError:`` / ``subprocess.CalledProcessError:`` / ``ImportError / ValueError:``
_ENTRY = re.compile(r"^\s*([A-Za-z_][\w. ,/]*?)\s*:")

# One entry label may name several classes that share a description.
_LABEL_SPLIT = re.compile(r"\s*(?:/|,|\bor\b)\s*")

# A dotted or bare identifier - a prose entry ("Nothing. An audit write ...") is
# not a class reference and contributes nothing.
_CLASS_REF = re.compile(r"^[A-Za-z_][\w.]*$")

# Permissive evidence: a class named anywhere in the block, including inside a
# wrapped description, still counts as documented.
_EXCEPTION_TOKEN = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception|Warning))\b")

_BUILTIN_EXCEPTIONS = frozenset(
    name
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type) and issubclass(getattr(builtins, name), BaseException)
)


def documented_exceptions(doc: str) -> tuple[frozenset[str], bool]:
    """Classes ``doc``'s ``Raises:`` block names, and whether it has one.

    Args:
        doc: A dedented docstring (as :func:`ast.get_docstring` returns).

    Returns:
        ``(names, has_section)``. ``names`` holds the last dotted segment of
        every class the block references - from an entry label, a combined
        label, or an exception-looking token in any description - so it is a
        superset of the strictly-parsed labels. It is empty when there is no
        ``Raises:`` section, so callers must consult ``has_section``.
    """
    lines = doc.splitlines()
    header_indent = None
    body: list[str] = []
    for index, line in enumerate(lines):
        if _SECTION_HEADER.match(line) and line.strip().startswith("Raises"):
            header_indent = len(line) - len(line.lstrip())
            body = lines[index + 1 :]
            break
    if header_indent is None:
        return frozenset(), False

    block: list[str] = []
    for line in body:
        if not line.strip():
            block.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= header_indent:
            break  # section ended (next header or a dedented paragraph)
        block.append(line)

    names: set[str] = set()
    entry_indent = None
    for line in block:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if entry_indent is None:
            entry_indent = indent
        if indent == entry_indent:
            match = _ENTRY.match(line)
            if match:
                for part in _LABEL_SPLIT.split(match.group(1)):
                    cleaned = part.strip()
                    if _CLASS_REF.match(cleaned):
                        names.add(cleaned.rsplit(".", 1)[-1])
    # Permissive second pass over the whole block, descriptions included.
    for token in _EXCEPTION_TOKEN.findall("\n".join(block)):
        names.add(token.rsplit(".", 1)[-1])
    return frozenset(names), True


def _exception_factories(tree: ast.Module) -> dict[str, str]:
    """Module-level functions that *return* an exception, by return annotation.

    ``raise _factory_import_error(exc)`` raises whatever the factory returns, so
    the class the block must name is the annotation rather than the callee.
    """
    factories = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
            factories[node.name] = ast.unparse(node.returns).rsplit(".", 1)[-1]
    return factories


def own_raises(fn: ast.FunctionDef | ast.AsyncFunctionDef, factories: dict[str, str]) -> list[tuple[int, str]]:
    """Classes raised in ``fn``'s own scope, as ``(lineno, class name)``.

    A nested ``def``/``class``/``lambda`` is skipped: its refusals belong to its
    own docstring. A bare ``raise`` and a raise of a non-name expression are
    skipped too - neither names a class at the raise site.
    """
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
                continue
            if isinstance(child, ast.Raise) and child.exc is not None:
                target = child.exc.func if isinstance(child.exc, ast.Call) else child.exc
                if isinstance(target, ast.Name | ast.Attribute):
                    name = ast.unparse(target).rsplit(".", 1)[-1]
                    found.append((child.lineno, factories.get(name, name)))
            visit(child)

    visit(fn)
    return found


def caught_in_scope(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    """Classes ``fn`` catches anywhere inside itself.

    A class raised and caught in one function never reaches the caller, so
    requiring an entry for it would be the opposite error.
    """
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            parts = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            for part in parts:
                names.add(ast.unparse(part).rsplit(".", 1)[-1])
    return frozenset(names)


def _covered(raised: str, documented: frozenset[str]) -> bool:
    """Whether ``documented`` names ``raised`` or a builtin superclass of it.

    Documenting ``OSError`` for a ``FileNotFoundError`` is a complete promise, so
    a documented superclass credits the subclass. Only builtins are resolved:
    a package class is compared by name, which is what the block writes.
    """
    if raised in documented:
        return True
    raised_cls = getattr(builtins, raised, None)
    if not isinstance(raised_cls, type):
        return False
    for name in documented:
        documented_cls = getattr(builtins, name, None)
        if isinstance(documented_cls, type) and issubclass(raised_cls, documented_cls):
            return True
    return False


def uncovered_refusals(root: Path) -> list[tuple[str, list[str]]]:
    """Every surface with a ``Raises:`` block that raises a class it does not name.

    Args:
        root: Package directory to walk. Every ``*.py`` beneath it is parsed by
            AST, so no optional backend needs to be importable.

    Returns:
        ``(surface_id, missing)`` pairs, where ``surface_id`` is
        ``"<relative path>:<lineno> <qualified name>"``.
    """
    findings = []
    for surface, documented, raised, caught, where in _surfaces(root):
        missing = sorted(
            {
                f"{name} (line {lineno})"
                for lineno, name in raised
                if name not in caught and not _covered(name, documented)
            }
        )
        if missing:
            findings.append((f"{where} {surface}", missing))
    return findings


def _surfaces(
    root: Path,
) -> list[tuple[str, frozenset[str], list[tuple[int, str]], frozenset[str], str]]:
    """Every function under ``root`` whose docstring has a ``Raises:`` block."""
    out = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        factories = _exception_factories(tree)
        owners: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                        owners[id(child)] = f"{node.name}."
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(fn)
            if not doc:
                continue
            documented, has_section = documented_exceptions(doc)
            if not has_section:
                continue
            rel = path.relative_to(root.parent)
            qualified = f"{owners.get(id(fn), '')}{fn.name}"
            out.append((qualified, documented, own_raises(fn, factories), caught_in_scope(fn), f"{rel}:{fn.lineno}"))
    return out


_SURFACES = _surfaces(_PACKAGE_ROOT)


def test_every_refusal_a_function_makes_is_in_its_own_raises_block() -> None:
    """A class raised here and named nowhere is a handler the caller cannot write."""
    findings = uncovered_refusals(_PACKAGE_ROOT)
    assert not findings, "\n".join(f"{surface} raises {missing} with no Raises: entry" for surface, missing in findings)


class TestTheDaggerPreflightIsDocumented:
    """The surface this guard was written for, pinned by name.

    Kept alongside the sweep: the sweep would still pass if this block were
    rewritten to document nothing at all, because a block that exists but names
    no class is out of the forward direction's scope.
    """

    def test_the_teleoperate_builder_documents_both_classes_it_raises(self) -> None:
        source = _PACKAGE_ROOT / "tools" / "lerobot_teleoperate.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        builder = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_lerobot_command"
        )
        doc = ast.get_docstring(builder)
        assert doc is not None
        documented, has_section = documented_exceptions(doc)
        assert has_section, "build_lerobot_command lost its Raises: block"
        assert {"ValueError", "RuntimeError"} <= documented, f"documents only {sorted(documented)}"


class TestTheFiltersAreLoadBearing:
    """Each filter is calibrated against the shipped shape that needs it."""

    def test_a_re_raised_captured_exception_names_no_class(self) -> None:
        """``raise error_holder[0]`` has no class name at the raise site."""
        source = "\n".join(
            [
                "def f(holder):",
                '    """Do a thing.',
                "",
                "    Raises:",
                "        TimeoutError: If it times out.",
                '    """',
                "    raise holder[0]",
            ]
        )
        tree = ast.parse(source)
        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert own_raises(fn, {}) == []

    def test_a_factory_call_is_read_as_its_return_annotation(self) -> None:
        """``raise _make(exc)`` where ``_make`` is ``-> ImportError`` raises that."""
        source = "\n".join(
            [
                "def _make(exc) -> ImportError:",
                "    return ImportError(str(exc))",
                "",
                "def f():",
                '    """Do a thing.',
                "",
                "    Raises:",
                "        ImportError: If a dependency is absent.",
                '    """',
                "    raise _make(ValueError())",
            ]
        )
        tree = ast.parse(source)
        factories = _exception_factories(tree)
        assert factories == {"_make": "ImportError"}
        fn = tree.body[1]
        assert isinstance(fn, ast.FunctionDef)
        assert [name for _, name in own_raises(fn, factories)] == ["ImportError"]

    def test_a_class_caught_in_the_same_function_is_not_required(self) -> None:
        """A refusal the caller never sees must not be demanded in the block."""
        source = "\n".join(
            [
                "def f():",
                '    """Do a thing.',
                "",
                "    Raises:",
                "        RuntimeError: If the world is gone.",
                '    """',
                "    try:",
                "        raise CooperativeStop()",
                "    except CooperativeStop:",
                "        return None",
            ]
        )
        tree = ast.parse(source)
        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert "CooperativeStop" in caught_in_scope(fn)

    def test_a_documented_superclass_credits_the_subclass(self) -> None:
        assert _covered("FileNotFoundError", frozenset({"OSError"}))
        assert not _covered("ValueError", frozenset({"OSError"}))

    def test_a_nested_def_keeps_its_own_refusals(self) -> None:
        """An inner function's raise belongs to the inner docstring."""
        source = "\n".join(
            [
                "def outer():",
                '    """Do a thing.',
                "",
                "    Raises:",
                "        ValueError: If asked to.",
                '    """',
                "    def inner():",
                "        raise KeyError('nope')",
                "",
                "    raise ValueError('asked')",
            ]
        )
        tree = ast.parse(source)
        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert [name for _, name in own_raises(fn, {})] == ["ValueError"]


class TestTheParserReadsShippedBlocks:
    """Calibrated against real label shapes rather than an invented grammar."""

    def test_a_dotted_label_is_read_as_its_class(self) -> None:
        doc = "Do a thing.\n\nRaises:\n    subprocess.CalledProcessError: If git fails.\n"
        documented, has_section = documented_exceptions(doc)
        assert has_section
        assert "CalledProcessError" in documented

    def test_a_slash_combined_label_documents_every_class(self) -> None:
        doc = "Do a thing.\n\nRaises:\n    ImportError / ValueError: If either.\n"
        documented, _ = documented_exceptions(doc)
        assert {"ImportError", "ValueError"} <= documented

    def test_a_class_named_only_in_a_description_still_counts(self) -> None:
        """Evidence is read permissively, so the scan can only under-report."""
        doc = "Do a thing.\n\nRaises:\n    ValueError: ... and a RuntimeError when busy.\n"
        documented, _ = documented_exceptions(doc)
        assert {"ValueError", "RuntimeError"} <= documented

    def test_a_prose_entry_names_no_class(self) -> None:
        doc = "Do a thing.\n\nRaises:\n    Nothing. A failed write is logged and swallowed.\n"
        documented, has_section = documented_exceptions(doc)
        assert has_section
        assert not documented

    def test_the_block_ends_at_the_next_section(self) -> None:
        doc = "Do a thing.\n\nRaises:\n    ValueError: If asked.\n\nReturns:\n    A KeyError-free list.\n"
        documented, _ = documented_exceptions(doc)
        assert documented == frozenset({"ValueError"})

    def test_a_docstring_with_no_raises_block_is_out_of_scope(self) -> None:
        documented, has_section = documented_exceptions("Do a thing.\n\nReturns:\n    None.\n")
        assert not has_section
        assert not documented


class TestTheScanIsNonVacuous:
    """A mis-rooted or over-filtered scan must fail rather than sweep nothing."""

    def test_the_scan_root_is_the_whole_package(self) -> None:
        assert _PACKAGE_ROOT.name == "strands_robots"
        assert (_PACKAGE_ROOT / "simulation" / "base.py").is_file()
        assert (_PACKAGE_ROOT / "tools" / "lerobot_teleoperate.py").is_file()

    def test_enough_surfaces_are_scanned_to_be_meaningful(self) -> None:
        assert len(_SURFACES) >= 150, f"only {len(_SURFACES)} surfaces scanned"

    def test_the_population_includes_module_level_functions(self) -> None:
        """The Args: guard grades methods only; the surface this found is not one."""
        module_level = [surface for surface, _, _, _, _ in _SURFACES if "." not in surface]
        assert len(module_level) >= 30, f"only {len(module_level)} module-level functions scanned"

    def test_a_planted_undocumented_refusal_is_reported(self, tmp_path: Path) -> None:
        """A clean sweep means the blocks are complete, not that nothing is graded."""
        pkg = tmp_path / "strands_robots"
        pkg.mkdir()
        (pkg / "planted.py").write_text(
            "\n".join(
                [
                    "def f(value):",
                    '    """Do a thing.',
                    "",
                    "    Raises:",
                    "        ValueError: If value is bad.",
                    '    """',
                    "    if value is None:",
                    "        raise RuntimeError('busy')",
                    "    raise ValueError('bad')",
                ]
            ),
            encoding="utf-8",
        )
        findings = uncovered_refusals(pkg)
        assert len(findings) == 1, findings
        surface, missing = findings[0]
        assert "f" in surface
        assert any("RuntimeError" in entry for entry in missing), missing

    def test_the_builtin_exception_table_is_populated(self) -> None:
        assert {"ValueError", "RuntimeError", "OSError"} <= _BUILTIN_EXCEPTIONS
