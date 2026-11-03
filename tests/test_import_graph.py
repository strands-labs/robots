"""The package's load-time import graph has no cycle, read from the source (the check behind the CodeQL filter).

An edge is an import that runs when its module loads. An import under ``if TYPE_CHECKING:`` never runs, and one
inside a function runs at call time, after both modules have loaded, so neither can leave a module half-built.
CodeQL's ``py/cyclic-import`` counts both; this test does not, and it covers modules whose optional extra is not
installed, which ``test_modules_import_alone`` has to skip.
"""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "strands_robots"


def _modules() -> dict[str, tuple[Path, str]]:
    out = {}
    for path in PKG.rglob("*.py"):
        parts = list(path.relative_to(PKG.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
            out[".".join(parts)] = (path, ".".join(parts))
        else:
            out[".".join(parts)] = (path, ".".join(parts[:-1]))
    return out


def _load_time_imports(tree: ast.Module, package: str, known: set[str]) -> set[str]:
    found: set[str] = set()

    def target(module: str | None, level: int, name: str) -> str:
        base = package.split(".")[: len(package.split(".")) - level + 1] if level else []
        full = ".".join([*base, module] if module else base)
        return f"{full}.{name}" if f"{full}.{name}" in known else full

    def walk(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
                walk(node.orelse)
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.ClassDef)):
                for field in ("body", "orelse", "finalbody"):
                    walk(getattr(node, field, []))
                for handler in getattr(node, "handlers", []):
                    walk(handler.body)
            elif isinstance(node, ast.ImportFrom):
                found.update(target(node.module, node.level, a.name) for a in node.names)
            elif isinstance(node, ast.Import):
                found.update(a.name for a in node.names)

    walk(tree.body)
    return found & known


def _cycle(graph: dict[str, set[str]]) -> list[str]:
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str]:
        state[node] = 1
        stack.append(node)
        for nxt in sorted(graph[node]):
            if state.get(nxt) == 1:
                return [*stack[stack.index(nxt) :], nxt]
            if nxt not in state and (found := visit(nxt)):
                return found
        state[node] = 2
        stack.pop()
        return []

    for node in sorted(graph):
        if node not in state and (found := visit(node)):
            return found
    return []


def test_no_module_imports_itself_back_at_load_time() -> None:
    modules = _modules()
    graph = {
        name: _load_time_imports(ast.parse(path.read_text()), package, set(modules)) - {name}
        for name, (path, package) in modules.items()
    }
    assert len(graph) > 100
    assert _cycle(graph) == [], " -> ".join(_cycle(graph))


def test_the_check_sees_a_cycle() -> None:
    assert _cycle({"a": {"b"}, "b": {"c"}, "c": {"a"}}) == ["a", "b", "c", "a"]
    assert _cycle({"a": {"b"}, "b": set()}) == []
