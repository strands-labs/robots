"""Mesh examples must release the Zenoh session through the attribute the SDK sets.

A mesh-enabled ``Robot`` holds a Zenoh session whose Rust runtime callbacks run
on **non-daemon** threads (six ``pyo3-closure`` threads on this build). Releasing
it is therefore not tidiness: an example that fails to call ``Mesh.stop()`` never
terminates, and the user has to interrupt it.

The observed defect: ``examples/04_mesh_peer_discovery.py`` cleaned up with
``getattr(sim, "_mesh", None)``, but :func:`strands_robots.robot.Robot` assigns
the *public* ``sim.mesh`` (and ``strands_robots.robot``'s own teardown reads
``getattr(instance, "mesh", None)`` before calling ``stop()``). ``_mesh`` never
exists, so ``getattr`` returned ``None``, the ``if mesh:`` guard skipped, and the
script whose docstring promised "~3 seconds" ran until it was killed. Because the
name was read through ``getattr`` with a default, nothing raised - the cleanup
silently did nothing.

Two rules, both derived from the SDK rather than hand-listed so they track a
rename instead of drifting from it:

1. No example may reach for a mesh attribute that nothing assigns - neither the
   factory nor the module itself. (A module that sets its own ``self._mesh`` and
   reads it back holds ordinary private state and is not affected.)
2. The peer-discovery example must actually stop the session it opened, so
   deleting the cleanup cannot pass rule 1 by having nothing to check.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_ROBOT_PY = _REPO_ROOT / "strands_robots" / "robot.py"

# Attribute names that plausibly hold the live Mesh: "mesh", "_mesh", "__mesh".
# Deliberately tight - it must not match unrelated names such as ``skip_mesh``.
_MESH_ATTR_RE = re.compile(r"^_*mesh$")


def _factory_mesh_attributes() -> set[str]:
    """Instance attributes the Robot factory assigns the live ``Mesh`` to.

    Derived from ``strands_robots/robot.py``: find the locals bound from
    ``init_mesh(...)`` (``sim_mesh`` / ``hw_mesh``), then the attributes those
    locals are assigned to (``sim.mesh`` / ``hw.mesh``). Reading the factory
    means a rename updates this rule automatically.
    """
    tree = ast.parse(_ROBOT_PY.read_text(encoding="utf-8"))
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if called == "init_mesh":
                bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}

    attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in bound:
            attrs |= {t.attr for t in node.targets if isinstance(t, ast.Attribute)}
    return attrs


def _locally_assigned_mesh_attributes(tree: ast.AST) -> set[str]:
    """Mesh-ish attributes this module assigns on its own objects.

    A module that writes ``self._mesh = mesh`` in its own ``__init__`` and then
    reads it back is holding ordinary private state - not reaching for an
    attribute the SDK was expected to provide. Those names are legitimate, so
    they are excluded from the rule below.
    """
    assigned: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute) and _MESH_ATTR_RE.match(target.attr):
                assigned.add(target.attr)
    return assigned


def _unknown_mesh_reads(source: str, known: set[str]) -> list[str]:
    """Mesh-ish attribute reads in ``source`` that nothing ever assigns.

    Flags a read only when the name is neither (a) the attribute the Robot
    factory assigns nor (b) one this module assigns itself. That is exactly the
    silent-no-op case: the value can only ever be the ``getattr`` default.
    """
    found: list[str] = []
    tree = ast.parse(source)
    known = known | _locally_assigned_mesh_attributes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _MESH_ATTR_RE.match(node.attr) and node.attr not in known:
            found.append(f"line {node.lineno}: .{node.attr}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and _MESH_ATTR_RE.match(node.args[1].value)
            and node.args[1].value not in known
        ):
            found.append(f'line {node.lineno}: getattr(..., "{node.args[1].value}")')
    return found


def _example_scripts() -> list[Path]:
    return sorted(_EXAMPLES_DIR.rglob("*.py")) if _EXAMPLES_DIR.is_dir() else []


def test_the_factory_assigns_exactly_one_public_mesh_attribute():
    """Non-vacuity: the derived rule must name the real attribute.

    An empty or renamed result would make the per-example scan below pass
    trivially, so the expected value is pinned here.
    """
    assert _factory_mesh_attributes() == {"mesh"}


@pytest.mark.parametrize("script", _example_scripts(), ids=[p.name for p in _example_scripts()])
def test_examples_only_read_the_mesh_attribute_the_factory_sets(script: Path):
    """An example must not read a mesh attribute the SDK never assigns."""
    known = _factory_mesh_attributes()
    offenders = _unknown_mesh_reads(script.read_text(encoding="utf-8"), known)
    assert not offenders, (
        f"{script.relative_to(_REPO_ROOT)} reads a mesh attribute the Robot factory "
        f"never assigns: {offenders}. The factory sets {sorted(known)} (see "
        f"strands_robots/robot.py), so this read returns None and any "
        f"`if mesh:` cleanup silently does nothing - leaving the Zenoh session's "
        f"non-daemon threads running and the example unable to exit."
    )


def test_the_peer_discovery_example_stops_the_session_it_opened():
    """The mesh example must call ``stop()`` on the attribute it read.

    Without this, deleting the cleanup entirely would satisfy the scan above
    while reintroducing the hang.
    """
    example = _EXAMPLES_DIR / "04_mesh_peer_discovery.py"
    source = example.read_text(encoding="utf-8")
    attr = sorted(_factory_mesh_attributes())[0]
    assert f'getattr(sim, "{attr}"' in source, (
        f"{example.name} must read the live mesh via getattr(sim, {attr!r}, None)."
    )
    stops = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "stop"
    ]
    assert stops, (
        f"{example.name} never calls .stop(); the Zenoh session runs on "
        f"non-daemon threads, so the script would not terminate."
    )


@pytest.mark.parametrize(
    "planted",
    [
        'mesh = getattr(sim, "_mesh", None)\nif mesh:\n    mesh.stop()\n',
        "sim._mesh.stop()\n",
    ],
    ids=["getattr-private-name", "direct-private-attribute"],
)
def test_the_scan_detects_a_planted_private_mesh_read(planted: str):
    """Meta: an empty result must mean clean examples, not a blind scanner."""
    assert _unknown_mesh_reads(planted, {"mesh"})


# Own private state: assigned in ``__init__``, then read back. Not a missing SDK
# attribute, so the rule must leave it alone (examples/fleet/dashboard.py does
# exactly this). Kept as a named constant rather than adjacent string literals so
# that a dropped comma in the list below cannot silently merge two cases into one.
_OWN_PRIVATE_MESH_ROUND_TRIP = textwrap.dedent(
    """\
    class D:
        def __init__(self, mesh):
            self._mesh = mesh

        def go(self):
            return self._mesh.peers
    """
)


@pytest.mark.parametrize(
    "clean",
    [
        'mesh = getattr(sim, "mesh", None)\nif mesh:\n    mesh.stop()\n',
        "if args.skip_mesh:\n    pass\n",
        'sim = Robot("so100", mesh=True)\n',
        _OWN_PRIVATE_MESH_ROUND_TRIP,
    ],
    ids=[
        "public-getattr",
        "unrelated-skip_mesh",
        "mesh-keyword-argument",
        "own-private-attribute-round-trip",
    ],
)
def test_the_scan_does_not_flag_correct_or_unrelated_code(clean: str):
    """The rule must not fire on the public form or on names merely containing 'mesh'."""
    assert _unknown_mesh_reads(clean, {"mesh"}) == []
