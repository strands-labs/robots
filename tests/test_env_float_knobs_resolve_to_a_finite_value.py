# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""No environment variable resolves to a non-finite float anywhere in the package.

``float()`` accepts ``"nan"``, ``"inf"``, ``"Infinity"`` and an overflowing
``"1e999"``, and no comparison-based bound recovers from that: a ``> 0`` floor is
``True`` for ``inf`` and ``False`` for ``nan``, so a positivity test lets the
first through as a resolved knob and refuses the second only by accident. Every
resolver that reads a float out of the environment therefore has to test
finiteness itself, and the library states that rule in two shapes - a direct
:func:`math.isfinite`, as :func:`strands_robots.mesh.security._env_pos_float`
does, or the shared numeric domain
:func:`strands_robots.utils.finite_number_error`, as
:func:`strands_robots.tools.robot_mesh._gateway_discovery_wait_s` does.

The rule is package-wide but was graded per package: ``tests/mesh`` held a sweep
rooted at ``strands_robots/mesh`` whose docstring said "a sixth resolver cannot
ship without the test". A resolver rooted anywhere else was not a sixth resolver
that failed the sweep - it was invisible to it, so the sweep read as a clean
tree. ``simulation/isaac/simulation.py::_env_float`` was exactly that, and it
admitted ``inf`` for the Isaac idle live-preview period. This sweep replaces the
mesh-rooted one and walks the whole package, so its root is the scope of the rule
it enforces.

A resolver's *position* is the same kind of blind spot as its root. The
population was every such **function**, and a knob resolved by a module-level
statement is not a function that failed the scan - it is invisible to it, so the
sweep read as a clean tree again. That position is the one where an unusable
value costs the most: the coercion runs while the module body executes, so a
typo does not degrade one knob but raises ``ValueError`` out of the import, from
a frame that names ``float`` rather than the variable. Module-level statements
are therefore classified too, keyed ``module::<module>``, and a knob resolved
through a resolver leaves that population by construction - it no longer
coerces in the statement at all.

The population is derived, never listed: a function - or a module-level
statement - that both reads the environment and coerces with ``float()``. Only the exemptions are named, each
with the reason it is not a resolver-level domain, and
:meth:`TestEveryEnvFloatResolverIsFinitenessBounded.test_every_exemption_is_still_discovered`
fails when one stops matching, so an exemption cannot outlive the code it
describes and quietly cover a new site.

The detector is structural rather than textual because the safety handlers
mention ``os.getenv`` in comments explaining why they cache their knobs, and a
text scan reads those comments as env-float sites.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

import strands_robots

#: The two spellings of "this resolver tested finiteness". ``isfinite`` is the
#: direct form; ``finite_number_error`` is the shared numeric domain, which
#: decides numeric-ness and finiteness together and returns the operator-facing
#: reason. A resolver satisfying either has the bound.
FINITENESS_GUARDS = frozenset({"isfinite", "finite_number_error"})

#: Sites that read a float out of the environment but are not the place its
#: domain is decided, each with the reason. Kept as small as the tree allows: an
#: exemption is a promise that some *other* named surface refuses the value.
NOT_A_RESOLVER_DOMAIN = {
    # Returns the raw value to a single caller, ``VeraConfig.__post_init__``,
    # which refuses a non-finite ``motion_plan_scale`` on the *effective* value
    # through utils.positive_finite_number_error - deliberately there, because
    # that funnel is also the only place a keyword-supplied value can be
    # refused. Pinned behaviourally by
    # tests/policies/vera/test_vera_motion_plan_scale_domain.py.
    "policies/vera/config.py::_env_float": "checked on the effective value at the VeraConfig funnel",
}

#: Resolvers the sweep must keep finding, one per top-level area, so a scan that
#: silently stopped matching - or one rooted at a single package again - fails
#: here rather than passing over nothing.
LANDMARK_RESOLVERS = (
    "mesh/core.py::_parse_positive_float_env",
    "simulation/isaac/simulation.py::_env_float",
    "tools/robot_mesh.py::_gateway_discovery_wait_s",
)

#: Non-vacuity floor, well below the count measured on this tree, so a resolver
#: added or removed does not send a contributor to edit a number. Counted over
#: the function population alone: a module-level knob resolved correctly leaves
#: the population entirely, so a floor over the whole classification would be a
#: floor on how many modules still coerce at import.
MINIMUM_RESOLVERS = 6

#: Stands in for the function name in a module-level entry's key. Not a legal
#: Python identifier, so it cannot collide with a function called this.
MODULE_SCOPE = "<module>"


#: The scanned tree, derived from the imported package rather than from a path
#: literal, and bound at module level rather than returned from a helper so that
#: ``scripts/check_whole_tree_graders.py`` can resolve this module's walk root
#: and collect it in the preflight - a grader whose root is only reachable
#: through a call argument is invisible to that derivation.
PACKAGE_ROOT = pathlib.Path(inspect.getfile(strands_robots)).parent


#: Node types whose bodies belong to a different member of the population. A
#: module-level statement must not be credited with a guard - nor charged with a
#: coercion - that lives inside a function it merely encloses, and a class body
#: is only ever a container for methods the function scan already classifies.
_OWN_SCOPE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _within_one_scope(node: ast.AST) -> list[ast.AST]:
    """*node* and its descendants, stopping at anything that owns its own scope."""
    seen: list[ast.AST] = []
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        seen.append(current)
        stack.extend(child for child in ast.iter_child_nodes(current) if not isinstance(child, _OWN_SCOPE))
    return seen


def _reads_the_environment(fn: ast.AST) -> bool:
    """True when *fn* really reads ``os.environ`` / ``os.getenv``."""
    for node in _within_one_scope(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if node.func.attr == "getenv" and isinstance(value, ast.Name) and value.id == "os":
                return True
            if node.func.attr == "get" and isinstance(value, ast.Attribute) and value.attr == "environ":
                return True
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
            return True
    return False


def _calls_any(fn: ast.AST, names: frozenset[str] | set[str]) -> bool:
    """True when *fn* calls any function in *names*, bare or as an attribute."""
    for node in _within_one_scope(fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in names:
                return True
            if isinstance(func, ast.Attribute) and func.attr in names:
                return True
    return False


def _classify(paths: list[pathlib.Path], root: pathlib.Path) -> dict[str, bool]:
    """Map ``module::function`` - and ``module::<module>`` - to whether it tests finiteness.

    A module contributes at most one ``<module>`` entry however many statements
    resolve knobs there, and it is bounded only when every one of them is: the
    import either survives an operator's typo or it does not, so one unguarded
    statement is the whole module's answer.
    """
    found: dict[str, bool] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        name = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _calls_any(node, {"float"}) and _reads_the_environment(node):
                found[f"{name}::{node.name}"] = _calls_any(node, FINITENESS_GUARDS)
        for statement in tree.body:
            if isinstance(statement, _OWN_SCOPE):
                continue
            if _calls_any(statement, {"float"}) and _reads_the_environment(statement):
                bounded = _calls_any(statement, FINITENESS_GUARDS)
                key = f"{name}::{MODULE_SCOPE}"
                found[key] = found.get(key, True) and bounded
    return found


def _resolvers() -> dict[str, bool]:
    """Every env-float resolver in the package, and whether each is bounded."""
    return _classify(sorted(PACKAGE_ROOT.rglob("*.py")), PACKAGE_ROOT)


def _scan(root: pathlib.Path) -> dict[str, bool]:
    """The same classification over an arbitrary tree, for planted source."""
    return _classify(sorted(root.rglob("*.py")), root)


class TestEveryEnvFloatResolverIsFinitenessBounded:
    """The rule holds over the package, not over one of its subpackages."""

    def test_no_resolver_admits_a_non_finite_value(self) -> None:
        adrift = sorted(
            name for name, guarded in _resolvers().items() if not guarded and name not in NOT_A_RESOLVER_DOMAIN
        )
        assert adrift == [], (
            "these read a float out of the environment and test neither math.isfinite nor the shared "
            f"utils.finite_number_error domain, so nan/inf/1e999 resolve to a knob no consumer can honor: {adrift}. "
            "Test finiteness there, or add the site to NOT_A_RESOLVER_DOMAIN naming the surface that does."
        )

    def test_every_exemption_is_still_discovered(self) -> None:
        """An exemption for code that moved would silently cover a new site."""
        stale = sorted(set(NOT_A_RESOLVER_DOMAIN) - set(_resolvers()))
        assert stale == [], f"exemptions matching no discovered resolver: {stale}"

    @pytest.mark.parametrize("landmark", LANDMARK_RESOLVERS)
    def test_the_scan_still_finds_each_landmark(self, landmark: str) -> None:
        """One landmark per area: a scan rooted at one package fails here."""
        assert landmark in _resolvers()

    def test_the_scan_is_not_vacuous(self) -> None:
        found = _resolvers()
        assert len(found) >= MINIMUM_RESOLVERS, f"only {len(found)} resolvers discovered: {sorted(found)}"


class TestTheDetectorAnswersOnPlantedSource:
    """A sweep that cannot fail, or cannot pass, proves nothing."""

    def test_an_unguarded_resolver_is_reported(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "planted.py").write_text(
            'import os\n\n\ndef _resolve(name: str) -> float:\n    return float(os.getenv(name, "1"))\n',
            encoding="utf-8",
        )
        assert _scan(tmp_path) == {"planted.py::_resolve": False}

    @pytest.mark.parametrize(
        ("label", "body"),
        [
            ("isfinite", "    value = float(os.environ[name])\n    return value if math.isfinite(value) else 1.0\n"),
            (
                "shared_domain",
                "    value = float(os.environ[name])\n"
                '    return 1.0 if finite_number_error(value, name, "planted") else value\n',
            ),
        ],
    )
    def test_either_guard_spelling_is_accepted(self, tmp_path: pathlib.Path, label: str, body: str) -> None:
        """Both forms the library uses count; neither is privileged."""
        (tmp_path / "planted.py").write_text(
            f"import math\nimport os\n\n\ndef _resolve(name: str) -> float:\n{body}",
            encoding="utf-8",
        )
        assert _scan(tmp_path) == {"planted.py::_resolve": True}, label

    def test_a_module_level_resolution_is_reported(self, tmp_path: pathlib.Path) -> None:
        """The position an unusable value raises out of the import from."""
        (tmp_path / "planted.py").write_text(
            'import os\n\nTTL = float(os.getenv("TTL", "300"))\n',
            encoding="utf-8",
        )
        assert _scan(tmp_path) == {"planted.py::<module>": False}

    def test_a_module_level_knob_read_through_a_resolver_is_not_in_the_population(self, tmp_path: pathlib.Path) -> None:
        """Routing the knob is the remedy, and it leaves nothing to grade."""
        (tmp_path / "planted.py").write_text(
            "import math\nimport os\n\n\n"
            "def _env_float(name: str, default: float) -> float:\n"
            "    value = float(os.getenv(name, str(default)))\n"
            "    return value if math.isfinite(value) else default\n\n\n"
            'TTL = _env_float("TTL", 300.0)\n',
            encoding="utf-8",
        )
        assert _scan(tmp_path) == {"planted.py::_env_float": True}

    def test_one_unguarded_statement_answers_for_the_whole_module(self, tmp_path: pathlib.Path) -> None:
        """A guarded sibling does not vouch for the statement beside it."""
        (tmp_path / "planted.py").write_text(
            "import math\nimport os\n\n"
            'A = float(os.getenv("A", "1")) if math.isfinite(float(os.getenv("A", "1"))) else 1.0\n'
            'B = float(os.getenv("B", "2"))\n',
            encoding="utf-8",
        )
        assert _scan(tmp_path) == {"planted.py::<module>": False}

    def test_a_statement_is_not_credited_with_a_guard_from_a_scope_it_encloses(self, tmp_path: pathlib.Path) -> None:
        """A guard inside an enclosed ``def`` belongs to that ``def``, not to its host.

        The coercion here runs at import and is unguarded; the ``isfinite`` sits in
        a function the block merely contains, and is never applied to it. A walk
        that descends into the nested scope reads the block as bounded.
        """
        (tmp_path / "planted.py").write_text(
            "import math\nimport os\n\n"
            "if True:\n"
            '    TTL = float(os.getenv("TTL", "300"))\n\n'
            "    def _unrelated(value: float) -> bool:\n"
            "        return math.isfinite(value)\n",
            encoding="utf-8",
        )
        assert _scan(tmp_path) == {"planted.py::<module>": False}

    def test_a_resolver_is_not_credited_with_a_guard_from_its_own_closure(self, tmp_path: pathlib.Path) -> None:
        """The same rule one scope down: a helper defined but not applied is not a bound."""
        (tmp_path / "planted.py").write_text(
            "import math\nimport os\n\n\n"
            "def _resolve(name: str) -> float:\n"
            "    def _ok(value: float) -> bool:\n"
            "        return math.isfinite(value)\n\n"
            '    return float(os.getenv(name, "1"))\n',
            encoding="utf-8",
        )
        assert _scan(tmp_path) == {"planted.py::_resolve": False}

    def test_a_comment_mentioning_getenv_is_not_a_resolver(self, tmp_path: pathlib.Path) -> None:
        """The detector must not re-acquire the text scan's false positive."""
        (tmp_path / "commented.py").write_text(
            "def _f(raw: str) -> float:\n"
            "    # Reading them per-use parsed os.getenv on every reference.\n"
            "    return float(raw)\n",
            encoding="utf-8",
        )
        assert _scan(tmp_path) == {}
