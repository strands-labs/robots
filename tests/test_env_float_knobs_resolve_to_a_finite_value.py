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

The population is derived, never listed: a function that both reads the
environment and coerces with ``float()``. Only the exemptions are named, each
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
#: added or removed does not send a contributor to edit a number.
MINIMUM_RESOLVERS = 6


#: The scanned tree, derived from the imported package rather than from a path
#: literal, and bound at module level rather than returned from a helper so that
#: ``scripts/check_whole_tree_graders.py`` can resolve this module's walk root
#: and collect it in the preflight - a grader whose root is only reachable
#: through a call argument is invisible to that derivation.
PACKAGE_ROOT = pathlib.Path(inspect.getfile(strands_robots)).parent


def _reads_the_environment(fn: ast.AST) -> bool:
    """True when *fn* really reads ``os.environ`` / ``os.getenv``."""
    for node in ast.walk(fn):
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
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in names:
                return True
            if isinstance(func, ast.Attribute) and func.attr in names:
                return True
    return False


def _classify(paths: list[pathlib.Path], root: pathlib.Path) -> dict[str, bool]:
    """Map ``module::function`` to whether it tests finiteness."""
    found: dict[str, bool] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _calls_any(node, {"float"}) and _reads_the_environment(node):
                found[f"{path.relative_to(root).as_posix()}::{node.name}"] = _calls_any(node, FINITENESS_GUARDS)
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

    def test_a_comment_mentioning_getenv_is_not_a_resolver(self, tmp_path: pathlib.Path) -> None:
        """The detector must not re-acquire the text scan's false positive."""
        (tmp_path / "commented.py").write_text(
            "def _f(raw: str) -> float:\n"
            "    # Reading them per-use parsed os.getenv on every reference.\n"
            "    return float(raw)\n",
            encoding="utf-8",
        )
        assert _scan(tmp_path) == {}
