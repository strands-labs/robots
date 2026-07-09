"""Tests for the package-root lazy-import contract in ``strands_robots/__init__.py``.

``import strands_robots`` must stay cheap: heavy symbols (Robot, Simulation,
Gr00tPolicy, tools, ...) are resolved on first attribute access via PEP 562
``__getattr__``, while light symbols (Policy, MockPolicy, create_policy) import
eagerly. These tests pin the observable behavior of that loader:

- light symbols are importable with no extra dependencies,
- lazy symbols resolve, get cached, and return a stable identity,
- an unknown attribute raises ``AttributeError`` with the standard message,
- a lazy symbol whose backing module is missing warns and raises
  ``AttributeError`` chained from the original ``ImportError`` (so callers
  without an optional extra get a clear, recoverable failure).
"""

import ast
import warnings
from pathlib import Path

import pytest

import strands_robots


class TestEagerLightSymbols:
    """Light-weight policy symbols import without torch/lerobot/mujoco."""

    def test_policy_symbols_available_immediately(self):
        # These are real top-level imports, not lazy entries.
        assert "Policy" in strands_robots.__all__
        assert isinstance(strands_robots.MockPolicy, type)
        assert callable(strands_robots.create_policy)

    def test_light_symbols_are_not_lazy_entries(self):
        for name in ("Policy", "MockPolicy", "create_policy"):
            assert name not in strands_robots._LAZY_IMPORTS


class TestLazyResolution:
    """First attribute access resolves, caches, and returns stable identity."""

    def test_lazy_symbol_resolves_and_caches(self):
        # list_robots is registry-backed (no torch) so it always resolves.
        assert "list_robots" in strands_robots._LAZY_IMPORTS

        resolved = strands_robots.list_robots
        assert callable(resolved)

        # After first access the name is cached in the module dict so
        # __getattr__ is not invoked again.
        assert "list_robots" in vars(strands_robots)

        # Subsequent access returns the same object identity.
        assert strands_robots.list_robots is resolved

    def test_every_lazy_name_is_exported(self):
        # __all__ and _LAZY_IMPORTS must not drift apart: every lazy symbol
        # is part of the public surface.
        for name in strands_robots._LAZY_IMPORTS:
            assert name in strands_robots.__all__


class TestUnknownAttribute:
    """Unknown attributes raise AttributeError with the standard message."""

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError, match="has no attribute 'does_not_exist'"):
            strands_robots.does_not_exist

    def test_dunder_attribute_raises_attributeerror(self):
        # Spurious dunder lookups (e.g. by copy/pickle) must not be swallowed.
        with pytest.raises(AttributeError):
            strands_robots.__wrapped__


class TestMissingDependencyContract:
    """A lazy symbol backed by an unimportable module warns then raises.

    Simulates the "optional extra not installed" path without uninstalling
    anything: register a temporary lazy entry pointing at a non-existent
    module, then assert the warn-and-raise contract.
    """

    def test_missing_module_warns_and_raises_chained_attributeerror(self):
        sentinel = "FakeMissingSymbolForTest"
        strands_robots._LAZY_IMPORTS[sentinel] = (
            "strands_robots._this_module_does_not_exist",
            "Thing",
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pytest.raises(AttributeError) as excinfo:
                    getattr(strands_robots, sentinel)

            # AttributeError carries the requested name and is chained from
            # the underlying ImportError so callers can introspect the cause.
            assert excinfo.value.args[0] == sentinel
            assert isinstance(excinfo.value.__cause__, ImportError)

            messages = [str(w.message) for w in caught]
            assert any(f"{sentinel} not available (missing dependencies)" in m for m in messages)
        finally:
            strands_robots._LAZY_IMPORTS.pop(sentinel, None)
            # The failed access must not leave a cached entry behind.
            assert sentinel not in vars(strands_robots)


class TestStaticExportContract:
    """Every ``__all__`` name must be statically resolvable.

    CodeQL's ``py/undefined-export`` (and most type-checkers) require that a
    name listed in ``__all__`` is defined in the module namespace by static
    analysis. The package resolves heavy symbols lazily via ``__getattr__``,
    which is invisible to a static analyzer, so each lazy name is also imported
    inside an ``if TYPE_CHECKING:`` block. This test pins that contract: a lazy
    symbol added to ``_LAZY_IMPORTS``/``__all__`` without a matching
    ``TYPE_CHECKING`` import would otherwise only be caught later by CodeQL.
    """

    @staticmethod
    def _type_checking_imported_names() -> set[str]:
        source = Path(strands_robots.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        names: set[str] = set()
        for node in ast.walk(tree):
            # Match the top-level ``if TYPE_CHECKING:`` guard.
            if isinstance(node, ast.If):
                test = node.test
                is_type_checking = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                    isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                )
                if not is_type_checking:
                    continue
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.ImportFrom):
                        for alias in stmt.names:
                            names.add(alias.asname or alias.name)
        return names

    def test_every_lazy_name_has_a_type_checking_import(self):
        type_checking_names = self._type_checking_imported_names()
        missing = sorted(name for name in strands_robots._LAZY_IMPORTS if name not in type_checking_names)
        assert not missing, (
            "Lazy symbols missing a TYPE_CHECKING import (CodeQL py/undefined-export "
            f"will flag these as exported-but-undefined): {missing}"
        )


class TestToolsReexportedAtTopLevel:
    """Every ``@tool`` in ``strands_robots.tools`` is re-exported at the package root.

    Agents and tool loaders address tools as ``strands_robots:<tool>`` (the
    top-level package), so a tool present in ``strands_robots.tools.__all__``
    but absent from the package root fails to load with
    ``module 'strands_robots' has no attribute '<tool>'``. This pins parity so
    a newly added tool cannot drift out of the top-level surface.
    """

    def test_every_tool_is_in_top_level_all(self):
        import strands_robots.tools as tools_pkg

        missing = sorted(name for name in tools_pkg.__all__ if name not in strands_robots.__all__)
        assert not missing, f"tools missing from top-level strands_robots.__all__: {missing}"

    def test_every_tool_resolves_from_top_level(self):
        import importlib

        import strands_robots.tools as tools_pkg

        # Each tool lives in a submodule named after it, defining an attribute
        # of the same name. The top-level lazy export must resolve to that same
        # object (not, for example, the submodule).
        for name in tools_pkg.__all__:
            submodule = importlib.import_module(f"strands_robots.tools.{name}")
            assert getattr(strands_robots, name) is getattr(submodule, name), name


class TestPolicyFactorySymbolsReexported:
    """The policy factory's constructor *and* its discovery/registration peers
    are all re-exported at the package root.

    ``create_policy`` is the eager top-level entry point, but an agent that
    reaches it also needs the two calls that make it usable blind:
    ``list_providers()`` (what can I pass to ``create_policy``?) and
    ``register_policy()`` (add my own). All three live in the light-weight
    ``strands_robots.policies.factory`` module (no torch/lerobot), so they
    import eagerly alongside ``create_policy`` rather than lazily. This pins
    that the discovery counterparts sit next to the constructor instead of
    forcing a reach into the ``strands_robots.policies`` subpackage.
    """

    def test_discovery_symbols_in_top_level_all(self):
        for name in ("create_policy", "list_providers", "register_policy"):
            assert name in strands_robots.__all__

    def test_discovery_symbols_are_eager_not_lazy(self):
        # Same provenance as create_policy: light imports, never lazy entries.
        for name in ("list_providers", "register_policy"):
            assert name not in strands_robots._LAZY_IMPORTS

    def test_discovery_symbols_resolve_to_policies_objects(self):
        import strands_robots.policies as policies_pkg

        for name in ("create_policy", "list_providers", "register_policy"):
            assert getattr(strands_robots, name) is getattr(policies_pkg, name), name

    def test_list_providers_callable_from_top_level(self):
        # The whole point: call it straight off the package root.
        providers = strands_robots.list_providers()
        assert isinstance(providers, (list, tuple, set))
        assert "mock" in providers


class TestPolicyTypeDiscoveryReexported:
    """``list_policy_types`` is re-exported as a peer of ``list_providers``.

    ``list_providers()`` answers "which providers can I pass to
    ``create_policy``?" and resolves ``lerobot_local``. The blind follow-up is
    "which ``policy_type`` strings does ``lerobot_local`` accept?", answered by
    ``list_policy_types()``. Pre-fix that discovery surface was reachable only
    via the buried ``strands_robots.policies.lerobot_local`` submodule, so an
    agent that found the provider had no peer-level way to enumerate its types.
    These pin the surface at both the package root and ``strands_robots.policies``.

    It is exported *lazily* (in ``_LAZY_IMPORTS``, not eagerly like
    ``list_providers``) because the ``lerobot_local`` package import chain pulls
    in torch -- resolving on first access keeps the package imports torch-free.
    """

    def test_in_top_level_all_and_is_lazy(self):
        assert "list_policy_types" in strands_robots.__all__
        # Unlike list_providers (torch-free factory), this peer is lazy.
        assert "list_policy_types" in strands_robots._LAZY_IMPORTS

    def test_in_policies_subpackage_all(self):
        import strands_robots.policies as policies_pkg

        assert "list_policy_types" in policies_pkg.__all__

    def test_resolves_to_canonical_object_at_both_levels(self):
        import strands_robots.policies as policies_pkg
        from strands_robots.policies.lerobot_local.resolution import (
            list_policy_types as canonical,
        )

        assert strands_robots.list_policy_types is canonical
        assert policies_pkg.list_policy_types is canonical

    def test_callable_returns_list_from_top_level(self):
        # Discovery surface: callable straight off the package root, and it
        # degrades to an empty list (never raises) when lerobot is absent.
        result = strands_robots.list_policy_types()
        assert isinstance(result, list)

    def test_policies_import_stays_torch_free(self):
        # The new lazy export must not regress the torch-free import contract:
        # a clean interpreter importing strands_robots.policies must not pull
        # torch, yet must still advertise list_policy_types in __all__.
        import subprocess
        import sys

        code = (
            "import sys, strands_robots.policies as p; "
            "assert 'torch' not in sys.modules, 'torch eagerly imported'; "
            "assert 'list_policy_types' in p.__all__"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


class TestImportResilience:
    """``import strands_robots`` must never fail because an optional import-time
    autoconfig step raised.

    The package body runs two best-effort autoconfig blocks at import:

    - a MuJoCo GL-backend probe (headless EGL/OSMesa selection), which can
      raise ``OSError`` when no usable GL device is present, and
    - a macOS dyld-path shim for torchcodec's ffmpeg discovery.

    Both are convenience shims, not correctness invariants, so a failure in
    either must be swallowed and leave ``import strands_robots`` succeeding.
    These tests reload the package with each shim forced to raise and assert
    the import still completes and the light API stays usable.
    """

    def test_import_survives_gl_backend_configure_failure(self):
        # The GL autoconfig block only runs when mujoco is importable.
        pytest.importorskip("mujoco")
        import importlib

        import strands_robots.simulation.mujoco.backend as backend_mod

        real = backend_mod._configure_gl_backend

        def _raise_oserror():
            # Mirrors a headless box with no usable GL device: EGL/OSMesa
            # initialisation surfaces as an OSError.
            raise OSError("simulated EGL device-open failure")

        backend_mod._configure_gl_backend = _raise_oserror  # type: ignore[assignment]
        try:
            # Reload re-runs the package body, which now calls the raising
            # shim. The except-guard must swallow it: reload must not raise.
            reloaded = importlib.reload(strands_robots)
            assert callable(reloaded.create_policy)
        finally:
            backend_mod._configure_gl_backend = real  # type: ignore[assignment]
            importlib.reload(strands_robots)

    def test_import_survives_dyld_shim_failure(self):
        import importlib

        import strands_robots._dyld as dyld_mod

        real = dyld_mod.ensure_ffmpeg_on_dyld_path

        def _raise_runtime(*args, **kwargs):
            raise RuntimeError("simulated dyld shim failure")

        dyld_mod.ensure_ffmpeg_on_dyld_path = _raise_runtime  # type: ignore[assignment]
        try:
            reloaded = importlib.reload(strands_robots)
            assert callable(reloaded.create_policy)
        finally:
            dyld_mod.ensure_ffmpeg_on_dyld_path = real  # type: ignore[assignment]
            importlib.reload(strands_robots)
