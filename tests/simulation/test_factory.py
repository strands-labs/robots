"""Tests for strands_robots.simulation.factory - backend registration + creation."""

from __future__ import annotations

import importlib.util

import pytest

from strands_robots.simulation import factory
from strands_robots.simulation.factory import (
    DEFAULT_BACKEND,
    create_simulation,
    list_backends,
    register_backend,
)


@pytest.fixture(autouse=True)
def _clear_runtime():
    """Each test starts with a clean runtime registry and plugin cache."""
    factory._runtime_registry.clear()
    factory._runtime_aliases.clear()
    factory._PLUGIN_BACKENDS_CACHE = None
    yield
    factory._runtime_registry.clear()
    factory._runtime_aliases.clear()
    factory._PLUGIN_BACKENDS_CACHE = None


class _FakeSim:
    """Plain class stand-in for a simulation backend.

    Not a real ``SimEngine`` subclass - the factory only calls the loader
    callable and the returned class's ``__init__``; it does not enforce the
    ABC contract. Using a plain class here keeps the test focused on the
    factory's own logic (registration, lookup, aliasing).
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestListBackends:
    def test_includes_builtin_mujoco(self):
        assert "mujoco" in list_backends()

    def test_includes_builtin_aliases(self):
        backends = list_backends()
        assert "mj" in backends
        assert "mjc" in backends
        assert "mjx" in backends

    def test_is_sorted_and_deduped(self):
        backends = list_backends()
        assert backends == sorted(set(backends))

    def test_includes_runtime_backends(self):
        register_backend("fake_sim", lambda: _FakeSim, aliases=["fk"])
        backends = list_backends()
        assert "fake_sim" in backends
        assert "fk" in backends


class TestRegisterBackend:
    def test_register_and_create(self):
        register_backend("fake_sim", lambda: _FakeSim)
        sim = create_simulation("fake_sim")
        assert isinstance(sim, _FakeSim)

    def test_register_with_aliases(self):
        register_backend("fake_sim", lambda: _FakeSim, aliases=["fs", "fake"])
        assert isinstance(create_simulation("fs"), _FakeSim)
        assert isinstance(create_simulation("fake"), _FakeSim)

    def test_duplicate_name_rejected(self):
        register_backend("fake_sim", lambda: _FakeSim)
        with pytest.raises(ValueError, match="already registered"):
            register_backend("fake_sim", lambda: _FakeSim)

    def test_duplicate_conflicts_with_builtin(self):
        with pytest.raises(ValueError, match="already registered"):
            register_backend("mujoco", lambda: _FakeSim)

    def test_duplicate_conflicts_with_builtin_alias(self):
        with pytest.raises(ValueError, match="conflicts with built-in alias"):
            register_backend("mj", lambda: _FakeSim)

    def test_runtime_alias_conflict(self):
        register_backend("alpha", lambda: _FakeSim, aliases=["shared"])
        with pytest.raises(ValueError, match="already registered"):
            register_backend("beta", lambda: _FakeSim, aliases=["shared"])

    def test_alias_conflicts_with_builtin(self):
        with pytest.raises(ValueError, match="conflicts with existing backend"):
            register_backend("beta", lambda: _FakeSim, aliases=["mujoco"])

    def test_force_overrides_duplicate(self):
        register_backend("fake_sim", lambda: _FakeSim, aliases=["fk"])

        class _OtherSim(_FakeSim):
            pass

        register_backend("fake_sim", lambda: _OtherSim, aliases=["fk"], force=True)
        sim = create_simulation("fake_sim")
        assert type(sim).__name__ == "_OtherSim"


@pytest.mark.skipif(
    not importlib.util.find_spec("mujoco"),
    reason="mujoco not installed",
)
class TestCreateSimulation:
    def test_default_is_mujoco(self):
        pytest.importorskip("mujoco")
        sim = create_simulation()
        assert type(sim).__name__ == "MuJoCoSimEngine"
        sim.cleanup()

    def test_by_alias(self):
        pytest.importorskip("mujoco")
        sim = create_simulation("mj")
        assert type(sim).__name__ == "MuJoCoSimEngine"
        sim.cleanup()

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown simulation backend"):
            create_simulation("nonexistent_backend_xyz")

    def test_unknown_backend_error_lists_available(self):
        with pytest.raises(ValueError) as exc_info:
            create_simulation("nonexistent_backend_xyz")
        msg = str(exc_info.value)
        assert "mujoco" in msg  # should list available backends

    def test_kwargs_forwarded_to_backend(self):
        register_backend("fake_sim", lambda: _FakeSim)
        sim = create_simulation("fake_sim", tool_name="custom", timestep=0.005)
        assert sim.kwargs == {"tool_name": "custom", "timestep": 0.005}

    def test_runtime_alias_priority_over_builtin(self):
        """Runtime aliases can shadow built-in aliases when ``force=True``."""
        register_backend("fake_sim", lambda: _FakeSim, aliases=["mj"], force=True)
        sim = create_simulation("mj")
        assert isinstance(sim, _FakeSim)


class TestDefaultBackendConstant:
    def test_default_is_documented(self):
        assert DEFAULT_BACKEND == "mujoco"


class TestNewtonBackendRegistration:
    """Newton backend is wired into the built-in registry (no GPU needed)."""

    def test_newton_in_builtin_backends(self):
        assert "newton" in factory._BUILTIN_BACKENDS
        module_path, class_name = factory._BUILTIN_BACKENDS["newton"]
        assert module_path == "strands_robots.simulation.newton.simulation"
        assert class_name == "NewtonSimEngine"

    def test_newton_listed(self):
        assert "newton" in list_backends()

    def test_nt_alias_resolves_to_newton(self):
        assert factory._resolve_name("nt") == "newton"
        assert "nt" in list_backends()

    def test_missing_newton_module_error_names_sim_newton_extra(self, monkeypatch):
        """A missing backend module yields an ImportError naming its pip extra."""
        monkeypatch.setitem(
            factory._BUILTIN_BACKENDS,
            "newton",
            ("strands_robots.simulation.newton._absent_module", "NewtonSimEngine"),
        )
        with pytest.raises(ImportError, match="sim-newton"):
            create_simulation("newton")


class _FakeEntryPoint:
    """Minimal stand-in for an importlib.metadata.EntryPoint.

    ``entry_points(group=...)`` returns objects with ``.name`` and ``.load()``;
    that is the entire surface the factory touches, so we mock just those.
    """

    def __init__(self, name, target, *, raises=None):
        self.name = name
        self._target = target
        self._raises = raises

    def load(self):
        if self._raises is not None:
            raise self._raises
        return self._target


def _patch_entry_points(monkeypatch, eps):
    """Patch the factory's ``entry_points`` to return ``eps`` for our group."""

    def _fake_entry_points(*, group):
        assert group == factory._ENTRY_POINT_GROUP
        return list(eps)

    monkeypatch.setattr(factory, "entry_points", _fake_entry_points)


class _PluginSimA(_FakeSim):
    pass


class _PluginSimB(_FakeSim):
    pass


class TestEntryPointDiscovery:
    def test_create_resolves_via_entry_point(self, monkeypatch):
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("plugin_sim", _PluginSimA)])
        sim = create_simulation("plugin_sim", gpu_id=3)
        assert isinstance(sim, _PluginSimA)
        assert sim.kwargs == {"gpu_id": 3}

    def test_list_backends_includes_plugins(self, monkeypatch):
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("plugin_sim", _PluginSimA)])
        backends = list_backends()
        assert "plugin_sim" in backends
        assert "mujoco" in backends
        assert backends == sorted(set(backends))

    def test_aliases_via_multiple_entry_points(self, monkeypatch):
        """Two entry-point names may point at the same class (newton/warp)."""
        _patch_entry_points(
            monkeypatch,
            [
                _FakeEntryPoint("newton_plugin", _PluginSimB),
                _FakeEntryPoint("warp_plugin", _PluginSimB),
            ],
        )
        assert isinstance(create_simulation("newton_plugin"), _PluginSimB)
        assert isinstance(create_simulation("warp_plugin"), _PluginSimB)

    def test_builtin_wins_over_plugin_of_same_name(self, monkeypatch):
        """A plugin named ``mujoco`` must not shadow the built-in backend."""
        pytest.importorskip("mujoco")
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("mujoco", _PluginSimA)])
        sim = create_simulation("mujoco")
        # Built-in resolves, not the plugin stand-in.
        assert type(sim).__name__ == "MuJoCoSimEngine"
        assert not isinstance(sim, _PluginSimA)
        sim.cleanup()

    def test_broken_plugin_is_skipped_not_fatal(self, monkeypatch):
        """A plugin that fails to import is logged and skipped, others work."""
        _patch_entry_points(
            monkeypatch,
            [
                _FakeEntryPoint("broken", None, raises=RuntimeError("boom")),
                _FakeEntryPoint("good", _PluginSimA),
            ],
        )
        # Good plugin still resolves despite the broken sibling.
        assert isinstance(create_simulation("good"), _PluginSimA)
        # Broken one is simply absent.
        assert "broken" not in factory._load_plugin_backends()

    def test_discovery_is_cached(self, monkeypatch):
        """Entry points are scanned at most once per process."""
        calls = {"n": 0}

        def _counting_entry_points(*, group):
            calls["n"] += 1
            return [_FakeEntryPoint("plugin_sim", _PluginSimA)]

        monkeypatch.setattr(factory, "entry_points", _counting_entry_points)
        list_backends()
        list_backends()
        create_simulation("plugin_sim")
        assert calls["n"] == 1

    def test_unknown_backend_lists_plugins_in_error(self, monkeypatch):
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("plugin_sim", _PluginSimA)])
        with pytest.raises(ValueError) as exc_info:
            create_simulation("nope_xyz")
        msg = str(exc_info.value)
        assert "mujoco" in msg
        assert "plugin_sim" in msg

    def test_unknown_known_plugin_name_suggests_install(self, monkeypatch):
        """Known out-of-tree names surface a pip install hint when absent."""
        _patch_entry_points(monkeypatch, [])
        with pytest.raises(ValueError, match="strands-robots-sim") as exc_info:
            create_simulation("isaac")
        assert "pip install" in str(exc_info.value)

    def test_mjwarp_and_warp_names_suggest_newton_plugin(self, monkeypatch):
        """The GPU-parallel warp/MuJoCo path is the built-in ``newton`` backend.
        ``warp`` and ``mjwarp`` are common names users reach for; both must
        surface the ``strands-robots-sim[newton]`` install hint instead of a
        bare unknown-backend dead end."""
        _patch_entry_points(monkeypatch, [])
        for name in ("warp", "mjwarp"):
            with pytest.raises(ValueError, match="strands-robots-sim") as exc_info:
                create_simulation(name)
            assert "newton" in str(exc_info.value)


class TestIsaacBuiltinWinsOverPlugin:
    """Backward-compat regression guard for issue #1150.

    When the ``isaac`` backend is absorbed in-tree as a built-in (parent epic
    #1144), a user who still has ``strands-robots-sim`` installed advertises an
    ``isaac`` entry-point plugin. Resolution MUST prefer the built-in, the
    plugin must never shadow or duplicate-register it, and no ``ValueError``
    may fire from the ``register_backend`` conflict checks.

    These tests simulate the post-#1146 state by injecting an ``isaac`` entry
    into ``_BUILTIN_BACKENDS`` pointed at a real, importable stub module, so
    they hold regardless of whether the isaac implementation has landed yet.
    """

    _STUB_MODULE = "strands_robots._test_isaac_builtin_stub"

    @pytest.fixture
    def isaac_builtin(self, monkeypatch):
        """Register a stub ``isaac`` built-in backend for the duration of a test.

        Yields the stub class the built-in ``isaac`` resolves to. Uses a real
        module in ``sys.modules`` because ``_import_backend_class`` imports
        built-ins via ``importlib.import_module`` (unlike runtime backends,
        which are plain loader callables).
        """
        import sys
        import types

        stub_mod = types.ModuleType(self._STUB_MODULE)

        class IsaacSimEngine(_FakeSim):
            """Stand-in for the future in-tree Isaac built-in backend."""

        stub_mod.IsaacSimEngine = IsaacSimEngine
        monkeypatch.setitem(sys.modules, self._STUB_MODULE, stub_mod)
        monkeypatch.setitem(
            factory._BUILTIN_BACKENDS,
            "isaac",
            (self._STUB_MODULE, "IsaacSimEngine"),
        )
        monkeypatch.setitem(factory._BUILTIN_ALIASES, "isaacsim", "isaac")
        yield IsaacSimEngine

    def test_builtin_isaac_wins_over_entry_point_plugin(self, isaac_builtin, monkeypatch):
        """A stale ``strands-robots-sim`` isaac plugin must not shadow the builtin."""
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("isaac", _PluginSimA)])
        sim = create_simulation("isaac")
        assert isinstance(sim, isaac_builtin)
        assert not isinstance(sim, _PluginSimA)

    def test_builtin_isaac_alias_wins_over_entry_point_plugin(self, isaac_builtin, monkeypatch):
        """An entry-point named after a built-in *alias* is also shadowed."""
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("isaacsim", _PluginSimA)])
        sim = create_simulation("isaacsim")
        assert isinstance(sim, isaac_builtin)
        assert not isinstance(sim, _PluginSimA)

    def test_collided_isaac_plugin_is_not_loaded(self, isaac_builtin, monkeypatch):
        """The plugin's (heavy) class is never even loaded when a builtin wins.

        A plugin whose name collides with a built-in is dropped *before*
        ``ep.load()`` runs, so the sibling package's Isaac import cost is
        avoided entirely.
        """
        loaded = {"n": 0}

        class _CountingEntryPoint(_FakeEntryPoint):
            def load(self_inner):
                loaded["n"] += 1
                return super().load()

        _patch_entry_points(monkeypatch, [_CountingEntryPoint("isaac", _PluginSimA)])
        assert "isaac" not in factory._load_plugin_backends()
        assert loaded["n"] == 0

    def test_list_backends_reports_isaac_once(self, isaac_builtin, monkeypatch):
        """Even with a colliding plugin, ``isaac`` appears exactly once."""
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("isaac", _PluginSimA)])
        backends = list_backends()
        assert backends.count("isaac") == 1
        assert backends == sorted(set(backends))

    def test_entry_point_plugin_never_hits_register_backend(self, isaac_builtin, monkeypatch):
        """Resolving a colliding plugin fires no ``register_backend`` ValueError.

        Entry-point discovery never routes through ``register_backend`` (it
        populates the plugin cache directly), so the conflict checks that would
        reject a duplicate ``isaac`` name are never reached. Guard it by making
        ``register_backend`` explode if it is called at all during resolution.
        """

        def _boom(*args, **kwargs):
            raise AssertionError("register_backend must not be called during plugin resolution")

        monkeypatch.setattr(factory, "register_backend", _boom)
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("isaac", _PluginSimA)])
        sim = create_simulation("isaac")
        assert isinstance(sim, isaac_builtin)
        # list_backends also must not trigger a register_backend call.
        assert "isaac" in list_backends()


class TestEntryPointLazyImport:
    def test_no_eager_plugin_scan_on_import(self):
        """Importing strands_robots.simulation must not scan entry points.

        The plugin cache stays ``None`` until the first ``create_simulation``
        / ``list_backends`` call, so installing a plugin never slows cold
        ``import strands_robots.simulation``.
        """
        import importlib as _importlib

        mod = _importlib.import_module("strands_robots.simulation.factory")
        mod = _importlib.reload(mod)
        try:
            assert mod._PLUGIN_BACKENDS_CACHE is None
        finally:
            mod._PLUGIN_BACKENDS_CACHE = None
