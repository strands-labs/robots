"""Unit tests for the Newton backend helpers (no GPU / no Newton required)."""

from __future__ import annotations

import importlib.util

import pytest

from strands_robots.simulation.newton import backend
from strands_robots.simulation.newton.simulation import _short_joint_name

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None


class TestSolverRegistry:
    def test_registry_lists_rigid_body_solvers(self):
        reg = backend.solver_registry()
        assert reg["mujoco"] == "SolverMuJoCo"
        assert reg["featherstone"] == "SolverFeatherstone"
        assert reg["xpbd"] == "SolverXPBD"

    def test_registry_keys_are_lowercase(self):
        assert all(k == k.lower() for k in backend.solver_registry())


class TestShortJointName:
    def test_strips_hierarchical_path(self):
        label = "so_arm100/worldbody/Base/Rotation_Pitch/Rotation"
        assert _short_joint_name(label) == "Rotation"

    def test_plain_name_unchanged(self):
        assert _short_joint_name("Jaw") == "Jaw"


@pytest.mark.skipif(_HAS_NEWTON, reason="newton installed; missing-dep path not exercised")
class TestMissingDependency:
    def test_resolve_solver_class_raises_without_newton(self):
        with pytest.raises(ImportError, match="sim-newton"):
            backend.resolve_solver_class("mujoco")


@pytest.mark.skipif(not _HAS_NEWTON, reason="newton not installed")
class TestSolverResolution:
    def test_resolve_known_solver(self):
        cls = backend.resolve_solver_class("featherstone")
        assert cls.__name__ == "SolverFeatherstone"

    def test_resolve_unknown_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown Newton solver"):
            backend.resolve_solver_class("not_a_solver")


class _FakeSolvers:
    """Stand-in for ``newton.solvers`` exposing the registry class names."""

    class SolverMuJoCo:  # noqa: N801 - mirrors the real newton class name
        pass

    class SolverFeatherstone:  # noqa: N801
        pass


class _FakeNewton:
    """Minimal stub of the ``newton`` module used to drive resolution paths."""

    solvers = _FakeSolvers


class TestLazyImportCache:
    """The ``_modules`` cache is the documented single source of import state."""

    def test_ensure_newton_returns_cached_modules(self, monkeypatch):
        fake_nt, fake_wp = _FakeNewton(), object()
        monkeypatch.setattr(backend, "_modules", {"newton": fake_nt, "warp": fake_wp})
        # require_optional must NOT be consulted on a cache hit; if it is, fail loud.
        monkeypatch.setattr(
            backend,
            "require_optional",
            lambda *a, **k: pytest.fail("cache hit should not import"),
        )
        assert backend.ensure_newton() == (fake_nt, fake_wp)

    def test_ensure_newton_imports_and_populates_cache(self, monkeypatch):
        fake_nt, fake_wp = _FakeNewton(), object()

        def fake_require(name, **kwargs):
            return {"warp": fake_wp, "newton": fake_nt}[name]

        monkeypatch.setattr(backend, "_modules", {})
        monkeypatch.setattr(backend, "require_optional", fake_require)

        assert backend.ensure_newton() == (fake_nt, fake_wp)
        # Second call is served from the now-populated cache.
        assert backend._modules == {"newton": fake_nt, "warp": fake_wp}


class TestSolverResolutionWithStub:
    """resolve_solver_class maps friendly names to newton.solvers classes."""

    def test_resolve_known_solver_returns_class(self, monkeypatch):
        monkeypatch.setattr(backend, "_modules", {"newton": _FakeNewton(), "warp": object()})
        assert backend.resolve_solver_class("MuJoCo") is _FakeSolvers.SolverMuJoCo

    def test_resolve_unknown_solver_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(backend, "_modules", {"newton": _FakeNewton(), "warp": object()})
        with pytest.raises(ValueError, match="Unknown Newton solver 'nope'"):
            backend.resolve_solver_class("nope")
