"""Tests for the package-root lazy-export contract in
``strands_robots/simulation/newton/__init__.py``.

Importing the newton package must stay cheap: the ``NewtonSimEngine`` symbol is
resolved on first attribute access via PEP 562 ``__getattr__`` so that
``import strands_robots.simulation.newton`` does not pull in the heavy
newton/warp stack. These tests pin the observable behavior of that loader:

- ``NewtonSimEngine`` is the sole promised public name (``__all__``),
- first attribute access resolves it, caches it in the module dict, and
  returns the same object identity as the backing ``simulation`` submodule,
- an unknown attribute raises ``AttributeError`` with the standard message.
"""

import importlib

import pytest

import strands_robots.simulation.newton as newton_pkg
from strands_robots.simulation.newton.simulation import NewtonSimEngine


class TestPublicSurface:
    def test_all_promises_only_newton_sim_engine(self):
        assert newton_pkg.__all__ == ["NewtonSimEngine"]


class TestLazyResolution:
    """First attribute access resolves, caches, and returns stable identity."""

    def test_lazy_symbol_resolves_caches_and_matches_submodule(self):
        # Reload the package so its namespace starts without the cached name,
        # letting us observe the lazy __getattr__ populate globals() on first
        # access rather than reading a value a prior test already resolved.
        pkg = importlib.reload(newton_pkg)
        assert "NewtonSimEngine" not in vars(pkg)

        resolved = pkg.NewtonSimEngine

        # Resolves to the very class defined in the backing submodule.
        assert resolved is NewtonSimEngine

        # First access caches the name in the module dict so __getattr__ is
        # not invoked again; the identity stays stable.
        assert "NewtonSimEngine" in vars(pkg)
        assert pkg.NewtonSimEngine is resolved


class TestUnknownAttribute:
    def test_unknown_attribute_raises_standard_message(self):
        with pytest.raises(AttributeError, match="has no attribute 'does_not_exist'"):
            newton_pkg.does_not_exist
