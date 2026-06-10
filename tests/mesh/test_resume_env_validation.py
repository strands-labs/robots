"""Pin tests for STRANDS_MESH_RESUME_* env-var input validation.

Without input validation, importing :mod:`strands_robots.mesh.core` with
a malformed env var (e.g. ``STRANDS_MESH_RESUME_FRESHNESS_S=abc``) would
raise an opaque :class:`ValueError` at import-time, breaking the whole
package -- not just the mesh subsystem. Negative values would either
silently disable the replay cache (``maxlen=0``) or crash later when
the deque is sized (``maxlen=-1``).

These regressions were discovered by running the module under bad env
locally before any test would have flagged them. The fix adds
``_parse_positive_float_env`` / ``_parse_positive_int_env`` helpers that
log a warning and fall back to the default on bad input.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def reload_core(monkeypatch):
    """Reload ``strands_robots.mesh.core`` with a fresh env state."""

    def _reload():
        import strands_robots.mesh.core as core_module

        return importlib.reload(core_module)

    return _reload


# --- Float env vars ------------------------------------------------------


def test_freshness_non_numeric_falls_back(monkeypatch, reload_core, caplog):
    monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "abc")
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        core = reload_core()
    assert core.RESUME_FRESHNESS_WINDOW_S == 60.0
    assert any("STRANDS_MESH_RESUME_FRESHNESS_S" in r.message for r in caplog.records)


def test_freshness_negative_falls_back(monkeypatch, reload_core, caplog):
    monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "-10")
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        core = reload_core()
    assert core.RESUME_FRESHNESS_WINDOW_S == 60.0


def test_freshness_valid_passes_through(monkeypatch, reload_core):
    monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "120")
    core = reload_core()
    assert core.RESUME_FRESHNESS_WINDOW_S == 120.0


def test_forward_skew_non_numeric_falls_back(monkeypatch, reload_core, caplog):
    monkeypatch.setenv("STRANDS_MESH_RESUME_FORWARD_SKEW_S", "not-a-number")
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        core = reload_core()
    assert core.RESUME_FORWARD_SKEW_S == 5.0


def test_forward_skew_negative_falls_back(monkeypatch, reload_core, caplog):
    monkeypatch.setenv("STRANDS_MESH_RESUME_FORWARD_SKEW_S", "-3.5")
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        core = reload_core()
    assert core.RESUME_FORWARD_SKEW_S == 5.0


# --- Int env vars --------------------------------------------------------


def test_replay_cache_max_non_numeric_falls_back(monkeypatch, reload_core, caplog):
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "abc")
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        core = reload_core()
    assert core.RESUME_REPLAY_CACHE_MAX == 4096


def test_replay_cache_max_zero_falls_back(monkeypatch, reload_core, caplog):
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "0")
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        core = reload_core()
    assert core.RESUME_REPLAY_CACHE_MAX == 4096


def test_replay_cache_max_negative_falls_back(monkeypatch, reload_core, caplog):
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "-1")
    with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
        core = reload_core()
    assert core.RESUME_REPLAY_CACHE_MAX == 4096


def test_replay_cache_max_valid_passes_through(monkeypatch, reload_core):
    monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "8192")
    core = reload_core()
    assert core.RESUME_REPLAY_CACHE_MAX == 8192


# --- Module imports cleanly with all defaults missing --------------------


def test_module_imports_with_no_env_vars(monkeypatch, reload_core):
    """No env vars set -> defaults must apply, no warnings, no exceptions."""
    for name in (
        "STRANDS_MESH_RESUME_FRESHNESS_S",
        "STRANDS_MESH_RESUME_FORWARD_SKEW_S",
        "STRANDS_MESH_RESUME_REPLAY_CACHE_MAX",
    ):
        monkeypatch.delenv(name, raising=False)
    core = reload_core()
    assert core.RESUME_FRESHNESS_WINDOW_S == 60.0
    assert core.RESUME_FORWARD_SKEW_S == 5.0
    assert core.RESUME_REPLAY_CACHE_MAX == 4096


# === lazy resolution of RESUME_* env vars ===


class TestResumeEnvLazyResolution:
    """RESUME_* env vars are now re-read on every
    safety-handler call, not snapshotted at module import time. This
    means an operator setting STRANDS_MESH_RESUME_FRESHNESS_S AFTER
    importing strands_robots.mesh.core sees the new value without a
    process restart -- removing the import-order coupling reviewer
    yinsong1986 flagged in the 21:34 batch.
    """

    def test_freshness_window_resolves_at_call_time(self, monkeypatch):
        from strands_robots.mesh import core as core_mod

        monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "120")
        # Lazy resolver picks up the new value
        assert core_mod._resume_freshness_window_s() == 120.0

        monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "30")
        assert core_mod._resume_freshness_window_s() == 30.0

    def test_forward_skew_resolves_at_call_time(self, monkeypatch):
        from strands_robots.mesh import core as core_mod

        monkeypatch.setenv("STRANDS_MESH_RESUME_FORWARD_SKEW_S", "10")
        assert core_mod._resume_forward_skew_s() == 10.0

    def test_replay_cache_max_resolves_at_call_time(self, monkeypatch):
        from strands_robots.mesh import core as core_mod

        monkeypatch.setenv("STRANDS_MESH_RESUME_REPLAY_CACHE_MAX", "256")
        assert core_mod._resume_replay_cache_max() == 256

    def test_module_constants_remain_for_back_compat(self):
        """The module-level RESUME_* constants are still defined (some
        downstream code may read them); they are the import-time defaults.
        Hot paths use the lazy resolvers."""
        from strands_robots.mesh import core as core_mod

        assert hasattr(core_mod, "RESUME_FRESHNESS_WINDOW_S")
        assert hasattr(core_mod, "RESUME_FORWARD_SKEW_S")
        assert hasattr(core_mod, "RESUME_REPLAY_CACHE_MAX")
        assert isinstance(core_mod.RESUME_FRESHNESS_WINDOW_S, float)
        assert isinstance(core_mod.RESUME_FORWARD_SKEW_S, float)
        assert isinstance(core_mod.RESUME_REPLAY_CACHE_MAX, int)

    def test_bad_env_falls_back_to_default_at_call_time(self, monkeypatch):
        """Bad env values fall back to the documented defaults (matching
        the existing _parse_positive_*_env contracts)."""
        from strands_robots.mesh import core as core_mod

        monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "abc")
        # Falls back to default 60
        assert core_mod._resume_freshness_window_s() == 60.0
