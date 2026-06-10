"""Pin: ``_float_env`` rejects non-finite values (NaN, +/-inf).

The helper in ``strands_robots/mesh/_zenoh_config.py`` clamps with
``if value < lo or value > hi: raise``. Both comparisons return
``False`` for ``NaN`` (IEEE-754 semantics), so without the explicit
``math.isfinite`` guard ``STRANDS_MESH_CMD_RATE_HZ=nan`` would be
silently accepted and the downstream Zenoh ``downsampling`` rule's
``freq`` field would become NaN, disabling the rate cap with no
operator-visible signal.

These tests pin the rejection path on every published env-var entry
that flows through ``_float_env`` so a future helper rewrite that
drops the isfinite check is caught here.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import _zenoh_config as zc


@pytest.mark.parametrize("bad", ["nan", "NaN", "NAN", "inf", "-inf", "Infinity", "-Infinity"])
def test_float_env_rejects_non_finite(monkeypatch, bad):
    """Every non-finite literal Python ``float()`` accepts must raise."""
    monkeypatch.setenv("STRANDS_MESH_CMD_RATE_HZ", bad)
    with pytest.raises(ValueError, match="must be finite"):
        zc._float_env("STRANDS_MESH_CMD_RATE_HZ", default=20.0, lo=0.001, hi=10000.0)


def test_float_env_rejects_nan_in_safety_rate(monkeypatch):
    """``downsampling_block`` reads ``STRANDS_MESH_SAFETY_RATE_HZ`` via _float_env."""
    monkeypatch.setenv("STRANDS_MESH_SAFETY_RATE_HZ", "nan")
    with pytest.raises(ValueError, match="must be finite"):
        zc.downsampling_block()


def test_float_env_accepts_finite_value_unchanged(monkeypatch):
    """Sanity: legitimate finite values still pass through cleanly."""
    monkeypatch.setenv("STRANDS_MESH_CMD_RATE_HZ", "5.5")
    assert zc._float_env("STRANDS_MESH_CMD_RATE_HZ", default=20.0, lo=0.001, hi=10000.0) == 5.5


def test_float_env_unset_returns_default(monkeypatch):
    """Sanity: unset env var falls back to the default."""
    monkeypatch.delenv("STRANDS_MESH_CMD_RATE_HZ", raising=False)
    assert zc._float_env("STRANDS_MESH_CMD_RATE_HZ", default=20.0, lo=0.001, hi=10000.0) == 20.0


def test_float_env_still_rejects_out_of_bounds(monkeypatch):
    """Sanity: existing bounds rejection still works post-fix."""
    monkeypatch.setenv("STRANDS_MESH_CMD_RATE_HZ", "999999.0")
    with pytest.raises(ValueError, match="out of bounds"):
        zc._float_env("STRANDS_MESH_CMD_RATE_HZ", default=20.0, lo=0.001, hi=10000.0)
