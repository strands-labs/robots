"""Teardown safety when NewtonSimEngine construction fails part-way.

NewtonSimEngine.__init__ has fallible steps (importing the optional
Newton/Warp stack, validating the solver name) that can raise before the
engine is fully built. When that happens the half-constructed instance is
still garbage-collected, so __del__ -> cleanup() -> destroy() runs against it.
These tests pin that such teardown is a clean no-op instead of raising
AttributeError for state (the lock, viewer handles) that a naive __init__
ordering would not have set yet.

They are dependency-free: ensure_newton is monkeypatched to raise, so the
missing-Warp path is exercised even on machines without the Newton stack.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.simulation.newton import simulation as sim_mod
from strands_robots.simulation.newton.simulation import NewtonSimEngine


def _capture_partial_engine(**kwargs: object) -> NewtonSimEngine:
    """Construct a NewtonSimEngine whose __init__ raises, returning the
    half-built instance recovered from the exception traceback.

    Fails the test if construction unexpectedly succeeds or the instance
    cannot be recovered.
    """
    try:
        NewtonSimEngine(**kwargs)  # type: ignore[arg-type]
    except ImportError as exc:
        tb = exc.__traceback__
        engine: NewtonSimEngine | None = None
        while tb is not None:
            candidate = tb.tb_frame.f_locals.get("self")
            if isinstance(candidate, NewtonSimEngine):
                engine = candidate
            tb = tb.tb_next
        assert engine is not None, "could not recover the partially-built engine"
        return engine
    raise AssertionError("expected NewtonSimEngine.__init__ to raise ImportError")


@pytest.fixture
def force_missing_newton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ensure_newton() raise, simulating an absent Warp install."""

    def _boom() -> tuple[object, object]:
        raise ImportError("warp missing (simulated)")

    monkeypatch.setattr(sim_mod, "ensure_newton", _boom)


class TestPartialInitTeardown:
    def test_lock_set_before_fallible_ensure_newton(self, force_missing_newton: None) -> None:
        engine = _capture_partial_engine(solver="mujoco")
        # The lock and viewer handles must exist even though __init__ aborted
        # at ensure_newton(), so teardown can acquire the lock safely.
        assert hasattr(engine, "_lock")
        assert engine._viewer is None
        assert engine._viewer_kind is None
        assert engine._world is None

    def test_destroy_is_clean_noop_on_partial_engine(self, force_missing_newton: None) -> None:
        engine = _capture_partial_engine(solver="mujoco")
        result = engine.destroy()
        assert result["status"] == "success"

    def test_cleanup_does_not_raise_on_partial_engine(self, force_missing_newton: None) -> None:
        engine = _capture_partial_engine(solver="mujoco")
        # cleanup() delegates to destroy(); must not raise.
        engine.cleanup()

    def test_del_logs_no_cleanup_error_for_partial_engine(
        self, force_missing_newton: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = _capture_partial_engine(solver="mujoco")
        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.base"):
            engine.__del__()
        assert not any("Cleanup error during __del__" in record.getMessage() for record in caplog.records), (
            "teardown of a partially-built engine must not log a cleanup error"
        )
