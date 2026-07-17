"""Newton ``create_world`` rejects the terrain/difficulty contract it cannot honor.

Rough-ground heightfields are a MuJoCo-backend capability; the Newton backend
has no heightfield ground, so it rejects both a non-None ``terrain`` and a
non-default ``difficulty`` (which only scales a heightfield, so it is inert
here) with actionable errors rather than silently ignoring them - the base
``create_world`` contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

_engine_cls: type[NewtonSimEngine] | None
try:  # NewtonSimEngine imports without Warp (Warp is lazily loaded at build time)
    from strands_robots.simulation.newton.simulation import NewtonSimEngine as _engine_cls
except Exception:  # pragma: no cover - newton package genuinely absent
    _engine_cls = None

pytestmark = pytest.mark.skipif(_engine_cls is None, reason="newton package not importable")


def test_newton_rejects_terrain_before_any_build() -> None:
    # __new__ bypasses __init__ (no solver/GPU); the reject returns before the
    # lock/_rebuild, so no engine state is needed.
    assert _engine_cls is not None
    eng = _engine_cls.__new__(_engine_cls)
    r = eng.create_world(terrain="rough")
    assert r["status"] == "error"
    text = r["content"][0]["text"]
    assert "Newton" in text and "MuJoCo" in text and "rough" in text


def test_newton_terrain_none_is_not_rejected() -> None:
    # A flat (terrain=None) create_world must NOT hit the reject path; it falls
    # through to the real build (which __new__ cannot run), so we only assert
    # the reject branch does not fire by patching _rebuild/_lock to no-ops.
    import threading

    assert _engine_cls is not None
    eng = _engine_cls.__new__(_engine_cls)
    eng._lock = threading.RLock()
    eng.default_timestep = 0.002
    eng._solver_name = "mujoco"
    eng._rebuild = lambda: None  # type: ignore[method-assign]
    r = eng.create_world(terrain=None)
    assert r["status"] == "success"


def test_newton_accepts_difficulty_kwarg_and_still_rejects_terrain() -> None:
    # ``difficulty`` exists on both backends for signature parity; on Newton it
    # is inert (terrain is rejected outright), but passing it must not raise a
    # TypeError - the terrain rejection still fires with difficulty supplied.
    assert _engine_cls is not None
    eng = _engine_cls.__new__(_engine_cls)
    r = eng.create_world(terrain="rough", difficulty=0.5)
    assert r["status"] == "error"
    assert "Newton" in r["content"][0]["text"] and "rough" in r["content"][0]["text"]


def test_newton_rejects_difficulty_without_terrain() -> None:
    # Base create_world contract: a non-default ``difficulty`` with no
    # ``terrain`` is rejected with an actionable error rather than silently
    # having no effect. On Newton there is no heightfield terrain for
    # difficulty to scale, so any != 1.0 value is doubly inert here; the reject
    # is a pure Python guard returning before the lock/_rebuild, so __new__
    # exercises it (no Warp / GPU). Was a status=success / silent no-op before
    # this contract landed.
    assert _engine_cls is not None
    eng = _engine_cls.__new__(_engine_cls)
    r = eng.create_world(difficulty=2.0)
    assert r["status"] == "error"
    text = r["content"][0]["text"].lower()
    assert "difficulty" in text and "newton" in text and "mujoco" in text, text


def test_newton_terrain_reject_precedes_difficulty_reject() -> None:
    # When both an unsupported terrain AND a non-default difficulty are passed,
    # the terrain rejection (the primary user error on this backend) fires
    # first, not the difficulty one.
    assert _engine_cls is not None
    eng = _engine_cls.__new__(_engine_cls)
    r = eng.create_world(terrain="rough", difficulty=2.0)
    assert r["status"] == "error"
    text = r["content"][0]["text"]
    assert "rough" in text and "difficulty" not in text.lower()


def test_newton_default_difficulty_not_rejected() -> None:
    # difficulty=1.0 (the default) is a no-op that must fall through to the real
    # build, not the reject path. Patch _rebuild/_lock so __new__ can reach it.
    import threading

    assert _engine_cls is not None
    eng = _engine_cls.__new__(_engine_cls)
    eng._lock = threading.RLock()
    eng.default_timestep = 0.002
    eng._solver_name = "mujoco"
    eng._rebuild = lambda: None  # type: ignore[method-assign]
    r = eng.create_world(difficulty=1.0)
    assert r["status"] == "success"
