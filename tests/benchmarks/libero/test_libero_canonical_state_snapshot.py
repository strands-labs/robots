"""Snapshot-and-restore branch of ``LiberoAdapter._apply_canonical_state``.

When a LIBERO scene carries neither per-episode ``init_states`` nor a MuJoCo
``<keyframe>`` (the procedurally-generated MJCF path), the adapter caches
``data.qpos`` / ``data.qvel`` on the first episode and restores that snapshot
on every subsequent episode so each rollout starts from the same canonical
pose. These tests pin that round-trip with a real compiled MuJoCo model - both
the lock-held and lockless restore paths - plus the documented "best-effort,
never fatal" contract of the keyframe and snapshot branches when the underlying
MuJoCo calls raise (the branches log a warning and return instead of aborting
the eval).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.benchmarks.libero.adapter import LiberoAdapter

# Minimal BDDL: enough for parse_bddl to build a problem the adapter accepts.
_PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the red cube and place it on the plate")
  (:objects cube_1 plate_1 table_1 - object)
  (:init (on cube_1 table_1))
  (:goal (on cube_1 plate_1)))
"""

# A 1-DoF scene with NO <keyframe> and NO LIBERO ``robot0_`` joints, so
# ``_apply_canonical_state`` takes the snapshot branch and the home-pose write
# no-ops (leaving qpos exactly as the test set it).
_NO_KEYFRAME_XML = """
<mujoco>
  <worldbody>
    <body name="b">
      <joint name="j" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <actuator><motor joint="j" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


class _World:
    def __init__(self, model: Any, data: Any) -> None:
        self._model = model
        self._data = data
        self._backend_state: dict[str, Any] = {}


class _Sim:
    """Minimal sim exposing the ``_world._model`` / ``_world._data`` (+ optional
    ``_lock``) surface ``_apply_canonical_state`` reads."""

    def __init__(self, model: Any, data: Any, lock: Any = None) -> None:
        self._world = _World(model, data)
        if lock is not None:
            self._lock = lock


def _make_sim(lock: Any = None):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_NO_KEYFRAME_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return _Sim(model, data, lock=lock), mujoco


def _adapter() -> LiberoAdapter:
    # init_states defaults to None -> snapshot/keyframe branch (not init-state);
    # no keyframe in the XML -> snapshot-and-restore branch.
    return LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
    )


def test_snapshot_branch_captures_then_restores_across_episodes():
    """Episode 0 captures the canonical qpos; a later episode restores it even
    after a rollout perturbed the state (the lockless restore path)."""
    sim, mujoco = _make_sim()
    adapter = _adapter()

    # Episode 0: known canonical pose, then capture.
    sim._world._data.qpos[0] = 0.37
    mujoco.mj_forward(sim._world._model, sim._world._data)
    adapter._apply_canonical_state(sim)
    assert adapter._canonical_qpos is not None, "first episode should capture a snapshot"
    captured = float(sim._world._data.qpos[0])
    assert captured == pytest.approx(0.37, abs=1e-9)

    # A rollout drives qpos away from canonical.
    sim._world._data.qpos[0] = -1.25
    mujoco.mj_forward(sim._world._model, sim._world._data)

    # Next episode: restore the captured snapshot.
    adapter._apply_canonical_state(sim)
    assert float(sim._world._data.qpos[0]) == pytest.approx(captured, abs=1e-9), (
        "subsequent episode did not restore the canonical qpos snapshot"
    )


def test_snapshot_restore_holds_and_releases_sim_lock():
    """When the sim exposes ``_lock``, the restore path takes it and releases
    it (mirrors Simulation.reset / send_action's locking contract)."""
    lock = threading.Lock()
    sim, mujoco = _make_sim(lock=lock)
    adapter = _adapter()

    sim._world._data.qpos[0] = 0.21
    mujoco.mj_forward(sim._world._model, sim._world._data)
    adapter._apply_canonical_state(sim)  # capture

    sim._world._data.qpos[0] = 2.0
    mujoco.mj_forward(sim._world._model, sim._world._data)
    adapter._apply_canonical_state(sim)  # restore under lock

    assert float(sim._world._data.qpos[0]) == pytest.approx(0.21, abs=1e-9)
    assert not lock.locked(), "restore must release sim._lock"


def test_keyframe_branch_never_fatal_when_reset_raises(caplog):
    """A failing ``mj_resetDataKeyframe`` is swallowed with a warning - the
    keyframe branch is best-effort and must never abort the eval."""
    sim, _ = _make_sim()
    adapter = _adapter()

    class _BoomMj:
        def mj_resetDataKeyframe(self, *_a: Any, **_k: Any) -> None:
            raise RuntimeError("boom")

        def mj_forward(self, *_a: Any, **_k: Any) -> None:  # pragma: no cover - must not run
            raise AssertionError("mj_forward should not run after reset raised")

    with caplog.at_level(logging.WARNING):
        # nkey=1 with in-range index 0 -> tries the reset, which raises.
        adapter._apply_keyframe_branch(sim, sim._world._model, sim._world._data, _BoomMj(), None, 1)

    assert any("mj_resetDataKeyframe" in r.message for r in caplog.records), (
        "expected a warning when mj_resetDataKeyframe fails"
    )


def test_snapshot_restore_never_fatal_when_forward_raises(caplog):
    """A failing ``mj_forward`` during restore is swallowed with a warning
    rather than propagating out of the best-effort branch."""
    sim, _ = _make_sim()
    adapter = _adapter()

    # Prime a matching-shape snapshot so the branch takes the restore path.
    adapter._canonical_qpos = np.array(sim._world._data.qpos, copy=True)
    adapter._canonical_qvel = np.array(sim._world._data.qvel, copy=True)

    class _ForwardBoomMj:
        def mj_forward(self, *_a: Any, **_k: Any) -> None:
            raise RuntimeError("kaboom")

    with caplog.at_level(logging.WARNING):
        adapter._apply_snapshot_branch(sim, sim._world._model, sim._world._data, _ForwardBoomMj(), None)

    assert any("snapshot restore failed" in r.message for r in caplog.records), (
        "expected a warning when mj_forward fails during restore"
    )
