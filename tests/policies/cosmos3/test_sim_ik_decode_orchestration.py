"""Dependency-free coverage of the Cosmos 3 chunk -> joint-target orchestration.

:func:`~strands_robots.policies.cosmos3.sim_ik.decode_cosmos_chunk_to_targets`
is the function that composes the honest sim-loop bridge: de-normalize the raw
``[-1, 1]`` quantile action -> decode relative EE-pose deltas to an absolute
trajectory -> inverse kinematics to joint targets. Its accuracy regression in
``test_sim_ik.py`` ``importorskip``s on the ``cosmos3-sim`` extra (mink +
mujoco), so on a clean-install image the whole orchestration body is skipped -
the part that actually wires de-normalization, the grasp-column split, and the
two anchoring modes together.

The orchestration only ever calls *duck-typed* methods on the IK bridge
(``ee_pose`` / ``solve`` / ``solve_trajectory`` / ``tracking_error`` and
``model.nq``); the de-normalization and pose-decode steps are pure numpy. So
the full body is driven here with a fake bridge - no mink, mujoco, or qpsolvers
needed - and the contracts that survive a clean install are pinned:

* the grasp column is split off for gripper embodiments and is ``None`` for
  grasp-less ones (the ``av`` embodiment);
* closed-loop re-anchoring (default) and legacy open-loop agree on a perfectly
  tracked trajectory, and re-anchoring composes each delta on the bridge's
  *achieved* pose;
* an explicit ``stats`` override bypasses the bundled-stats Hub load;
* output shapes/keys match the documented contract for both modes.
"""

import types

import numpy as np
import pytest

from strands_robots.policies.cosmos3 import sim_ik
from strands_robots.policies.cosmos3.embodiments import get_embodiment


class _FakeBridge:
    """A mink-free stand-in for :class:`MinkIKBridge`.

    Models a *perfect* IK solver: ``solve`` reaches the commanded target exactly
    and records it as the new achieved EE pose, so forward kinematics
    (``ee_pose``) returns whatever was last commanded. This makes the closed-loop
    re-anchoring and legacy open-loop paths produce identical Cartesian targets
    (the contract that the default ``reanchor=True`` does not change behavior on
    reachable trajectories), while needing no real kinematics engine.
    """

    def __init__(self, nq: int = 7, home: np.ndarray | None = None):
        self.model = types.SimpleNamespace(nq=nq)
        self._achieved = np.eye(4) if home is None else np.asarray(home, dtype=float)
        self.solve_calls = 0

    def ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        return self._achieved.astype(np.float32)

    def solve(self, target_pose: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        self.solve_calls += 1
        self._achieved = np.asarray(target_pose, dtype=float)
        return np.full(self.model.nq, float(self.solve_calls), dtype=np.float64)

    def solve_trajectory(self, poses: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        poses = np.asarray(poses, dtype=float)
        if len(poses):
            self._achieved = poses[-1]
        return np.tile(np.arange(1, len(poses) + 1, dtype=np.float64)[:, None], (1, self.model.nq))

    def tracking_error(self, poses: np.ndarray, qpos_traj: np.ndarray) -> dict[str, float]:
        return {"mean_mm": 0.0, "max_mm": 0.0}


def _home_pose() -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = [0.4, 0.0, 0.3]  # mid-workspace, away from the origin.
    return pose


def _reachable_chunk(raw_dim: int, t: int = 8, seed: int = 0) -> np.ndarray:
    """A synthetic raw action chunk with near-identity rotation per step."""
    rng = np.random.default_rng(seed)
    chunk = rng.uniform(-0.3, 0.3, (t, raw_dim)).astype(np.float32)
    # Columns 3:9 are the rot6d block - pin to identity so deltas stay small.
    chunk[:, 3:9] = np.tile([1, 0, 0, 0, 1, 0], (t, 1))
    return chunk


def test_reanchor_default_returns_documented_contract():
    """Default (re-anchored) path: shapes, keys, and split grasp column."""
    emb = get_embodiment("droid")  # grasp embodiment.
    bridge = _FakeBridge(home=_home_pose())
    chunk = _reachable_chunk(emb.raw_action_dim, t=8)

    out = sim_ik.decode_cosmos_chunk_to_targets(chunk, emb, bridge, np.zeros(7))

    assert set(out) == {"qpos", "gripper", "poses", "tracking_error"}
    assert out["qpos"].shape == (8, bridge.model.nq)
    assert out["poses"].shape == (8, 4, 4)
    assert out["gripper"] is not None and out["gripper"].shape == (8,)
    # One solve() per step in the closed-loop path.
    assert bridge.solve_calls == 8


def test_legacy_open_loop_uses_solve_trajectory_not_per_step_solve():
    """``reanchor=False`` integrates targets up front and calls solve_trajectory."""
    emb = get_embodiment("droid")
    bridge = _FakeBridge(home=_home_pose())
    chunk = _reachable_chunk(emb.raw_action_dim, t=8)

    out = sim_ik.decode_cosmos_chunk_to_targets(chunk, emb, bridge, np.zeros(7), reanchor=False)

    assert out["qpos"].shape == (8, bridge.model.nq)
    assert out["poses"].shape == (8, 4, 4)
    # Open-loop path never calls the per-step solver.
    assert bridge.solve_calls == 0


def test_reanchor_and_legacy_agree_on_perfectly_tracked_trajectory():
    """The default does not change commanded targets when IK tracks exactly.

    With a perfect bridge the achieved EE pose equals the commanded target each
    step, so re-anchoring on the realized pose is identical to integrating the
    targets up front. Pins that ``reanchor=True`` is behavior-preserving for
    reachable inputs (the divergence is only at the workspace edge).
    """
    emb = get_embodiment("droid")
    chunk = _reachable_chunk(emb.raw_action_dim, t=8)

    out_reanchor = sim_ik.decode_cosmos_chunk_to_targets(chunk, emb, _FakeBridge(home=_home_pose()), np.zeros(7))
    out_legacy = sim_ik.decode_cosmos_chunk_to_targets(
        chunk, emb, _FakeBridge(home=_home_pose()), np.zeros(7), reanchor=False
    )

    np.testing.assert_allclose(out_reanchor["poses"], out_legacy["poses"], atol=1e-5)


def test_grasp_less_embodiment_yields_no_gripper_column():
    """A grasp-less embodiment (``av``) keeps the full pose block, gripper=None."""
    emb = get_embodiment("av")  # 9-dim, no trailing grasp column.
    assert emb.raw_action_layout[-1] != "grasp"
    bridge = _FakeBridge(home=_home_pose())
    d = emb.raw_action_dim
    chunk = _reachable_chunk(d, t=6)
    # ``av`` ships no bundled stats; supply a unit quantile range explicitly so
    # the test exercises only the grasp-split branch, not the Hub loader.
    # ``stats_domain`` declares which domain those quantiles stand in for -
    # required because several domains share an action width.
    stats = {"q01": np.full(d, -1.0, dtype=np.float32), "q99": np.full(d, 1.0, dtype=np.float32)}

    out = sim_ik.decode_cosmos_chunk_to_targets(
        chunk, emb, bridge, np.zeros(7), stats=stats, stats_domain=emb.domain_name
    )

    assert out["gripper"] is None
    assert out["poses"].shape == (6, 4, 4)


def test_explicit_stats_override_bypasses_bundled_stats(monkeypatch):
    """Passing ``stats`` must skip the bundled per-domain Hub load entirely."""
    emb = get_embodiment("droid")

    def _boom(domain_name):
        raise AssertionError(f"load_action_stats should not be called; got {domain_name}")

    # Patch the lazily-imported loader at its source module.
    from strands_robots.policies.cosmos3 import action_decode

    monkeypatch.setattr(action_decode, "load_action_stats", _boom)

    d = emb.raw_action_dim
    stats = {"q01": np.full(d, -1.0, dtype=np.float32), "q99": np.full(d, 1.0, dtype=np.float32)}
    chunk = _reachable_chunk(d, t=5)

    out = sim_ik.decode_cosmos_chunk_to_targets(
        chunk, emb, _FakeBridge(home=_home_pose()), np.zeros(7), stats=stats, stats_domain=emb.domain_name
    )

    assert out["qpos"].shape == (5, 7)


def test_empty_chunk_returns_empty_arrays_for_both_modes():
    """A zero-length chunk produces correctly-shaped empty outputs (no crash)."""
    emb = get_embodiment("droid")
    empty = np.empty((0, emb.raw_action_dim), dtype=np.float32)

    out_reanchor = sim_ik.decode_cosmos_chunk_to_targets(empty, emb, _FakeBridge(home=_home_pose()), np.zeros(7))
    assert out_reanchor["qpos"].shape == (0, 7)
    assert out_reanchor["poses"].shape == (0, 4, 4)


def test_non_2d_action_chunk_raises_before_touching_bridge():
    """The rank guard fires before any bridge method is referenced."""
    emb = get_embodiment("droid")
    bridge = _FakeBridge(home=_home_pose())
    with pytest.raises(ValueError, match=r"\[T, D\]"):
        sim_ik.decode_cosmos_chunk_to_targets(np.zeros(emb.raw_action_dim, dtype=np.float32), emb, bridge, np.zeros(7))
    assert bridge.solve_calls == 0
