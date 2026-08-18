"""The reference-motion cache's body rows are the order the tracker indexes.

The tracker does not look a body up by name. ``ProtoMotionsPolicy`` reads its
anchor orientation as ``future["body_rot"][:, config.anchor_body_index]``, and
``anchor_body_index``/``root_body_index`` are offsets into
:data:`~strands_robots.policies.protomotions.config.GTP_G1_BODY_NAMES` - a fixed
33-name list pinned from the checkpoint's own sidecar, whose comment states the
order outright ("Index 0 = pelvis (root), 16 = torso_link (anchor)").

:func:`~strands_robots.policies.protomotions.bridge.qpos_to_motion_data` fills
those rows by running forward kinematics through a caller-supplied MJCF. A G1
variant with a different body set therefore yields rows in *its* order, and every
row after the first gap names a different link than the tracker asked for. The
G1 family is full of such variants - the fingerless 30-body models omit ``head``
and both ``rubber_hand`` placeholders - so the pairing has to be made by name,
and refused when a body the tracker reads is absent, rather than taken on trust
positionally.

These tests grade the cache's rows against an independent MuJoCo oracle that
resolves each body by name, so a row can never be confirmed by the same
index arithmetic that produced it.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.protomotions import bridge
from strands_robots.policies.protomotions.config import (
    GTP_G1_ANCHOR_BODY_INDEX,
    GTP_G1_BODY_NAMES,
    GTP_G1_JOINT_NAMES,
    GTP_G1_ROOT_BODY_INDEX,
)

FPS = 50.0
FRAMES = 12

# The three bodies the tracker reads that carry no joint of their own. Real G1
# MJCFs either ship them as fixed placeholder links or omit them entirely.
JOINTLESS = ("head", "left_rubber_hand", "right_rubber_hand")


def g1_like_mjcf(
    path: pathlib.Path,
    *,
    body_names: tuple[str, ...] = GTP_G1_BODY_NAMES,
    joint_names: tuple[str, ...] = GTP_G1_JOINT_NAMES,
    free_root: bool = True,
    extra_hinges: int = 0,
) -> pathlib.Path:
    """Write a hermetic G1-shaped chain MJCF carrying the given names.

    MuJoCo orders bodies depth-first, so a single chain makes the compiled body
    order exactly ``body_names`` - which is what lets a test choose an order that
    does or does not match the tracker's.

    Each jointed link hinges about a different axis so no two bodies share a world
    rotation; a fixture whose bodies rotated together could not tell a
    correctly-aligned row from a shifted one.

    Args:
        path: File to write.
        body_names: Body names, outermost first. ``body_names[0]`` is the root.
        joint_names: Hinge names handed out to the links in order. Bodies named in
            :data:`JOINTLESS` are skipped, mirroring the real model.
        free_root: Write a ``<freejoint/>`` on the root. When ``False`` the root's
            seven qpos entries are made up with hinges instead, so ``nq`` is
            unchanged and only the root's joint TYPE differs.
        extra_hinges: Additional hinges, widening ``nq`` past the tracker's.

    Returns:
        ``path``, for convenience.
    """
    axes = ("1 0 0", "0 1 0", "0 0 1")
    joints: list[list[str]] = [[] for _ in body_names]
    dofs = [0] * len(body_names)

    if free_root:
        joints[0].append("<freejoint/>")
        dofs[0] = 6  # a free joint fills the per-body dof budget

    pending = list(joint_names)
    for index, name in enumerate(body_names):
        if index == 0 or name in JOINTLESS or not pending:
            continue
        joints[index].append(f'<joint name="{pending.pop(0)}" type="hinge" axis="{axes[index % 3]}"/>')
        dofs[index] += 1
    assert not pending, f"chain has no room for {pending}"

    # MuJoCo caps a body at six dofs, so filler hinges spread along the chain.
    remaining = (0 if free_root else 7) + extra_hinges
    made = 0
    for index in range(len(body_names)):
        while remaining and dofs[index] < 6:
            joints[index].append(f'<joint name="filler{made}" type="hinge" axis="{axes[made % 3]}"/>')
            dofs[index] += 1
            remaining -= 1
            made += 1
    assert not remaining, "chain has no room for the requested hinges"

    chunks: list[str] = []
    for index, name in enumerate(body_names):
        chunks.append(f'<body name="{name}" pos="0 0.01 0.04">')
        chunks.extend(joints[index])
        chunks.append('<geom type="box" size="0.02 0.02 0.02" mass="0.1"/>')
    body_xml = "".join(chunks) + "</body>" * len(body_names)
    path.write_text(f"<mujoco><worldbody>{body_xml}</worldbody></mujoco>")
    return path


def _qpos(frames: int = FRAMES) -> np.ndarray:
    """A ``[T, 36]`` clip: the root turning about world Z, every hinge sweeping."""
    qpos = np.zeros((frames, 36))
    for t in range(frames):
        half = 0.35 * t / max(frames - 1, 1)
        qpos[t, 2] = 1.0
        qpos[t, 3] = np.cos(half)
        qpos[t, 6] = np.sin(half)
        qpos[t, 7:] = 0.25 * np.sin(np.arange(29) * 0.7 + t * 0.2)
    return qpos


def _world_rotation_by_name(mjcf: pathlib.Path, qpos_frame: np.ndarray) -> dict[str, np.ndarray]:
    """Oracle: each body's world ``xyzw`` rotation, resolved by NAME not index."""
    mujoco = pytest.importorskip("mujoco")
    mj_model, mj_data = bridge._patch_and_load_mjcf(mjcf)[1:]
    mj_data.qpos[:] = qpos_frame
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    out: dict[str, np.ndarray] = {}
    for body_id in range(1, mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name:
            w, x, y, z = mj_data.xquat[body_id]
            out[name] = np.array([x, y, z, w], dtype=np.float64)
    return out


#: Componentwise tolerance for one cached rotation against the float64 oracle.
#: Well above float32 storage error and far below any two distinct G1 links.
ROW_ATOL = 1e-5


def _same_rotation(cached: np.ndarray, oracle: np.ndarray) -> bool:
    """Whether a cached ``xyzw`` row is the oracle rotation, up to float32 storage.

    Compared componentwise rather than as a geodesic angle: ``arccos`` is
    ill-conditioned near identity, where float32 storage alone reads as tenths of
    a degree, which would make the tolerance say more about the metric than about
    the row.
    """
    both = np.asarray(oracle, dtype=np.float64)
    return bool(
        np.allclose(np.asarray(cached, dtype=np.float64), both, atol=ROW_ATOL)
        or np.allclose(np.asarray(cached, dtype=np.float64), -both, atol=ROW_ATOL)
    )


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two ``xyzw`` rotations, in degrees - for messages."""
    a = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    return float(2.0 * np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def _bridge(mjcf: pathlib.Path) -> dict[str, Any]:
    pytest.importorskip("mujoco")
    return bridge.qpos_to_motion_data(_qpos(), fps=FPS, proto_mjcf_path=mjcf, control_dt=1.0 / FPS)


# A complete embodiment whose compiled body order is NOT the tracker's: `head`
# moved to the end of the chain. Every body the tracker reads is present, so the
# cache is fully servable - only the positional pairing is wrong.
_SHUFFLED = (GTP_G1_BODY_NAMES[0], *GTP_G1_BODY_NAMES[2:], GTP_G1_BODY_NAMES[1])


class TestEveryCacheRowHoldsTheBodyTheTrackerReadsThere:
    """Rows are paired to the tracker's list by name, not by MuJoCo body index."""

    def test_the_anchor_row_holds_the_anchor_body(self, tmp_path: pathlib.Path) -> None:
        mjcf = g1_like_mjcf(tmp_path / "shuffled.xml", body_names=_SHUFFLED)
        cache = _bridge(mjcf)
        oracle = _world_rotation_by_name(mjcf, _qpos()[0])
        anchor_name = GTP_G1_BODY_NAMES[GTP_G1_ANCHOR_BODY_INDEX]
        row = cache["body_rot"][0, GTP_G1_ANCHOR_BODY_INDEX]
        error = _angle_deg(row, oracle[anchor_name])
        nearest = min(oracle, key=lambda n: _angle_deg(row, oracle[n]))
        assert _same_rotation(row, oracle[anchor_name]), (
            f"the tracker reads row {GTP_G1_ANCHOR_BODY_INDEX} as its {anchor_name!r} anchor, but that row is "
            f"{error:.2f} deg from {anchor_name!r} and matches {nearest!r} instead"
        )

    def test_the_root_row_holds_the_root_body(self, tmp_path: pathlib.Path) -> None:
        mjcf = g1_like_mjcf(tmp_path / "shuffled.xml", body_names=_SHUFFLED)
        cache = _bridge(mjcf)
        oracle = _world_rotation_by_name(mjcf, _qpos()[0])
        root_name = GTP_G1_BODY_NAMES[GTP_G1_ROOT_BODY_INDEX]
        row = cache["body_rot"][0, GTP_G1_ROOT_BODY_INDEX]
        error = _angle_deg(row, oracle[root_name])
        assert _same_rotation(row, oracle[root_name]), (
            f"row {GTP_G1_ROOT_BODY_INDEX} is {error:.2f} deg from the root body {root_name!r}"
        )

    def test_no_row_names_a_different_body_than_the_tracker_expects(self, tmp_path: pathlib.Path) -> None:
        mjcf = g1_like_mjcf(tmp_path / "shuffled.xml", body_names=_SHUFFLED)
        cache = _bridge(mjcf)
        oracle = _world_rotation_by_name(mjcf, _qpos()[0])
        wrong = [
            (row, name, round(_angle_deg(cache["body_rot"][0, row], oracle[name]), 2))
            for row, name in enumerate(GTP_G1_BODY_NAMES)
            if not _same_rotation(cache["body_rot"][0, row], oracle[name])
        ]
        assert not wrong, f"{len(wrong)} of {len(GTP_G1_BODY_NAMES)} rows hold another body's rotation: {wrong[:4]}"

    def test_body_positions_are_aligned_the_same_way_as_rotations(self, tmp_path: pathlib.Path) -> None:
        mujoco = pytest.importorskip("mujoco")
        mjcf = g1_like_mjcf(tmp_path / "shuffled.xml", body_names=_SHUFFLED)
        cache = _bridge(mjcf)
        mj_model, mj_data = bridge._patch_and_load_mjcf(mjcf)[1:]
        mj_data.qpos[:] = _qpos()[0]
        mujoco.mj_forward(mj_model, mj_data)
        anchor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, GTP_G1_BODY_NAMES[GTP_G1_ANCHOR_BODY_INDEX])
        assert cache["body_pos"][0, GTP_G1_ANCHOR_BODY_INDEX] == pytest.approx(mj_data.xpos[anchor_id], abs=1e-5)


class TestAnMjcfThatIsNotTheTrackerEmbodimentIsRefused:
    """A model that cannot fill the tracker's rows is refused, not shifted."""

    @staticmethod
    def _fingerless(tmp_path: pathlib.Path) -> pathlib.Path:
        """The real 30-body shape: no ``head``, no ``rubber_hand`` placeholders."""
        kept = tuple(n for n in GTP_G1_BODY_NAMES if n not in JOINTLESS)
        return g1_like_mjcf(tmp_path / "fingerless.xml", body_names=kept)

    def test_a_variant_missing_tracker_bodies_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError) as excinfo:
            _bridge(self._fingerless(tmp_path))
        assert "missing" in str(excinfo.value).lower()

    def test_the_refusal_names_every_missing_body(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError) as excinfo:
            _bridge(self._fingerless(tmp_path))
        message = str(excinfo.value)
        assert [name for name in JOINTLESS if name not in message] == [], message

    def test_the_refusal_says_the_rows_are_read_by_index(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError) as excinfo:
            _bridge(self._fingerless(tmp_path))
        message = str(excinfo.value)
        assert "row index" in message and "GTP_G1_BODY_NAMES" in message, message

    def test_a_model_whose_qpos_layout_is_wider_is_refused_as_a_model_problem(self, tmp_path: pathlib.Path) -> None:
        mjcf = g1_like_mjcf(tmp_path / "wide.xml", extra_hinges=14)  # nq=50, as a G1 with hands
        with pytest.raises(ValueError) as excinfo:
            _bridge(mjcf)
        message = str(excinfo.value)
        assert str(mjcf) in message, f"a model-side mismatch has to name the model: {message}"
        assert "layout" in message and "nq=" in message, message

    def test_a_model_with_no_free_root_is_refused(self, tmp_path: pathlib.Path) -> None:
        mjcf = g1_like_mjcf(tmp_path / "fixed_base.xml", free_root=False)
        with pytest.raises(ValueError) as excinfo:
            _bridge(mjcf)
        assert "free root" in str(excinfo.value), str(excinfo.value)

    def test_the_layout_refusal_distinguishes_itself_from_a_bad_qpos_argument(self, tmp_path: pathlib.Path) -> None:
        mjcf = g1_like_mjcf(tmp_path / "fixed_base.xml", free_root=False)
        with pytest.raises(ValueError) as excinfo:
            _bridge(mjcf)
        assert "not the qpos argument" in str(excinfo.value).replace("'", ""), str(excinfo.value)


class TestTheTrackerOwnEmbodimentIsUnchanged:
    """Controls: the accepted path keeps every verdict it already had."""

    def test_the_tracker_order_model_produces_the_declared_row_count(self, tmp_path: pathlib.Path) -> None:
        cache = _bridge(g1_like_mjcf(tmp_path / "tracker.xml"))
        assert cache["body_rot"].shape[1] == len(GTP_G1_BODY_NAMES)
        assert cache["body_pos"].shape[1] == len(GTP_G1_BODY_NAMES)

    def test_the_declared_anchor_and_root_indices_are_in_range(self, tmp_path: pathlib.Path) -> None:
        cache = _bridge(g1_like_mjcf(tmp_path / "tracker.xml"))
        rows = cache["body_rot"].shape[1]
        assert GTP_G1_ANCHOR_BODY_INDEX < rows and GTP_G1_ROOT_BODY_INDEX < rows

    def test_the_joint_block_is_still_read_straight_off_qpos(self, tmp_path: pathlib.Path) -> None:
        cache = _bridge(g1_like_mjcf(tmp_path / "tracker.xml"))
        assert cache["dof_pos"].shape[1] == len(GTP_G1_JOINT_NAMES)
        assert cache["dof_pos"][0] == pytest.approx(_qpos()[0, 7:], abs=1e-5)

    def test_a_too_narrow_qpos_is_still_refused_as_a_qpos_problem(self, tmp_path: pathlib.Path) -> None:
        pytest.importorskip("mujoco")
        mjcf = g1_like_mjcf(tmp_path / "tracker.xml")
        with pytest.raises(ValueError) as excinfo:
            bridge.qpos_to_motion_data(_qpos()[:, :30], fps=FPS, proto_mjcf_path=mjcf, control_dt=1.0 / FPS)
        assert "qpos" in str(excinfo.value)

    def test_a_padded_qpos_is_still_truncated_rather_than_refused(self, tmp_path: pathlib.Path) -> None:
        pytest.importorskip("mujoco")
        mjcf = g1_like_mjcf(tmp_path / "tracker.xml")
        padded = np.concatenate([_qpos(), np.zeros((FRAMES, 4))], axis=1)
        cache = bridge.qpos_to_motion_data(padded, fps=FPS, proto_mjcf_path=mjcf, control_dt=1.0 / FPS)
        assert cache["dof_pos"].shape[1] == len(GTP_G1_JOINT_NAMES)


class TestTheDocumentedRowOrderMatchesTheCode:
    """The Returns section is the only place a caller can read the row order."""

    def test_the_returns_section_names_the_body_row_order(self) -> None:
        doc = bridge.qpos_to_motion_data.__doc__ or ""
        returns = re.split(r"\n\s*Raises:", re.split(r"\n\s*Returns:", doc)[1])[0]
        assert "GTP_G1_BODY_NAMES" in returns, (
            "the tracker indexes the cache's body rows positionally, so the Returns "
            f"section has to name the order they are in; it reads {returns.strip()[:160]!r}"
        )
