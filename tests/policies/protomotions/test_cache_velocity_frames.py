"""The reference-motion cache's velocity channels are world-frame, and say so.

``MotionPlayer`` accepts a hand-built cache dict as a first-class input mode, so
the frame of its ``body_vel`` / ``body_ang_vel`` rows is a contract a caller has
to be able to read. The frame is not a detail: the tracker's own root input is a
*local*-frame angular velocity that
:func:`~strands_robots.policies.protomotions.state_utils.compute_root_local_ang_vel`
derives by rotating a world-frame row, so a cache built in the local frame is
rotated a second time instead of used as-is. On a walking G1 clip the two frames
differ by whole rad/s.

These tests grade every frame claim in the package against what the code
measurably produces, rather than against each other, so the claim and the
behaviour cannot drift apart in either direction.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.protomotions import bridge, motion_utils, state_utils
from strands_robots.policies.protomotions.state_utils import quat_rotate_inverse

from .test_cache_body_rows_match_the_tracker_order import g1_like_mjcf

# A quarter turn about world X, xyzw. Chosen so a world-Z spin has a *different*
# local-frame representation - a frame-agnostic pose would make every assertion
# below pass for either convention.
TILT_X90_XYZW = np.array([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])
OMEGA = 0.4  # rad/s about world Z
FPS = 200.0


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``xyzw`` quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _world_z_spin_of_a_tilted_body(frames: int, dt: float) -> np.ndarray:
    """``[T, 1, 4]`` xyzw trajectory: a tilted body spinning about world Z."""
    out = np.zeros((frames, 1, 4))
    for t in range(frames):
        half = OMEGA * t * dt * 0.5
        spin = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
        out[t, 0] = _qmul(spin, TILT_X90_XYZW)  # left-multiply == world-frame spin
    return out


def _measured_frame_of_the_helper() -> str:
    """Return ``"world"`` or ``"local"`` - whichever the helper actually emits."""
    dt = 1.0 / FPS
    quats = _world_z_spin_of_a_tilted_body(40, dt)
    mid = 20
    got = bridge._quat_finite_diff_ang_vel(quats, dt)[mid, 0]
    world = np.array([0.0, 0.0, OMEGA], dtype=np.float32)
    local = quat_rotate_inverse(quats[mid, 0].astype(np.float32), world)
    assert not np.allclose(world, local, atol=1e-3), "premise: the probe pose must distinguish the two frames"
    return "world" if np.linalg.norm(got - world) < np.linalg.norm(got - local) else "local"


# --- docstring reading -----------------------------------------------------

_FRAME_RE = re.compile(r"\b(world|local)[- ]frame\b", re.IGNORECASE)


def _frames_named(text: str) -> set[str]:
    """Frame words named in ``text``, lowercased."""
    return {m.group(1).lower() for m in _FRAME_RE.finditer(text)}


def _section(doc: str, name: str) -> str:
    """The body of a Google-style ``name:`` section of ``doc`` (or "")."""
    lines = (doc or "").splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"{name}:":
            indent = len(line) - len(line.lstrip())
            body = []
            for nxt in lines[i + 1 :]:
                if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                    break
                body.append(nxt)
            return "\n".join(body)
    return ""


def _docstrings_in_the_package() -> list[tuple[str, str]]:
    """``(qualname, docstring)`` for every module/class/function in the package."""
    pkg_dir = pathlib.Path(inspect.getfile(state_utils)).parent
    out: list[tuple[str, str]] = []
    for path in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(path.read_text())
        mod_doc = ast.get_docstring(tree)
        if mod_doc:
            out.append((f"{path.name}:<module>", mod_doc))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                doc = ast.get_docstring(node)
                if doc:
                    out.append((f"{path.name}:{node.name}", doc))
    return out


def _multi_body_ang_vel_entries() -> list[tuple[str, str, str]]:
    """``(qualname, section, line)`` for each per-body angular-velocity claim.

    An entry qualifies when one line of an ``Args:``/``Returns:`` section names
    both a ``num_bodies`` axis and an angular velocity - i.e. it describes the
    full body array, not the single root row the tracker consumes.
    """
    found: list[tuple[str, str, str]] = []
    for qualname, doc in _docstrings_in_the_package():
        for section in ("Args", "Returns"):
            for line in _section(doc, section).splitlines():
                if "num_bodies" in line and re.search(r"angular velocit", line, re.IGNORECASE):
                    found.append((qualname, section, line.strip()))
    return found


# --- the frame the code produces -------------------------------------------


class TestTheFiniteDiffHelperProducesAWorldFrameVelocity:
    """Pin the frame itself, so a change to the math has to change these."""

    def test_a_world_axis_spin_of_a_tilted_body_is_reported_in_the_world_frame(self) -> None:
        dt = 1.0 / FPS
        quats = _world_z_spin_of_a_tilted_body(40, dt)
        got = bridge._quat_finite_diff_ang_vel(quats, dt)[20, 0]
        assert got == pytest.approx([0.0, 0.0, OMEGA], abs=1e-4)

    def test_it_is_not_the_body_local_representation_of_that_same_spin(self) -> None:
        dt = 1.0 / FPS
        quats = _world_z_spin_of_a_tilted_body(40, dt)
        got = bridge._quat_finite_diff_ang_vel(quats, dt)[20, 0]
        local = quat_rotate_inverse(quats[20, 0].astype(np.float32), np.array([0.0, 0.0, OMEGA], dtype=np.float32))
        assert np.linalg.norm(got - local) > 0.1, (
            f"the two frames must stay distinguishable here: got {got}, local would be {local}"
        )

    def test_a_row_of_it_composes_with_compute_root_local_ang_vel(self) -> None:
        """The pair contract: rotating one row yields the local-frame quantity."""
        dt = 1.0 / FPS
        quats = _world_z_spin_of_a_tilted_body(40, dt)
        body_ang_vel = bridge._quat_finite_diff_ang_vel(quats, dt)
        got_local = state_utils.compute_root_local_ang_vel(quats[20].astype(np.float32), body_ang_vel[20], 0)
        want_local = quat_rotate_inverse(quats[20, 0].astype(np.float32), np.array([0.0, 0.0, OMEGA], dtype=np.float32))
        assert got_local == pytest.approx(want_local, abs=1e-3)


# --- every claim matches that frame ----------------------------------------


class TestEveryFrameClaimNamesTheFrameTheCodeProduces:
    """A frame word is a claim about behaviour; grade it against the behaviour."""

    def test_no_per_body_angular_velocity_claim_calls_the_array_local_frame(self) -> None:
        measured = _measured_frame_of_the_helper()
        entries = _multi_body_ang_vel_entries()
        assert entries, "premise: the package must document at least one per-body angular-velocity array"
        wrong = [(q, s, line) for q, s, line in entries if _frames_named(line) - {measured}]
        assert not wrong, (
            f"the per-body angular-velocity array is measurably {measured}-frame, but "
            f"{len(wrong)} documented entr(y/ies) name another frame: "
            + "; ".join(f"{q} {s}: {line!r}" for q, s, line in wrong)
        )

    def test_the_helper_returns_section_names_a_frame_and_it_is_the_measured_one(self) -> None:
        measured = _measured_frame_of_the_helper()
        returns = _section(bridge._quat_finite_diff_ang_vel.__doc__ or "", "Returns")
        assert _frames_named(returns) == {measured}, (
            f"_quat_finite_diff_ang_vel returns a {measured}-frame velocity but its "
            f"Returns section names {sorted(_frames_named(returns)) or 'no frame'}: {returns.strip()!r}"
        )

    def test_the_public_bridge_returns_section_names_the_velocity_frame(self) -> None:
        returns = _section(bridge.qpos_to_motion_data.__doc__ or "", "Returns")
        assert _frames_named(returns) == {"world"}, (
            "qpos_to_motion_data builds the cache's velocity channels, so its Returns "
            f"section has to say which frame they are in; it names {sorted(_frames_named(returns)) or 'none'}"
        )

    def test_the_cache_format_contract_names_the_frame_of_both_velocity_channels(self) -> None:
        doc = motion_utils.__doc__ or ""
        bullets = {
            channel: [ln for ln in doc.splitlines() if f"``{channel}``" in ln]
            for channel in ("body_vel", "body_ang_vel")
        }
        for channel, lines in bullets.items():
            assert lines, f"premise: the cache-format list must describe {channel}"
            assert any("world" in ln.lower() for ln in lines), (
                f"a hand-built cache is a documented input mode, so the cache-format entry for "
                f"{channel} has to name its frame; it reads {lines[0].strip()!r}"
            )


# --- end to end through MuJoCo forward kinematics --------------------------


class TestTheBridgedCacheIsWorldFrameEndToEnd:
    """Drive the real public entry point on a hermetic 36-dof MJCF."""

    @staticmethod
    def _mjcf(tmp_path: pathlib.Path) -> pathlib.Path:
        """The tracker's embodiment: a freejoint root plus 29 hinges, ``nq`` 36.

        Carries the tracker's own body and joint names, because the bridge fills
        the cache's rows against
        :data:`~strands_robots.policies.protomotions.config.GTP_G1_BODY_NAMES` and
        refuses a model that cannot supply them. Row 0 is still the root, which is
        the row every assertion below reads.
        """
        return g1_like_mjcf(tmp_path / "tracker36.xml")

    @staticmethod
    def _qpos(frames: int, dt: float, speed: float) -> np.ndarray:
        """Root tilted a quarter turn about Y, spinning about world Z, sliding along world X."""
        tilt_y90 = np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
        qpos = np.zeros((frames, 36))
        for t in range(frames):
            half = OMEGA * t * dt * 0.5
            spin = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
            q_xyzw = _qmul(spin, tilt_y90)
            qpos[t, 0] = speed * t * dt
            qpos[t, 2] = 1.0
            qpos[t, 3:7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # wxyz for MuJoCo
        return qpos

    def _cache(self, tmp_path: pathlib.Path, speed: float = 0.0, control_dt: float | None = None) -> dict[str, Any]:
        pytest.importorskip("mujoco")
        dt = 1.0 / FPS
        return bridge.qpos_to_motion_data(
            self._qpos(40, dt, speed),
            fps=FPS,
            proto_mjcf_path=self._mjcf(tmp_path),
            control_dt=dt if control_dt is None else control_dt,
        )

    def test_the_root_row_of_body_ang_vel_is_the_world_frame_spin(self, tmp_path: pathlib.Path) -> None:
        cache = self._cache(tmp_path)
        got = cache["body_ang_vel"][20, 0]
        assert got == pytest.approx([0.0, 0.0, OMEGA], abs=1e-3)

    def test_resampling_does_not_change_the_frame(self, tmp_path: pathlib.Path) -> None:
        cache = self._cache(tmp_path, control_dt=2.0 / FPS)
        got = cache["body_ang_vel"][10, 0]
        assert got == pytest.approx([0.0, 0.0, OMEGA], abs=1e-3)

    def test_body_vel_is_a_world_frame_linear_velocity(self, tmp_path: pathlib.Path) -> None:
        speed = 0.7
        cache = self._cache(tmp_path, speed=speed)
        got = cache["body_vel"][20, 0]
        assert got == pytest.approx([speed, 0.0, 0.0], abs=1e-3)
        local = quat_rotate_inverse(cache["body_rot"][20, 0], np.array([speed, 0.0, 0.0], dtype=np.float32))
        assert np.linalg.norm(got - local) > 0.1, (
            f"premise: the probe pose must distinguish the frames (local would be {local})"
        )
