"""MotionPlayer - a fixed-rate reference-motion window for the ProtoMotions tracker.

The ONNX Generalist Tracking Policy (GTP) that
:class:`~strands_robots.policies.protomotions.policy.ProtoMotionsPolicy` wraps
expects, each control tick, a window of FUTURE reference frames: joint pos +
joint vel + anchor rotation at a handful of step-offsets ahead (default
``[1, 2, 4, 8]`` control steps). This module is the source of that window.

Two input modes:

1. A pre-resampled cache dict (from
   :func:`~strands_robots.policies.protomotions.bridge.qpos_to_motion_data` or
   from an on-disk ``.npz`` produced by this module) - fast, no torch.
2. A raw motion library or single-motion ``.pt`` file in the ProtoMotions
   format. Handled lazily so a caller with only a cache dict never imports
   torch or scipy.

Cache format (dict-of-numpy):

* ``dof_pos``       - ``[num_frames, num_dofs]``   float32
* ``dof_vel``       - ``[num_frames, num_dofs]``   float32
* ``body_rot``      - ``[num_frames, num_bodies, 4]``  float32 (xyzw, world)
* ``body_pos``      - ``[num_frames, num_bodies, 3]``  float32 (world)
* ``body_vel``      - ``[num_frames, num_bodies, 3]``  float32 (world frame)
* ``body_ang_vel``  - ``[num_frames, num_bodies, 3]``  float32 (world frame)
* ``control_dt``    - python float, seconds per control tick
* ``num_frames``    - python int

The two velocity channels are WORLD-frame, the same convention as a raw
ProtoMotions motion library's ``rigid_body_ang_vel``. A hand-built cache must
follow it: the tracker's own root input is a local-frame angular velocity that
:func:`~strands_robots.policies.protomotions.state_utils.compute_root_local_ang_vel`
derives by rotating a world-frame row, so supplying local-frame rows here is
rotated a second time rather than used as-is.

Clean-room from ProtoMotions (Apache 2.0) deployment/motion_utils.py.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np

from strands_robots.utils import require_optional

__all__ = ["MotionPlayer", "slerp", "lerp"]

logger = logging.getLogger(__name__)

# The keys a full motion state carries. Kept as a module constant so a caller
# a stubs one for a test can round-trip identical field names.
_STATE_KEYS = ("dof_pos", "dof_vel", "body_rot", "body_pos", "body_vel", "body_ang_vel")


# ---------------------------------------------------------------------------
# Vendored quaternion + scalar interpolation
# ---------------------------------------------------------------------------


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Unit-normalise along the last axis with a soft floor."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-8)


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray | np.floating[Any] | float) -> np.ndarray:
    """Spherical linear interpolation between two ``xyzw`` quaternions.

    Falls back to linear interpolation on near-parallel pairs so the result is
    stable when ``sin(theta) -> 0``.

    Args:
        q0: Shape ``[..., 4]`` start quaternion.
        q1: Shape ``[..., 4]`` end quaternion.
        t: Interpolation factor in ``[0, 1]`` - a python float, a numpy
            scalar, or an array broadcastable against ``q0``.

    Returns:
        Shape ``[..., 4]`` interpolated, unit-normalised.
    """
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)  # ensure shortest path
    dot = np.abs(dot).clip(-1.0, 1.0)

    if np.ndim(t) < np.ndim(dot):
        t = np.asarray(t)[..., np.newaxis]

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    s0 = np.sin((1.0 - t) * theta) / np.maximum(sin_theta, 1e-8)
    s1 = np.sin(t * theta) / np.maximum(sin_theta, 1e-8)
    result = s0 * q0 + s1 * q1

    # Linear-blend fallback where sin(theta) ~ 0 (parallel quats).
    linear_result = (1.0 - t) * q0 + t * q1
    use_linear = np.abs(sin_theta) < 1e-6
    result = np.where(use_linear, linear_result, result)
    return _normalize_quat(result)


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray | np.floating[Any] | float) -> np.ndarray:
    """Linear interpolation ``a + t * (b - a)`` with sensible broadcasting.

    Args:
        a: Any-shape array.
        b: Same shape as ``a``.
        t: Scalar or broadcast-compatible factor.

    Returns:
        Interpolated array, same shape as ``a``.
    """
    if np.ndim(t) < np.ndim(a):
        t = np.asarray(t)[..., np.newaxis]
    return a + t * (b - a)


def _calc_frame_blend(time_s: float, motion_length_s: float, num_frames: int, src_dt: float) -> tuple[int, int, float]:
    """Map a query time onto ``(f0, f1, blend)`` inside a fixed-step motion.

    Args:
        time_s: Query time, seconds since the start.
        motion_length_s: Total motion duration in seconds.
        num_frames: Number of source frames.
        src_dt: Source period ``1/fps``. Currently informational - kept in the
            signature so an at-a-glance reader can verify the frame count and
            duration are consistent.

    Returns:
        ``(f0, f1, blend)`` - the two frames to interpolate between and the
        blend factor in ``[0, 1)``.
    """
    del src_dt  # informational only; motion_length + num_frames define the rate
    phase = max(0.0, min(time_s / max(motion_length_s, 1e-8), 1.0))
    frame_f = phase * (num_frames - 1)
    f0 = int(frame_f)
    f1 = min(f0 + 1, num_frames - 1)
    blend = frame_f - f0
    return f0, f1, blend


# ---------------------------------------------------------------------------
# MotionPlayer
# ---------------------------------------------------------------------------


class MotionPlayer:
    """Playhead over a pre-resampled reference motion for a single clip.

    Args:
        source: Either a cache dict (see module docstring) or a path to a
            ProtoMotions ``.pt`` file. String paths are loaded lazily - no
            torch import when only a dict is passed.
        control_dt: Target control period in seconds (default ``0.02s`` =
            50Hz). Only used when ``source`` is a raw ``.pt`` that needs
            resampling; a cache dict carries its own ``control_dt``.
        motion_index: For a packaged multi-motion ``.pt`` library, which entry
            to play.
    """

    def __init__(
        self,
        source: dict[str, Any] | str,
        control_dt: float = 0.02,
        motion_index: int = 0,
    ) -> None:
        self._control_dt = float(control_dt)
        if isinstance(source, dict):
            self._load_cache(source)
        elif isinstance(source, str):
            self._load_file(source, motion_index)
        else:
            raise TypeError(f"MotionPlayer source must be a cache dict or a .pt path, got {type(source).__name__}.")

    # ---- Public API --------------------------------------------------------

    @property
    def total_frames(self) -> int:
        """Number of pre-resampled frames in the loaded clip."""
        return int(self._num_frames)

    @property
    def num_bodies(self) -> int:
        """Number of rigid bodies in the loaded clip's rotation tensor."""
        return int(self._body_rot.shape[1])

    @property
    def num_dofs(self) -> int:
        """Number of joint DOFs in the loaded clip's joint tensor."""
        return int(self._dof_pos.shape[1])

    @property
    def control_dt(self) -> float:
        """Control period, seconds. Matches the ONNX tracker's ``control_dt``."""
        return float(self._control_dt)

    def get_state_at_frame(self, frame_idx: int) -> dict[str, np.ndarray]:
        """Return a single-frame state, index-clamped to the valid range."""
        idx = int(np.clip(frame_idx, 0, self._num_frames - 1))
        return {
            "dof_pos": self._dof_pos[idx],
            "dof_vel": self._dof_vel[idx],
            "body_rot": self._body_rot[idx],
            "body_pos": self._body_pos[idx],
            "body_vel": self._body_vel[idx],
            "body_ang_vel": self._body_ang_vel[idx],
        }

    def get_future_references(self, frame_idx: int, step_indices: list[int]) -> dict[str, np.ndarray]:
        """Stack N future frames at the given step offsets.

        Args:
            frame_idx: Current playhead frame.
            step_indices: Positive integer offsets in control steps (e.g. the
                GTP tracker's ``[1, 2, 4, 8]``).

        Returns:
            Dict with each state key stacked along a new leading axis of size
            ``len(step_indices)``.
        """
        future_states = [self.get_state_at_frame(frame_idx + s) for s in step_indices]
        return {key: np.stack([s[key] for s in future_states], axis=0) for key in _STATE_KEYS}

    def as_cache(self) -> dict[str, Any]:
        """Return the loaded state as a plain dict (for saving or re-loading)."""
        return {
            "dof_pos": self._dof_pos,
            "dof_vel": self._dof_vel,
            "body_rot": self._body_rot,
            "body_pos": self._body_pos,
            "body_vel": self._body_vel,
            "body_ang_vel": self._body_ang_vel,
            "control_dt": self._control_dt,
            "num_frames": self._num_frames,
        }

    def save_cache_npz(self, path: str) -> None:
        """Write the loaded state to an ``.npz`` file for later re-use."""
        np.savez(
            path,
            dof_pos=self._dof_pos,
            dof_vel=self._dof_vel,
            body_rot=self._body_rot,
            body_pos=self._body_pos,
            body_vel=self._body_vel,
            body_ang_vel=self._body_ang_vel,
            control_dt=np.float32(self._control_dt),
            num_frames=np.int64(self._num_frames),
        )
        logger.info(
            "MotionPlayer cached %d frames @ %.0f Hz -> %s",
            self._num_frames,
            1.0 / self._control_dt,
            path,
        )

    # ---- Private loaders ---------------------------------------------------

    def _load_cache(self, data: dict[str, Any]) -> None:
        """Bind a pre-resampled cache dict in-place."""
        missing = [k for k in _STATE_KEYS if k not in data]
        if missing:
            raise KeyError(f"MotionPlayer cache is missing required keys: {missing!r}.")
        self._dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
        self._dof_vel = np.asarray(data["dof_vel"], dtype=np.float32)
        self._body_rot = np.asarray(data["body_rot"], dtype=np.float32)
        self._body_pos = np.asarray(data["body_pos"], dtype=np.float32)
        self._body_vel = np.asarray(data["body_vel"], dtype=np.float32)
        self._body_ang_vel = np.asarray(data["body_ang_vel"], dtype=np.float32)
        self._control_dt = float(data.get("control_dt", self._control_dt))
        self._num_frames = int(data.get("num_frames", self._dof_pos.shape[0]))
        logger.info(
            "MotionPlayer loaded cache: %d frames @ %.0f Hz",
            self._num_frames,
            1.0 / self._control_dt,
        )

    def _load_file(self, path: str, motion_index: int) -> None:
        """Load an ``.npz`` cache or a raw ProtoMotions ``.pt`` file."""
        if path.endswith(".npz"):
            data = dict(np.load(path))
            self._load_cache(
                {
                    "dof_pos": data["dof_pos"],
                    "dof_vel": data["dof_vel"],
                    "body_rot": data["body_rot"],
                    "body_pos": data["body_pos"],
                    "body_vel": data["body_vel"],
                    "body_ang_vel": data["body_ang_vel"],
                    "control_dt": float(data["control_dt"]),
                    "num_frames": int(data["num_frames"]),
                }
            )
            return

        # .pt raw ProtoMotions format - needs torch to unpickle. torch is not
        # part of [protomotions]: the tracker itself runs on onnxruntime, and a
        # caller with a cache dict or .npz never needs it.
        torch = require_optional(
            "torch",
            extra="kimodo",
            purpose="unpickling a raw ProtoMotions .pt motion (a cache dict or .npz needs no torch)",
        )

        # A motion file travels: it gets downloaded, shared between machines and
        # committed to dataset repos. The unrestricted unpickler runs whatever
        # __reduce__ the file names while reading it, so accepting one would make
        # playing a motion enough to execute code on this host. The documented
        # payload is tensors plus two scalars (see the cache format above), which
        # weights_only=True reads, so the restriction costs nothing. Same loader
        # as the checkpoint read in strands_robots.training.rl.base_algo.
        try:
            data = torch.load(  # type: ignore[attr-defined]
                path, map_location="cpu", weights_only=True
            )
        except pickle.UnpicklingError as e:
            raise ValueError(
                f"{path} carries more than tensors and plain scalars, so only "
                "the unrestricted unpickler could read it - and that executes "
                "arbitrary code from the file while loading. Re-save the motion "
                "as a dict of tensors, or convert it once with "
                "MotionPlayer.save_cache_npz and load the .npz instead."
            ) from e
        if "control_dt" in data and "body_rot" in data:
            # A .pt that is already a cache.
            self._load_cache({k: np.asarray(v) for k, v in data.items()})
            return
        self._resample_raw(data, motion_index)

    def _resample_raw(self, data: dict[str, Any], motion_index: int) -> None:
        """Resample raw ProtoMotions motion data onto ``control_dt``."""
        if "length_starts" in data:
            length_starts = data["length_starts"]
            motion_num_frames = data["motion_num_frames"]
            motion_dt_all = data["motion_dt"]

            start = int(length_starts[motion_index].item())
            nf = int(motion_num_frames[motion_index].item())
            end = start + nf
            src_dt = float(motion_dt_all[motion_index].item())

            gts = np.asarray(data["gts"][start:end], dtype=np.float32)
            grs = np.asarray(data["grs"][start:end], dtype=np.float32)
            gvs = np.asarray(data["gvs"][start:end], dtype=np.float32)
            gavs = np.asarray(data["gavs"][start:end], dtype=np.float32)
            dps = np.asarray(data["dps"][start:end], dtype=np.float32)
            dvs = np.asarray(data["dvs"][start:end], dtype=np.float32)
        elif "rigid_body_pos" in data:
            fps = float(data["fps"])
            src_dt = 1.0 / fps
            gts = np.asarray(data["rigid_body_pos"], dtype=np.float32)
            grs = np.asarray(data["rigid_body_rot"], dtype=np.float32)
            gvs = np.asarray(data["rigid_body_vel"], dtype=np.float32)
            gavs = np.asarray(data["rigid_body_ang_vel"], dtype=np.float32)
            dps = np.asarray(data["dof_pos"], dtype=np.float32)
            dvs = np.asarray(data["dof_vel"], dtype=np.float32)
            nf = gts.shape[0]
        else:
            raise ValueError(
                "Unrecognised raw motion format. Expected either "
                "'length_starts' (packed library) or 'rigid_body_pos' "
                "(single motion)."
            )

        motion_length = src_dt * (nf - 1)
        num_ctrl_frames = max(1, int(round(motion_length / self._control_dt)) + 1)

        body_pos_list: list[np.ndarray] = []
        body_rot_list: list[np.ndarray] = []
        body_vel_list: list[np.ndarray] = []
        body_ang_vel_list: list[np.ndarray] = []
        dof_pos_list: list[np.ndarray] = []
        dof_vel_list: list[np.ndarray] = []

        for i in range(num_ctrl_frames):
            t = i * self._control_dt
            f0, f1, blend = _calc_frame_blend(t, motion_length, nf, src_dt)
            bl = np.float32(blend)
            body_pos_list.append(lerp(gts[f0], gts[f1], bl))
            body_rot_list.append(slerp(grs[f0], grs[f1], bl))
            body_vel_list.append(lerp(gvs[f0], gvs[f1], bl))
            body_ang_vel_list.append(lerp(gavs[f0], gavs[f1], bl))
            dof_pos_list.append(lerp(dps[f0], dps[f1], bl))
            dof_vel_list.append(lerp(dvs[f0], dvs[f1], bl))

        self._body_pos = np.stack(body_pos_list).astype(np.float32)
        self._body_rot = np.stack(body_rot_list).astype(np.float32)
        self._body_vel = np.stack(body_vel_list).astype(np.float32)
        self._body_ang_vel = np.stack(body_ang_vel_list).astype(np.float32)
        self._dof_pos = np.stack(dof_pos_list).astype(np.float32)
        self._dof_vel = np.stack(dof_vel_list).astype(np.float32)
        self._num_frames = num_ctrl_frames

        logger.info(
            "MotionPlayer resampled: %d @ %.1f Hz -> %d @ %.0f Hz",
            nf,
            1.0 / src_dt,
            num_ctrl_frames,
            1.0 / self._control_dt,
        )
