"""Observation builder for the WBC policy.

Reproduces the observation layout of NVIDIA's GR00T-WholeBodyControl reference
controller. One observation *frame* is laid out as (for ``n = num_actions``,
``c = command_dim``)::

    [0       : c        ]  command  (vx, vy, omega, + gait/style fields)
    [c       : c+3      ]  base angular velocity      (scaled by obs_scales.ang_vel)
    [c+3     : c+6      ]  projected gravity          (orientation cue, unscaled)
    [c+6     : c+6+n    ]  joint positions qj         (defaults subtracted, * dof_pos)
    [c+6+n   : c+6+2n   ]  joint velocities dqj       (scaled by obs_scales.dof_vel)
    [c+6+2n  : c+6+2n+n ]  previous action            (n-dim)

With the upstream GEAR-SONIC defaults (c=7, n=15) that is
7 + 3 + 3 + 15 + 15 + 15 = 58 populated values; the remaining width up to
``single_obs_dim`` (86) is zero-padded so a checkpoint trained with extra
command/style channels loads without code changes. The frame is then stacked
over ``obs_history_len`` via a ``deque`` to form the ``num_obs``-wide network
input (oldest frame first).

Pure NumPy - no torch / onnxruntime - so the layout is unit-testable on any
machine (issue #466: "observation builder produces the exact 86-dim layout").
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from .config import WBCConfig


def build_single_frame(
    config: WBCConfig,
    *,
    command: NDArray[np.float64],
    base_ang_vel: NDArray[np.float64],
    proj_gravity: NDArray[np.float64],
    qj: NDArray[np.float64],
    dqj: NDArray[np.float64],
    prev_action: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble one ``single_obs_dim``-wide observation frame.

    Applies the upstream scaling (``obs_scales``), subtracts ``default_angles``
    from ``qj``, and writes each sub-vector at its fixed offset, zero-padding
    any remaining width.

    Args:
        config: The policy config (dims, scales, default angles).
        command: Locomotion command, length ``command_dim``. Shorter inputs
            (e.g. just ``[vx, vy, omega]``) are zero-padded to ``command_dim``.
        base_ang_vel: Base angular velocity (rad/s), length 3.
        proj_gravity: Gravity direction in the body frame, length 3.
        qj: Measured joint positions, length ``num_actions``.
        dqj: Measured joint velocities, length ``num_actions``.
        prev_action: Previous network action, length ``num_actions``.

    Returns:
        A ``(single_obs_dim,)`` float64 array.

    Raises:
        ValueError: If any sub-vector has the wrong length, or the assembled
            frame would overflow ``single_obs_dim``.
    """
    n = config.num_actions
    c = config.command_dim

    command = np.asarray(command, dtype=np.float64).ravel()
    if command.shape[0] > c:
        raise ValueError(f"command length {command.shape[0]} exceeds command_dim {c}")
    # Right-pad a short command (e.g. [vx, vy, omega]) up to command_dim.
    if command.shape[0] < c:
        command = np.concatenate([command, np.zeros(c - command.shape[0], dtype=np.float64)])

    base_ang_vel = _require_len(base_ang_vel, 3, "base_ang_vel")
    proj_gravity = _require_len(proj_gravity, 3, "proj_gravity")
    qj = _require_len(qj, n, "qj")
    dqj = _require_len(dqj, n, "dqj")
    prev_action = _require_len(prev_action, n, "prev_action")

    defaults = np.asarray(config.default_angles, dtype=np.float64) if config.default_angles else np.zeros(n)
    ang_vel_scale = config.obs_scales.get("ang_vel", 1.0)
    dof_pos_scale = config.obs_scales.get("dof_pos", 1.0)
    dof_vel_scale = config.obs_scales.get("dof_vel", 1.0)

    frame = np.zeros(config.single_obs_dim, dtype=np.float64)
    end = c + 6 + 3 * n
    if end > config.single_obs_dim:
        raise ValueError(
            f"observation layout needs {end} values (command_dim={c}, num_actions={n}) "
            f"but single_obs_dim={config.single_obs_dim}; raise single_obs_dim or check the config."
        )

    frame[0:c] = command
    frame[c : c + 3] = base_ang_vel * ang_vel_scale
    frame[c + 3 : c + 6] = proj_gravity
    frame[c + 6 : c + 6 + n] = (qj - defaults) * dof_pos_scale
    frame[c + 6 + n : c + 6 + 2 * n] = dqj * dof_vel_scale
    frame[c + 6 + 2 * n : c + 6 + 3 * n] = prev_action
    # Indices [end:single_obs_dim] remain zero (gait/style padding).
    return frame


def _require_len(vec: NDArray[np.float64], n: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}")
    return arr


class ObservationHistory:
    """Fixed-length history of observation frames, stacked into the network input.

    Wraps a ``deque(maxlen=obs_history_len)`` warm-started with ZERO frames,
    matching the upstream reference exactly (run_mujoco_gear_wbc.py:47-50 inits
    the deque with ``[np.zeros(single_obs_dim)] * obs_history_len`` then appends
    the live frame each tick). So for the first ``obs_history_len - 1`` ticks the
    older history slots hold zeros, not copies of the first frame - the network
    sees the same warm-start transient the reference controller was validated
    with. The stacked vector is oldest-frame-first.
    """

    def __init__(self, config: WBCConfig) -> None:
        self._maxlen = config.obs_history_len
        self._single_dim = config.single_obs_dim
        self._num_obs = config.num_obs
        self._buffer: deque[NDArray[np.float64]] = deque(maxlen=self._maxlen)
        self._fill_zeros()

    def _fill_zeros(self) -> None:
        """Pre-fill the deque with ``obs_history_len`` zero-frames (upstream warm-start)."""
        zero = np.zeros(self._single_dim, dtype=np.float64)
        for _ in range(self._maxlen):
            self._buffer.append(zero.copy())

    def reset(self) -> None:
        """Reset the history to the zero warm-start (call at episode boundaries).

        Matches a fresh controller: the deque is re-seeded with zero-frames so
        the next ``obs_history_len - 1`` pushes reproduce the upstream
        zero-warm-start transient (not a stale rolling window from the prior
        episode).
        """
        self._buffer.clear()
        self._fill_zeros()

    def push(self, frame: NDArray[np.float64]) -> NDArray[np.float64]:
        """Append ``frame`` (evicting the oldest) and return the stacked input.

        The buffer is always full (zero-warm-started), so the returned vector is
        immediately ``num_obs`` wide. Oldest frame first; on early ticks the
        older slots are still the zero warm-start frames.
        """
        frame = np.asarray(frame, dtype=np.float64).ravel()
        if frame.shape[0] != self._single_dim:
            raise ValueError(f"frame must have length single_obs_dim={self._single_dim}, got {frame.shape[0]}")
        self._buffer.append(frame)
        stacked = np.concatenate(list(self._buffer))
        if stacked.shape[0] != self._num_obs:
            raise ValueError(f"stacked obs width {stacked.shape[0]} != num_obs {self._num_obs}")
        return stacked

    def __len__(self) -> int:
        return len(self._buffer)


__all__ = ["build_single_frame", "ObservationHistory"]
