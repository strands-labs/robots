"""MotionPlayer reads a ``.pt`` motion without executing code out of the file.

``MotionPlayer`` accepts a reference motion as a path, and a motion file is an
artifact that travels: checkpoints and motion libraries get downloaded, shared
between machines, and committed to dataset repos. Reading one with the legacy
unpickler runs whatever ``__reduce__`` the file names *while loading it*, so a
motion file is enough to run code on the machine that plays it.

The documented cache format is tensors plus two scalars (see
``strands_robots.policies.protomotions.motion_utils``), which is exactly what
``torch.load(..., weights_only=True)`` accepts, so the restricted unpickler is
sufficient for the format and the unrestricted one buys nothing. The same
loader reads training checkpoints in ``strands_robots.training.rl.base_algo``.

What is pinned here:

* A well-formed payload still round-trips (the restriction is not a regression).
* A payload carrying a ``__reduce__`` is refused, and its side effect never
  runs. This is the one that fails against an unrestricted loader.
* The refusal names the ``.npz`` route, so a caller holding a legitimately
  exotic file has somewhere to go instead of a dead end.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.protomotions import MotionPlayer
from tests.mocks.torch_mock import real_torch_installed

pytestmark = pytest.mark.skipif(
    not real_torch_installed(),
    reason="reads a real .pt through torch.load; the torch mock has no serializer",
)

_NUM_FRAMES = 6
_NUM_DOFS = 29
_NUM_BODIES = 33


def _cache_payload() -> dict[str, Any]:
    """A well-formed MotionPlayer cache, in the documented dict-of-arrays shape."""
    return {
        "dof_pos": np.zeros((_NUM_FRAMES, _NUM_DOFS), dtype=np.float32),
        "dof_vel": np.zeros((_NUM_FRAMES, _NUM_DOFS), dtype=np.float32),
        "body_rot": np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (_NUM_FRAMES, _NUM_BODIES, 1),
        ),
        "body_pos": np.zeros((_NUM_FRAMES, _NUM_BODIES, 3), dtype=np.float32),
        "body_vel": np.zeros((_NUM_FRAMES, _NUM_BODIES, 3), dtype=np.float32),
        "body_ang_vel": np.zeros((_NUM_FRAMES, _NUM_BODIES, 3), dtype=np.float32),
        "control_dt": 0.02,
        "num_frames": _NUM_FRAMES,
    }


class _RunsCodeOnLoad:
    """Stand-in for a motion file that carries an executable payload.

    ``__reduce__`` names a directory creation rather than anything harmful: the
    point is only that it is observable, so the test can tell "the loader ran
    the file's code" from "the loader refused the file".
    """

    def __init__(self, marker: Path) -> None:
        self._marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.mkdir, (str(self._marker),))


def test_well_formed_pt_motion_round_trips(tmp_path: Path) -> None:
    """A tensor+scalar payload still loads, so the hardened loader is sufficient."""
    import torch

    payload = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in _cache_payload().items()}
    path = tmp_path / "motion.pt"
    torch.save(payload, path)

    player = MotionPlayer(str(path))

    assert player.total_frames == _NUM_FRAMES
    assert player.num_dofs == _NUM_DOFS
    assert player.num_bodies == _NUM_BODIES


def test_pt_motion_carrying_executable_payload_is_refused(tmp_path: Path) -> None:
    """The file's ``__reduce__`` must not run while the motion is being read."""
    import torch

    marker = tmp_path / "code-ran"
    payload = dict(_cache_payload())
    payload["dof_pos"] = torch.zeros(_NUM_FRAMES, _NUM_DOFS)
    payload["trailer"] = _RunsCodeOnLoad(marker)
    path = tmp_path / "hostile.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError) as excinfo:
        MotionPlayer(str(path))

    assert not marker.exists(), (
        f"loading the motion executed code named by the file: the marker at {marker} was created while reading {path}"
    )
    message = str(excinfo.value)
    assert ".npz" in message, (
        "the refusal must point at the .npz cache route, else a caller with an "
        f"exotic-but-legitimate file has no next step. Got: {message}"
    )


def test_npz_cache_needs_no_unpickler(tmp_path: Path) -> None:
    """The ``.npz`` route is unchanged: NumPy arrays, no pickle, no torch."""
    path = tmp_path / "motion.npz"
    np.savez(str(path), **_cache_payload())

    player = MotionPlayer(str(path))

    assert player.total_frames == _NUM_FRAMES
    assert player.control_dt == pytest.approx(0.02)
