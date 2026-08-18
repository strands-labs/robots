# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac's rollout-video recorder refuses a rate it cannot encode at.

``IsaacSimulation.start_cameras_recording`` stores ``fps`` in the recording
state and ``stop_cameras_recording`` hands it to
:func:`strands_robots.rendering.encode_clip`, which refuses a rate it cannot
honor rather than inventing one. Without a pre-flight the mistake surfaced only
at flush time - after a whole rollout's frames had been buffered - as a
``ValueError`` escaping ``stop_cameras_recording``, an agent-facing tool whose
flush contract is best-effort and never-raise, with the recording state already
cleared so every buffered frame was lost with no structured response.

So ``fps`` (and the in-memory frame cap, whose sub-1 values drop every captured
frame) is checked at ``start`` against the same domain the MuJoCo recorder uses,
and the flush reports a refusal on its artifact line either way.

The engine is a skeleton ``IsaacSimulation`` built with ``__new__`` (the fixture
shape ``test_dataset_recording.py`` uses) so the recording lifecycle runs
without the Isaac Sim Kit runtime.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import (
    IsaacSimulation,
    _CameraState,
)

# Every value the shared positive-whole-number domain refuses: zero and
# negative rates, a fractional rate, non-finite floats, a string that looks
# like a number, ``True`` (an ``int`` subclass that would act as a silent 1),
# and ``None``.
_UNUSABLE = [0, -5, 2.7, float("nan"), float("inf"), "30", True, None]


class _FakeCameraHandle:
    """Stub RTX camera handle: ``get_rgba()`` returns a fixed RGBA buffer."""

    def __init__(self, rgba: np.ndarray) -> None:
        self.rgba = rgba

    def get_rgba(self) -> np.ndarray:
        return self.rgba


def _camera(name: str, width: int = 64, height: int = 48) -> _CameraState:
    cam = _CameraState(name=name, prim_path=f"/World/Cameras/{name}", width=width, height=height)
    cam.handle = _FakeCameraHandle(np.zeros((height, width, 4), dtype=np.uint8))
    return cam


def _make_engine() -> IsaacSimulation:
    """Skeleton IsaacSimulation with one camera and no Kit runtime."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = IsaacConfig()
    engine._lock = threading.RLock()
    engine._world = None
    engine._world_created = True
    engine._robots = {}
    engine._cameras = {"front": _camera("front")}
    engine._objects = {}
    engine._prim_registry = []
    engine._cams_rec_state = None
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._main_tid = threading.get_ident()
    return engine


@pytest.mark.parametrize("fps", _UNUSABLE)
def test_start_cameras_recording_refuses_an_unusable_fps(fps: object, tmp_path) -> None:
    """An fps the encoder cannot honor is refused before any frame is buffered."""
    engine = _make_engine()

    result = engine.start_cameras_recording(output_dir=str(tmp_path), fps=fps)  # type: ignore[arg-type]

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "fps" in text and repr(fps) in text
    # No recording started, so no rollout's frames are buffered against a rate
    # the flush would refuse.
    assert engine._cams_rec_state is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("cap", _UNUSABLE)
def test_start_cameras_recording_refuses_an_unusable_frame_cap(cap: object, tmp_path) -> None:
    """A frame cap below 1 drops every frame, so it is refused up front."""
    engine = _make_engine()

    result = engine.start_cameras_recording(output_dir=str(tmp_path), max_frames_per_camera=cap)  # type: ignore[arg-type]

    assert result["status"] == "error"
    assert "max_frames_per_camera" in result["content"][0]["text"]
    assert engine._cams_rec_state is None


def test_start_cameras_recording_accepts_a_usable_rate(tmp_path) -> None:
    """The guard admits the rates the encoder honors, including a real-valued one."""
    engine = _make_engine()

    result = engine.start_cameras_recording(output_dir=str(tmp_path), fps=np.int64(24), max_frames_per_camera=30.0)  # type: ignore[arg-type]

    assert result["status"] == "success"
    assert engine._cams_rec_state is not None
    assert callable(result["content"][0]["json"]["on_frame"])


def test_isaac_refuses_the_same_rate_as_the_mujoco_recorder() -> None:
    """One domain across the two recording surfaces, only the prefix differs."""
    from strands_robots.simulation.isaac.simulation import _cameras_recording_option_error as isaac_error
    from strands_robots.simulation.mujoco.rendering import _cameras_recording_option_error as mujoco_error

    for fps in _UNUSABLE:
        assert isaac_error("start_cameras_recording", fps, 3000) is not None
        assert mujoco_error("start_cameras_recording", fps, None, None, 3000) is not None
    assert isaac_error("start_cameras_recording", 30, 3000) is None
    assert mujoco_error("start_cameras_recording", 30, None, None, 3000) is None


def test_stop_cameras_recording_reports_a_refused_rate_instead_of_raising(tmp_path) -> None:
    """A rate that reaches the flush is reported, never raised past the tool.

    The pre-flight above makes this unreachable through the public pair, but the
    flush is the last chance to hand back the buffered frames' fate: its
    documented contract is best-effort and never-raise, so a state carrying a
    rate ``encode_clip`` refuses has to come back as a structured response.
    """
    engine = _make_engine()
    path = str(tmp_path / "rec__front.mp4")
    engine._cams_rec_state = {
        "running": True,
        "name": "rec",
        "cameras": ["front"],
        "fps": 0,
        "buffers": {"front": [np.zeros((48, 64, 3), dtype=np.uint8)] * 4},
        "paths": {"front": path},
        "errors": {"front": 0},
        "output_dir": str(tmp_path),
        "started_mono": 0.0,
        "max_frames": 3000,
    }

    result = engine.stop_cameras_recording()

    assert result["status"] == "success"
    artifact = result["content"][1]["json"]["artifacts"][0]
    assert "ValueError" in artifact["flush_error"]
    assert "fps" in artifact["flush_error"]
    # Frames that reached no file are not counted as written.
    assert artifact["frames"] == 0
    assert engine._cams_rec_state is None
