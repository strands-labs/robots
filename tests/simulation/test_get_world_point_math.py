# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic ``SimEngine.get_world_point`` contract (issue #1647).

``get_world_point`` is pure math over ``get_frame`` + ``get_camera_params``,
so its unprojection, median robustness, invalid-pixel handling, input
validation, and the failure arms of BOTH backend reads are all pinned here
against a pure-Python stub engine -- no MuJoCo, no GL, no GPU. The two reads
fail independently: ``get_camera_params`` is called after a frame has already
rendered, so its failure is not reachable by making ``get_frame`` fail, and
the two must stay distinguishable in the reported text. Backend-specific
accuracy lives in
``tests/simulation/mujoco/test_get_world_point.py`` (regression against a box
at a known pose) and the gated Isaac GPU test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.rendering import CameraParams
from strands_robots.simulation.base import SimEngine

_W, _H = 64, 48
_FX = _FY = 100.0
_CX, _CY = _W / 2.0, _H / 2.0
_ZFAR = 1000.0


class _StubSim(SimEngine):
    """Minimal engine with a synthetic depth buffer and known camera params."""

    def __init__(self, depth: np.ndarray | None, T_world_cam: np.ndarray | None = None) -> None:
        self._depth = depth
        self._T = np.eye(4, dtype=np.float64) if T_world_cam is None else T_world_cam

    # -- raw-frame surface under test -- #

    def get_frame(self, camera_name="default", width=None, height=None):
        rgb = np.zeros((_H, _W, 3), dtype=np.uint8)
        return rgb, self._depth

    def get_camera_params(self, camera_name="default", width=None, height=None):
        K = np.array([[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]], dtype=np.float64)
        return CameraParams(K=K, T_world_cam=self._T, width=_W, height=_H, znear=0.01, zfar=_ZFAR)

    # -- SimEngine abstract boilerplate -- #

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        return {"status": "success"}

    def get_state(self):
        return {"sim_time": 0.0, "step_count": 0}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return []

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return []

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {}

    def send_action(self, action, robot_name=None, n_substeps=1):
        return {"status": "success"}

    def render(self, camera_name="default", width=None, height=None):
        return {"status": "success", "content": [{"text": "ok"}]}


def _flat_depth(value: float = 2.0) -> np.ndarray:
    return np.full((_H, _W), value, dtype=np.float32)


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    assert result["status"] == "success", result
    return result["content"][1]["json"]


# ----- Unprojection math ----- #


def test_identity_pose_unprojection_is_exact() -> None:
    """Pixel-center unprojection through K^-1 into the OpenGL optical frame.

    Camera at the origin with the identity pose: +X right, +Y up, looking
    along -Z. A flat depth plane at 2 m must unproject the pixel (u, v) to
    exactly ``[(u+0.5-cx)/fx * d, -(v+0.5-cy)/fy * d, -d]`` -- pinning the
    pixel-center convention, the image-v-grows-down y-flip, and the -Z-forward
    z sign in one shot. A wrong sign on any axis fails by centimeters.
    """
    sim = _StubSim(_flat_depth(2.0))
    result = sim.get_world_point("default", pixels=[[32, 24], [0, 0], [63, 47]])
    data = _json_block(result)
    assert data["n_valid"] == 3 and data["n_requested"] == 3
    expected = [
        [(u + 0.5 - _CX) / _FX * 2.0, -((v + 0.5 - _CY) / _FY) * 2.0, -2.0] for u, v in ((32, 24), (0, 0), (63, 47))
    ]
    for got, want in zip(data["points"], expected, strict=True):
        assert got == pytest.approx(want, abs=1e-12)


def test_world_pose_is_applied_after_unprojection() -> None:
    """``p_world = T_world_cam @ p_cam``: a pure translation shifts the point."""
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    sim = _StubSim(_flat_depth(2.0), T_world_cam=T)
    data = _json_block(sim.get_world_point("default", pixels=[[32, 24]]))
    cam_frame = [0.5 / _FX * 2.0, -(0.5 / _FY) * 2.0, -2.0]
    assert data["point"] == pytest.approx([cam_frame[0] + 1.0, cam_frame[1] + 2.0, cam_frame[2] + 3.0], abs=1e-12)


def test_median_rejects_a_depth_outlier() -> None:
    """The paper's robustness rule is built in: one bad sample cannot move the
    answer. Three same-column pixels at 2 m plus one outlier at 9 m must
    report the 2 m surface, not the mean (which would drift ~1.75 m)."""
    depth = _flat_depth(2.0)
    depth[10, 32] = 9.0  # outlier sample
    sim = _StubSim(depth)
    data = _json_block(sim.get_world_point("default", pixels=[[32, 20], [32, 24], [32, 28], [32, 10]]))
    assert data["n_valid"] == 4
    assert data["point"][2] == pytest.approx(-2.0, abs=1e-9)


# ----- Invalid-depth handling ----- #


def test_background_and_nonfinite_pixels_are_dropped_not_zero_filled() -> None:
    """zfar (MuJoCo background), 0 and non-finite (Isaac background) samples
    are excluded from the median and reported as ``None`` in ``points`` --
    never unprojected into a bogus world coordinate."""
    depth = _flat_depth(2.0)
    depth[0, 0] = _ZFAR  # MuJoCo pins sky to exactly zfar
    depth[1, 1] = 0.0  # Isaac reports no-geometry as 0
    depth[2, 2] = np.inf  # ... or non-finite
    sim = _StubSim(depth)
    data = _json_block(sim.get_world_point("default", pixels=[[0, 0], [1, 1], [2, 2], [32, 24]]))
    assert data["n_valid"] == 1 and data["n_requested"] == 4
    assert data["points"][0] is None and data["points"][1] is None and data["points"][2] is None
    assert data["points"][3] is not None
    assert data["point"][2] == pytest.approx(-2.0, abs=1e-9)


def test_all_invalid_pixels_is_a_structured_error_not_a_zero_point() -> None:
    sim = _StubSim(_flat_depth(_ZFAR))
    result = sim.get_world_point("default", pixels=[[0, 0], [32, 24]])
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "no valid depth" in text.lower()
    assert "far plane" in text


def test_no_depth_backend_is_a_structured_error() -> None:
    """A Newton-shaped backend (``get_frame`` -> ``(rgb, None)``) degrades to
    the documented no-depth error, never a silent zero point."""
    sim = _StubSim(depth=None)
    result = sim.get_world_point("default", pixels=[[1, 1]])
    assert result["status"] == "error"
    assert "no" in result["content"][0]["text"].lower()
    assert "depth" in result["content"][0]["text"]


def test_base_facade_without_raw_frames_is_a_structured_error() -> None:
    """An engine that never implemented get_frame gets a clear error dict
    (tool-envelope contract: never raises), naming the missing path."""

    class _NoFrames(_StubSim):
        def get_frame(self, camera_name="default", width=None, height=None):
            raise NotImplementedError("get_frame not implemented by this backend")

    result = _NoFrames(_flat_depth()).get_world_point("default", pixels=[[1, 1]])
    assert result["status"] == "error"
    assert "get_frame" in result["content"][0]["text"]


def test_base_facade_without_camera_params_is_a_structured_error() -> None:
    """The mirror of the ``get_frame`` case one backend read later: an engine
    that renders frames but never implemented ``get_camera_params`` gets a
    clear error dict naming the missing path, not a ``NotImplementedError``
    out of a method documented never to raise.

    ``NotImplementedError`` subclasses ``RuntimeError``, so the handled tuple
    below this arm would catch it too and still return an envelope. The
    dedicated arm therefore earns its place on the WORDING -- "this backend
    has no camera-params path" instead of a generic read failure carrying the
    exception text -- which is what these assertions pin. Checking only
    ``status``, or only that the method name appears somewhere in the text,
    passes either way.
    """

    class _NoParams(_StubSim):
        def get_camera_params(self, camera_name="default", width=None, height=None):
            raise NotImplementedError("get_camera_params not implemented by this backend")

    result = _NoParams(_flat_depth()).get_world_point("default", pixels=[[1, 1]])
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "has no camera-params path" in text, text
    assert "get_camera_params" in text, text
    # Only the dedicated arm produces the wording above; the tuple arm would
    # report a generic read failure instead.
    assert "failed to read camera parameters" not in text, text
    # The frame read succeeded, so a test that only checked ``status`` could
    # be satisfied by the WRONG read being reported. Pin that it was not.
    assert "get_frame" not in text
    assert "render camera frame" not in text


def test_the_missing_path_stubs_reproduce_the_base_defaults() -> None:
    """Premise for the two missing-path cases above: :class:`SimEngine`'s own
    defaults raise exactly what those stubs raise, so they exercise the arm
    rather than an invented exception."""
    naked = _StubSim.__new__(_StubSim)
    for method in ("get_frame", "get_camera_params"):
        with pytest.raises(NotImplementedError, match=method):
            getattr(SimEngine, method)(naked)


@pytest.mark.parametrize(
    "exc",
    [
        KeyError("camera 'front' not found"),
        ValueError("no pinhole K for an orthographic projection"),
        RuntimeError("renderer went away between the two reads"),
        TypeError("camera name must be a string"),
    ],
    ids=["KeyError", "ValueError", "RuntimeError", "TypeError"],
)
def test_a_failed_camera_params_read_reports_its_reason(exc: Exception) -> None:
    """Every member of the handled tuple is reported, with the backend's own
    reason carried through -- the caller cannot act on "it failed" alone, and
    the render already succeeded so nothing else names the cause."""

    class _ParamsRaise(_StubSim):
        def get_camera_params(self, camera_name="default", width=None, height=None):
            raise exc

    result = _ParamsRaise(_flat_depth()).get_world_point("default", pixels=[[1, 1]])
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "camera parameters" in text
    assert str(exc.args[0]) in text


def test_the_two_backend_reads_are_distinguishable() -> None:
    """One failure per read, same exception type, different text: a caller
    that has to fix the camera must be able to tell a bad render from
    unreadable intrinsics. The success control shares the fixture, so the
    difference is the failing read and nothing else."""
    boom = RuntimeError("backend went away")

    class _FrameRaise(_StubSim):
        def get_frame(self, camera_name="default", width=None, height=None):
            raise boom

    class _ParamsRaise(_StubSim):
        def get_camera_params(self, camera_name="default", width=None, height=None):
            raise boom

    depth = _flat_depth()
    assert _StubSim(depth).get_world_point("default", pixels=[[1, 1]])["status"] == "success"
    frame_text = _FrameRaise(depth).get_world_point("default", pixels=[[1, 1]])["content"][0]["text"]
    params_text = _ParamsRaise(depth).get_world_point("default", pixels=[[1, 1]])["content"][0]["text"]
    assert frame_text != params_text
    assert "render camera frame" in frame_text and "camera parameters" not in frame_text
    assert "camera parameters" in params_text and "render camera frame" not in params_text


# ----- Input validation ----- #


@pytest.mark.parametrize(
    "pixels",
    [None, [], "12,24", [[1]], [[1, 2, 3]], [["a", 2]], [[True, 2]], [[np.nan, 2]]],
)
def test_malformed_pixels_are_rejected(pixels) -> None:
    result = _StubSim(_flat_depth()).get_world_point("default", pixels=pixels)
    assert result["status"] == "error", result
    assert "pixel" in result["content"][0]["text"].lower()


def test_fractional_pixels_are_rejected_not_truncated() -> None:
    result = _StubSim(_flat_depth()).get_world_point("default", pixels=[[10.5, 20]])
    assert result["status"] == "error"
    assert "fractional" in result["content"][0]["text"]


def test_integral_floats_are_accepted() -> None:
    """LLMs emit 32.0 for 32; integer-valued floats must not be rejected."""
    data = _json_block(_StubSim(_flat_depth()).get_world_point("default", pixels=[[32.0, 24.0]]))
    assert data["n_valid"] == 1


@pytest.mark.parametrize("bad", [[-1, 0], [_W, 0], [0, -1], [0, _H]])
def test_out_of_bounds_pixels_are_rejected_with_the_valid_range(bad) -> None:
    result = _StubSim(_flat_depth()).get_world_point("default", pixels=[[1, 1], bad])
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "outside" in text
    assert f"{_W}x{_H}" in text


def test_pixel_count_is_capped() -> None:
    too_many = [[1, 1]] * (SimEngine._WORLD_POINT_MAX_PIXELS + 1)
    result = _StubSim(_flat_depth()).get_world_point("default", pixels=too_many)
    assert result["status"] == "error"
    assert str(SimEngine._WORLD_POINT_MAX_PIXELS) in result["content"][0]["text"]
