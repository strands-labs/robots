"""``get_frame`` refuses a manufactured depth buffer, and depth is shape-guarded.

Two asymmetries in the Isaac render path, both of which handed a numeric consumer
a buffer it was promised could not arrive.

**1. A substituted zero-depth buffer reached ``get_frame``.** When a camera carries
no depth annotator, ``_render_frame`` logs a WARNING and returns
``np.zeros(rgb.shape[:2])``. That is a fair degradation for the *envelope* path it
also serves - ``render`` only needs pixels - but ``get_frame`` passed it straight
through, while its own docstring promised the opposite:

    Unlike :meth:`_render_frame` -- whose blank-frame fallbacks exist for the
    envelope path -- this method **raises** on every degraded path [...] so a
    compositing consumer can never silently receive black pixels with zero depth.

and the ``SimEngine.get_frame`` contract on the ABC says backends "must never
substitute silently wrong pixels -- failures raise", reserving ``None`` for
backends with no depth path at all.

The consequence is not a subtly wrong image. ``HybridCompositor``'s per-pixel rule
is ``valid_fg = isfinite(fg_depth) & (fg_depth > depth_epsilon) & ...``, and its
documented convention treats ``0`` as sky - so an all-zero foreground depth loses
*every* pixel and composites a frame containing the backdrop alone, with the
simulated robot entirely absent. A WARNING in a log the compositor does not read is
not a refusal.

**2. Depth had no shape guard while RGB had one.** ``_render_frame`` refuses a
malformed RGB buffer by shape, naming the shape::

    arr = np.asarray(rgba)
    if arr.ndim < 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None, None, {"error": f"camera {name!r} returned a malformed RGB buffer (shape ...)"}

Depth was ``depth = np.asarray(depth_raw)`` and nothing else, so a 0-D scalar or an
``(H, W, 4)`` annotator frame passed through to a consumer promised ``(H, W)``, and
a *ragged* buffer raised a bare NumPy ``ValueError: setting an array element with a
sequence. The requested array has an inhomogeneous shape ...`` which the handler
reported as a generic "Failed to render camera", naming neither the buffer nor its
shape. Nor was depth's size ever compared against RGB's.

Scope: the RTX camera handle is stood in, which is what lets these run without a
GPU - the handle's two methods are the whole seam. The live-Kit half is exercised
on GPU.
"""

from __future__ import annotations

import threading
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacConfig,
    IsaacSimulation,
    _CameraState,
)

#: A well-formed RTX frame pair for a 4x3 camera.
_RGB = np.zeros((3, 4, 3), dtype=np.uint8)
_DEPTH = np.full((3, 4), 2.5, dtype=np.float32)


class _Handle:
    """An RTX camera handle: ``get_rgba`` and ``get_depth`` are the whole seam."""

    def __init__(self, rgba: Any, depth: Any) -> None:
        self._rgba, self._depth = rgba, depth

    def get_rgba(self) -> Any:
        return self._rgba

    def get_depth(self) -> Any:
        return self._depth


def _engine(rgba: Any = None, depth: Any = _DEPTH, *, render_mode: str = "rtx_realtime") -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode=render_mode, headless=False)
    engine._world = types.SimpleNamespace()
    engine._world_created = True
    engine._robots = {}
    engine._objects = {}
    rgba = np.dstack([_RGB, np.full((3, 4, 1), 255, dtype=np.uint8)]) if rgba is None else rgba
    engine._cameras = {"cam": _CameraState(name="cam", prim_path="/World/Cameras/cam", width=4, height=3)}
    engine._cameras["cam"].handle = _Handle(rgba, depth)
    return engine


class TestASubstitutedDepthBufferIsRefused:
    def test_get_frame_raises_when_the_annotator_is_absent(self) -> None:
        """The docstring's own promise, which the code did not keep."""
        engine = _engine(depth=None)

        with pytest.raises(RuntimeError) as exc:
            engine.get_frame("cam")

        text = str(exc.value)
        assert "cam" in text
        assert "depth annotator" in text
        # The refusal explains why zeros are not an acceptable answer, because a
        # caller who only wanted pixels has a different method.
        assert "sky" in text
        assert "render()" in text

    def test_no_zero_depth_array_is_ever_returned(self) -> None:
        """The specific value that made the robot vanish from a composite."""
        engine = _engine(depth=None)

        try:
            _rgb, depth = engine.get_frame("cam")
        except RuntimeError:
            return  # refused, which is the contract
        raise AssertionError(f"get_frame returned a substituted depth buffer: {depth!r}")

    def test_the_envelope_path_still_degrades(self) -> None:
        """``render`` only needs pixels, so zeros stay a fair degradation there.

        This is the half that must NOT change: the substitution exists for the
        envelope, and removing it would turn a renderable frame into a failure for
        a caller who never asked about depth.
        """
        engine = _engine(depth=None)

        rgb, depth, meta = engine._render_frame("cam")

        assert rgb is not None
        assert depth is not None
        assert np.all(depth == 0.0)
        # And it is marked, which is what lets get_frame tell the two apart.
        assert meta["json"]["depth_is_real"] is False

    def test_a_real_depth_buffer_is_marked_and_returned(self) -> None:
        """The control: a camera with an annotator is unaffected."""
        engine = _engine()

        rgb, depth, meta = engine._render_frame("cam")
        assert meta["json"]["depth_is_real"] is True

        rgb, depth = engine.get_frame("cam")
        assert rgb.shape == (3, 4, 3)
        assert rgb.dtype == np.uint8
        assert depth is not None
        assert depth.shape == (3, 4)
        assert depth.dtype == np.float32
        assert np.allclose(depth, 2.5)


class TestDepthIsShapeGuardedLikeRgb:
    """RGB was refused by shape and depth was not. Same guard, same wording shape."""

    @pytest.mark.parametrize(
        "bad",
        [
            np.float32(2.5),  # 0-D scalar
            np.zeros((0, 4), dtype=np.float32),  # empty
            np.zeros((3, 0), dtype=np.float32),  # empty
            np.zeros(12, dtype=np.float32),  # 1-D
        ],
    )
    def test_a_malformed_depth_buffer_is_refused_by_shape(self, bad: Any) -> None:
        engine = _engine(depth=bad)

        rgb, depth, meta = engine._render_frame("cam")

        assert rgb is None and depth is None
        text = str(meta["error"])
        assert "depth buffer" in text
        # The shape is named, which is the whole point of the RGB guard this
        # mirrors: "malformed" without the shape is not diagnosable.
        assert "shape" in text

    def test_a_ragged_depth_buffer_names_the_buffer(self) -> None:
        """A ragged buffer raises inside ``np.asarray`` BEFORE the shape guard can
        see it, so the guard alone does not cover this case.

        Unwrapped it surfaced as NumPy's "setting an array element with a sequence.
        The requested array has an inhomogeneous shape ..." reported as a generic
        "Failed to render camera", naming neither the depth buffer nor what shape
        was expected. Asserting only the camera name would pass on that message
        too, which is why this reads the words the wrapping adds.
        """
        engine = _engine(depth=[[1.0, 2.0], [3.0]])

        rgb, depth, meta = engine._render_frame("cam")

        assert rgb is None and depth is None
        text = str(meta["error"])
        assert "camera 'cam'" in text
        assert "depth buffer" in text
        assert "(H, W)" in text

    def test_a_depth_buffer_of_the_wrong_size_is_refused(self) -> None:
        """Never compared against RGB before, so a mismatched pair was returned
        as though the two described one frame."""
        engine = _engine(depth=np.full((7, 9), 2.5, dtype=np.float32))

        rgb, depth, meta = engine._render_frame("cam")

        assert rgb is None and depth is None
        text = str(meta["error"])
        assert "(7, 9)" in text
        assert "(3, 4)" in text

    def test_a_three_dimensional_depth_frame_is_accepted_only_if_it_matches(self) -> None:
        """An ``(H, W, 1)`` annotator frame has the right footprint, so it is not
        malformed by this guard - the size comparison is on the first two axes.

        Pinned so the guard's boundary is deliberate rather than incidental: what
        it refuses is a buffer that cannot describe this frame, not every buffer
        whose ``ndim`` differs from 2.
        """
        engine = _engine(depth=np.full((3, 4, 1), 2.5, dtype=np.float32))

        rgb, depth, meta = engine._render_frame("cam")

        assert rgb is not None, meta
        assert depth is not None

    def test_the_rgb_guard_is_unchanged(self) -> None:
        """Control: the guard this one was modelled on still fires the same way."""
        engine = _engine(rgba=np.zeros((0, 4, 4), dtype=np.uint8))

        rgb, depth, meta = engine._render_frame("cam")

        assert rgb is None and depth is None
        assert "malformed RGB buffer" in str(meta["error"])


class TestTheContractThisRestores:
    def test_the_abc_reserves_none_for_a_backend_with_no_depth_path(self) -> None:
        """Why this raises rather than returning ``None``.

        The ABC gives ``None`` a specific meaning - "backends with no depth path
        (Newton)" - and Isaac has one; a camera merely missing its annotator is a
        misconfiguration with a remedy. Raising is what lets the message name that
        remedy, which the compositor's generic ``None`` diagnosis cannot.
        """
        import inspect

        from strands_robots.simulation.base import SimEngine

        doc = inspect.getdoc(SimEngine.get_frame) or ""
        assert "substitute silently wrong pixels" in doc
        assert "Newton" in doc

    def test_get_frame_documents_the_refusal_it_makes(self) -> None:
        """AGENTS.md: a ``Raises:`` block names every refusal the function makes,
        because it is the only place a caller learns which handler to write."""
        import inspect

        doc = inspect.getdoc(IsaacSimulation.get_frame) or ""
        raises = doc.split("Raises:", 1)[-1]
        # The new refusal, in the block a caller reads to decide which handler to
        # write. "depth annotator" already appeared in the prose above it, so
        # asserting on the whole docstring would pass without the Raises: entry.
        assert "depth annotator" in raises
        assert "substituted zero buffer" in raises
