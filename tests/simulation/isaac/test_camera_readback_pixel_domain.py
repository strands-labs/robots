# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: Isaac's two camera *readback* surfaces refuse an unusable
pixel count, and each names the parameter that was wrong.

:func:`~strands_robots.utils.positive_count_error` is the shared pixel floor,
and its docstring names the whole family it backs: "the ``width`` / ``height``
of ``add_camera`` and of the render family (``render``, ``get_frame``,
``get_camera_params``) on every simulation backend", with the invariant that
"the same camera configuration cannot be refused on one backend and accepted on
another". Four Isaac methods apply it. Two of them were driven:
``add_camera`` (by ``test_add_camera_numeric_validation.py``, whose docstring
states the same parity invariant against the MuJoCo sibling) and the internal
``_render_frame`` that ``render`` forwards to. The two **readback** surfaces
were not: no test in this package called ``get_frame`` or
``get_camera_params`` at all, so neither their refusal nor their success path
had ever run.

The two undriven surfaces are exactly the two whose failure *channel* differs.
``add_camera`` returns the agent-tool ``{"status": "error"}`` envelope and
``_render_frame`` returns ``(None, None, {"error": ...})``; ``get_frame`` and
``get_camera_params`` are the raw in-process surfaces documented to **raise** on
every degraded path, so a compositing consumer can never silently receive black
pixels. A structural sweep can show the guard is *wired* at all four; only a
behavioural test can show it *refuses*, and a regression that kept the call and
dropped the ``raise`` would satisfy the former.

What each class pins:

* ``TestBothReadbackSurfacesRefuseAnUnusablePixelCount`` - the refusal text is
  the shared domain's verdict verbatim, so neither surface can drift into a
  locally reworded copy, and the guard runs before the RTX handle is touched,
  so a refused size costs no render.
* ``TestUsableSizesAndTheOmissionStillWork`` - the accepted side, including
  ``None`` (membership, not truthiness, decides "omitted"), and the over-reach
  control: a usable count that differs from the camera's native size still
  gets the native-resolution message, not the domain message.
* ``TestTheReadbackContract`` - the values these surfaces exist to return: the
  ``(rgb, depth)`` shapes and dtypes, and the fixed USD-prim-to-OpenGL basis
  correction ``get_camera_params`` documents.
* ``TestTheDomainIsDocumented`` - ``get_frame``'s ``Raises:`` entry named only
  the native-resolution mismatch while its code refuses two distinct causes,
  and its sibling ``get_camera_params`` named both. A caller reading the
  narrower docstring could not discover the floor.
* ``TestEveryPublicDimsSurfaceIsAccountedFor`` - a fifth public surface taking
  ``width``/``height`` must either apply the shared floor or forward the value
  to a sibling that does.

None of this needs Isaac Sim or a GPU. The dims guard runs before the handle is
read, and the handle itself is duck-typed, so the accepted paths are exercised
too - the skeleton-via-``__new__`` fixture shape the sibling
``add_camera`` module uses.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _CameraState
from strands_robots.utils import positive_count_error
from tests._package_ast import parse_source
from tests.simulation._isaac_engine import isaac_engine

#: The camera's native render size. Both readback surfaces accept only this
#: size or ``None``, so every "usable" probe below uses it.
NATIVE_W, NATIVE_H = 64, 48

#: Pixel counts outside the shared floor. ``0`` and a negative are the counts
#: no renderer can produce; ``True``/``False`` are ``int`` subclasses that would
#: read as the sizes 1 and 0; ``2.7`` and ``64.0`` are the fractional and
#: integral floats ``int()`` would silently accept; ``nan``/``inf`` are the
#: non-finite pair; ``"64"``, ``[64]`` and ``np.int64(64)`` are the spellings
#: whose value is right but whose type the pixel consumers refuse.
UNUSABLE: list[Any] = [0, -8, True, False, 2.7, 64.0, float("nan"), float("inf"), "64", [64], np.int64(64)]


class _FakeCameraHandle:
    """Duck-typed stand-in for the Isaac ``Camera`` sensor handle.

    Only the four reads the two surfaces under test perform are implemented.
    Values are distinct constants so a transposed or dropped read is visible.
    """

    def __init__(self) -> None:
        self.reads: list[str] = []

    def get_intrinsics_matrix(self) -> np.ndarray:
        self.reads.append("intrinsics")
        return np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def get_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        self.reads.append("pose")
        return np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0])

    def get_rgba(self) -> np.ndarray:
        self.reads.append("rgba")
        return np.full((NATIVE_H, NATIVE_W, 4), 200, dtype=np.uint8)

    def get_depth(self) -> np.ndarray:
        self.reads.append("depth")
        return np.full((NATIVE_H, NATIVE_W), 1.5, dtype=np.float32)


class _FatalCameraHandle(_FakeCameraHandle):
    """A handle whose every read fails, so reaching it cannot go unnoticed."""

    def _boom(self) -> Any:
        raise AssertionError("the refused size reached the RTX handle")

    get_intrinsics_matrix = _boom  # type: ignore[assignment]
    get_world_pose = _boom  # type: ignore[assignment]
    get_rgba = _boom  # type: ignore[assignment]
    get_depth = _boom  # type: ignore[assignment]


def _engine(handle: _FakeCameraHandle | None = None) -> IsaacSimulation:
    """Skeleton ``IsaacSimulation`` carrying only what the readbacks read.

    ``render_mode`` must not be ``"headless"``: both surfaces refuse that mode
    before they reach the dims guard, and ``get_frame`` says so explicitly.
    """
    engine = isaac_engine()
    engine._config = IsaacConfig(render_mode="rtx_realtime", camera_width=NATIVE_W, camera_height=NATIVE_H)
    engine._world_created = True
    engine._camera_warmup_steps = 0

    cam = _CameraState("cam", "/World/cameras/cam", NATIVE_W, NATIVE_H)
    cam.handle = handle if handle is not None else _FakeCameraHandle()
    engine._cameras["cam"] = cam
    return engine


def _call(engine: IsaacSimulation, method: str, **kwargs: Any) -> Any:
    """Invoke a readback surface with deliberately off-type pixel counts.

    ``width``/``height`` are annotated ``int | None``, and half the probe set is
    outside that annotation on purpose - the runtime domain is what is under
    test. Splatting through one funnel keeps that intent in a single place
    instead of a suppression at every call site.
    """
    return getattr(engine, method)(**kwargs)


#: The two raw readback surfaces and the context string each passes to the
#: shared domain, so a refusal names the method the caller actually called.
READBACKS = [("get_frame", "get_frame"), ("get_camera_params", "get_camera_params")]


class TestBothReadbackSurfacesRefuseAnUnusablePixelCount:
    """The refusal is the shared domain's verdict, verbatim, and costs no render.

    Asserting equality rather than a substring is what pins "one rule, one
    wording": a surface that grew its own message would still name the
    parameter and still raise, and only this comparison would notice. The
    camera's handle fails on contact, so the guard is also shown to sit above
    every ``cam.handle`` read.
    """

    @pytest.mark.parametrize("method,context", READBACKS)
    def test_the_message_is_the_shared_domains_verdict(self, method: str, context: str) -> None:
        for param in ("width", "height"):
            for value in UNUSABLE:
                expected = positive_count_error(value, param, context)
                assert expected is not None, f"probe {param}={value!r} must be outside the shared domain"
                handle = _FatalCameraHandle()
                with pytest.raises(ValueError) as excinfo:
                    _call(_engine(handle), method, camera_name="cam", **{param: value})
                assert str(excinfo.value) == expected
                assert handle.reads == []


class TestUsableSizesAndTheOmissionStillWork:
    """The accepted side, including the documented ``None``, reaches the handle.

    Membership decides "omitted", not truthiness - reading ``0`` as absent is
    the failure the shared floor exists to prevent, and it would look identical
    to a passing test here without the ``0`` case in ``UNUSABLE`` above.
    """

    @pytest.mark.parametrize("method", ["get_frame", "get_camera_params"])
    def test_the_readback_succeeds(self, method: str) -> None:
        accepted: list[dict[str, Any]] = [
            {},
            {"width": None},
            {"height": None},
            {"width": NATIVE_W},
            {"height": NATIVE_H},
            {"width": NATIVE_W, "height": NATIVE_H},
        ]
        for kwargs in accepted:
            handle = _FakeCameraHandle()
            assert _call(_engine(handle), method, camera_name="cam", **kwargs) is not None, kwargs
            assert handle.reads, f"{kwargs} never reached the handle"

    def test_a_usable_but_mismatched_size_reports_the_resolution(self) -> None:
        """Over-reach control: the floor did not swallow the resolution check.

        A positive integer is inside the shared domain, so it must reach the
        Isaac-specific native-resolution comparison and be refused by *that*
        message instead. The two surfaces word their mismatch differently;
        what both must do is name the requested size, the camera's actual
        size, and the remedy.
        """
        for method, param, value in [
            ("get_frame", "width", NATIVE_W * 2),
            ("get_frame", "height", NATIVE_H + 1),
            ("get_camera_params", "width", 1),
            ("get_camera_params", "height", NATIVE_H * 3),
        ]:
            assert positive_count_error(value, param, method) is None
            with pytest.raises(ValueError) as excinfo:
                _call(_engine(), method, camera_name="cam", **{param: value})
            message = str(excinfo.value)
            assert f"requested {param}={value}" in message, message
            assert f"{NATIVE_W}x{NATIVE_H}" in message, message
            assert "Re-add the camera with the desired size." in message, message
            assert "must be a positive integer" not in message, message


class TestTheReadbackContract:
    """The values these two surfaces exist to return."""

    def test_get_frame_returns_rgb_and_metric_depth(self) -> None:
        rgb, depth = _engine().get_frame("cam")
        assert rgb.shape == (NATIVE_H, NATIVE_W, 3)
        assert rgb.dtype == np.uint8
        assert depth is not None
        assert depth.shape == (NATIVE_H, NATIVE_W)
        assert depth.dtype == np.float32
        # The alpha channel is dropped, not averaged in.
        assert np.array_equal(np.unique(rgb), np.array([200], dtype=np.uint8))
        assert float(depth[0, 0]) == pytest.approx(1.5)

    def test_get_camera_params_reports_the_native_size_and_intrinsics(self) -> None:
        params = _engine().get_camera_params("cam")
        assert (params.width, params.height) == (NATIVE_W, NATIVE_H)
        assert np.allclose(params.K, [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]])

    def test_the_prim_to_opengl_basis_correction_is_applied(self) -> None:
        """The fixed USD-prim-to-GL rotation the docstring documents.

        With an identity prim orientation the returned basis is exactly the
        documented mapping - prim +X -> GL -Z, prim +Y -> GL -X, prim +Z ->
        GL +Y - so a dropped or transposed correction is visible rather than
        merely plausible.
        """
        params = _engine().get_camera_params("cam")
        rotation = np.asarray(params.T_world_cam)[:3, :3]
        expected = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert np.allclose(rotation, expected), rotation
        # The prim's world position is carried through untouched.
        assert np.allclose(np.asarray(params.T_world_cam)[:3, 3], [1.0, 2.0, 3.0])


class TestTheDomainIsDocumented:
    """Both readback docstrings must name the pixel-count cause.

    They apply a byte-identical guard, so a caller reading either one must be
    able to discover the floor. ``get_frame``'s ``Raises:`` named only the
    native-resolution mismatch, which is the narrower of its two ValueError
    causes.
    """

    def test_the_args_and_raises_entries_name_the_pixel_floor(self) -> None:
        for method in ("get_frame", "get_camera_params"):
            doc = inspect.getdoc(getattr(IsaacSimulation, method)) or ""
            args = doc.split("Args:", 1)
            raises = doc.split("Raises:", 1)
            assert len(args) == 2 and len(raises) == 2, f"{method} documents no Args:/Raises: section"
            assert "positive integer when supplied" in " ".join(args[1].split("Raises:", 1)[0].split()), method
            assert "positive integer" in " ".join(raises[1].split()), method


def _dims_surfaces() -> dict[str, tuple[bool, bool]]:
    """Map every public ``IsaacSimulation`` method taking ``width``/``height``.

    Returns ``{name: (applies_the_floor, forwards_the_value)}``. Scoping to
    public methods excludes the internal ``_render_frame`` (which guards
    anyway) and ``_create_camera_prim`` (a terminal consumer of already
    validated dims) by construction rather than by an exemption list.
    """
    src = inspect.getsource(inspect.getmodule(IsaacSimulation))  # type: ignore[arg-type]
    tree = parse_source(src)
    out: dict[str, tuple[bool, bool]] = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != "IsaacSimulation":
            continue
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name.startswith("_"):
                continue
            names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            dims = names & {"width", "height"}
            if not dims:
                continue
            guards = False
            forwards = False
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                called = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
                if called == "positive_count_error":
                    guards = True
                    continue
                passed = {a.id for a in node.args if isinstance(a, ast.Name)}
                passed |= {k.value.id for k in node.keywords if isinstance(k.value, ast.Name)}
                if dims <= passed:
                    forwards = True
            out[fn.name] = (guards, forwards)
    return out


class TestEveryPublicDimsSurfaceIsAccountedFor:
    """A fifth public dims surface must guard or forward, not neither."""

    def test_the_known_surfaces_are_the_ones_measured(self) -> None:
        """Non-vacuity: a scan that found nothing would pass every assertion."""
        assert set(_dims_surfaces()) == {"render", "get_frame", "get_camera_params", "add_camera"}

    def test_it_applies_the_floor_or_forwards_the_value(self) -> None:
        for name, (guards, forwards) in _dims_surfaces().items():
            assert guards or forwards, f"{name} neither applies the pixel floor nor forwards width/height"

    def test_the_two_readbacks_apply_the_floor_themselves(self) -> None:
        """They raise rather than returning an envelope, so they cannot delegate."""
        for name in ("get_frame", "get_camera_params"):
            assert _dims_surfaces()[name][0], f"{name} must apply the shared floor itself"
