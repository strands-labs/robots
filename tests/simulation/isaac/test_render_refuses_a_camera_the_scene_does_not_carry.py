# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: Isaac's ``render`` refuses a camera the scene does not carry.

:meth:`~strands_robots.simulation.isaac.simulation.IsaacSimulation.render`
documents four blank-frame fallback paths, and one of them named *two* causes
with one answer: "``camera_name`` is unknown to ``self._cameras``. Caller
probably forgot to call ``add_camera`` (or typo'd the name)." Those two causes
need different answers, and the typo half was answered with success.

Measured on the pre-fix branch, one camera named ``wrist`` registered and
``render_mode="rtx_realtime"``:

* ``render("wrst")`` returned ``status="success"``, text ``Rendered (no
  camera): 640x480``, ``pixel_mean`` **0.0**, and a ``json`` block naming
  ``camera: "wrst"`` - a measurement of a camera that does not exist, sized
  from the config default rather than from any camera.
* the PNG block above that json feeds the shared
  ``PolicyRunner._extract_frame_ndarray`` (``_rgb_png_block``'s own docstring
  says so), so a rollout recording that camera writes an all-black video and
  reports success.
* the same name through this same class's :meth:`get_frame` raised ``KeyError:
  Camera 'wrst' not found. Available: ['wrist']`` - one registry, one lock, one
  question, two answers.
* the MuJoCo and Newton ``render`` refuse it too, each with that same sentence,
  so Isaac was the one cell of six (three backends x two readback surfaces)
  that succeeded.

The fix splits the branch by the docstring's own two causes.
:data:`~strands_robots.utils.FREE_CAMERA_TOKENS` is the shared spelling of "no
camera was *named*" - ``None``, ``""``, ``"default"`` and ``"free"``, one of
which is this method's own signature default - and Isaac has no free camera to
fall back to, so for those the blank frame stays exactly as documented. A name
outside that set is a caller mistake this method can name, and it is refused
with the verdict its own ``get_frame`` already gives.

What each class pins:

* ``TestRenderRefusesACameraTheSceneDoesNotCarry`` - the headline: a named
  camera nothing carries is an error, and the message names the alternatives.
* ``TestBothSurfacesGiveOneVerdictForOneName`` - the root cause: ``render`` and
  ``get_frame`` report the *same* sentence for the same name, so the two cannot
  drift into locally reworded copies again.
* ``TestARefusalCarriesNoFrame`` - a refusal carries no PNG block and no pixel
  statistics, so nothing downstream can mistake it for a frame.
* ``TestTheRefusalIsTotal`` - a name that is not a string, and one that is not
  even hashable, are refused rather than raising past the envelope.
* ``TestTheDocumentedDegradationsAreUnchanged`` - the over-reach control, and
  the reason the fix is scoped to a *named* camera: every one of the four
  documented fallbacks still answers exactly as before, byte-for-byte, on both
  the token path and in ``headless`` mode.
* ``TestEveryBackendGivesTheSameVerdict`` - the parity pin: all three backends'
  render path carry the one refusal sentence.
* ``TestTheContractIsDocumented`` - the docstring states the split, so a caller
  can discover which half of the old bullet they are in.

None of this needs Isaac Sim or a GPU: the camera lookup runs before any RTX
handle is read, and the handle itself is duck-typed - the skeleton-via-``__new__``
fixture shape the sibling ``test_camera_readback_pixel_domain.py`` uses.
"""

from __future__ import annotations

import inspect
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _CameraState
from strands_robots.utils import FREE_CAMERA_TOKENS

#: The registered camera's own render size, deliberately different from the
#: config default below so a frame sized from the wrong source is visible.
NATIVE_W, NATIVE_H = 64, 48

#: The config's blank-frame size, which the pre-fix miss branch used.
CONFIG_W, CONFIG_H = 640, 480

#: The registered camera's name. Every "not carried" probe below is a name this
#: scene does not hold.
CAM = "wrist"

#: Names that *name* a camera this scene does not carry: a typo of the
#: registered one, a plausible camera name from another scene, and a name that
#: differs only in case.
NOT_CARRIED = ["wrst", "front", "agentview", "Wrist"]


class _FakeCameraHandle:
    """Duck-typed stand-in for the Isaac ``Camera`` sensor handle.

    ``get_rgba`` returns a constant, distinctly non-black frame so a real
    render is never confused with a blank fallback.
    """

    def __init__(self) -> None:
        self.reads: list[str] = []

    def get_rgba(self) -> np.ndarray:
        self.reads.append("rgba")
        return np.full((NATIVE_H, NATIVE_W, 4), 200, dtype=np.uint8)

    def get_depth(self) -> np.ndarray:
        self.reads.append("depth")
        return np.full((NATIVE_H, NATIVE_W), 1.5, dtype=np.float32)


def _engine(
    *,
    render_mode: str = "rtx_realtime",
    handle: _FakeCameraHandle | None = None,
    with_handle: bool = True,
) -> IsaacSimulation:
    """Skeleton ``IsaacSimulation`` carrying only what the render path reads.

    ``render_mode`` must not be ``"headless"`` for the camera lookup to be
    reached: that mode returns its own documented blank frame first.
    """
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = IsaacConfig(render_mode=render_mode, camera_width=CONFIG_W, camera_height=CONFIG_H)
    engine._lock = threading.RLock()
    engine._world = None
    engine._world_created = True
    engine._robots = {}
    engine._objects = {}
    engine._cameras = {}
    engine._prim_registry = []
    engine._cam_out_size = {}
    engine._camera_warmup_steps = 0
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._main_tid = threading.get_ident()

    cam = _CameraState(CAM, f"/World/cameras/{CAM}", NATIVE_W, NATIVE_H)
    if with_handle:
        cam.handle = handle if handle is not None else _FakeCameraHandle()
    engine._cameras[CAM] = cam
    return engine


def _text(result: dict[str, Any]) -> str:
    """The result's first text block, or ``""`` when it carries none."""
    return next((b["text"] for b in result.get("content", []) if "text" in b), "")


def _json(result: dict[str, Any]) -> dict[str, Any]:
    """The result's json block, or ``{}`` when it carries none."""
    return next((b["json"] for b in result.get("content", []) if "json" in b), {})


def _get_frame_verdict(engine: IsaacSimulation, name: Any) -> str:
    """The message ``get_frame`` reports for ``name``, as the sibling surface."""
    try:
        engine.get_frame(name)
    except KeyError as refused:
        # ``KeyError``'s ``str()`` quotes its argument; the message itself is arg 0.
        return str(refused.args[0])
    raise AssertionError(f"get_frame({name!r}) did not refuse a camera the scene does not carry")


class TestRenderRefusesACameraTheSceneDoesNotCarry:
    """A named camera nothing carries is an error, and the message names the alternatives."""

    @pytest.mark.parametrize("name", NOT_CARRIED)
    def test_the_result_is_an_error(self, name: str) -> None:
        result = _engine().render(camera_name=name)
        assert result["status"] == "error", (
            f"render({name!r}) reported {result['status']!r} for a camera this scene does not "
            f"carry, with text {_text(result)!r} and json {_json(result)!r}"
        )

    @pytest.mark.parametrize("name", NOT_CARRIED)
    def test_the_message_names_the_cameras_that_are_carried(self, name: str) -> None:
        text = _text(_engine().render(camera_name=name))
        assert f"Camera '{name}' not found" in text, text
        assert f"Available: ['{CAM}']" in text, text


class TestBothSurfacesGiveOneVerdictForOneName:
    """``render`` and ``get_frame`` report the same sentence for the same name.

    They read the same registry under the same lock, so a name one refuses the
    other cannot answer - and a locally reworded copy of the verdict is how the
    two drifted apart in the first place.
    """

    @pytest.mark.parametrize("name", NOT_CARRIED)
    def test_the_two_verdicts_are_identical(self, name: str) -> None:
        engine = _engine()
        rendered = _text(engine.render(camera_name=name))
        raw = _get_frame_verdict(engine, name)
        assert rendered == raw, f"render said {rendered!r}; get_frame said {raw!r}"


class TestARefusalCarriesNoFrame:
    """A refusal carries no PNG block and no pixel statistics.

    ``render``'s success envelope carries a PNG block precisely so the shared
    ``PolicyRunner._extract_frame_ndarray`` can pull frames for video
    recording. A refusal that still carried one would put black pixels into a
    recording under an error.
    """

    def test_no_image_block_is_emitted(self) -> None:
        result = _engine().render(camera_name="wrst")
        images = [b for b in result.get("content", []) if "image" in b]
        assert images == [], f"a refusal carried {len(images)} image block(s)"

    def test_no_pixel_statistics_are_reported(self) -> None:
        payload = _json(_engine().render(camera_name="wrst"))
        assert "pixel_mean" not in payload, payload
        assert "pixel_variance" not in payload, payload

    def test_the_refused_name_is_not_reported_as_the_camera(self) -> None:
        payload = _json(_engine().render(camera_name="wrst"))
        assert payload.get("camera") != "wrst", (
            "the json block attributed the frame to a camera the scene does not carry"
        )


class TestTheRefusalIsTotal:
    """A name of any type reaches a verdict rather than raising past the envelope.

    ``registered`` is total by contract, and the token test is membership in a
    *tuple*, whose ``==``-per-element comparison stays total for a name that is
    not hashable at all.
    """

    @pytest.mark.parametrize("name", [7, 0, ["wrist"], {"name": "wrist"}, 1.5], ids=["7", "0", "list", "dict", "1.5"])
    def test_a_non_string_name_is_refused(self, name: Any) -> None:
        result = _engine().render(camera_name=name)
        assert result["status"] == "error", f"render({name!r}) reported {result['status']!r}"
        assert "not found" in _text(result), _text(result)


class TestTheDocumentedDegradationsAreUnchanged:
    """Every documented blank-frame fallback still answers exactly as before.

    This is the over-reach control and the reason the fix is scoped to a
    *named* camera: broadening it to every registry miss would turn Isaac's own
    signature default into an error.
    """

    def test_a_carried_camera_still_renders_its_own_frame(self) -> None:
        result = _engine().render(camera_name=CAM)
        assert result["status"] == "success", _text(result)
        assert _text(result) == "Rendered (RTX rtx_realtime): 64x48"
        assert _json(result)["pixel_mean"] == pytest.approx(200.0 * 3 / 3)
        assert _json(result)["camera"] == CAM

    @pytest.mark.parametrize("token", list(FREE_CAMERA_TOKENS), ids=["None", "empty", "default", "free"])
    def test_a_token_that_names_no_camera_keeps_the_blank_frame(self, token: str | None) -> None:
        result = _engine().render(camera_name=token)  # type: ignore[arg-type]
        assert result["status"] == "success", _text(result)
        assert _text(result) == f"Rendered (no camera): {CONFIG_W}x{CONFIG_H}"

    def test_the_signature_default_is_one_of_those_tokens(self) -> None:
        default = inspect.signature(IsaacSimulation.render).parameters["camera_name"].default
        assert default in FREE_CAMERA_TOKENS, f"render's default camera_name is {default!r}, which the fix would refuse"

    @pytest.mark.parametrize("name", [CAM, "wrst"])
    def test_headless_mode_answers_first_as_documented(self, name: str) -> None:
        result = _engine(render_mode="headless").render(camera_name=name)
        assert result["status"] == "success", _text(result)
        assert _text(result) == f"Rendered (headless, no RTX): {CONFIG_W}x{CONFIG_H}"

    def test_a_carried_camera_without_a_handle_keeps_its_blank_frame(self) -> None:
        result = _engine(with_handle=False).render(camera_name=CAM)
        assert result["status"] == "success", _text(result)
        assert _text(result) == f"Rendered (Phase-1 camera, no RTX handle): {NATIVE_W}x{NATIVE_H}"

    def test_a_refused_name_costs_no_render(self) -> None:
        handle = _FakeCameraHandle()
        engine = _engine(handle=handle)
        engine.render(camera_name="wrst")
        assert handle.reads == [], f"the refused name reached the RTX handle: {handle.reads}"


#: The function on each backend that resolves ``render``'s ``camera_name``, and
#: therefore owns the verdict for a name the scene does not carry. Isaac's
#: ``render`` forwards to a private helper; the other two resolve inline.
_RENDER_NAME_RESOLVERS = [
    ("isaac", "strands_robots.simulation.isaac.simulation", "IsaacSimulation", "_render_frame"),
    ("newton", "strands_robots.simulation.newton.simulation", "NewtonSimEngine", "render"),
    ("mujoco", "strands_robots.simulation.mujoco.rendering", "RenderingMixin", "render"),
]


class TestEveryBackendGivesTheSameVerdict:
    """All three backends' render path carry the one refusal sentence.

    ``camera_fov_error``'s docstring states the invariant these backends share:
    a camera configuration one refuses must be refused by the others. The name
    of a camera is the same kind of fact, and this was the entry point that
    answered a name nothing carries with a frame.
    """

    @pytest.mark.parametrize(("label", "module", "cls", "func"), _RENDER_NAME_RESOLVERS, ids=str)
    def test_the_resolver_refuses_an_uncarried_name(self, label: str, module: str, cls: str, func: str) -> None:
        import importlib

        owner = getattr(importlib.import_module(module), cls)
        source = inspect.getsource(getattr(owner, func))
        assert "not found. Available" in source, (
            f"the {label} backend's render name resolver ({cls}.{func}) carries no "
            "refusal for a camera name the scene does not carry"
        )

    def test_the_scan_reached_all_three_backends(self) -> None:
        assert {row[0] for row in _RENDER_NAME_RESOLVERS} == {"isaac", "newton", "mujoco"}


class TestTheContractIsDocumented:
    """``render``'s docstring states the split, so a caller can find their half."""

    def test_the_blank_frame_bullet_is_scoped_to_an_unnamed_camera(self) -> None:
        doc = inspect.getdoc(IsaacSimulation.render) or ""
        bullet = doc[doc.index("``Rendered (no camera)``") :]
        bullet = bullet[: bullet.index("``Rendered (Phase-1")]
        assert "FREE_CAMERA_TOKENS" in bullet, bullet

    def test_the_docstring_names_the_refusal(self) -> None:
        doc = inspect.getdoc(IsaacSimulation.render) or ""
        assert "not found. Available" in doc, "render's docstring does not state the refusal it produces"
