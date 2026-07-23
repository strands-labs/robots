"""Import + contract smoke tests for the ``examples/isaac_gs`` hybrid-render demo.

Pins two clean-checkout breakages from #1536:

1. **Example-vs-repo drift**: ``examples/isaac_gs`` hard-imports its sibling
   ``examples.mujoco_gs`` (``CameraParams``, ``BackgroundRenderer``,
   ``PanoramaBackground``, the gsplat preset helpers). #1280 originally merged
   isaac_gs before that dependency existed on ``main``; it only appeared to work
   on dev boxes where a stale ``robots-sim`` checkout was on ``sys.path``.
   Importing the modules here makes CI fail loudly if the sibling example (or a
   symbol it re-exports) ever drifts again - per AGENTS.md "test import paths
   must match production".

2. **Stale ``render()`` envelope**: ``render_rgb_and_depth`` used to read raw
   ``result["rgb"]`` / ``result["depth"]`` off the ``sim.render()`` tool-result
   envelope, which carries ONLY ``{status, content}`` (the tool-result contract
   forbids extra top-level keys) - a guaranteed ``KeyError``. It now consumes
   the public raw-frame API ``sim.get_frame`` (issue #1537 promoted the
   private ``_render_frame`` reach-through into ``SimEngine.get_frame``,
   which raises on degraded paths instead of returning blank frames). The
   behavioral tests below pin that consumption contract with a stub sim -
   no Isaac install needed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ``examples`` is a PEP 420 namespace package rooted at the repo top level.
# pytest's rootdir insertion usually covers this, but make it explicit so the
# test cannot silently resolve ``examples.mujoco_gs`` from some OTHER checkout
# earlier on sys.path (the exact failure mode that let #1280 ship broken).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_ISAAC_GS_MODULES = [
    "examples.isaac_gs.camera_utils",
    "examples.isaac_gs.compositor",
    "examples.isaac_gs.background",
]


@pytest.mark.parametrize("module_name", _ISAAC_GS_MODULES)
def test_isaac_gs_module_imports_from_this_repo(module_name: str) -> None:
    """Each isaac_gs module imports on a clean checkout, resolved from THIS repo."""
    module = importlib.import_module(module_name)
    assert module.__file__ is not None
    resolved = Path(module.__file__).resolve()
    assert resolved.is_relative_to(_REPO_ROOT), (
        f"{module_name} resolved to {resolved}, outside this checkout "
        f"({_REPO_ROOT}). A stale sibling checkout on sys.path is masking "
        "missing files - the exact failure mode of #1536."
    )


def test_isaac_gs_mujoco_gs_dependency_resolves_in_repo() -> None:
    """The ``examples.mujoco_gs`` symbols isaac_gs depends on exist in this repo."""
    backgrounds = importlib.import_module("examples.mujoco_gs.backgrounds")
    for symbol in (
        "BackgroundRenderer",
        "PanoramaBackground",
        "GsplatBackground",
        "download_gsplat_scene",
        "gsplat_skybox_align_for",
        "gsplat_skybox_scene_names",
    ):
        assert hasattr(backgrounds, symbol), f"examples.mujoco_gs.backgrounds lost {symbol}"
    camera_utils = importlib.import_module("examples.mujoco_gs.camera_utils")
    assert hasattr(camera_utils, "CameraParams")
    assert backgrounds.__file__ is not None
    assert Path(backgrounds.__file__).resolve().is_relative_to(_REPO_ROOT)


class _StubSim:
    """Stub of the public ``sim.get_frame`` raw-frame surface (issue #1537)."""

    def __init__(self, rgb, depth, error: str | None = None):
        self._rgb = rgb
        self._depth = depth
        self._error = error

    def get_frame(self, camera_name: str = "default", width=None, height=None):
        if self._error is not None:
            raise RuntimeError(self._error)
        return self._rgb, self._depth

    def _render_frame(self, camera_name: str = "default", width=None, height=None):
        raise AssertionError(
            "render_rgb_and_depth must NOT reach into the private _render_frame - "
            "the public get_frame API replaced it (#1537)."
        )

    def render(self, camera_name: str = "default", **_: object):
        raise AssertionError(
            "render_rgb_and_depth must NOT call the public render() - its "
            "{status, content} envelope carries no raw arrays (#1536)."
        )


def test_render_rgb_and_depth_consumes_render_frame() -> None:
    """Happy path: raw ``get_frame`` arrays come back as (uint8 RGB, float32 depth)."""
    from examples.isaac_gs.camera_utils import render_rgb_and_depth

    h, w = 4, 6
    rgb_in = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_in[..., 0] = 200  # red channel survives the round-trip
    depth_in = np.full((h, w), 1.5, dtype=np.float32)

    rgb, depth = render_rgb_and_depth(_StubSim(rgb_in, depth_in), "cam0")

    assert rgb.shape == (h, w, 3)
    assert rgb.dtype == np.uint8
    assert int(rgb[0, 0, 0]) == 200
    assert depth.shape == (h, w)
    assert depth.dtype == np.float32
    assert float(depth[0, 0]) == pytest.approx(1.5)


def test_render_rgb_and_depth_raises_on_render_failure() -> None:
    """Failure path: ``get_frame`` raises on degraded paths and the wrapper
    propagates it - never a KeyError off a tool-result envelope (#1536)."""
    from examples.isaac_gs.camera_utils import render_rgb_and_depth

    sim = _StubSim(None, None, error="No world created.")
    with pytest.raises(RuntimeError, match="No world created"):
        render_rgb_and_depth(sim, "cam0")
