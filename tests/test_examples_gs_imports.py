# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import smoke test: GS example shims stay wired to the library layer.

The mujoco_gs / isaac_gs hybrid-render examples now delegate their shared
rendering layer to ``strands_robots.rendering`` (issue #1537). This pins the
example/library seam so a rename or removal in either place fails CI instead
of breaking a clean checkout at demo time (the drift class from the
companion bug of #1537: examples importing unmerged sibling examples).

These imports must succeed with only the base install (numpy + Pillow):
none of the shim modules may import mujoco / omni / gradio / torch at
module load.
"""

import importlib

import pytest


def test_mujoco_gs_shims_reexport_library_layer() -> None:
    pkg = importlib.import_module("examples.mujoco_gs")
    from strands_robots import rendering

    # Background + camera layer come straight from the library.
    assert pkg.BackgroundRenderer is rendering.BackgroundRenderer
    assert pkg.PanoramaBackground is rendering.PanoramaBackground
    assert pkg.GsplatBackground is rendering.GsplatBackground
    assert pkg.CameraParams is rendering.CameraParams
    assert pkg.CompositeFrame is rendering.CompositeFrame
    # The example compositor is the library compositor + MuJoCo demo glue.
    assert issubclass(pkg.HybridCompositor, rendering.HybridCompositor)


def test_isaac_gs_shims_reexport_library_layer() -> None:
    pkg = importlib.import_module("examples.isaac_gs")
    from strands_robots import rendering

    assert pkg.IsaacCameraParams is rendering.CameraParams
    assert issubclass(pkg.IsaacHybridCompositor, rendering.HybridCompositor)


def test_isaac_gs_background_resolution_runs_without_gs_deps() -> None:
    from examples.isaac_gs.background import resolve_background
    from strands_robots.rendering import PanoramaBackground

    bg = resolve_background(prefer_gs=False)
    assert isinstance(bg, PanoramaBackground)


def test_gs_shim_modules_do_not_import_heavy_deps_at_load() -> None:
    # Fresh-interpreter check (immune to sys.modules pollution from other
    # tests): importing the shim seam must not pull sim/UI/GPU deps.
    import subprocess
    import sys as _sys

    code = (
        "import importlib, sys\n"
        "mods = ['examples.mujoco_gs.backgrounds', 'examples.mujoco_gs.camera_utils',\n"
        "        'examples.mujoco_gs.compositor', 'examples.isaac_gs.camera_utils',\n"
        "        'examples.isaac_gs.compositor', 'examples.isaac_gs.background']\n"
        "for m in mods:\n"
        "    importlib.import_module(m)\n"
        "heavy = [h for h in ('gradio', 'omni', 'isaacsim', 'mujoco', 'gsplat') if h in sys.modules]\n"
        "assert not heavy, f'heavy deps imported at shim load: {heavy}'\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr


def test_library_rendering_package_has_no_private_exports() -> None:
    from strands_robots import rendering

    private = [name for name in rendering.__all__ if name.startswith("_")]
    assert private == []
    missing = [name for name in rendering.__all__ if not hasattr(rendering, name)]
    assert missing == []


@pytest.mark.parametrize("backend_attr", ["get_frame", "get_camera_params"])
def test_sim_engine_declares_raw_frame_apis(backend_attr: str) -> None:
    from strands_robots.simulation.base import SimEngine

    assert hasattr(SimEngine, backend_attr)
