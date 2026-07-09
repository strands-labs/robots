"""Regression: sim.render() must produce a non-black frame immediately after
replace_scene_mjcf / patch_scene_mjcf.

Surfaced by the 50-cycle demo-artifact generation run
(`/tmp/pr85_artifacts/`). Cycle 21 (patch_scene_mjcf block-tower build) and
cycle 26 (add/remove lifecycle) produced rendered frames with 68% / 55%
black pixels - the hero collage exposed what looked like "black tiles".
Root cause: neither scene-op called `mj_forward` after swapping/recompiling
model+data, so body `xpos` / camera `cam_xmat` arrays were still zero from
`mj.MjData(model)` initialisation. The renderer saw all bodies stacked at
the origin behind the free camera.

Fix: call `mj_forward(model, data)` inside `replace_scene_mjcf` and
`patch_scene_mjcf` right after the recompile, so the very first `render()`
call has valid geom transforms.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("PIL")

from io import BytesIO  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "glfw")

from PIL import Image  # noqa: E402

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from tests.simulation.mujoco._gl_probe import gl_available  # noqa: E402


def _black_fraction(img: np.ndarray) -> float:
    return float((img.mean(axis=2) < 20).mean())


def _render_np(sim: Simulation, camera: str, w: int = 240, h: int = 180) -> np.ndarray:
    import base64

    r = sim.render(camera_name=camera, width=w, height=h)
    assert r["status"] == "success", r
    for block in r.get("content", []):
        if "image" in block:
            raw = block["image"]["source"]["bytes"]
            if isinstance(raw, str):
                raw = base64.b64decode(raw)
            return np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
    raise AssertionError("render returned no image")


@pytest.fixture
def sim():
    s = Simulation(tool_name="render_after_scene_edit", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _skip_if_no_gl():
    """Skip when no usable offscreen GL context is available."""
    if not gl_available():
        pytest.skip("no usable OpenGL context; force-skip with ROBOT_TEST_MUJOCO=0")


class TestRenderAfterSceneEdit:
    def test_replace_scene_mjcf_render_not_black(self, sim: Simulation) -> None:
        _skip_if_no_gl()
        sim.create_world()
        sim.replace_scene_mjcf(
            "<mujoco><worldbody>"
            '<light pos="0 0 3" dir="0 0 -1" directional="true"/>'
            '<geom type="plane" size="1 1 0.01" rgba="0.85 0.85 0.9 1"/>'
            '<body name="b" pos="0 0 0.1">'
            '<geom type="box" size="0.1 0.1 0.05" rgba="1 0 0 1"/></body>'
            '<camera name="c" pos="0.8 -0.8 0.4" xyaxes="0.707 0.707 0 -0.2 0.2 0.96"/>'
            "</worldbody></mujoco>"
        )
        img = _render_np(sim, "c")
        # Before the fix this was 100% black. A correct render has the plane
        # + red box visible; generous upper bound lets CI tolerate
        # different tone-maps / depth-clear colours.
        assert _black_fraction(img) < 0.7, (
            f"render immediately after replace_scene_mjcf is still mostly black "
            f"(black_frac={_black_fraction(img):.2%}). mj_forward probably "
            "missing from scene_ops.replace_scene_mjcf."
        )
        assert img.mean() > 20, f"frame looks empty: mean={img.mean():.1f}"

    def test_patch_scene_mjcf_render_not_black(self, sim: Simulation) -> None:
        _skip_if_no_gl()
        sim.create_world()
        sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "block_0", "pos": [0, 0, 0.1]},
                {
                    "op": "add_geom",
                    "body": "block_0",
                    "type": "box",
                    "size": [0.1, 0.1, 0.05],
                    "rgba": [1, 0, 0, 1],
                },
            ]
        )
        img = _render_np(sim, "default")
        assert _black_fraction(img) < 0.5, (
            f"render immediately after patch_scene_mjcf is still mostly black "
            f"(black_frac={_black_fraction(img):.2%}). mj_forward probably "
            "missing from scene_ops.patch_scene_mjcf."
        )

    def test_iterative_patch_renders_each_step(self, sim: Simulation) -> None:
        """Simulates the pr85_artifacts cycle 21 'iterative block tower' demo:
        repeated patches, render after each one. Every frame must be non-black.
        """
        _skip_if_no_gl()
        sim.create_world()
        frames = []
        for i in range(5):
            sim.patch_scene_mjcf(
                [
                    {"op": "add_body", "name": f"blk_{i}", "pos": [0, 0, 0.05 + i * 0.12]},
                    {
                        "op": "add_geom",
                        "body": f"blk_{i}",
                        "type": "box",
                        "size": [0.1, 0.1, 0.05],
                        "rgba": [0.2 + 0.15 * i, 0.5, 0.95 - 0.15 * i, 1.0],
                    },
                ]
            )
            frames.append(_render_np(sim, "default"))
        for i, img in enumerate(frames):
            bf = _black_fraction(img)
            assert bf < 0.5, f"frame after block_{i} is {bf:.1%} black"

    def test_render_returns_raw_png_bytes_not_base64(self, sim: Simulation) -> None:
        """Pin the image-content contract: render() must emit RAW PNG bytes in
        ``source.bytes``, not a base64 str.

        The boto3 Bedrock Converse client base64-encodes ``source.bytes`` on the
        wire. Pre-encoding to a base64 str in render() double-encoded the image
        and Bedrock rejected it ("ConverseStream: Could not process image"). The
        other render tests defensively b64decode both str and bytes, so they
        never pinned this contract and the double-encode shipped unnoticed.
        """
        _skip_if_no_gl()
        sim.create_world()
        r = sim.render(camera_name="default", width=320, height=240)
        assert r["status"] == "success", r
        block = next((c for c in r.get("content", []) if "image" in c), None)
        assert block is not None, "render returned no image block"
        raw = block["image"]["source"]["bytes"]
        assert isinstance(raw, (bytes, bytearray)), (
            "render() must return raw bytes in source.bytes; boto3 base64-encodes "
            f"on the wire, so a str double-encodes. Got {type(raw).__name__}."
        )
        assert bytes(raw[:8]) == b"\x89PNG\r\n\x1a\n", "source.bytes is not a valid PNG payload"
