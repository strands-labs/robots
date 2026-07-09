"""render() honors a named camera's configured resolution.

A camera added via ``add_camera(width=, height=)`` declares its own resolution.
``get_observation`` already keys off that per-camera config, but ``render()``
historically ignored it and fell back to the engine default whenever the caller
omitted ``width``/``height`` - contradicting its documented contract and
silently producing a different frame size than the camera (and the recorded
dataset) used. These tests pin the agreed contract: omitted dims -> the named
camera's configured resolution; explicit dims -> override; free/unknown camera
-> engine default.
"""

from __future__ import annotations

import io
import os

import pytest

pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl as _requires_mujoco  # noqa: E402


def _rendered_size(result: dict) -> tuple[int, int]:
    """Decode the PNG image block and return its ``(width, height)``."""
    from PIL import Image

    assert result["status"] == "success", result
    for block in result["content"]:
        if "image" in block:
            png = block["image"]["source"]["bytes"]
            return Image.open(io.BytesIO(png)).size
    raise AssertionError(f"no image block in render result: {result}")


@_requires_mujoco
def test_render_uses_camera_configured_resolution() -> None:
    """Omitted width/height -> the camera's own configured resolution."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    # A non-default resolution distinct from the 640x480 engine default.
    sim.add_camera("cam_hi", position=[0.5, 0.0, 0.4], target=[0.0, 0.0, 0.1], width=320, height=240)

    result = sim.render(camera_name="cam_hi")  # no width/height
    assert _rendered_size(result) == (320, 240)


@_requires_mujoco
def test_render_explicit_dims_override_camera_config() -> None:
    """Explicit width/height still win over the camera's configured resolution."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_camera("cam_hi", position=[0.5, 0.0, 0.4], target=[0.0, 0.0, 0.1], width=320, height=240)

    result = sim.render(camera_name="cam_hi", width=128, height=96)
    assert _rendered_size(result) == (128, 96)


@_requires_mujoco
def test_render_free_camera_uses_engine_default() -> None:
    """The free ('default') camera has no SimCamera config -> engine default."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()

    result = sim.render(camera_name="default")  # free camera, no config
    assert _rendered_size(result) == (sim.default_width, sim.default_height)


@_requires_mujoco
def test_render_matches_get_observation_resolution() -> None:
    """render() and get_observation() agree on a configured camera's frame size."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import numpy as np

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam_obs", position=[0.5, 0.0, 0.4], target=[0.0, 0.0, 0.1], width=200, height=160)

    rendered_w, rendered_h = _rendered_size(sim.render(camera_name="cam_obs"))
    obs = sim.get_observation()
    arr = np.asarray(obs["cam_obs"])
    obs_h, obs_w = arr.shape[0], arr.shape[1]
    assert (rendered_w, rendered_h) == (obs_w, obs_h) == (200, 160)


@_requires_mujoco
def test_render_depth_uses_camera_configured_resolution() -> None:
    """render_depth() also honors the camera's configured resolution.

    render_depth historically fell back to the engine default whenever the
    caller omitted width/height, so the depth map came back at a different size
    than render()'s RGB frame for the same camera - breaking pixel alignment
    for any depth-aware downstream consumer. Omitted dims must resolve to the
    camera's own configured resolution, matching render() / render_all().
    """
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_camera("cam_hi", position=[0.5, 0.0, 0.4], target=[0.0, 0.0, 0.1], width=320, height=240)

    result = sim.render_depth(camera_name="cam_hi")  # no width/height
    assert _rendered_size(result) == (320, 240)


@_requires_mujoco
def test_render_depth_explicit_dims_override_camera_config() -> None:
    """Explicit width/height still win over the camera's configured resolution."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_camera("cam_hi", position=[0.5, 0.0, 0.4], target=[0.0, 0.0, 0.1], width=320, height=240)

    result = sim.render_depth(camera_name="cam_hi", width=128, height=96)
    assert _rendered_size(result) == (128, 96)


@_requires_mujoco
def test_render_depth_free_camera_uses_engine_default() -> None:
    """The free ('default') camera has no SimCamera config -> engine default."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()

    result = sim.render_depth(camera_name="default")  # free camera, no config
    assert _rendered_size(result) == (sim.default_width, sim.default_height)


@_requires_mujoco
def test_render_depth_matches_render_resolution() -> None:
    """render() and render_depth() agree on a configured camera's frame size.

    Depth-aware consumers pair the RGB frame with the depth map per pixel; the
    two must be the same size for the same named camera.
    """
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam_obs", position=[0.5, 0.0, 0.4], target=[0.0, 0.0, 0.1], width=200, height=160)

    rgb_size = _rendered_size(sim.render(camera_name="cam_obs"))
    depth_size = _rendered_size(sim.render_depth(camera_name="cam_obs"))
    assert rgb_size == depth_size == (200, 160)
