"""GPU integration test for ``get_world_point`` on the Isaac backend (#1647).

Exercises the pixel-to-world grounding path against a real RTX depth
annotator (``distance_to_image_plane``): a camera looking at the origin of
the default ground plane must unproject its center pixel to a world point on
that plane (z ~ 0), and out-of-bounds pixels must be rejected with the
structured error contract.

Requirements match ``test_isaac_recording_gpu.py``: NVIDIA GPU + CUDA, Isaac
Sim 6.0+ installed out-of-band, ``pip install 'strands-robots[sim-isaac]'``,
and ``STRANDS_GPU_TEST=1``. Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ \\
        tests_integ/simulation/test_isaac_world_point_gpu.py -m gpu -v
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]


def _skip_if_isaac_unavailable() -> None:
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


class TestIsaacWorldPoint:
    def test_center_pixel_grounds_to_the_ground_plane(self):
        """Camera looking at the origin: the center pixel's median world point
        lands on the ground plane (z ~ 0) within tolerance -- the same
        acceptance shape as the MuJoCo box regression, on real RTX depth."""
        import numpy as np

        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=False, render_mode="rtx_realtime"))
        try:
            r = sim.create_world(ground_plane=True)
            assert r["status"] == "success", f"create_world: {r}"
            r = sim.add_camera("front", position=[0.0, 2.0, 1.5], target=[0.0, 0.0, 0.0])
            assert r["status"] == "success", f"add_camera: {r}"
            sim.reset()
            sim.step(5)

            cam = sim.get_camera_params("front")
            cu, cv = int(cam.width // 2), int(cam.height // 2)
            pixels = [[cu, cv], [cu - 3, cv], [cu + 3, cv], [cu, cv - 3], [cu, cv + 3]]

            result = sim.get_world_point("front", pixels=pixels)
            assert result["status"] == "success", result
            data = result["content"][1]["json"]
            assert data["n_valid"] >= 3, data
            point = np.asarray(data["point"], dtype=float)
            assert np.isfinite(point).all()
            # The camera-to-origin ray hits the ground plane at/near the
            # origin; the depth axis (plane height) is the tight assertion.
            assert abs(point[2]) < 0.05, data
            assert np.linalg.norm(point[:2]) < 0.5, data

            # Out-of-bounds pixels degrade to the structured error contract.
            bad = sim.get_world_point("front", pixels=[[cam.width, 0]])
            assert bad["status"] == "error"
            assert "outside" in bad["content"][0]["text"]
        finally:
            sim.destroy()
