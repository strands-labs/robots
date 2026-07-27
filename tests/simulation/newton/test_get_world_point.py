# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``get_world_point`` on the Newton backend (issue #1647).

Newton's ray-traced tiled camera has no depth output path (``get_frame``
returns ``(rgb, None)``), so pixel-to-world grounding must degrade to the
documented structured no-depth error -- never a silent zero point. Skipped
when Newton/Warp are not installed.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


@pytest.fixture
def engine_with_camera():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    sim = NewtonSimEngine(solver="mujoco")
    sim.create_world()
    sim.add_robot("so101")
    sim.add_camera("front", position=[0.5, -0.5, 0.4], target=[0.0, 0.0, 0.1], width=64, height=48)
    yield sim
    sim.destroy()


def test_get_world_point_reports_no_depth_structured_error(engine_with_camera) -> None:
    result = engine_with_camera.get_world_point("front", pixels=[[10, 10], [12, 12]])
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "no" in text.lower() and "depth" in text
    # The error names the remedy, not just the failure.
    assert "MuJoCo" in text or "RGB-D" in text
