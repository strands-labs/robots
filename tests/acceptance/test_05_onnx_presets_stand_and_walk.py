"""Check 5: the ONNX presets: holosoma keeps a G1 standing for 10 s, microduck walks forward on command."""

from __future__ import annotations

import pytest
from checks import needs_module

pytest.importorskip("mujoco")


def lowest_height(robot, policy, seconds: float) -> float:
    zs: list[float] = []
    robot.run(policy, hz=50, seconds=seconds, stop_when=lambda o: zs.append(float(o.pose.position[2])) or False)
    return min(zs)


def test_holosoma_keeps_the_g1_standing():
    needs_module("strands_robots.policy.onnx", "part 4: ONNX policies")
    pytest.importorskip("onnxruntime")
    from strands_robots import Policy, Robot

    g1 = Robot("g1", keyframe="stand")
    z = lowest_height(g1, Policy("holosoma"), 10)
    print(f"[measured] g1 under holosoma: pelvis min {z:.3f} m over 10 s")
    assert z >= 0.74


def test_microduck_walks_a_metre_in_ten_seconds():
    needs_module("strands_robots.policy.onnx", "part 4: ONNX policies")
    pytest.importorskip("onnxruntime")
    from strands_robots import Policy, Robot

    duck = Robot("microduck")
    x0 = float(duck.pose.position[0])
    z = lowest_height(duck, Policy("microduck?vx=0.3"), 10)
    dx = float(duck.pose.position[0]) - x0
    print(f"[measured] microduck vx=0.3: dx {dx:.2f} m, trunk min {z:.3f} m over 10 s")
    assert dx >= 1.0 and z > 0.10
