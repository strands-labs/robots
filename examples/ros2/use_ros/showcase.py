#!/usr/bin/env python3
"""Live showcase of every use_ros action against a real ROS 2 turtlesim.

Runs in-process through rclpy (no ros2 CLI, no code-generation) and exercises
the full action surface against a running turtlesim node:

    status / list_topics / list_nodes / list_services / info /
    echo / publish / service_call  + the two error paths.

It captures real data - including a closed sense->act->sense loop: read the
pose, publish a Twist, read the pose again and confirm it changed - plus the
two structured-error contracts (a valid-shaped but non-installed type returns
an error dict instead of crashing; a metacharacter topic is rejected).

Run it inside a sourced ROS 2 environment with turtlesim running:

    # in a ros:jazzy container (or any machine with ROS 2 sourced):
    source /opt/ros/jazzy/setup.bash
    QT_QPA_PLATFORM=offscreen ros2 run turtlesim turtlesim_node &
    pip install strands-agents          # rclpy comes from the sourced distro
    # /turtle1/cmd_vel is a gated command surface and this script runs outside
    # an agent loop, so there is no operator to prompt - pre-approve it:
    STRANDS_ROS2_COMMAND_ALLOW=/turtle1/cmd_vel python3 showcase.py

See ./README.md for a one-command docker-compose runner.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from typing import Any

from strands_robots.tools.use_ros import use_ros as _use_ros_tool

# The @tool decorator returns a DecoratedFunctionTool instance whose
# __call__ static analysis (CodeQL) does not recognize. Bind it to a
# Callable-typed name so the call sites are seen as calling a function.
use_ros: Callable[..., dict[str, Any]] = _use_ros_tool


def run(**kw: object) -> dict:
    args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
    print(f"\n### use_ros({args})")
    r = use_ros(**kw)  # type: ignore[arg-type]
    print(f"[{r['status']}] {r['content'][0]['text'][:600]}")
    return r


def _pose_xyt(result: dict) -> tuple[float, float, float] | None:
    txt = result["content"][0]["text"]
    if result["status"] != "success" or ":\n" not in txt:
        return None
    sample = json.loads(txt.split(":\n", 1)[1])[0]
    return sample["x"], sample["y"], sample["theta"]


def main() -> int:
    print("=" * 70)
    print("use_ros LIVE SHOWCASE - in-process rclpy against a real turtlesim")
    print("=" * 70)

    status = run(action="status")
    if "rclpy (in-process)" not in status["content"][0]["text"]:
        print("\nERROR: rclpy backend not available - source a ROS 2 distro first.", file=sys.stderr)
        return 2

    # Graph introspection.
    run(action="list_topics")
    run(action="list_nodes")
    run(action="list_services")
    run(action="info", topic="/turtle1/cmd_vel")

    # Closed sense -> act -> sense loop.
    before = run(action="echo", topic="/turtle1/pose", count=1, timeout=3.0)
    run(
        action="publish",
        topic="/turtle1/cmd_vel",
        type="geometry_msgs/msg/Twist",
        fields={"linear": {"x": 2.0}, "angular": {"z": 1.8}},
        count=20,
        rate=10.0,
    )
    time.sleep(0.4)
    after = run(action="echo", topic="/turtle1/pose", count=1, timeout=3.0)

    # Service call: spawn a second turtle, then confirm it joined the graph.
    run(
        action="service_call",
        service="/spawn",
        type="turtlesim/srv/Spawn",
        fields={"x": 2.0, "y": 2.0, "theta": 0.0, "name": "t2"},
    )
    time.sleep(0.5)
    run(action="list_topics")

    # Structured-error contracts (never raise).
    run(action="echo", topic="/turtle1/pose", type="nonexistent_pkg/msg/Foo")
    run(action="echo", topic="/bad; rm -rf")

    b, a = _pose_xyt(before), _pose_xyt(after)
    print("\n" + "=" * 70)
    print("PROOF: pose changed via use_ros publish (in-process rclpy, no CLI):")
    print(f"  before: {b}")
    print(f"  after:  {a}")
    print("=" * 70)
    if b and a and (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) > 0.1:
        print("PASS: a real ROS 2 turtle was driven and read back through use_ros.")
        return 0
    print("FAIL: pose did not change", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
