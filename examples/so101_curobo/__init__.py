# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SO-101 synthetic-data generation with cuRobo on the Isaac/MuJoCo backend.

Design + feasibility: strands-labs/robots-sim#67. The control + data-collection
loop runs today on the **MuJoCo** backend (which loads a real SO-101) with a
**scripted** planner fallback; it flips to the **Isaac** backend and **cuRobo**
collision-aware planning once those runtimes are installed — only the
``make_sim(...)`` / ``make_planner(...)`` choices change.

Public pieces:
    scene.make_sim / scene.build_pick_place_scene
    planner.make_planner (ScriptedPlanner | CuroboMotionPlanner)
    collector.LeRobotDataCollector
    controller.SO101CuroboDemo
"""

__all__ = ["scene", "planner", "collector", "controller"]
