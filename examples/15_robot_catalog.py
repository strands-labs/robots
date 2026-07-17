#!/usr/bin/env python3
"""Discover the robot catalog: what can I simulate, and how do I spawn it?

Goal: Answer the first question a new user has - "which robots can I use?" - from
the registry, with no sim or hardware. ``list_robots(mode="sim")`` returns every
robot that has a simulation model; ``list_robots_by_category(...)`` groups them
(arm, humanoid, hand, quadruped/mobile, bimanual, aerial, ...); ``get_robot(name)``
returns one robot's detail. Any name printed here is a valid ``Robot(name)``.

This is pure registry lookup - no MuJoCo, no GPU, no hardware, no Hugging Face
credentials - so it is the fastest way to explore what the SDK supports before
you build a scene.

Dependencies: pip install "strands-robots"
Expected output: a per-category count table of simulatable robots, then the
detail record for one robot. Runtime: <1 second.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from strands_robots import get_robot, list_robots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", default="go2", help="a robot name to show the detail record for")
    parser.add_argument("--category", default=None, help="list every simulatable robot in one category")
    args = parser.parse_args()

    # list_robots(mode="sim") returns every robot that ships a simulation model;
    # group them by their "category" field.
    sim_robots = list_robots(mode="sim")
    by_category: dict[str, list[dict]] = defaultdict(list)
    for robot in sim_robots:
        by_category[robot["category"]].append(robot)

    print(f"Simulatable robots: {len(sim_robots)} across {len(by_category)} categories\n")
    print(f"  {'category':14} count  examples")
    for category, robots in sorted(by_category.items()):
        names = ", ".join(r["name"] for r in robots[:3])
        more = ", ..." if len(robots) > 3 else ""
        print(f"  {category:14} {len(robots):>5}  {names}{more}")

    if args.category:
        robots = by_category.get(args.category)
        if not robots:
            raise SystemExit(f"unknown category {args.category!r}; choose one of {sorted(by_category)}")
        print(f"\nAll simulatable {args.category} robots:")
        for r in robots:
            print(f"  {r['name']:24} {r['joints']:>3} DOF  {r['description']}")

    # One robot's detail record. Every name above is a valid Robot(name).
    detail = get_robot(args.inspect)
    print(f"\nget_robot({args.inspect!r}):")
    print(f"  category   : {detail.get('category')}")
    print(f"  joints     : {detail.get('joints')}")
    print(f"  aliases    : {detail.get('aliases', [])}")
    print(f'\nSpawn any of them in sim with: Robot("{args.inspect}")')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
