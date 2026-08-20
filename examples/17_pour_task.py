#!/usr/bin/env python3
"""Author and score a contact-rich pouring task with particle-proxy contents.

Goal: Show the articulated-container task objects and the pour predicates -
how "open the carton and pour its contents into the tray" becomes a measurable
benchmark TODAY, with zero fluid simulation. The carton's contents are proxied
by a handful of rigid beads (small spheres via ``add_object``), and the
declarative benchmark DSL scores the pour with three predicates:

* ``joint_above(cap_slide)``            - the cap actually opened
* ``particles_inside(beads, tray)``     - enough beads landed in the receptacle
* ``particles_spilled(beads, ...)``     - too many beads ended up in neither
                                          container (failure clause)

The scene is composed from bundled MJCF task objects
(:mod:`strands_robots.simulation.task_objects`): an ``open_tray`` receptacle
and a ``sliding_carton`` mounted upside-down above it, so gravity does the
pouring the moment something slides the cap open. (The ``hinged_carton``
variant is the same task with the lid on a hinge - mount that one upright,
where gravity holds the lid shut; see the asset's own comment.) The scripted
teaser at the end drives the cap directly to demonstrate the physics and the
predicates flipping; the benchmark evaluation itself runs the ``mock`` policy,
which does not act, so success is expected to be 0 - the point is authoring
and scoring a contact-rich task the same way examples 10/11 score locomotion.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: the compiled benchmark's metadata, its evaluation metrics,
then the scripted pour flipping the success predicates.
Runtime: ~20 seconds on CPU.
"""

from __future__ import annotations

import argparse

from strands_robots import Robot
from strands_robots.simulation.benchmark import register_benchmark
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.predicates import make_predicate
from strands_robots.simulation.task_objects import task_object_path

# Proxy contents: 6 beads, spawned inside the carton cavity.
BEADS = [f"bead_{i}" for i in range(6)]

# The whole task, as data - same closed-registry DSL as example 11, applied to
# a contact-rich manipulation task instead of locomotion.
BENCHMARK_SPEC = {
    "name": "pour_carton_into_tray",
    "instruction": "Slide the carton cap open so the beads pour into the tray.",
    "default_robot": "so100",
    "max_steps": 300,
    "success": {
        "all": [
            {"predicate": "joint_above", "joint": "cap_slide", "value": 0.06},
            {
                "predicate": "particles_inside",
                "particles": BEADS,
                "container": "tray",
                "min_fraction": 0.8,
                "xy_tol": 0.12,
                "z_tol": 0.08,
            },
        ]
    },
    # FAILURE: more than one bead in neither the tray nor the carton.
    "failure": {
        "any": [
            {
                "predicate": "particles_spilled",
                "particles": BEADS,
                "containers": ["tray", "carton"],
                "max_spilled": 1,
                "xy_tol": 0.12,
                "z_tol": 0.30,
            }
        ]
    },
    # DENSE REWARD: every bead landed pays, and so does progress on the cap.
    "dense_reward": [
        {
            "predicate": "particles_inside_fraction",
            "particles": BEADS,
            "container": "tray",
            "xy_tol": 0.12,
            "z_tol": 0.08,
        },
        {"predicate": "joint_progress", "joint": "cap_slide", "target": 0.09, "weight": 5.0},
    ],
}


def build_scene(sim) -> None:
    """Compose the pour scene: tray + upside-down carton + beads inside it.

    The task objects attach through the EXISTING scene APIs -
    ``add_robot(urdf_path=...)`` namespaces their bodies and makes the
    carton's ``cap_slide`` observable by the joint predicates. The carton is
    mounted lid-down 0.30 m above the tray and out of the arm's reach (so a
    flailing mock policy cannot bump it open); its closed cap is the only
    thing holding the beads.
    """
    tray = sim.add_robot(name="tray", urdf_path=task_object_path("open_tray"), position=[0.55, 0.0, 0.0])
    if tray["status"] != "success":
        raise RuntimeError(f"failed to attach tray: {tray}")
    carton = sim.add_robot(
        name="carton",
        urdf_path=task_object_path("sliding_carton"),
        position=[0.55, 0.0, 0.30],
        orientation=[0.0, 1.0, 0.0, 0.0],  # pi about X: cavity opens downward, cap underneath
    )
    if carton["status"] != "success":
        raise RuntimeError(f"failed to attach carton: {carton}")
    # Beads inside the flipped cavity (world z ~[0.18, 0.30] at x=0.55).
    positions = [
        [0.53, 0.0, 0.19],
        [0.57, 0.0, 0.19],
        [0.53, 0.0, 0.22],
        [0.57, 0.0, 0.22],
        [0.53, 0.0, 0.25],
        [0.57, 0.0, 0.25],
    ]
    for name, pos in zip(BEADS, positions, strict=True):
        bead = sim.add_object(name=name, shape="sphere", size=[0.024] * 3, position=pos, mass=0.05, color=[0.9, 0.6, 0.1])
        if bead["status"] != "success":
            raise RuntimeError(f"failed to add {name}: {bead}")
    sim.step(200)  # let the beads settle onto the closed cap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2, help="number of evaluation episodes")
    parser.add_argument("--policy", default="mock", help="policy provider to score (mock needs no GPU)")
    parser.add_argument("--seed", type=int, default=0, help="evaluation seed (scores are deterministic under it)")
    args = parser.parse_args()

    benchmark = DeclarativeBenchmark.from_dict(BENCHMARK_SPEC)
    register_benchmark(benchmark.name, benchmark)
    print(f"Registered pour benchmark: {benchmark.name}")
    print(f"  instruction : {benchmark.instruction}")

    sim = Robot("so100", mesh=False)
    build_scene(sim)

    # Score it exactly like a built-in benchmark. mock does not act, so the
    # cap stays shut and success_rate is expected to be 0.
    result = sim.evaluate_benchmark(
        benchmark.name,
        robot_name="so100",
        policy_provider=args.policy,
        n_episodes=args.episodes,
        seed=args.seed,
    )
    if result.get("status") == "error":
        msg = "; ".join(c.get("text", "") for c in result.get("content", []))
        raise RuntimeError(f"evaluate_benchmark failed: {msg}")
    metrics = next((c["json"] for c in result.get("content", []) if "json" in c), {})
    print(f"\n{benchmark.name} with the {args.policy!r} policy (seed {args.seed}):")
    print(f"  success_rate : {metrics.get('success_rate')}")
    print(f"  avg_reward   : {metrics.get('avg_reward')}")
    print(f"  avg_steps    : {metrics.get('avg_steps')}")

    # Scripted teaser: slide the cap open and watch the same predicates the
    # benchmark scores with flip as the beads pour out.
    cap_open = make_predicate("joint_above", joint="cap_slide", value=0.06)
    poured = make_predicate(
        "particles_inside", particles=BEADS, container="tray", min_fraction=0.8, xy_tol=0.12, z_tol=0.08
    )
    print(f"\nScripted pour: cap_open={cap_open(sim)} poured={poured(sim)}")
    sim.set_joint_positions({"cap_slide": 0.09}, robot_name="carton")
    sim.step(600)
    print(f"After sliding the cap open: cap_open={cap_open(sim)} poured={poured(sim)}")
    print("\n(mock does not open the cap, so the benchmark scores 0 - point --policy at a")
    print(" trained provider, or record the scripted pour as demonstration data.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
