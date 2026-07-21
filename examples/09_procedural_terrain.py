#!/usr/bin/env python3
"""Build each procedural terrain, drop a quadruped on it, and confirm the ground.

Goal: Show that ``create_world(terrain=...)`` lays down a deterministic
heightfield ground - rough bumps, stairs, a pyramid, or a slope - instead of
the flat plane, so a locomotion scene tests robustness to ground the robot can
trip on. The terrain curriculum knob is the ``difficulty`` scalar: it scales the
peak elevation without changing the terrain kind, so the same world gets harder
across resets.

For each terrain the example inspects the generated heightfield (its distinct
signature is the proof each kind is different: ``stairs`` is five discrete
plateaus, ``slope`` is a monotonic ramp, ``rough`` is continuous noise,
``pyramid`` is symmetric concentric plateaus), then rebuilds the world, spawns
the robot, steps physics so it settles onto the ground, and saves a rendered
frame. Note the default ~8 cm terrain is deliberately subtle in the image -
raise ``--difficulty`` to make the relief obvious.

No policy, no checkpoint, no GPU, no Hugging Face credentials: the heightfield
generator is pure Python and MuJoCo steps on CPU. This is the sim-only
foundation the locomotion examples (``examples/locomotion/``) run their WBC
policy on top of.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: one line per terrain (kind, peak elevation, distinct
heightfield levels) and one PNG per terrain under the output dir. Runtime:
~5 seconds on CPU.
"""

from __future__ import annotations

import argparse
import pathlib

import imageio.v3 as iio

from strands_robots import Robot
from strands_robots.simulation.terrain import (
    SUPPORTED_TERRAINS,
    generate_heightfield,
    terrain_elevation,
)


def _check(result: dict, what: str) -> None:
    """Raise if a sim action returned an error dict - never continue silently."""
    if isinstance(result, dict) and result.get("status") == "error":
        msg = "; ".join(c.get("text", "") for c in result.get("content", []))
        raise RuntimeError(f"{what} failed: {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="go2", help="registered robot to spawn (default: go2 quadruped)")
    parser.add_argument("--difficulty", type=float, default=1.0, help="terrain curriculum scale (1.0 = full height)")
    parser.add_argument("--steps", type=int, default=40, help="physics steps to settle the robot before capture")
    parser.add_argument("--out", default="/tmp/strands_terrain", help="directory for the per-terrain PNG frames")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sim by default - no hardware. mesh=False keeps this a single local process.
    sim = Robot(args.robot, mesh=False)

    for kind in SUPPORTED_TERRAINS:
        # The heightfield is pure data, generated deterministically before any
        # physics: its distinct-level count is the signature that each kind is a
        # different ground (stairs -> a few discrete plateaus, slope -> a
        # monotonic ramp, rough -> continuous noise, pyramid -> concentric rings).
        heightfield = generate_heightfield(kind)
        distinct_levels = len({round(h, 4) for h in heightfield})

        # Tear down the current world (the factory built a flat one, and
        # create_world refuses to overwrite a live world), then lay this
        # heightfield ground. Each step returns a status dict - check it rather
        # than assume success. difficulty scales the terrain's peak elevation.
        _check(sim.destroy(), f"destroy before {kind}")
        _check(sim.create_world(terrain=kind, difficulty=args.difficulty), f"create_world(terrain={kind!r})")
        _check(sim.add_robot(args.robot), f"add_robot({args.robot!r}) on {kind}")
        # Elevated 3/4 camera: terrain relief reads best from above and to the side.
        sim.add_camera(name="view", position=[4.0, -4.0, 3.0], target=[0.0, 0.0, 0.2])

        # Let the robot settle so it rests on the heightfield (real contact).
        for _ in range(args.steps):
            sim.step()

        frame = sim.get_observation(args.robot)["view"]
        path = out_dir / f"terrain_{kind}.png"
        iio.imwrite(path, frame)
        print(f"{kind:8} peak={terrain_elevation(args.difficulty):.3f} m  levels={distinct_levels:>4}  ->  {path}")

    print(f"Saved {len(SUPPORTED_TERRAINS)} terrain frames under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
