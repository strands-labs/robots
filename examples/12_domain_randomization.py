#!/usr/bin/env python3
"""Randomize appearance, physics, and sensor noise for sim2real robustness.

Goal: Show the two sim2real primitives that make a recorded dataset (or an
in-sim eval) robust instead of overfit to one pristine scene:

  - ``randomize`` perturbs the world: geom colors, lighting, per-geom friction,
    and per-body mass - seeded, so a run is reproducible.
  - ``set_obs_noise`` adds sensor noise to observations: joint position/velocity
    Gaussian noise and camera pixel jitter, mimicking real encoders/cameras.

Training on randomized episodes teaches a policy to generalize across the
appearance and dynamics gap between sim and hardware. This example renders a
baseline frame, randomizes, renders again to show the change, then applies
observation noise and prints the perturbed joint reading.

Runs on CPU, no GPU/checkpoint/hardware. On macOS the offscreen renderer needs
``MUJOCO_GL=cgl``; on headless Linux use ``MUJOCO_GL=egl``.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: two PNGs (baseline + randomized) under the output dir, plus a
before/after joint reading showing the injected noise. Runtime: ~5 seconds.
"""

from __future__ import annotations

import argparse
import pathlib

import imageio.v3 as iio

from strands_robots import Robot


def _check(result: dict, what: str) -> dict:
    """Raise if a sim action returned an error dict - never continue silently."""
    if isinstance(result, dict) and result.get("status") == "error":
        msg = "; ".join(c.get("text", "") for c in result.get("content", []))
        raise RuntimeError(f"{what} failed: {msg}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="randomization seed (reproducible)")
    parser.add_argument("--out", default="/tmp/strands_domain_rand", help="directory for the PNG frames")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = Robot("so100", mesh=False)
    sim.add_object(
        name="cube",
        shape="box",
        size=[0.025, 0.025, 0.025],
        position=[0.2, 0.0, 0.05],
        color=[1, 0, 0, 1],
        mass=0.05,
    )
    sim.add_camera(name="view", position=[0.5, 0.0, 0.4], target=[0.2, 0.0, 0.05])

    # Baseline frame (pristine scene).
    baseline = out_dir / "01_baseline.png"
    iio.imwrite(baseline, sim.get_observation("so100")["view"])
    print(f"baseline frame           -> {baseline}")

    # Perturb the world: colors + lighting + per-geom friction + per-body mass.
    r = _check(
        sim._dispatch_action(
            "randomize",
            {"randomize_colors": True, "randomize_lighting": True, "randomize_physics": True, "seed": args.seed},
        ),
        "randomize",
    )
    print("randomize:", (r.get("content") or [{}])[0].get("text", "").splitlines()[0])

    randomized = out_dir / "02_randomized.png"
    iio.imwrite(randomized, sim.get_observation("so100")["view"])
    print(f"randomized frame (seed={args.seed}) -> {randomized}")

    # Read a clean joint value, then turn on sensor noise and read again. A
    # scalar joint reading is any float obs key that is not a velocity (.vel),
    # a camera image, or a floating-base pose field.
    obs = sim.get_observation("so100")
    joint_key = next(
        k for k, v in obs.items() if not k.endswith(".vel") and not k.startswith("base") and isinstance(v, float)
    )
    clean = float(sim.get_observation("so100")[joint_key])
    _check(
        sim._dispatch_action("set_obs_noise", {"joint_pos_std": 0.02, "joint_vel_std": 0.05, "camera_jitter_px": 1.0}),
        "set_obs_noise",
    )
    noisy = float(sim.get_observation("so100")[joint_key])
    print(f"\njoint {joint_key!r}: clean={clean:.4f} rad  noisy={noisy:.4f} rad  (delta={noisy - clean:+.4f})")
    print("Sensor noise is now applied to every observation until reset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
