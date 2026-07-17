#!/usr/bin/env python3
"""Gait-clock WBC variant for the Unitree G1 - usage + clock visualization.

NVIDIA's GR00T-WholeBodyControl ships two MuJoCo reference controllers for the
G1: the non-gait balance/walk pair (see ``examples/wbc/wbc_g1_torque_deploy.py``)
and a *gait-clock* variant - a single ONNX policy fed a 95-dim observation that
carries an 8-wide command (with a ``freq_cmd`` step-frequency slot) plus a
2-dim bipedal phase clock. :class:`strands_robots.policies.wbc.WBCGaitPolicy`
ports that variant.

This example has two modes:

* ``--plot-clock`` (no checkpoint, no GPU): drive the :class:`GaitClock` through
  a static -> walk -> static command schedule and save a PNG of the two phase
  signals. This visualizes exactly what the variant adds on top of the non-gait
  controller - the locomotion rhythm - and runs anywhere.
* ``--checkpoint <dir>``: run the gait policy in the MuJoCo torque-deploy loop.
  Requires a gait-clock ONNX checkpoint whose input is ``[batch, 570]`` (95 x 6);
  the shipped Balance/Walk weights are the *non-gait* 516-wide family and will
  not load here.

Usage::

    # Clock visualization (the demonstrable behavior, no weights needed):
    python examples/wbc/wbc_g1_gait.py --plot-clock --out /tmp/g1_gait_clock.png

    # Policy usage (needs a 95x6 gait checkpoint):
    python examples/wbc/wbc_g1_gait.py --checkpoint /path/to/gait-g1 --vx 0.5 --freq 1.5
"""

from __future__ import annotations

import argparse

import numpy as np

from strands_robots.policies.wbc import GaitClock


def plot_clock(out_path: str, freq: float = 1.0, duration_s: float = 6.0) -> None:
    """Drive a GaitClock through static -> walk -> static and save a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt = 0.02
    n = int(duration_s / dt)
    cmd_scale = np.array([2.0, 2.0, 0.5])
    clk = GaitClock()

    t = np.arange(n) * dt
    fl, fr, walking = [], [], []
    for i in range(n):
        # static for the first and last third, walk forward in the middle.
        vx = 0.5 if (n // 3) <= i < (2 * n // 3) else 0.0
        scaled = np.array([vx, 0.0, 0.0]) * cmd_scale
        c = clk.update(scaled, freq=freq)
        fl.append(c[0])
        fr.append(c[1])
        walking.append(clk.walking_mask)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, fl, label="clock_FL (left foot)", lw=1.6)
    ax.plot(t, fr, label="clock_FR (right foot)", lw=1.6)
    walk = np.array(walking, dtype=bool)
    ax.fill_between(t, -1.1, 1.1, where=walk.tolist(), color="0.85", label="walking", zorder=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("clock signal")
    ax.set_title(f"WBC gait-clock phase signal (freq_cmd = {freq} Hz): static -> walk -> static")
    ax.set_ylim(-1.15, 1.15)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


def run_policy(checkpoint: str, vx: float, freq: float, duration: float, mp4: str | None) -> None:
    """Run the gait policy in the MuJoCo torque-deploy loop (needs a gait checkpoint)."""
    from strands_robots import Robot
    from strands_robots.policies.wbc import WBCGaitPolicy

    policy = WBCGaitPolicy(checkpoint=checkpoint, target_velocity=[vx, 0.0, 0.0], gait_frequency=freq)
    sim = Robot("unitree_g1")
    video = {"path": mp4, "fps": 30, "camera": "track", "width": 640, "height": 480} if mp4 else None
    sim.run_policy(
        robot_name="unitree_g1",
        policy_object=policy,
        policy_kwargs={"target_velocity": [vx, 0.0, 0.0], "gait_frequency": freq},
        duration=duration,
        control_frequency=50.0,
        video=video,
    )
    print("gait rollout complete")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-clock", action="store_true", help="save a gait-clock signal PNG (no checkpoint needed)")
    ap.add_argument("--out", default="/tmp/g1_gait_clock.png", help="PNG path for --plot-clock")
    ap.add_argument("--checkpoint", default=None, help="gait-clock ONNX checkpoint dir (95x6 input)")
    ap.add_argument("--vx", type=float, default=0.5, help="forward velocity command (m/s)")
    ap.add_argument("--freq", type=float, default=1.0, help="step-frequency command freq_cmd")
    ap.add_argument("--duration", type=float, default=5.0, help="rollout duration (s)")
    ap.add_argument("--mp4", default=None, help="optional MP4 output path for the rollout")
    args = ap.parse_args()

    if args.plot_clock:
        plot_clock(args.out, freq=args.freq)
    elif args.checkpoint:
        run_policy(args.checkpoint, args.vx, args.freq, args.duration, args.mp4)
    else:
        ap.error("pass --plot-clock (no weights) or --checkpoint <dir> (gait ONNX)")


if __name__ == "__main__":
    main()
