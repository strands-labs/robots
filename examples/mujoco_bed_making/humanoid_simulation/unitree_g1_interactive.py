#!/usr/bin/env python3
"""Interactive Unitree G1 MuJoCo demo driven by natural-language instructions.

Uses the normal Strands policy abstraction layer. By default it runs the local
`g1_demo` policy provider, which turns simple natural-language instructions into
pose sequences for the simulated G1. You can swap `--policy` to another provider
once you have one configured.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("STRANDS_ASSETS_DIR", str(REPO_ROOT / ".strands_robots" / "assets"))


def _maybe_reexec_with_mjpython() -> None:
    """MuJoCo passive viewer on macOS must run under mjpython."""
    if sys.platform != "darwin":
        return
    if "--no-viewer" in sys.argv:
        return
    if os.path.basename(sys.executable) == "mjpython":
        return
    if os.environ.get("STRANDS_MJPYTHON_REEXEC") == "1":
        return

    mjpython = shutil.which("mjpython")
    if not mjpython:
        return

    env = dict(os.environ)
    env["STRANDS_MJPYTHON_REEXEC"] = "1"
    os.execve(mjpython, [mjpython, *sys.argv], env)


_maybe_reexec_with_mjpython()

from strands_robots import Robot
from strands_robots.tools.download_assets import download_robots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="g1_demo", help="Policy provider name.")
    parser.add_argument("--duration", type=float, default=2.5, help="Default seconds per instruction.")
    parser.add_argument("--fast-mode", action="store_true", help="Run without wall-clock sleeps.")
    parser.add_argument("--no-viewer", action="store_true", help="Skip opening the MuJoCo viewer.")
    return parser.parse_args()


def _ensure_assets() -> None:
    result = download_robots(names=["unitree_g1"])
    failures = result.get("failed_details", {})
    if failures:
        raise RuntimeError(f"Failed to prepare assets: {failures}")


def _build_scene(sim) -> None:
    sim.add_object(
        name="demo_table",
        shape="box",
        position=[0.55, 0.0, 0.34],
        size=[0.35, 0.55, 0.34],
        color=[0.45, 0.37, 0.29, 1.0],
        is_static=True,
    )
    sim.add_object(
        name="demo_cube",
        shape="box",
        position=[0.45, -0.10, 0.73],
        size=[0.03, 0.03, 0.03],
        color=[0.88, 0.18, 0.12, 1.0],
        mass=0.08,
    )


def _print_help() -> None:
    print("Commands:")
    print("  pick up the cube")
    print("  reach to the cube")
    print("  wave with the right arm")
    print("  turn left")
    print("  reset to neutral")
    print("  /state   show sim state")
    print("  /reset   reset sim")
    print("  /help    show commands")
    print("  /quit    exit")


def main() -> int:
    args = _parse_args()
    _ensure_assets()

    sim = Robot("unitree_g1", mode="sim", backend="mujoco")
    try:
        _build_scene(sim)

        if not args.no_viewer:
            viewer_result = sim.open_viewer()
            text = viewer_result["content"][0]["text"]
            print(text)

        print(f"Using policy provider: {args.policy}")
        _print_help()

        while True:
            try:
                instruction = input("\ng1> ").strip()
            except EOFError:
                print()
                break

            if not instruction:
                continue
            if instruction in ("/quit", "quit", "exit"):
                break
            if instruction == "/help":
                _print_help()
                continue
            if instruction == "/state":
                print(sim.get_state()["content"][0]["text"])
                continue
            if instruction == "/reset":
                print(sim.reset()["content"][0]["text"])
                continue

            result = sim.run_policy(
                robot_name="unitree_g1",
                policy_provider=args.policy,
                instruction=instruction,
                duration=args.duration,
                fast_mode=args.fast_mode,
            )
            print(result["content"][0]["text"])

    finally:
        try:
            sim.close_viewer()
        except Exception:
            pass
        sim.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
