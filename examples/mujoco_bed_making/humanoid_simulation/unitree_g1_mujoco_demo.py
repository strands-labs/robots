#!/usr/bin/env python3
"""Unitree G1 MuJoCo visualization demo for Strands Robots.

Creates a simple manipulation-style scene around the MuJoCo-backed G1 model:
table, object, deterministic upper-body pose sequence, and rendered keyframes.

Usage:
    python3 examples/mujoco_bed_making/humanoid_simulation/unitree_g1_mujoco_demo.py
    python3 examples/mujoco_bed_making/humanoid_simulation/unitree_g1_mujoco_demo.py --output-dir artifacts/g1_demo --width 1280 --height 720
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("STRANDS_ASSETS_DIR", str(REPO_ROOT / ".strands_robots" / "assets"))

from strands_robots import Robot
from strands_robots.tools.download_assets import download_robots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640, help="Render width.")
    parser.add_argument("--height", type=int, default=480, help="Render height.")
    parser.add_argument("--render-every", type=int, default=12, help="Save every Nth control frame.")
    parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per control update.")
    parser.add_argument(
        "--strict-render",
        action="store_true",
        help="Fail immediately if a frame cannot be rendered.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "unitree_g1_demo"),
        help="Directory for exported keyframes.",
    )
    return parser.parse_args()


def _ensure_assets() -> None:
    result = download_robots(names=["unitree_g1"])
    failures = result.get("failed_details", {})
    if failures:
        raise RuntimeError(f"Failed to prepare assets: {failures}")


def _extract_png_bytes(render_result: Dict) -> bytes:
    for item in render_result.get("content", []):
        image = item.get("image")
        if image and image.get("format") == "png":
            return image["source"]["bytes"]
    raise RuntimeError(f"Render result did not include a PNG image: {render_result}")


def _save_frame(sim, frame_path: Path, width: int, height: int) -> None:
    render_result = sim.render(width=width, height=height)
    if render_result.get("status") != "success":
        raise RuntimeError(f"Render failed: {render_result}")
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(_extract_png_bytes(render_result))


def _side_factor(name: str) -> int:
    lower = name.lower()
    if any(token in lower for token in ("left", "l_", "_l", "l_")):
        return 1
    if any(token in lower for token in ("right", "r_", "_r")):
        return -1
    return 0


def _pose_for_stage(joint_names: Iterable[str], stage: str) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    for joint_name in joint_names:
        lower = joint_name.lower()
        side = _side_factor(lower)

        if "waist" in lower or "torso" in lower:
            if stage in ("reach", "grasp", "lift"):
                targets[joint_name] = 0.08
            elif stage == "retract":
                targets[joint_name] = 0.03
            continue

        if "shoulder" in lower:
            if "pitch" in lower:
                targets[joint_name] = {"reach": -0.45, "grasp": -0.50, "lift": -0.25, "retract": -0.12}.get(stage, 0.0)
            elif "roll" in lower:
                targets[joint_name] = side * {"reach": 0.22, "grasp": 0.25, "lift": 0.12, "retract": 0.08}.get(stage, 0.0)
            elif "yaw" in lower:
                targets[joint_name] = side * {"reach": 0.18, "grasp": 0.22, "lift": 0.10, "retract": 0.06}.get(stage, 0.0)
            continue

        if "elbow" in lower:
            targets[joint_name] = {"reach": 0.75, "grasp": 0.95, "lift": 0.65, "retract": 0.35}.get(stage, 0.0)
            continue

        if "wrist" in lower:
            if "pitch" in lower:
                targets[joint_name] = {"reach": -0.20, "grasp": -0.30, "lift": -0.10, "retract": -0.05}.get(stage, 0.0)
            elif "yaw" in lower:
                targets[joint_name] = side * {"reach": 0.10, "grasp": 0.12, "lift": 0.05, "retract": 0.03}.get(stage, 0.0)
            elif "roll" in lower:
                targets[joint_name] = side * {"reach": 0.08, "grasp": 0.10, "lift": 0.04, "retract": 0.02}.get(stage, 0.0)
            continue

        if any(token in lower for token in ("hand", "thumb", "index", "middle", "ring", "pinky", "finger")):
            targets[joint_name] = 0.45 if stage in ("grasp", "lift") else 0.0

    return targets


def _blend_pose(start: Dict[str, float], target: Dict[str, float], alpha: float) -> Dict[str, float]:
    blended: Dict[str, float] = {}
    keys = set(start) | set(target)
    for key in keys:
        start_val = start.get(key, 0.0)
        target_val = target.get(key, 0.0)
        blended[key] = start_val + (target_val - start_val) * alpha
    return blended


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


def _run_sequence(sim, joint_names: List[str], args: argparse.Namespace, output_dir: Path) -> str | None:
    stages: List[Tuple[str, int]] = [
        ("reach", 72),
        ("grasp", 48),
        ("lift", 72),
        ("retract", 72),
    ]

    frame_index = 0
    current_pose: Dict[str, float] = {}
    render_error = None
    try:
        _save_frame(sim, output_dir / f"frame_{frame_index:04d}_start.png", args.width, args.height)
    except Exception as exc:
        render_error = str(exc)
        if args.strict_render:
            raise

    for stage_name, steps in stages:
        target_pose = _pose_for_stage(joint_names, stage_name)
        for step in range(steps):
            alpha = (step + 1) / steps
            command = _blend_pose(current_pose, target_pose, alpha)
            sim.send_action(command, robot_name="unitree_g1", n_substeps=args.substeps)
            if render_error is None and (step % args.render_every == 0 or step == steps - 1):
                frame_index += 1
                try:
                    _save_frame(
                        sim,
                        output_dir / f"frame_{frame_index:04d}_{stage_name}.png",
                        args.width,
                        args.height,
                    )
                except Exception as exc:
                    render_error = str(exc)
                    if args.strict_render:
                        raise
        current_pose = target_pose

    return render_error


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_assets()
    sim = Robot("unitree_g1", mode="sim", backend="mujoco")

    try:
        _build_scene(sim)
        features = sim.get_features()
        joint_names = features["content"][1]["json"]["features"]["robots"]["unitree_g1"]["joint_names"]
        render_error = _run_sequence(sim, joint_names, args, output_dir)
        state = sim.get_state()
        (output_dir / "summary.txt").write_text(
            "\n".join(
                [
                    "Unitree G1 MuJoCo demo complete.",
                    f"Asset cache: {os.environ['STRANDS_ASSETS_DIR']}",
                    f"Output dir: {output_dir}",
                    f"Render status: {'disabled after failure' if render_error else 'frames exported'}",
                    f"Render detail: {render_error or 'ok'}",
                    state["content"][0]["text"],
                ]
            )
        )
        if render_error:
            print(f"Simulation ran, but rendering failed in this environment: {render_error}")
            print(f"Wrote run summary to {output_dir / 'summary.txt'}")
        else:
            print(f"Saved demo frames to {output_dir}")
        return 0
    finally:
        sim.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
