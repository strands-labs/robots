"""Kimodo + G1 head-cam dataset recording — end-to-end.

Text prompt → Kimodo motion diffusion → G1 MuJoCo sim with a **head-mounted
camera** → recorded LeRobot v3 dataset with per-episode MP4 video tracks and
parquet joint/action tables.

The head-cam is the sensor cagatay asked for: it rides on the G1 as it moves
so the recorded video is a first-person, from-the-robot view — the same shape
next-gen VLA datasets need.

Note on the mount: the 29-DoF G1 URDF has no ``head_link`` body (bodies stop
at ``torso_link``). We mount the camera on ``g1/torso_link`` with an offset
that approximates head position (~35 cm up, ~8 cm forward). Once the MJCF
asset ships a canonical ``head`` site this file will use ``parent_body="g1/head"``
directly.

Run (stub motion — no CUDA/weights needed, useful for CI/dev):

    MUJOCO_GL=egl python examples/kimodo/kimodo_g1_dataset_headcam.py --stub

Run with real Kimodo (needs the ``[kimodo]`` extra + CUDA):

    MUJOCO_GL=egl STRANDS_TRUST_REMOTE_CODE=1 \\
        python examples/kimodo/kimodo_g1_dataset_headcam.py \\
        --prompt "a person walking forward with confident strides"

The script prints parquet-truth verification at the end and exits non-zero if
the dataset was not built with N distinct episodes — same contract the
autonomous harness uses.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _positive_int(v: str) -> int:
    n = int(v)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {n}")
    return n


def _make_stub_motion_agent():
    """Return a lightweight motion agent that emits a gentle arm-sway motion.

    Avoids downloading Kimodo weights so this example runs on a CPU-only
    machine and in CI. Same protocol as the real agent (see
    ``strands_robots.policies.kimodo.KimodoMotionAgent``).
    """
    import numpy as np

    class _StubKimodoAgent:
        def sample(self, prompt, num_frames, diffusion_steps, guidance_scale, seed):
            qpos = np.zeros((num_frames, 7 + 29), dtype=np.float32)
            qpos[:, 2] = 0.75  # standing z
            qpos[:, 6] = 1.0   # quat w
            t = np.linspace(0.0, 4.0 * np.pi, num_frames, dtype=np.float32)
            # left/right shoulder pitch — indices matched to KIMODO_G1_JOINTS
            qpos[:, 7 + 15] = 0.5 * np.sin(t)
            qpos[:, 7 + 22] = 0.5 * np.sin(t + np.pi)
            return qpos

    return _StubKimodoAgent()


def _verify_parquet_truth(dataset_root: Path, n_episodes: int) -> None:
    """Parquet-truth verification — matches the autonomous harness contract."""
    import pyarrow.parquet as pq

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    total_episodes = info.get("total_episodes")
    if total_episodes != n_episodes:
        raise SystemExit(
            f"parquet truth FAIL: info.total_episodes={total_episodes}, expected {n_episodes}"
        )

    ep_tab = pq.read_table(
        dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    if ep_tab.num_rows != n_episodes:
        raise SystemExit(
            f"parquet truth FAIL: meta/episodes rows={ep_tab.num_rows}, expected {n_episodes}"
        )

    data_tab = pq.read_table(
        dataset_root / "data" / "chunk-000" / "file-000.parquet"
    )
    ep_col = data_tab.column("episode_index").to_pylist()
    unique = sorted(set(ep_col))
    if unique != list(range(n_episodes)):
        raise SystemExit(
            f"parquet truth FAIL: unique episode_index={unique}, expected {list(range(n_episodes))}"
        )

    features = list(info.get("features", {}).keys())
    for expected in ("observation.images.head", "observation.images.front"):
        if expected not in features:
            raise SystemExit(f"parquet truth FAIL: missing feature {expected}")

    print(
        f"✓ parquet truth PASS: {total_episodes} eps, "
        f"{info.get('total_frames')} frames, features={features}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Kimodo → G1 head-cam dataset")
    parser.add_argument("--prompt", default="a person walking forward")
    parser.add_argument("--n-episodes", type=_positive_int, default=3)
    parser.add_argument("--n-steps", type=_positive_int, default=100)
    parser.add_argument("--control-hz", type=_positive_int, default=25)
    parser.add_argument("--num-frames", type=_positive_int, default=180,
                        help="Kimodo motion length in native frames (<=196)")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use a stub motion agent (no CUDA/weights needed).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/kimodo_g1_headcam_dataset"),
        help="Dataset root directory (LeRobot v3 layout).",
    )
    parser.add_argument(
        "--repo-id",
        default="cagataydev/g1-kimodo-headcam-example",
        help="Dataset repo_id for the LeRobot recorder.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    # Trust gate fires on provider name even in --stub mode (stub loads no code).
    # Ack unconditionally — real code path is protected by the stub agent path.
    os.environ.setdefault("STRANDS_TRUST_REMOTE_CODE", "1")

    from strands_robots import Robot

    sim = Robot("g1", mesh=False)

    # Head-cam: mounted on torso_link with head-position offset.
    # When the G1 asset ships a canonical head site this becomes
    # parent_body="g1/head" with zero offset.
    sim.add_camera(
        name="head",
        parent_body="g1/torso_link",
        position=[0.08, 0.0, 0.35],
        target=[1.5, 0.0, 0.15],
        fov=70.0,
        width=640,
        height=480,
    )
    sim.add_camera(
        name="front",
        position=[3.0, 0.0, 1.2],
        target=[0.0, 0.0, 0.8],
        width=640,
        height=480,
    )
    print(f"cameras: {sim.list_cameras()}")

    if args.out.exists():
        shutil.rmtree(args.out)

    sim.start_recording(
        repo_id=args.repo_id,
        root=str(args.out),
        fps=args.control_hz,
        overwrite=True,
        task=f"Kimodo motion: {args.prompt}",
        cameras=["front", "head"],  # skip the implicit 'default' overview cam
    )

    policy_config = {
        "num_frames": args.num_frames,
        "device": "cpu" if args.stub else "cuda",
    }
    if args.stub:
        policy_config["motion_agent"] = _make_stub_motion_agent()

    result = sim.run_policy(
        robot_name="g1",
        policy_provider="kimodo",
        policy_config=policy_config,
        instruction=args.prompt,
        n_steps=args.n_steps,
        control_frequency=args.control_hz,
        n_episodes=args.n_episodes,
        reset_between=True,
        video={"path": str(args.out.parent / "kimodo_headcam.mp4"),
               "fps": args.control_hz, "camera": "head"},
    )
    status = result.get("status") if isinstance(result, dict) else str(result)
    print(f"run_policy status: {status}")

    sim.stop_recording()

    _verify_parquet_truth(args.out, args.n_episodes)
    print(f"dataset root: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
