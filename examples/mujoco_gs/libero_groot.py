# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a real NVIDIA GR00T policy on a LIBERO task and record the result.

This is the "real policy" companion to the scripted-wave MuJoCo-GS demo. Where
``agent.py`` drives a small SO-101/SO-100 arm with a hand-scripted trajectory,
this script hands control to a **real GR00T vision-language-action policy**
served over ZMQ, driving a Franka Panda through a **LIBERO** manipulation task,
and records the run as an MP4 through the same :class:`HybridCompositor`.

Why a Panda + LIBERO (not the SO-101 wave scene)?
    GR00T policies are *embodiment-locked*: a checkpoint is trained for a
    specific robot + camera/state layout. The checkpoint used here
    (``LIBERO_PANDA`` / ``data_config="libero_panda"``) expects LIBERO's
    ``image`` + ``wrist_image`` cameras and a 7-DoF end-effector action space,
    so it only makes sense to run it on a LIBERO Panda task. Feeding it the
    SO-101 wave scene would produce garbage actions.

Prerequisites:
    * A GR00T inference server reachable over **ZMQ** (NVIDIA's
      ``gr00t.eval.run_gr00t_server``). Point ``--port`` at it (default 8000).
      Note: strands-robots' GR00T client is ZMQ-only; an HTTP-only server
      will not work.
    * ``libero`` + ``robosuite`` installed (``pip install`` them; they ship the
      BDDL task definitions and RoboSuite scenes).
    * ``strands-robots[sim-mujoco]`` and this example's deps
      (``imageio``/``imageio-ffmpeg`` for MP4 writing).

Note on the hybrid background:
    LIBERO scenes are fully enclosed (table, walls, floor, props), so a
    panorama / 3DGS backdrop has no "sky" to show through and is effectively
    hidden — the composite is dominated by the LIBERO scene itself. The
    compositor is still used (so the same code path renders both demos), but
    the GS backdrop only adds visible value for open scenes like the SO-101
    cube demo. Set ``--no-composite`` to skip it and render the raw scene.

Example:
    python -m examples.mujoco_gs.libero_groot \\
        --suite libero_10 --task 0 --port 8000 --max-steps 200 \\
        --out /tmp/libero_groot.mp4

    # Sanity-check the pipeline without a policy server:
    python -m examples.mujoco_gs.libero_groot --provider mock --max-steps 60
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Make the example importable both as a module and as a script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent.parent))

from examples.mujoco_gs.backgrounds import PanoramaBackground  # noqa: E402
from examples.mujoco_gs.compositor import HybridCompositor  # noqa: E402

logger = logging.getLogger("mujoco_gs.libero_groot")


def run(
    suite: str = "libero_10",
    task_index: int = 0,
    provider: str = "groot",
    host: str = "127.0.0.1",
    port: int = 8000,
    data_config: str = "libero_panda",
    groot_version: str = "n1.7",
    max_steps: int | None = None,
    action_horizon: int | None = None,
    seed: int = 42,
    camera: str = "image",
    width: int = 512,
    height: int = 384,
    composite: bool = True,
    out: str | None = None,
    fps: int = 20,
) -> dict:
    """Run one LIBERO episode under ``provider`` and record an MP4.

    Returns a small summary dict ``{task, steps, success, video, ...}``.
    """
    try:
        from strands_robots.benchmarks.libero import load_libero_suite
        from strands_robots.simulation.mujoco.simulation import Simulation
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "strands-robots[sim-mujoco] + libero + robosuite are required. See this file's module docstring."
        ) from e

    # 1. Register the LIBERO tasks (auto-discovers BDDL + RoboSuite scenes).
    # Leave max_steps at the adapter default unless overridden — LIBERO-Long
    # episodes need ~500 steps; capping them truncates before the policy can
    # finish and tanks the success rate.
    logger.info("Loading LIBERO suite %r (max_steps=%s)…", suite, max_steps or "default")
    suite_kwargs = {"max_steps": max_steps} if max_steps else {}
    adapters = load_libero_suite(suite, **suite_kwargs)
    names = list(adapters.keys())
    if not names:
        raise RuntimeError(
            f"No LIBERO tasks loaded from suite {suite!r} (some BDDL predicates "
            "may be unsupported). Try a different suite."
        )
    task_index = max(0, min(task_index, len(names) - 1))
    task = names[task_index]
    instruction = adapters[task].problem.language
    logger.info("Task %d/%d: %s", task_index, len(names), task)
    logger.info("Instruction: %s", instruction)

    # 2. Build the sim and pre-add a Panda (the benchmark compatibility check
    #    runs before the adapter loads the LIBERO scene, so a Panda-typed robot
    #    named "robot" must exist up front; the adapter then re-wraps the
    #    scene's own Panda under the same name).
    sim = Simulation(tool_name="libero_groot", mesh=False)
    sim.create_world()
    add = sim.add_robot(name="robot", data_config="panda")
    if isinstance(add, dict) and add.get("status") != "success":
        raise RuntimeError(f"Failed to add Panda robot: {add}")

    compositor = HybridCompositor(sim, background=PanoramaBackground()) if composite else None

    frames: list[np.ndarray] = []
    render_errors: list[str] = []

    def on_frame(step: int, obs: dict, action: dict) -> None:
        # The LIBERO scene is loaded on episode start; drop any stale renderer
        # / camera caches on the first frame so we render the new model.
        try:
            if step == 0 and compositor is not None:
                compositor.clear_caches()
            if compositor is not None:
                frames.append(compositor.render(camera_name=camera, width=width, height=height).rgb)
            else:
                # Raw scene render via the sim's own renderer.
                from examples.mujoco_gs.camera_utils import render_rgb_and_depth

                rgb, _ = render_rgb_and_depth(sim, camera, width, height)
                frames.append(rgb)
        except Exception as e:  # pragma: no cover
            if len(render_errors) < 5:
                render_errors.append(f"step {step}: {type(e).__name__}: {e}")

    # 3. Build the policy config and run one episode.
    policy_config = None
    if provider == "groot":
        policy_config = {
            "host": host,
            "port": port,
            "data_config": data_config,
            "groot_version": groot_version,
        }
        logger.info("Using GR00T policy at zmq://%s:%d (data_config=%s, %s)", host, port, data_config, groot_version)

    # Pre-warm the LIBERO scene (generate BDDL scene → load → prewarm) so the
    # image/wrist_image cameras + Panda exist before recording/inference, and
    # drop stale compositor caches for the freshly loaded model.
    if provider == "groot":
        from strands_robots.simulation.benchmark import get_benchmark

        spec = get_benchmark(task)
        if spec.scene_path is None and getattr(spec, "_auto_generate_scene", False):
            generated = spec._generate_scene_from_bddl()
            if generated:
                spec.scene_path = generated
        if spec.scene_path:
            sim.load_scene(spec.scene_path)
            if hasattr(spec, "prewarm"):
                spec.prewarm(sim)
            if "robot" not in sim.list_robots():
                sim.add_robot(name="robot", data_config="panda")
        if compositor is not None:
            compositor.clear_caches()

    logger.info("Running 1 episode under provider=%r…", provider)
    eval_kwargs = dict(
        benchmark_name=task,
        policy_provider=provider,
        policy_config=policy_config,
        n_episodes=1,
        seed=seed,
        instruction=instruction,
        on_frame=on_frame,
    )
    # Omit robot_name (the LIBERO scene renames its Panda to "robot" on
    # episode start; evaluate_benchmark auto-picks the single robot). Only
    # override action_horizon if explicitly requested.
    if action_horizon:
        eval_kwargs["action_horizon"] = action_horizon
    result = sim.evaluate_benchmark(**eval_kwargs)

    status = result.get("status") if isinstance(result, dict) else "unknown"
    if status != "success":
        msg = result["content"][0].get("text", "") if isinstance(result, dict) else str(result)
        raise RuntimeError(f"evaluate_benchmark failed: {msg[:400]}")

    payload = next((c["json"] for c in result["content"] if "json" in c), {})
    ep = (payload.get("episodes") or [{}])[0]
    if render_errors:
        logger.warning("render errors (first few): %s", render_errors)

    # 4. Encode the captured frames to MP4.
    video_path = None
    if frames:
        out_path = out or os.path.join(_THIS_DIR.parent.parent, f"libero_{provider}.mp4")
        _encode_mp4(frames, out_path, fps=fps)
        video_path = out_path
        logger.info("Saved %d frames → %s (%.0f KB)", len(frames), out_path, os.path.getsize(out_path) / 1024)

    # 5. Cleanup (closes the render thread / GL contexts).
    if compositor is not None:
        compositor.close()
    try:
        sim.cleanup()
    except Exception:  # pragma: no cover
        pass

    summary = {
        "task": task,
        "instruction": instruction,
        "provider": provider,
        "steps": ep.get("steps"),
        "success": ep.get("success"),
        "video": video_path,
        "n_frames": len(frames),
    }
    logger.info("Done: %s", summary)
    return summary


def _encode_mp4(frames: list[np.ndarray], path: str, fps: int = 20) -> None:
    from strands_robots.rendering import encode_clip  # shared library encoder (issue #1537)

    encode_clip(frames, path, fps=fps, quality=7, macro_block_size=8)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--suite", default="libero_10", help="LIBERO suite (libero_10, libero_spatial, libero_object, …).")
    p.add_argument("--task", type=int, default=0, help="Task index within the suite.")
    p.add_argument("--provider", default="groot", choices=["groot", "mock"], help="Policy provider.")
    p.add_argument("--host", default="127.0.0.1", help="GR00T ZMQ host.")
    p.add_argument("--port", type=int, default=8000, help="GR00T ZMQ port.")
    p.add_argument("--data-config", default="libero_panda", help="GR00T data_config / embodiment.")
    p.add_argument("--groot-version", default="n1.7", help="GR00T wire-format version (n1.5/n1.6/n1.7).")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps per episode (default: adapter default; don't cap LIBERO-Long).",
    )
    p.add_argument(
        "--action-horizon", type=int, default=None, help="Actions applied per inference (default: eval default)."
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--camera", default="image", help="LIBERO camera to render (image=agentview, wrist_image).")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--no-composite", action="store_true", help="Render the raw scene without the GS/panorama backdrop.")
    p.add_argument("--out", default=None, help="Output MP4 path.")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.environ.setdefault("MUJOCO_GL", "egl")

    run(
        suite=args.suite,
        task_index=args.task,
        provider=args.provider,
        host=args.host,
        port=args.port,
        data_config=args.data_config,
        groot_version=args.groot_version,
        max_steps=args.max_steps,
        action_horizon=args.action_horizon,
        seed=args.seed,
        camera=args.camera,
        width=args.width,
        height=args.height,
        composite=not args.no_composite,
        out=args.out,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
