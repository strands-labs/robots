#!/usr/bin/env python3
"""Render an Isaac RTX robot composited into a 3DGS / panorama background.

Entry point for the Isaac-Sim + 3D Gaussian Splatting hybrid-render
example -- the digital-twin companion to ``examples/mujoco_gs``.

Builds the default scene (real Franka + red cube + RTX camera),
renders one or more depth-composited frames (Isaac RTX foreground
z-composited over a captured-real 3DGS scene by default; the
procedural panorama only via ``--panorama`` or ``--allow-fallback``),
writes them to disk as PNG stills, and — for a multi-frame run —
assembles them into a video clip (MP4/GIF).

This ships a **render-stills / short-clip** entry point rather than a
live Gradio view (like ``mujoco_gs/app.py``): Isaac's RTX renderer
isn't real-time-cheap the way MuJoCo's offscreen renderer is, and the
SimulationApp boot is heavyweight (~200 s), so a render-and-save shape
is the honest fit. A live-view / agent-driven variant can layer on
once the per-frame RTX cost is budgeted.

Output
------
By default each composited frame is written as a PNG still. When more
than one frame is rendered (``--frames > 1`` or ``--wave``) the frames
are *also* assembled into a video clip, so the example delivers the
"short-clip" its name promises — at output parity with
``examples/mujoco_gs/libero_groot.py`` (``imageio`` libx264, matching
``mimsave(..., codec="libx264", quality=7, macro_block_size=8)``).
``--mp4`` / ``--gif`` pick the clip container (default: MP4 for a
multi-frame run); ``--out-video`` overrides the path; ``--fps`` sets the
frame rate; ``--no-stills`` skips the per-frame PNGs.

Usage
-----
::

    # Default: the real captured 3DGS tabletop scene (auto-downloaded).
    # Fails loud with an install hint if gsplat can't rasterize:
    python -m examples.isaac_gs.render_demo --frames 1 --out rollouts/isaac_gs

    # Zero-ML-deps run: demote to the procedural panorama when GS is
    # unavailable (the pre-#2321 behavior, now opt-in):
    python -m examples.isaac_gs.render_demo --frames 1 --allow-fallback

    # Real captured 3DGS background from your own capture (requires gsplat + a .ply):
    python -m examples.isaac_gs.render_demo --gsplat-ply /path/to/kitchen.ply

    # Sweep a joint across frames -> PNG stills + an assembled MP4 clip:
    python -m examples.isaac_gs.render_demo --frames 24 --wave

    # Same, but write a GIF (and skip the per-frame PNGs):
    python -m examples.isaac_gs.render_demo --frames 24 --wave --gif --no-stills

Requires
--------
``pip install "strands-robots[sim-isaac]"`` + a working Isaac Sim
install (RTX GPU). The default (real-3DGS) path needs a **pre-built**
``gsplat`` wheel that can CUDA-rasterize (see the README -- a plain
``pip install gsplat`` imports fine but can't rasterize where nvcc is
absent). The procedural panorama path (``--panorama`` /
``--allow-fallback``) needs neither.

Depends at runtime on PR #61 (add_camera) + PR #62 (render frame-path);
see the package docstring.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--frames", type=int, default=1, help="Number of composited frames to render.")
    p.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: rollouts/<date>/isaac_gs.",
    )
    p.add_argument(
        "--gsplat-ply",
        default=None,
        help="Path to a .ply / .spz 3DGS capture for the background. "
        "Requires `pip install gsplat`. Overrides the default preset scene.",
    )
    p.add_argument(
        "--gsplat-scene",
        default=None,
        help="Named built-in 3DGS preset for the background (e.g. "
        "'tabletop (indoor room)'). Default: the tabletop preset, "
        "auto-downloaded + skybox-aligned, when gsplat is installed.",
    )
    p.add_argument(
        "--robot-usd",
        default=None,
        help="Override the robot asset USD. Default: bundled Franka Panda.",
    )
    p.add_argument(
        "--panorama",
        default=None,
        help="Path to an equirectangular panorama image for the background "
        "(used by PanoramaBackground when no --gsplat-ply is given).",
    )
    p.add_argument(
        "--wave",
        action="store_true",
        help="Sweep the arm's first joint across frames so the composite "
        "shows the robot moving on the backdrop (needs --frames > 1).",
    )
    p.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Demote to the procedural panorama when the 3DGS background can't "
        "initialize (gsplat missing / CUDA rasterizer disabled / scene load "
        "failure) instead of failing. Off by default so the photoreal path "
        "never silently degrades (issue #2321).",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--mp4",
        dest="mp4",
        action="store_true",
        default=None,
        help="Assemble the rendered frames into an MP4 clip (libx264). "
        "This is the default for a multi-frame run (--frames > 1 / --wave).",
    )
    p.add_argument(
        "--gif",
        dest="gif",
        action="store_true",
        default=None,
        help="Assemble the rendered frames into an animated GIF instead of an MP4.",
    )
    p.add_argument(
        "--out-video",
        default=None,
        help="Output path for the assembled clip. Default: " "<out>/<ts>--isaac_gs.{mp4,gif}. Implies clip assembly.",
    )
    p.add_argument("--fps", type=int, default=20, help="Frame rate for the assembled clip (default: 20).")
    p.add_argument(
        "--no-stills",
        dest="stills",
        action="store_false",
        default=True,
        help="Skip writing the per-frame PNG stills (only emit the assembled clip).",
    )
    p.add_argument(
        "--no-ibl",
        dest="ibl",
        action="store_false",
        default=True,
        help="Keep the legacy hardcoded key/dome lights instead of deriving "
        "the lighting from the background scene (an environment map baked "
        "from the 3DGS scene / the panorama image textures the dome light "
        "and aims the key light).",
    )
    p.add_argument(
        "--no-shadow-catcher",
        dest="shadow_catcher",
        action="store_false",
        default=True,
        help="Skip the matte shadow-catcher plane (the robot then casts no "
        "contact shadow onto the backdrop).",
    )
    return p


def _want_video(args: argparse.Namespace) -> bool:
    """Decide whether to assemble a clip from the rendered frames.

    A clip is produced when (a) the user asked for one explicitly
    (``--mp4`` / ``--gif`` / ``--out-video``), or (b) the run is
    multi-frame (``--frames > 1`` or ``--wave``) — mirroring
    ``mujoco_gs/libero_groot.py``'s always-a-video output. A single
    still has nothing to animate, so a bare ``--frames 1`` stays
    PNG-only unless a clip flag is passed.
    """
    if args.mp4 or args.gif or args.out_video:
        return True
    return bool(args.wave or args.frames > 1)


def _video_path(args: argparse.Namespace, out_dir: str, ts: str) -> str:
    if args.out_video:
        d = os.path.dirname(args.out_video)
        if d:
            os.makedirs(d, exist_ok=True)
        return args.out_video
    ext = "gif" if args.gif else "mp4"
    return os.path.join(out_dir, f"{ts}--isaac_gs.{ext}")


def _encode_clip(frames, path: str, fps: int = 20) -> None:
    """Assemble RGB frames into a clip at ``path`` (.mp4 or .gif).

    Delegates to the shared library encoder
    (``strands_robots.rendering.encode_clip``, issue #1537) with this demo's
    historical knobs (libx264 ``quality=7``, ``macro_block_size=8``; GIF
    output takes per-frame duration instead).
    """
    from strands_robots.rendering import encode_clip

    encode_clip(frames, path, fps=fps, quality=7, macro_block_size=8)


def _date_out(out: "str | None") -> str:
    if out:
        os.makedirs(out, exist_ok=True)
        return out
    d = os.path.join("rollouts", _dt.date.today().strftime("%Y_%m_%d"), "isaac_gs")
    os.makedirs(d, exist_ok=True)
    return d


def _make_background(args: argparse.Namespace):
    """Construct the background renderer from CLI args.

    Defaults to the real 3DGS ``tabletop`` scene and **fails loud** (with the
    pre-built ``gsplat`` wheel install hint) when it can't initialize;
    ``--allow-fallback`` opts into the procedural-panorama demotion -- see
    ``examples.isaac_gs.background.resolve_background``.
    """
    from examples.isaac_gs.background import resolve_background

    return resolve_background(
        gsplat_ply=args.gsplat_ply,
        gsplat_scene=args.gsplat_scene,
        panorama=args.panorama,
        allow_fallback=args.allow_fallback,
    )


def _save_png(path: str, rgb) -> None:
    """Write an (H, W, 3) uint8 array to PNG without a hard PIL dep at import."""
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except ImportError:
        # Fallback: numpy .npy so the demo still produces output without PIL.
        import numpy as np

        np.save(path.replace(".png", ".npy"), rgb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()

    from strands_robots.simulation import create_simulation

    from strands_robots.simulation.isaac import IsaacSimulation

    # Fail-fast on non-Isaac hosts (cheap probe, no omni import).
    available, reason = IsaacSimulation.is_available()
    if not available:
        raise RuntimeError(
            f"Isaac Sim is not available on this host: {reason}. "
            "Install Isaac Sim (RTX GPU) and the strands-robots[sim-isaac] extra."
        )

    from examples.isaac_gs.compositor import IsaacHybridCompositor
    from examples.isaac_gs.scene import build_default_scene

    out_dir = _date_out(args.out)
    # rtx_realtime so render() takes the RTX frame path (not headless blanks).
    sim = create_simulation("isaac", headless=True, num_envs=1, render_mode="rtx_realtime")
    try:
        # The background is resolved before the scene so its baked environment
        # map can light the robot (issue #2323): the dome light is textured
        # with the room the robot stands in, and the key light aims the way
        # the room's dominant light falls.
        background = _make_background(args)
        env_map_path = None
        if args.ibl:
            from examples.isaac_gs.background import resolve_ibl_env_map

            env_map_path = resolve_ibl_env_map(
                background,
                gsplat_ply=args.gsplat_ply,
                gsplat_scene=args.gsplat_scene,
                panorama=args.panorama,
            )

        build = build_default_scene(
            sim,
            robot_usd=args.robot_usd,
            camera_name="front",
            camera_width=args.width,
            camera_height=args.height,
            env_map_path=env_map_path,
            shadow_catcher=args.shadow_catcher,
        )
        print(f"[scene] robot={build.robot_name} joints={build.robot_joint_count} objects={build.object_names}")

        compositor = IsaacHybridCompositor(
            sim,
            background=background,
            # The scene reports the catcher plane's height; the compositor
            # turns that plane's shading into a shadow on the backdrop
            # (None when --no-shadow-catcher / the plane failed to add).
            shadow_plane_z=build.shadow_plane_z,
        )

        want_video = _want_video(args)
        rendered = []
        ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        for i in range(max(1, args.frames)):
            if args.wave and build.robot_joint_count > 0:
                # Sweep the first joint to show motion on the backdrop.
                import math

                jn = sim.robot_joint_names(build.robot_name)
                if jn:
                    angle = 0.6 * math.sin(2.0 * math.pi * i / max(1, args.frames))
                    sim.send_action({jn[0]: angle}, robot_name=build.robot_name)
                    sim.step(5)

            frame = compositor.render(camera_name=build.camera_name)
            fg_px = int(frame.foreground_mask.sum())
            if want_video:
                rendered.append(frame.rgb)
            if args.stills:
                path = os.path.join(out_dir, f"{ts}--isaac_gs--frame{i:03d}.png")
                _save_png(path, frame.rgb)
                print(f"[frame {i}] saved {path}  foreground_px={fg_px}  ({frame.rgb.shape[1]}x{frame.rgb.shape[0]})")
            else:
                print(f"[frame {i}] rendered  foreground_px={fg_px}  ({frame.rgb.shape[1]}x{frame.rgb.shape[0]})")

        video_path = None
        if want_video and rendered:
            video_path = _video_path(args, out_dir, ts)
            _encode_clip(rendered, video_path, fps=args.fps)
            print(f"[video] assembled {video_path}  frames={len(rendered)}  fps={args.fps}")

        # Grep-stable summary line.
        print(
            f"isaac_gs  frames={args.frames}  robot={build.robot_name}  "
            f"out={out_dir}  video={video_path}  backend=isaac"
        )
    finally:
        sim.destroy()


if __name__ == "__main__":
    # Isaac's SimulationApp installs a fast-shutdown path (``simulation_app
    # .close()`` / ``os._exit``-style teardown) that can swallow a non-zero
    # process exit even when ``main`` raised -- so a failed
    # ``build_default_scene`` would otherwise exit 0 and hide the failure
    # from CI / scripts (see strands-labs/robots-sim#110). Catch any
    # exception here, log it, and force a non-zero exit *after* the
    # SimulationApp teardown via ``os._exit`` so the status survives.
    try:
        main()
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("render_demo failed")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
