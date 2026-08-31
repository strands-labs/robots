#!/usr/bin/env python3
"""Render a polished MuJoCo video of the Pollen Microduck under an ONNX policy.

Unlike :mod:`microduck_walk_sim` (which hands the rollout to the internal
``run_policy`` seam), this script steps the MuJoCo world **manually** at the
control frequency so it owns every frame:

    reset -> for each control tick:
        obs      = sim.get_observation()
        actions  = await policy.get_actions(obs, target_velocity=[vx, vy, vyaw])
        sim.send_action(actions[0])
        sim.step(substeps)
        renderer.update_scene(data, camera=tracking_cam); frame = renderer.render()

Frames are captured with a **body-tracking camera** locked to the duck's
pelvis (``microduck/trunk_base``) so the duck stays centered as it walks - this
reads far better than the fixed "default" cam. It talks to the underlying
``mujoco.Renderer`` + ``mujoco.MjvCamera`` directly (via ``sim.mj_model`` /
``sim.mj_data``); if that offscreen GL path is unreachable, it falls back to
``sim.render(camera_name=...)``.

Encodes an MP4 (h264 / yuv420p) with imageio + imageio-ffmpeg, and optionally a
looping GIF. Import-clean and headless.

Examples::

    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
    python examples/microduck/render_video.py \
        --onnx ../microduck/policies/alpha_walking.onnx \
        --vx 0.3 --duration 8 --out /tmp/microduck_viz/walk_forward.mp4

    python examples/microduck/render_video.py \
        --onnx ../microduck/policies/alpha_walking.onnx \
        --vx 0.1 --vyaw 0.6 --duration 6 --out /tmp/microduck_viz/turn.mp4

    python examples/microduck/render_video.py \
        --onnx ../microduck/policies/alpha_stand.onnx \
        --duration 5 --out /tmp/microduck_viz/stand.mp4

    # A skill trained in a variant scene names it with --scene. Without one the
    # duck rolls on a floor with no wheels under its feet, or swings at a ball
    # that is not there.
    python examples/microduck/render_video.py \
        --onnx ../microduck/policies/roller.onnx --scene scene_rollers.xml \
        --vx 0.3 --duration 8 --out /tmp/microduck_viz/roller.mp4

    python examples/microduck/render_video.py \
        --onnx ../microduck/policies/ball_kick_left.onnx --scene scene_ball.xml \
        --vx 0 --duration 4 --out /tmp/microduck_viz/kick.mp4

Dependencies::

    pip install "strands-robots[sim-mujoco,microduck]"   # rollout + video
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import numpy as np

BASE_BODY = "microduck/trunk_base"


def _load_mujoco():
    """Import mujoco, returning the module (or raise a clear error)."""
    try:
        import mujoco  # noqa: PLC0415

        return mujoco
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "mujoco is required. Install with: pip install 'strands-robots[sim-mujoco]'"
        ) from exc


def _make_tracking_camera(mujoco, model, body_name, distance, azimuth, elevation):
    """Build a MjvCamera that tracks ``body_name`` (falls back to free look)."""
    cam = mujoco.MjvCamera()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id >= 0:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = body_id
    else:  # pragma: no cover - defensive
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _encode(frames, out_path, fps, gif_path=None, gif_fps=13, gif_width=480):
    """Write frames to an MP4 (h264/yuv420p); optionally a looping GIF."""
    import imageio.v2 as imageio  # noqa: PLC0415

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    imageio.mimwrite(
        out_path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,  # allow arbitrary WxH (no forced /16)
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    print(f"  wrote mp4 {out_path} ({len(frames)} frames @ {fps} fps)")

    if gif_path:
        _encode_gif(frames, gif_path, gif_fps, gif_width)


def _encode_gif(frames, gif_path, gif_fps, gif_width):
    """Write a compact looping GIF, downscaled to ``gif_width`` px wide."""
    import imageio.v2 as imageio  # noqa: PLC0415

    os.makedirs(os.path.dirname(os.path.abspath(gif_path)) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    scale = gif_width / float(w)
    new_w, new_h = gif_width, max(1, int(round(h * scale)))
    # Nearest-neighbour subsample keeps it dependency-light (no PIL/cv2 needed).
    ys = (np.linspace(0, h - 1, new_h)).astype(int)
    xs = (np.linspace(0, w - 1, new_w)).astype(int)
    small = [f[ys][:, xs] for f in frames]
    imageio.mimwrite(gif_path, small, fps=gif_fps, loop=0)
    size_mb = os.path.getsize(gif_path) / 1e6
    print(f"  wrote gif {gif_path} ({new_w}x{new_h}, {size_mb:.2f} MB)")


def _resolve_scene(name: str) -> str:
    """Resolve a Microduck scene file by name through the asset search paths.

    A shipped weight and the scene it was trained in are one pair: ``roller`` and
    ``roller_crouch`` need the four passive ankle wheels only ``scene_rollers.xml``
    carries, and the ``ball_kick_*`` pair needs the prop only ``scene_ball.xml``
    places. All three scenes ship in the one asset directory the registry entry
    already downloads, so this resolves by name through
    :func:`~strands_robots.utils.get_search_paths` - the same route
    ``docs/policies/microduck.md`` documents - rather than taking a path the
    caller has to spell out.

    Args:
        name: A scene file name under the ``microduck`` asset directory, such as
            ``scene_rollers.xml``.

    Returns:
        The absolute path of the first match, searching the asset roots in the
        order :func:`get_search_paths` returns them.

    Raises:
        SystemExit: If no asset root carries ``microduck/<name>``, naming the
            roots that were searched. A misspelled scene is refused rather than
            silently rendering the entry's declared scene, which reports success
            and shows a duck standing still with no indication why.
    """
    from strands_robots.utils import get_search_paths  # noqa: PLC0415

    roots = get_search_paths()
    for root in roots:
        candidate = Path(root) / "microduck" / name
        if candidate.is_file():
            return str(candidate)
    searched = ", ".join(str(Path(root) / "microduck") for root in roots)
    raise SystemExit(f"scene {name!r} not found; searched {searched}")


def _sim_kwargs(args) -> dict[str, str]:
    """The ``Robot(...)`` keyword arguments the requested scene needs.

    Empty when no ``--scene`` was given, so the registry entry resolves its own
    declared scene exactly as before.
    """
    if not getattr(args, "scene", None):
        return {}
    return {"urdf_path": _resolve_scene(args.scene)}


async def _rollout(args):
    from strands_robots import Robot  # noqa: PLC0415
    from strands_robots.policies.microduck import MicroduckPolicy  # noqa: PLC0415

    mujoco = _load_mujoco()

    sim = Robot("microduck", mesh=False, **_sim_kwargs(args))
    sim.reset()
    model, data = sim.mj_model, sim.mj_data

    policy = MicroduckPolicy(onnx_path=os.path.abspath(args.onnx))

    dt = model.opt.timestep
    substeps = max(1, int(round((1.0 / args.control_frequency) / dt)))
    n_ticks = int(round(args.duration * args.control_frequency))
    tv = [args.vx, args.vy, args.vyaw]

    # --- renderer: direct mujoco.Renderer with a tracking camera (preferred) ---
    renderer = None
    cam = None
    direct = args.camera != "default"
    if direct:
        try:
            renderer = mujoco.Renderer(model, args.height, args.width)
            cam = _make_tracking_camera(
                mujoco, model, BASE_BODY, args.distance, args.azimuth, args.elevation
            )
            # smoke-render one frame to confirm a GL context exists
            renderer.update_scene(data, camera=cam)
            _ = renderer.render()
        except Exception as exc:  # pragma: no cover - GL context guard
            print(f"  direct mujoco.Renderer unavailable ({exc}); falling back to sim.render")
            renderer = None
            direct = False

    frames = []
    for _ in range(n_ticks):
        obs = sim.get_observation()
        actions = await policy.get_actions(obs, "", target_velocity=tv)
        sim.send_action(actions[0])
        sim.step(substeps)
        if direct and renderer is not None:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
        else:
            img = sim.render(width=args.width, height=args.height, camera_name="default")
            frames.append(np.asarray(img))

    if renderer is not None:
        renderer.close()

    if not frames:
        raise SystemExit("no frames rendered")

    # static-frame sanity check
    spread = float(np.mean(np.abs(frames[-1].astype(np.int16) - frames[0].astype(np.int16))))
    print(f"  rendered {len(frames)} frames; first/last mean abs diff = {spread:.2f}")

    _encode(
        frames,
        args.out,
        args.fps,
        gif_path=args.gif,
        gif_fps=args.gif_fps,
        gif_width=args.gif_width,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="../microduck/policies/alpha_walking.onnx")
    ap.add_argument(
        "--scene",
        default=None,
        help="scene file under the microduck asset dir, for a skill trained in a "
        "variant scene (scene_rollers.xml for the roller pair, scene_ball.xml for "
        "the ball-kick pair); omit to use the scene the registry entry declares",
    )
    ap.add_argument("--duration", type=float, default=8.0, help="seconds of rollout")
    ap.add_argument("--vx", type=float, default=0.3, help="forward velocity command (m/s)")
    ap.add_argument("--vy", type=float, default=0.0, help="lateral velocity command (m/s)")
    ap.add_argument("--vyaw", type=float, default=0.0, help="yaw-rate command (rad/s)")
    ap.add_argument("--control-frequency", type=float, default=50.0)
    ap.add_argument("--camera", default="track", help="'track' (body-tracking) or 'default'")
    ap.add_argument("--out", default="/tmp/microduck_viz/microduck.mp4")
    ap.add_argument("--gif", default=None, help="also write a looping GIF here")
    ap.add_argument("--gif-fps", type=float, default=13.0)
    ap.add_argument("--gif-width", type=int, default=480)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--distance", type=float, default=1.4, help="tracking cam distance")
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-15.0)
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        raise SystemExit(f"no such ONNX: {args.onnx}")

    asyncio.run(_rollout(args))


if __name__ == "__main__":
    main()
