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

Encodes an MP4 (h264 / yuv420p) - and optionally a looping GIF - through
:func:`strands_robots.rendering.encode_clip`, the encoder every recorder in the
package writes with, so this clip and a ``run_policy(video=...)`` clip are the
same bytes for the same frames. Import-clean and headless.

Examples::

    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
    python examples/microduck/render_video.py \
        --onnx alpha_walking.onnx \
        --vx 0.3 --duration 8 --out /tmp/microduck_viz/walk_forward.mp4

    python examples/microduck/render_video.py \
        --onnx alpha_walking.onnx \
        --vx 0.1 --vyaw 0.6 --duration 6 --out /tmp/microduck_viz/turn.mp4

    python examples/microduck/render_video.py \
        --onnx alpha_stand.onnx \
        --duration 5 --out /tmp/microduck_viz/stand.mp4

    # A skill trained in a variant scene names it with --scene. Without one the
    # duck rolls on a floor with no wheels under its feet, or swings at a ball
    # that is not there. The world's rendered floor reaches 5 m from the origin
    # and the roller covers about 0.7 m/s at --vx 0.3, so 6 s keeps it on the
    # checkerboard; the script says so when a ride runs off the edge.
    python examples/microduck/render_video.py \
        --onnx roller.onnx --scene scene_rollers.xml \
        --vx 0.3 --duration 6 --out /tmp/microduck_viz/roller.mp4

    # The ball scene declares its ball 0.3 m ahead; the kick weights were trained
    # with it 0.09 m ahead and 0.042 m to the side of the kicking foot. With a
    # ball in the scene the ball is placed there before the rollout, on the
    # side --kick-foot names (inferred from the weight's file name when its
    # stem says left or right), the way Pollen's runtime places it.
    python examples/microduck/render_video.py \
        --onnx ball_kick_left.onnx --scene scene_ball.xml \
        --vx 0 --duration 4 --out /tmp/microduck_viz/kick.mp4

``--onnx`` takes a local file or the bare name of a weight in Pollen's Hub
repository (``pollen-robotics/microduck-policies``), which is fetched on first
use and cached - the ``policies/`` directory of Pollen's git repository, where
the weights used to live, no longer exists.

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

# Where the two kick weights were trained to find the ball, in the robot's yaw
# frame: ahead of the trunk, and to the side of the foot that swings. Read off
# Pollen's reference runtime (microduck_rl ``scripts/infer_policy.py``,
# ``BALL_OFFSET_X`` / ``BALL_OFFSET_ABS_Y``), whose ``_place_ball`` this mirrors.
# ``scene_ball.xml`` itself declares the ball 0.3 m straight ahead - 3.3x too
# far, and centred - so a kick driven from the declared position swings at air.
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
#: The ball's free joint, as ``ball.xml`` names it; ``add_robot(name=...)``
#: prefixes it with the robot's name, so it is matched by suffix.
BALL_JOINT = "ball_free"


def _load_mujoco():
    """Import mujoco, returning the module (or raise a clear error)."""
    try:
        import mujoco  # noqa: PLC0415

        return mujoco
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit("mujoco is required. Install with: pip install 'strands-robots[sim-mujoco]'") from exc


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
    """Write frames to an MP4 through the package encoder; optionally a looping GIF.

    :func:`~strands_robots.rendering.encode_clip` is what ``run_policy(video=...)``
    and every recorder in the package write with (libx264, yuv420p, quality 8,
    exact frame size). Going through it rather than a private ``imageio`` call
    keeps this showcase on the encoder the release ships - and on its refusals:
    a frame rate that is not a positive whole number is named here, not
    discovered as a clip that plays at the wrong speed.
    """
    from strands_robots.rendering import encode_clip  # noqa: PLC0415

    out = encode_clip(frames, out_path, fps=fps, quality=8)
    print(f"  wrote mp4 {out} ({len(frames)} frames @ {fps} fps)")

    if gif_path:
        _encode_gif(frames, gif_path, gif_fps, gif_width)


def _downscale(frames, width):
    """Nearest-neighbour downscale to ``width`` px wide (no PIL/cv2 needed)."""
    h, w = frames[0].shape[:2]
    scale = width / float(w)
    new_w, new_h = width, max(1, int(round(h * scale)))
    ys = (np.linspace(0, h - 1, new_h)).astype(int)
    xs = (np.linspace(0, w - 1, new_w)).astype(int)
    return [f[ys][:, xs] for f in frames]


def _encode_gif(frames, gif_path, gif_fps, gif_width):
    """Write a compact looping GIF, downscaled to ``gif_width`` px wide."""
    from strands_robots.rendering import encode_clip  # noqa: PLC0415

    small = _downscale(frames, gif_width)
    out = encode_clip(small, gif_path, fps=gif_fps)
    new_h, new_w = small[0].shape[:2]
    size_mb = os.path.getsize(out) / 1e6
    print(f"  wrote gif {out} ({new_w}x{new_h}, {size_mb:.2f} MB)")


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


def _floor_half_extent(mujoco, model) -> float | None:
    """How far from the origin the rendered ground reaches, in metres.

    The world strips the floor a scene ships and lays its own ``ground`` plane
    in its place, whose ``size`` bounds what is *drawn* - MuJoCo planes collide
    without limit, so a robot that rolls past that edge keeps rolling, on a floor
    the frame no longer shows. Returns ``None`` when there is no such plane or it
    is drawn without limit (``size`` 0), in which case nothing can be run off.
    """
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if gid < 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_PLANE:
        return None
    half = float(min(model.geom_size[gid][0], model.geom_size[gid][1]))
    return half if half > 0 else None


def first_tick_off_the_floor(path, half_extent: float | None) -> int | None:
    """The index of the first trunk position past the rendered floor, or ``None``.

    Args:
        path: The trunk's world ``(x, y[, z])`` per control tick, in order.
        half_extent: The floor's half-extent from :func:`_floor_half_extent`;
            ``None`` means the floor is unbounded and nothing is ever off it.
    """
    if half_extent is None:
        return None
    for i, pos in enumerate(path):
        if max(abs(float(pos[0])), abs(float(pos[1]))) > half_extent:
            return i
    return None


def distance_travelled(path) -> float:
    """How far the trunk went along ``path``, in metres.

    The sum of the step lengths, not the straight line from the first position to
    the last: the roller curves, so the chord understates the ride (a 10 s ride
    that ends 6.22 m from the start covers 7.21 m of ground).

    Args:
        path: The trunk's world ``(x, y[, z])`` per control tick, in order.
    """
    if len(path) < 2:
        return 0.0
    steps = np.diff(np.asarray(path, dtype=float)[:, :2], axis=0)
    return float(np.linalg.norm(steps, axis=1).sum())


def _kick_foot(args) -> str | None:
    """Which foot the weight kicks with: ``--kick-foot``, else read off the file name.

    Returns:
        ``"left"`` or ``"right"``, or ``None`` when neither was given and the
        ONNX stem says neither (a non-kick weight on the ball scene, which needs
        no placement).
    """
    if getattr(args, "kick_foot", None):
        return args.kick_foot
    stem = Path(args.onnx).stem.lower()
    for foot in ("left", "right"):
        if foot in stem.split("_"):
            return foot
    return None


def _ball_joint(mujoco, model) -> int:
    """The id of the ball's free joint, or ``-1`` when the scene carries no ball."""
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if name == BALL_JOINT or name.endswith("/" + BALL_JOINT):
            return jid
    return -1


def place_ball_for_kick(mujoco, model, data, foot: str) -> tuple[float, float] | None:
    """Teleport the ball to where the kick weights were trained to find it.

    Mirrors ``_place_ball`` in Pollen's reference runtime: the trained offset
    (:data:`BALL_OFFSET_X` ahead, :data:`BALL_OFFSET_ABS_Y` toward ``foot``) is
    rotated into the trunk's yaw frame and added to the trunk position; the ball
    keeps its declared height, its orientation is reset and its velocity zeroed,
    and the kinematics are recomputed so the first observation and frame see it
    there.

    Args:
        mujoco: The ``mujoco`` module.
        model: The compiled scene.
        data: Its state, after ``reset()``.
        foot: ``"left"`` or ``"right"``.

    Returns:
        The ball's new ``(x, y)`` in world coordinates, or ``None`` when the
        scene carries no ball joint (nothing was moved).
    """
    jid = _ball_joint(mujoco, model)
    if jid < 0:
        return None
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    x, y = float(data.xpos[body][0]), float(data.xpos[body][1])
    qw, qx, qy, qz = (float(v) for v in data.xquat[body])
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    off_y = -BALL_OFFSET_ABS_Y if foot == "right" else BALL_OFFSET_ABS_Y
    bx = x + np.cos(yaw) * BALL_OFFSET_X - np.sin(yaw) * off_y
    by = y + np.sin(yaw) * BALL_OFFSET_X + np.cos(yaw) * off_y
    qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    z = float(data.qpos[qadr + 2])
    data.qpos[qadr : qadr + 7] = [bx, by, z, 1.0, 0.0, 0.0, 0.0]
    data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return float(bx), float(by)


def _placement_report(args, foot: str, placed: tuple[float, float] | None) -> str:
    """What to say about the ball this rollout was going to kick.

    Three outcomes, one place: the ball was placed and the line says where; the
    caller asked for a placement the scene cannot hold, which is refused; or the
    foot was only inferred from the weight's name, in which case the rollout runs
    on the ball-less scene the caller chose - which is not an error, but is said
    out loud, because a duck kicking air renders as a duck standing still.

    Args:
        args: The parsed command line, read for an explicit ``--kick-foot``.
        foot: The foot the kick swings with, ``"left"`` or ``"right"``.
        placed: What :func:`place_ball_for_kick` returned - the ball's new
            ``(x, y)``, or ``None`` when the scene carries no ball.

    Returns:
        One line to print.

    Raises:
        SystemExit: If ``--kick-foot`` was passed for a scene with no ball,
            naming the scene that has one.
    """
    if placed is not None:
        return f"  ball placed at ({placed[0]:.3f}, {placed[1]:.3f}) in front of the {foot} foot"
    if getattr(args, "kick_foot", None):
        raise SystemExit(
            f"--kick-foot {foot}: the scene carries no {BALL_JOINT!r} joint; pass --scene scene_ball.xml"
        )
    return (
        f"  no {BALL_JOINT!r} joint in this scene: the {foot}-foot kick swings at nothing; "
        "pass --scene scene_ball.xml for the geometry the weight was trained on"
    )


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

    foot = _kick_foot(args)
    if foot is not None:
        print(_placement_report(args, foot, place_ball_for_kick(mujoco, model, data, foot)))

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
            cam = _make_tracking_camera(mujoco, model, BASE_BODY, args.distance, args.azimuth, args.elevation)
            # smoke-render one frame to confirm a GL context exists
            renderer.update_scene(data, camera=cam)
            _ = renderer.render()
        except Exception as exc:  # pragma: no cover - GL context guard
            print(f"  direct mujoco.Renderer unavailable ({exc}); falling back to sim.render")
            renderer = None
            direct = False

    base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    half_extent = _floor_half_extent(mujoco, model)
    path = []
    frames = []
    for _ in range(n_ticks):
        obs = sim.get_observation()
        actions = await policy.get_actions(obs, "", target_velocity=tv)
        sim.send_action(actions[0])
        sim.step(substeps)
        if base_body >= 0:
            path.append(data.xpos[base_body][:2].copy())
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

    # A ride that outruns the drawn floor keeps rolling on an invisible one; the
    # frames from that tick on show the duck over the void. Say so, with the
    # second it happened, rather than leave it to be found in the clip.
    off = first_tick_off_the_floor(path, half_extent)
    if off is not None:
        travelled = distance_travelled(path)
        print(
            f"  the duck left the rendered floor (±{half_extent:.1f} m) at "
            f"{off / args.control_frequency:.1f} s and travelled {travelled:.2f} m in all; "
            f"the frames after that show it over the void - shorten --duration or lower --vx"
        )

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
    ap.add_argument(
        "--onnx",
        default="alpha_walking.onnx",
        help="a local .onnx file, or the bare name of a weight in pollen-robotics/microduck-policies",
    )
    ap.add_argument(
        "--scene",
        default=None,
        help="scene file under the microduck asset dir, for a skill trained in a "
        "variant scene (scene_rollers.xml for the roller pair, scene_ball.xml for "
        "the ball-kick pair); omit to use the scene the registry entry declares",
    )
    ap.add_argument(
        "--kick-foot",
        choices=("left", "right"),
        default=None,
        help="place the ball where the kick weights were trained to find it, in front of "
        "this foot; inferred from the ONNX file name (ball_kick_left / ball_kick_right) "
        "when omitted",
    )
    ap.add_argument("--duration", type=float, default=8.0, help="seconds of rollout")
    ap.add_argument("--vx", type=float, default=0.3, help="forward velocity command (m/s)")
    ap.add_argument("--vy", type=float, default=0.0, help="lateral velocity command (m/s)")
    ap.add_argument("--vyaw", type=float, default=0.0, help="yaw-rate command (rad/s)")
    ap.add_argument("--control-frequency", type=float, default=50.0)
    ap.add_argument("--camera", default="track", help="'track' (body-tracking) or 'default'")
    ap.add_argument("--out", default="/tmp/microduck_viz/microduck.mp4")
    ap.add_argument("--gif", default=None, help="also write a looping GIF here")
    ap.add_argument("--gif-fps", type=int, default=13, help="GIF playback rate (whole frames per second)")
    ap.add_argument("--gif-width", type=int, default=480)
    ap.add_argument("--fps", type=int, default=50, help="MP4 playback rate (whole frames per second)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--distance", type=float, default=1.4, help="tracking cam distance")
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-15.0)
    args = ap.parse_args()

    from strands_robots.policies.microduck import resolve_microduck_weight  # noqa: PLC0415

    try:
        args.onnx = str(resolve_microduck_weight(args.onnx))
    except (FileNotFoundError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc

    asyncio.run(_rollout(args))


if __name__ == "__main__":
    main()
