#!/usr/bin/env python3
"""Generate simulation demo videos using MuJoCo headless rendering."""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import mujoco as mj
import numpy as np

ASSETS_DIR = Path("strands_robots/assets")
OUTPUT_DIR = Path("docs/assets/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

W, H, FPS = 854, 480, 24


def render_frames(model, data, ctrl_fn, n_frames, substeps=5):
    """Render n_frames with a control function."""
    model.vis.global_.offwidth = max(W, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(H, model.vis.global_.offheight)
    renderer = mj.Renderer(model, height=H, width=W)
    frames = []
    for i in range(n_frames):
        ctrl_fn(data, i, n_frames)
        for _ in range(substeps):
            mj.mj_step(model, data)
        renderer.update_scene(data)
        frames.append(renderer.render().copy())
    del renderer
    return frames


def smooth(start, end, steps):
    """Smooth sinusoidal interpolation."""
    t = np.linspace(0, 1, steps)
    t = (1 - np.cos(t * np.pi)) / 2
    return start + (end - start) * t[:, np.newaxis]


def encode(frames, filename):
    """Encode frames to MP4."""
    out = OUTPUT_DIR / filename
    writer = imageio.get_writer(
        str(out), fps=FPS, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=1,  # avoid resize warning
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    fsize = out.stat().st_size
    mean_px = np.mean([f.mean() for f in frames])
    print(f"  ✅ {filename}: {len(frames)} frames, {fsize/1024:.0f}KB, mean_px={mean_px:.1f}")
    return fsize


def write_scene_xml(robot_dir, extra_worldbody="", extra_top=""):
    """Write a temp scene XML in the robot's directory so mesh paths resolve correctly."""
    scene_xml = f"""<mujoco model="video_scene">
  <include file="so_arm100.xml"/>
  <statistic center="0.1 -0.01 0.05" extent="0.5"/>
  {extra_top}
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-25"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="vidgp" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="vidgp" texture="vidgp" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="vidfloor" size="0 0 0.05" type="plane" material="vidgp"/>
    {extra_worldbody}
  </worldbody>
</mujoco>"""
    path = robot_dir / "_temp_video_scene.xml"
    path.write_text(scene_xml)
    return str(path)


def write_custom_scene_xml(target_dir, include_file, scene_xml_content):
    """Write a fully custom scene XML into the right directory."""
    path = target_dir / "_temp_video_scene.xml"
    path.write_text(scene_xml_content)
    return str(path)


def video_tabletop():
    """SO-100 tabletop manipulation."""
    print("🎬 01: Tabletop Manipulation")

    robot_dir = ASSETS_DIR / "trs_so_arm100"
    objects = """
    <body name="red_cube" pos="0.15 0.08 0.015">
      <freejoint/>
      <geom type="box" size="0.015 0.015 0.015" rgba="0.9 0.2 0.2 1" mass="0.05"/>
    </body>
    <body name="blue_cyl" pos="0.2 0.0 0.02">
      <freejoint/>
      <geom type="cylinder" size="0.012 0.02" rgba="0.2 0.4 0.9 1" mass="0.04"/>
    </body>"""

    scene_path = write_scene_xml(robot_dir, extra_worldbody=objects)
    model = mj.MjModel.from_xml_path(scene_path)
    data = mj.MjData(model)

    for _ in range(200):
        mj.mj_step(model, data)

    # Keyframes: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw
    keyframes = [
        np.array([0.0, -0.5, 0.8, 0.0, 0.0, 0.5]),    # Home
        np.array([0.4, -1.2, 1.8, -0.3, 0.0, 1.2]),    # Reach
        np.array([0.4, -1.2, 1.8, -0.3, 0.0, 0.0]),    # Grasp
        np.array([0.4, -0.6, 1.0, -0.3, 0.0, 0.0]),    # Lift
        np.array([-0.4, -0.6, 1.0, -0.3, 0.0, 0.0]),   # Move
        np.array([-0.4, -1.1, 1.7, -0.3, 0.0, 0.0]),   # Place
        np.array([-0.4, -1.1, 1.7, -0.3, 0.0, 1.0]),   # Release
        np.array([0.0, -0.5, 0.8, 0.0, 0.0, 0.5]),     # Retract
    ]
    steps_per = [20, 25, 10, 20, 25, 18, 8, 25]

    # Build trajectory
    all_ctrl = []
    for i in range(len(keyframes) - 1):
        traj = smooth(keyframes[i], keyframes[i + 1], steps_per[i + 1])
        all_ctrl.extend(traj)
    for _ in range(10):
        all_ctrl.append(keyframes[-1])

    def ctrl_fn(data, i, n):
        if i < len(all_ctrl):
            data.ctrl[:6] = all_ctrl[i]

    frames = render_frames(model, data, ctrl_fn, len(all_ctrl), substeps=5)
    Path(scene_path).unlink(missing_ok=True)
    return encode(frames, "01_tabletop_manipulation.mp4")


def video_domain_rand():
    """Same scene with 4 visual variations."""
    print("🎬 03: Domain Randomization")

    robot_dir = ASSETS_DIR / "trs_so_arm100"
    variations = [
        ("Standard", "0.3 0.5 0.7", "0 0 0", "0.2 0.3 0.4", "0.1 0.2 0.3", "0.6 0.6 0.6", "0.3 0.3 0.3", "0.9 0.2 0.2 1"),
        ("Bright", "0.8 0.85 0.9", "0.5 0.55 0.6", "0.85 0.85 0.85", "0.75 0.75 0.75", "0.9 0.9 0.9", "0.5 0.5 0.5", "0.1 0.6 0.9 1"),
        ("Dark", "0.1 0.1 0.15", "0.05 0.05 0.08", "0.15 0.15 0.18", "0.1 0.1 0.12", "0.4 0.35 0.3", "0.15 0.12 0.1", "0.9 0.7 0.1 1"),
        ("Warm", "0.5 0.35 0.25", "0.15 0.1 0.05", "0.4 0.3 0.25", "0.3 0.2 0.15", "0.7 0.55 0.4", "0.35 0.28 0.2", "0.3 0.9 0.4 1"),
    ]

    all_frames = []
    for name, sky1, sky2, f1, f2, diff, amb, cube_rgba in variations:
        scene_xml = f"""<mujoco model="dr_{name}">
  <include file="so_arm100.xml"/>
  <statistic center="0.1 -0.01 0.05" extent="0.5"/>
  <visual>
    <headlight diffuse="{diff}" ambient="{amb}" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="135" elevation="-25"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="{sky1}" rgb2="{sky2}" width="512" height="3072"/>
    <texture type="2d" name="drgp" builtin="checker" mark="edge" rgb1="{f1}" rgb2="{f2}"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="drgp" texture="drgp" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="drfloor" size="0 0 0.05" type="plane" material="drgp"/>
    <body name="cube" pos="0.15 0.05 0.015">
      <freejoint/>
      <geom type="box" size="0.015 0.015 0.015" rgba="{cube_rgba}" mass="0.05"/>
    </body>
  </worldbody>
</mujoco>"""
        sp = write_custom_scene_xml(robot_dir, "so_arm100.xml", scene_xml)
        model = mj.MjModel.from_xml_path(sp)
        data = mj.MjData(model)
        for _ in range(100):
            mj.mj_step(model, data)

        n = 30  # ~1.25 seconds per variation
        def ctrl_fn(data, i, n_total):
            t = i / 120 * 2 * np.pi
            data.ctrl[0] = 0.3 * np.sin(t)
            data.ctrl[1] = -0.8 + 0.3 * np.sin(t * 0.7)
            data.ctrl[2] = 1.0 + 0.4 * np.sin(t * 0.5)
            data.ctrl[3] = 0.2 * np.sin(t * 1.2)
            data.ctrl[4] = 0.3 * np.sin(t * 0.8)
            data.ctrl[5] = 0.3 + 0.3 * np.sin(t)

        frames = render_frames(model, data, ctrl_fn, n, substeps=5)
        all_frames.extend(frames)
        Path(sp).unlink(missing_ok=True)

    return encode(all_frames, "03_domain_randomization.mp4")


def video_world_building():
    """Step-by-step scene construction."""
    print("🎬 04: World Building")

    robot_dir = ASSETS_DIR / "trs_so_arm100"
    all_frames = []

    # Stage 1: Empty world
    xml1 = """<mujoco model="wb1">
  <statistic center="0.1 0.0 0.1" extent="0.6"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="135" elevation="-25"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="wbgp" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="wbgp" texture="wbgp" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="wbfloor" size="0 0 0.05" type="plane" material="wbgp"/>
  </worldbody>
</mujoco>"""
    sp = write_custom_scene_xml(robot_dir, None, xml1)
    m = mj.MjModel.from_xml_path(sp)
    d = mj.MjData(m)
    mj.mj_step(m, d)
    frames = render_frames(m, d, lambda d, i, n: None, 20, substeps=1)
    all_frames.extend(frames)
    Path(sp).unlink(missing_ok=True)

    # Stage 2: Add robot
    xml2 = """<mujoco model="wb2">
  <include file="so_arm100.xml"/>
  <statistic center="0.1 0.0 0.05" extent="0.5"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="135" elevation="-25"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="wb2gp" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="wb2gp" texture="wb2gp" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="wb2floor" size="0 0 0.05" type="plane" material="wb2gp"/>
  </worldbody>
</mujoco>"""
    sp = write_custom_scene_xml(robot_dir, "so_arm100.xml", xml2)
    m = mj.MjModel.from_xml_path(sp)
    d = mj.MjData(m)
    for _ in range(100):
        mj.mj_step(m, d)
    frames = render_frames(m, d, lambda d, i, n: None, 24, substeps=1)
    all_frames.extend(frames)
    Path(sp).unlink(missing_ok=True)

    # Stage 3: Add objects + animate
    xml3 = """<mujoco model="wb3">
  <include file="so_arm100.xml"/>
  <statistic center="0.1 0.0 0.05" extent="0.5"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="135" elevation="-25"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="wb3gp" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="wb3gp" texture="wb3gp" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="wb3floor" size="0 0 0.05" type="plane" material="wb3gp"/>
    <body name="rc" pos="0.15 0.08 0.015">
      <freejoint/>
      <geom type="box" size="0.015 0.015 0.015" rgba="0.9 0.2 0.2 1" mass="0.05"/>
    </body>
    <body name="gc" pos="0.12 -0.06 0.015">
      <freejoint/>
      <geom type="box" size="0.012 0.012 0.012" rgba="0.2 0.8 0.3 1" mass="0.03"/>
    </body>
    <body name="bc" pos="0.2 0.0 0.02">
      <freejoint/>
      <geom type="cylinder" size="0.012 0.02" rgba="0.2 0.4 0.9 1" mass="0.04"/>
    </body>
  </worldbody>
</mujoco>"""
    sp = write_custom_scene_xml(robot_dir, "so_arm100.xml", xml3)
    m = mj.MjModel.from_xml_path(sp)
    d = mj.MjData(m)
    for _ in range(200):
        mj.mj_step(m, d)

    # Static pause
    frames = render_frames(m, d, lambda d, i, n: None, 20, substeps=1)
    all_frames.extend(frames)

    # Animate
    def anim_ctrl(data, i, n):
        t = i / max(n, 1) * 2 * np.pi
        data.ctrl[0] = 0.4 * np.sin(t)
        data.ctrl[1] = -0.8 + 0.4 * np.sin(t * 0.6)
        data.ctrl[2] = 1.0 + 0.5 * np.sin(t * 0.4)
        data.ctrl[3] = 0.3 * np.sin(t * 1.3)
        data.ctrl[4] = 0.2 * np.sin(t * 0.9)
        data.ctrl[5] = 0.4 + 0.4 * np.sin(t)

    frames = render_frames(m, d, anim_ctrl, 48, substeps=5)
    all_frames.extend(frames)
    Path(sp).unlink(missing_ok=True)

    return encode(all_frames, "04_world_building.mp4")


def video_bimanual():
    """ALOHA bimanual coordination."""
    print("🎬 05: Bimanual Coordination")

    aloha_path = ASSETS_DIR / "aloha" / "scene.xml"
    if not aloha_path.exists():
        print("  ⏭️ ALOHA not found")
        return 0

    model = mj.MjModel.from_xml_path(str(aloha_path))
    data = mj.MjData(model)
    for _ in range(200):
        mj.mj_step(model, data)

    nu = model.nu
    half = nu // 2

    def ctrl_fn(data, i, n):
        t = i / max(n, 1) * 2 * np.pi
        for j in range(min(half, nu)):
            if j < model.njnt:
                jr = model.jnt_range[j]
                mid, span = (jr[0] + jr[1]) / 2, (jr[1] - jr[0]) / 2
                data.ctrl[j] = mid + 0.3 * span * np.sin((0.4 + 0.2 * (j % 4)) * t + j * 0.5)
        for j in range(half, min(nu, model.nu)):
            if j < model.njnt:
                jr = model.jnt_range[j]
                mid, span = (jr[0] + jr[1]) / 2, (jr[1] - jr[0]) / 2
                data.ctrl[j] = mid + 0.3 * span * np.sin((0.4 + 0.2 * ((j-half) % 4)) * t + (j-half) * 0.5 + np.pi)

    frames = render_frames(model, data, ctrl_fn, 100, substeps=5)
    return encode(frames, "05_bimanual_coordination.mp4")


def main():
    t0 = time.time()
    print("🎬 Generating MuJoCo simulation videos")
    print(f"📐 {W}×{H} @ {FPS}fps | MuJoCo {mj.__version__} + OSMesa\n")

    total_size = 0
    generated = 0

    for gen_fn in [video_tabletop, video_domain_rand, video_world_building, video_bimanual]:
        try:
            size = gen_fn()
            if size and size > 0:
                generated += 1
                total_size += size
        except Exception as e:
            print(f"  ❌ {e}")
            import traceback
            traceback.print_exc()
        print()

    elapsed = time.time() - t0
    print(f"📊 {generated} videos, {total_size/1024/1024:.1f}MB total, {elapsed:.0f}s")
    return generated


if __name__ == "__main__":
    n = main()
    sys.exit(0 if n > 0 else 1)
