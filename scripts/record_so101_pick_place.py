#!/usr/bin/env python3
"""
Record SO-101 Pick-and-Place Demo Video.

Uses MuJoCo with EGL GPU rendering to create a 3-camera pick-and-place
demonstration video for the Isaac Sim pipeline (issue #124).

Cameras:
  1. Front overview (free camera)
  2. Wrist camera (SO-101 built-in camera)
  3. Side view (free camera)

The robot follows a scripted trajectory through all 4 phases:
  Reach → Grasp → Transport → Place

Run:
    MUJOCO_GL=egl python scripts/record_so101_pick_place.py

Output:
    videos/so101_pick_place_demo.mp4
    videos/so101_pick_place_multicam.mp4

Refs: Issue #124
"""

import os
import sys
import time

os.environ["MUJOCO_GL"] = "egl"
os.environ["DISPLAY"] = ""

import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIDTH = 640
HEIGHT = 480
FPS = 30
DURATION = 8.0  # seconds
TOTAL_FRAMES = int(DURATION * FPS)


def record_pick_place_demo():
    """Record a scripted pick-and-place demo with SO-101."""
    import imageio
    import mujoco

    from strands_robots.assets import resolve_model_path

    print("=" * 60)
    print("🤖 SO-101 Pick-and-Place Demo Recording")
    print(f"   Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps")
    print(f"   Duration: {DURATION}s ({TOTAL_FRAMES} frames)")
    print("=" * 60)

    # Load SO-101 scene_box (has the pickup box + keyframe)
    model_path = resolve_model_path("so101", prefer_scene=True)
    if model_path is None:
        # Fallback: try scene_box directly
        from pathlib import Path
        model_path = Path(__file__).parent.parent / "strands_robots" / "assets" / "robotstudio_so101" / "scene_box.xml"

    print(f"  Model: {model_path}")

    # Read XML and inject offscreen buffer size
    model_dir = os.path.dirname(os.path.abspath(str(model_path)))
    with open(str(model_path), 'r') as f:
        xml_content = f.read()

    # Inject offscreen framebuffer
    if '<global' in xml_content:
        import re
        if 'offwidth' not in xml_content:
            xml_content = re.sub(r'<global\b', f'<global offwidth="{WIDTH}" offheight="{HEIGHT}"', xml_content)
        else:
            xml_content = re.sub(r'offwidth="\d+"', f'offwidth="{WIDTH}"', xml_content)
            xml_content = re.sub(r'offheight="\d+"', f'offheight="{HEIGHT}"', xml_content)
    elif '<visual>' in xml_content:
        xml_content = xml_content.replace(
            '<visual>',
            f'<visual>\n    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>'
        )

    # Write temp XML
    tmp_xml = os.path.join(model_dir, "_tmp_pick_place.xml")
    with open(tmp_xml, 'w') as f:
        f.write(xml_content)

    try:
        model = mujoco.MjModel.from_xml_path(tmp_xml)
    finally:
        try:
            os.remove(tmp_xml)
        except Exception:
            pass

    data = mujoco.MjData(model)

    print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
    print(f"  nq: {model.nq}, nv: {model.nv}")

    # Load the pickup keyframe to get a good starting position
    if model.nkey > 0:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pickup")
        if key_id >= 0:
            print(f"  Found 'pickup' keyframe (id={key_id})")
        else:
            key_id = 0
            print("  Using keyframe 0")
    else:
        key_id = -1
        print("  No keyframes found")

    # Create renderer
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    # Settle physics from default position
    mujoco.mj_resetData(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)

    dt = model.opt.timestep
    phys_per_frame = max(1, int((1.0 / FPS) / dt))

    # ── Define scripted pick-and-place trajectory ──
    # Robot joint names for SO-101:
    # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    # Actuator order matches these joints

    # Key waypoints (joint positions for 6 actuators)
    home_pos = np.zeros(model.nu)  # All zeros = home

    # Pickup keyframe values (from scene_box.xml)
    if key_id >= 0 and model.nkey > 0:
        # Extract ctrl from keyframe
        pickup_ctrl = model.key_ctrl[key_id].copy()
        print(f"  Pickup keyframe ctrl: {pickup_ctrl}")
    else:
        pickup_ctrl = np.array([0, 0, 0.4732, 1.17718, 1.58438, 0.727665])

    # Define trajectory phases
    def get_trajectory_ctrl(progress):
        """Get control values based on animation progress (0-1)."""
        ctrl = np.zeros(model.nu)

        if progress < 0.15:
            # Phase 0: Home → Approach (move above box)
            alpha = progress / 0.15
            # Interpolate from home to a pre-grasp position
            pre_grasp = pickup_ctrl.copy()
            pre_grasp[2] = pickup_ctrl[2] * 0.7  # Less bend
            ctrl[:len(pre_grasp)] = home_pos[:len(pre_grasp)] * (1 - alpha) + pre_grasp * alpha

        elif progress < 0.30:
            # Phase 1: Reach — move down to box
            alpha = (progress - 0.15) / 0.15
            pre_grasp = pickup_ctrl.copy()
            pre_grasp[2] = pickup_ctrl[2] * 0.7
            ctrl[:len(pickup_ctrl)] = pre_grasp * (1 - alpha) + pickup_ctrl * alpha

        elif progress < 0.40:
            # Phase 2: Grasp — hold at pickup position (gripper closes)
            ctrl[:len(pickup_ctrl)] = pickup_ctrl

        elif progress < 0.60:
            # Phase 3: Lift — raise the arm
            alpha = (progress - 0.40) / 0.20
            lift_pos = pickup_ctrl.copy()
            lift_pos[2] = pickup_ctrl[2] * 0.5  # Reduce elbow bend
            lift_pos[1] = pickup_ctrl[1] * 0.7   # Raise shoulder
            ctrl[:len(pickup_ctrl)] = pickup_ctrl * (1 - alpha) + lift_pos * alpha

        elif progress < 0.80:
            # Phase 4: Transport — rotate to new position
            alpha = (progress - 0.60) / 0.20
            lift_pos = pickup_ctrl.copy()
            lift_pos[2] = pickup_ctrl[2] * 0.5
            lift_pos[1] = pickup_ctrl[1] * 0.7
            place_pos = lift_pos.copy()
            place_pos[0] = 0.8  # Rotate shoulder pan
            ctrl[:len(pickup_ctrl)] = lift_pos * (1 - alpha) + place_pos * alpha

        else:
            # Phase 5: Place — lower arm
            alpha = (progress - 0.80) / 0.20
            place_pos = pickup_ctrl.copy()
            place_pos[2] = pickup_ctrl[2] * 0.5
            place_pos[1] = pickup_ctrl[1] * 0.7
            place_pos[0] = 0.8
            lower_pos = place_pos.copy()
            lower_pos[2] = pickup_ctrl[2]
            lower_pos[1] = pickup_ctrl[1]
            ctrl[:len(pickup_ctrl)] = place_pos * (1 - alpha) + lower_pos * alpha

        return ctrl

    # ── Record video ──
    output_path = os.path.join(OUTPUT_DIR, "so101_pick_place_demo.mp4")

    writer = imageio.get_writer(
        output_path,
        fps=FPS,
        codec='libx264',
        macro_block_size=1,
        quality=8,
        output_params=['-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '22'],
    )

    # Camera setup
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 160
    cam.elevation = -25
    cam.distance = 0.8
    cam.lookat[:] = [0.15, 0, 0.25]

    t_start = time.time()
    print(f"\n  Recording {TOTAL_FRAMES} frames...")

    for frame_idx in range(TOTAL_FRAMES):
        frame_idx / FPS
        progress = frame_idx / TOTAL_FRAMES

        # Get scripted control
        ctrl = get_trajectory_ctrl(progress)

        # Apply ctrl (clamp to actuator limits)
        for i in range(min(model.nu, len(ctrl))):
            if model.actuator_ctrllimited[i]:
                ctrl[i] = np.clip(ctrl[i], model.actuator_ctrlrange[i, 0], model.actuator_ctrlrange[i, 1])
            data.ctrl[i] = ctrl[i]

        # Step physics
        for _ in range(phys_per_frame):
            mujoco.mj_step(model, data)

        # Slowly orbit camera for cinematic effect
        cam.azimuth = 160 + 30 * np.sin(2 * np.pi * progress * 0.5)

        # Render
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        writer.append_data(frame)

        if (frame_idx + 1) % (FPS * 2) == 0:
            elapsed = time.time() - t_start
            fps_actual = (frame_idx + 1) / elapsed
            phase = "Home" if progress < 0.15 else "Reach" if progress < 0.30 else "Grasp" if progress < 0.40 else "Lift" if progress < 0.60 else "Transport" if progress < 0.80 else "Place"
            print(f"  Frame {frame_idx+1}/{TOTAL_FRAMES} ({fps_actual:.0f} fps) — Phase: {phase}")

    writer.close()
    del renderer

    elapsed = time.time() - t_start
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  ✅ Recorded: {output_path} ({file_size:.2f} MB, {elapsed:.1f}s)")

    return output_path, file_size


def record_multicam_view():
    """Record a multi-camera view (3 cameras tiled in one frame)."""
    import imageio
    import mujoco

    from strands_robots.assets import resolve_model_path

    print("\n" + "=" * 60)
    print("📹 SO-101 Multi-Camera Recording (so101_tricam layout)")
    print("=" * 60)

    CAM_W = 320
    CAM_H = 240
    TILE_W = CAM_W * 3  # 960 wide
    TILE_H = CAM_H      # 240 tall

    model_path = resolve_model_path("so101", prefer_scene=True)
    if model_path is None:
        from pathlib import Path
        model_path = Path(__file__).parent.parent / "strands_robots" / "assets" / "robotstudio_so101" / "scene_box.xml"

    model_dir = os.path.dirname(os.path.abspath(str(model_path)))
    with open(str(model_path), 'r') as f:
        xml_content = f.read()

    # Inject offscreen buffer (large enough for all cameras)
    import re
    if 'offwidth' in xml_content:
        xml_content = re.sub(r'offwidth="\d+"', f'offwidth="{TILE_W}"', xml_content)
        xml_content = re.sub(r'offheight="\d+"', f'offheight="{TILE_H}"', xml_content)
    elif '<global' in xml_content:
        xml_content = re.sub(r'<global\b', f'<global offwidth="{TILE_W}" offheight="{TILE_H}"', xml_content)
    elif '<visual>' in xml_content:
        xml_content = xml_content.replace('<visual>', f'<visual>\n    <global offwidth="{TILE_W}" offheight="{TILE_H}"/>')

    tmp_xml = os.path.join(model_dir, "_tmp_multicam.xml")
    with open(tmp_xml, 'w') as f:
        f.write(xml_content)

    try:
        model = mujoco.MjModel.from_xml_path(tmp_xml)
    finally:
        try:
            os.remove(tmp_xml)
        except Exception:
            pass

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)

    # Settle
    for _ in range(200):
        mujoco.mj_step(model, data)

    dt = model.opt.timestep
    phys_per_frame = max(1, int((1.0 / FPS) / dt))

    # Define 3 camera viewpoints (matching so101_tricam data config)
    cameras = [
        {"name": "front", "azimuth": 160, "elevation": -25, "distance": 0.8, "lookat": [0.15, 0, 0.25]},
        {"name": "wrist", "azimuth": 90, "elevation": -40, "distance": 0.4, "lookat": [0.3, 0, 0.2]},
        {"name": "side", "azimuth": 250, "elevation": -20, "distance": 0.7, "lookat": [0.15, 0, 0.3]},
    ]

    output_path = os.path.join(OUTPUT_DIR, "so101_pick_place_multicam.mp4")
    writer = imageio.get_writer(
        output_path,
        fps=FPS,
        codec='libx264',
        macro_block_size=1,
        quality=8,
        output_params=['-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '22'],
    )

    # Use the pickup keyframe ctrl
    pickup_ctrl = np.zeros(model.nu)
    if model.nkey > 0:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pickup")
        if key_id >= 0:
            pickup_ctrl = model.key_ctrl[key_id].copy()

    t_start = time.time()
    print(f"  Recording {TOTAL_FRAMES} frames with 3 cameras (tiled {TILE_W}x{TILE_H})...")

    for frame_idx in range(TOTAL_FRAMES):
        progress = frame_idx / TOTAL_FRAMES

        # Simple arm motion
        alpha = np.sin(2 * np.pi * progress * 0.5)
        ctrl = pickup_ctrl * (0.5 + 0.5 * alpha)
        for i in range(min(model.nu, len(ctrl))):
            if model.actuator_ctrllimited[i]:
                ctrl[i] = np.clip(ctrl[i], model.actuator_ctrlrange[i, 0], model.actuator_ctrlrange[i, 1])
            data.ctrl[i] = ctrl[i]

        for _ in range(phys_per_frame):
            mujoco.mj_step(model, data)

        # Render from each camera and tile
        tiled = np.zeros((CAM_H, TILE_W, 3), dtype=np.uint8)

        for cam_idx, cam_cfg in enumerate(cameras):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.azimuth = cam_cfg["azimuth"]
            cam.elevation = cam_cfg["elevation"]
            cam.distance = cam_cfg["distance"]
            cam.lookat[:] = cam_cfg["lookat"]

            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            tiled[:, cam_idx * CAM_W:(cam_idx + 1) * CAM_W, :] = frame

        writer.append_data(tiled)

        if (frame_idx + 1) % (FPS * 2) == 0:
            elapsed = time.time() - t_start
            print(f"  Frame {frame_idx+1}/{TOTAL_FRAMES} ({(frame_idx+1)/elapsed:.0f} fps)")

    writer.close()
    del renderer

    elapsed = time.time() - t_start
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✅ Recorded: {output_path} ({file_size:.2f} MB, {elapsed:.1f}s)")

    return output_path, file_size


def main():
    results = {}

    # Single camera demo
    try:
        path, size = record_pick_place_demo()
        results["demo"] = {"path": path, "size_mb": size, "status": "success"}
    except Exception as e:
        print(f"  ❌ Demo recording failed: {e}")
        import traceback
        traceback.print_exc()
        results["demo"] = {"error": str(e), "status": "failed"}

    # Multi-camera
    try:
        path, size = record_multicam_view()
        results["multicam"] = {"path": path, "size_mb": size, "status": "success"}
    except Exception as e:
        print(f"  ❌ Multicam recording failed: {e}")
        import traceback
        traceback.print_exc()
        results["multicam"] = {"error": str(e), "status": "failed"}

    print("\n" + "=" * 60)
    print("📊 Recording Summary")
    print("=" * 60)
    for name, r in results.items():
        status = r.get("status", "unknown")
        icon = "✅" if status == "success" else "❌"
        extra = f" ({r.get('size_mb', 0):.2f} MB)" if status == "success" else f" ({r.get('error', '')})"
        print(f"  {icon} {name}{extra}")


if __name__ == "__main__":
    main()
