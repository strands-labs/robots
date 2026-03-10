#!/usr/bin/env python3
"""
Render all 32 bundled robot models to high-quality PNG images.

Usage:
    MUJOCO_GL=osmesa python scripts/render_all_robots.py

Generates:
    docs/assets/robots/<robot_name>.png     — Individual robot renders (1280×720)
    docs/assets/robots/gallery.png          — Grid gallery of all robots
    docs/assets/robots/renders.json         — Metadata about each render

Requires:
    - MuJoCo (pip install mujoco)
    - OSMesa for headless rendering (apt install libosmesa6-dev)
    - Pillow, numpy, matplotlib

Environment:
    MUJOCO_GL=osmesa for headless CI environments (no GPU required)
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# Must set before importing mujoco
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
OUTPUT_DIR = Path("docs/assets/robots")

# Robot definitions: (canonical_name, scene_xml_relative_path, display_name, category)
# Derived from strands_robots/assets/__init__.py _ROBOT_MODELS registry
ROBOTS = [
    # Arms
    ("so100", "trs_so_arm100/scene.xml", "SO-ARM100", "arm"),
    ("so101", "robotstudio_so101/scene_box.xml", "SO-101", "arm"),
    ("koch", "low_cost_robot_arm/scene.xml", "Koch v1.1", "arm"),
    ("panda", "franka_emika_panda/scene.xml", "Franka Panda", "arm"),
    ("fr3", "franka_fr3/scene.xml", "Franka FR3", "arm"),
    ("ur5e", "universal_robots_ur5e/scene.xml", "UR5e", "arm"),
    ("kuka_iiwa", "kuka_iiwa_14/scene.xml", "KUKA iiwa 14", "arm"),
    ("kinova_gen3", "kinova_gen3/scene.xml", "Kinova Gen3", "arm"),
    ("xarm7", "ufactory_xarm7/scene.xml", "xArm 7", "arm"),
    ("vx300s", "trossen_vx300s/scene.xml", "ViperX 300s", "arm"),
    ("arx_l5", "arx_l5/scene.xml", "ARX L5", "arm"),
    ("piper", "agilex_piper/scene.xml", "AgileX Piper", "arm"),
    ("z1", "unitree_z1/scene.xml", "Unitree Z1", "arm"),
    # Bimanual
    ("aloha", "aloha/scene.xml", "ALOHA Bimanual", "bimanual"),
    ("trossen_wxai", "trossen_wxai/scene.xml", "Trossen WX-AI", "bimanual"),
    # Hands
    ("shadow_hand", "shadow_hand/scene_left.xml", "Shadow Hand", "hand"),
    ("leap_hand", "leap_hand/scene_left.xml", "LEAP Hand", "hand"),
    ("robotiq_2f85", "robotiq_2f85/scene.xml", "Robotiq 2F-85", "hand"),
    # Humanoids
    ("fourier_n1", "fourier_n1/scene.xml", "Fourier N1", "humanoid"),
    ("unitree_g1", "unitree_g1/scene.xml", "Unitree G1", "humanoid"),
    ("unitree_h1", "unitree_h1/scene.xml", "Unitree H1", "humanoid"),
    ("apollo", "apptronik_apollo/scene.xml", "Apptronik Apollo", "humanoid"),
    ("cassie", "agility_cassie/scene.xml", "Agility Cassie", "humanoid"),
    ("open_duck_mini", "open_duck_mini_v2/scene.xml", "Open Duck Mini", "humanoid"),
    ("asimov_v0", "asimov_v0/scene.xml", "Asimov V0", "humanoid"),
    # Expressive
    ("reachy_mini", "reachy_mini/mjcf/scene.xml", "Reachy Mini", "expressive"),
    # Mobile
    ("unitree_go2", "unitree_go2/scene.xml", "Unitree Go2", "mobile"),
    ("unitree_a1", "unitree_a1/scene.xml", "Unitree A1", "mobile"),
    ("spot", "boston_dynamics_spot/scene.xml", "Spot (with arm)", "mobile"),
    ("stretch3", "hello_robot_stretch_3/scene.xml", "Stretch 3", "mobile"),
    # Mobile Manipulation
    ("google_robot", "google_robot/scene.xml", "Google Robot", "mobile_manip"),
]

# Camera angles per category (azimuth, elevation, distance_scale)
# These provide aesthetically pleasing views for each robot type
CATEGORY_CAMERA = {
    "arm": {"azimuth": 135, "elevation": -25, "distance": 1.5},
    "bimanual": {"azimuth": 160, "elevation": -20, "distance": 2.0},
    "hand": {"azimuth": 120, "elevation": -30, "distance": 1.2},
    "humanoid": {"azimuth": 150, "elevation": -15, "distance": 2.5},
    "expressive": {"azimuth": 140, "elevation": -20, "distance": 1.5},
    "mobile": {"azimuth": 145, "elevation": -20, "distance": 2.0},
    "mobile_manip": {"azimuth": 155, "elevation": -20, "distance": 2.2},
}


def render_robot(scene_path: str, width: int = RENDER_WIDTH,
                 height: int = RENDER_HEIGHT, category: str = "arm") -> np.ndarray:
    """Render a robot from its scene XML using MuJoCo headless renderer.

    Args:
        scene_path: Path to scene.xml
        width: Render width in pixels
        height: Render height in pixels
        category: Robot category (for camera angle selection)

    Returns:
        RGB numpy array of shape (height, width, 3)
    """
    model = mj.MjModel.from_xml_path(scene_path)
    data = mj.MjData(model)

    # Set offscreen buffer to match our desired resolution
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)

    # Step physics once so everything settles
    mj.mj_step(model, data)

    renderer = mj.Renderer(model, height=height, width=width)

    # Check if scene has a named camera
    cam_id = -1
    for i in range(model.ncam):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
        if name:
            cam_id = i
            break

    if cam_id >= 0:
        renderer.update_scene(data, camera=cam_id)
    else:
        # Use free camera with category-appropriate angle
        _cam_cfg = CATEGORY_CAMERA.get(category, CATEGORY_CAMERA["arm"])
        _scene = renderer._scene
        renderer.update_scene(data)

    img = renderer.render().copy()
    del renderer
    return img


def add_label(img_array: np.ndarray, label: str, category: str) -> np.ndarray:
    """Add a text label to the bottom of an image."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # Category color mapping
    cat_colors = {
        "arm": (52, 152, 219),       # Blue
        "bimanual": (155, 89, 182),   # Purple
        "hand": (231, 76, 60),        # Red
        "humanoid": (46, 204, 113),   # Green
        "expressive": (241, 196, 15), # Yellow
        "mobile": (230, 126, 34),     # Orange
        "mobile_manip": (26, 188, 156), # Teal
    }
    color = cat_colors.get(category, (200, 200, 200))

    # Draw label bar at bottom
    bar_height = 40
    draw.rectangle(
        [(0, img.height - bar_height), (img.width, img.height)],
        fill=(20, 20, 30, 220),
    )

    # Category badge
    badge_text = category.upper().replace("_", " ")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    # Badge background
    badge_bbox = draw.textbbox((0, 0), badge_text, font=font_small)
    badge_w = badge_bbox[2] - badge_bbox[0] + 16
    badge_h = badge_bbox[3] - badge_bbox[1] + 8
    badge_x = 10
    badge_y = img.height - bar_height + (bar_height - badge_h) // 2
    draw.rounded_rectangle(
        [(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)],
        radius=4, fill=color
    )
    draw.text((badge_x + 8, badge_y + 3), badge_text, fill=(255, 255, 255), font=font_small)

    # Robot name
    draw.text(
        (badge_x + badge_w + 12, img.height - bar_height + 10),
        label, fill=(255, 255, 255), font=font
    )

    return np.array(img)


def create_gallery(renders: dict, cols: int = 6) -> Image.Image:
    """Create a grid gallery from all rendered robots.

    Args:
        renders: Dict of robot_name -> (image_array, display_name, category)
        cols: Number of columns in grid

    Returns:
        PIL Image of the gallery
    """
    thumb_w, thumb_h = 400, 225  # 16:9 thumbnails
    padding = 4
    rows = math.ceil(len(renders) / cols)

    gallery_w = cols * (thumb_w + padding) + padding
    gallery_h = rows * (thumb_h + padding) + padding + 60  # +60 for title

    gallery = Image.new("RGB", (gallery_w, gallery_h), (15, 15, 25))
    draw = ImageDraw.Draw(gallery)

    # Title
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except (IOError, OSError):
        title_font = ImageFont.load_default()

    draw.text((padding + 10, 15), "Strands Robots — 31 Bundled MuJoCo Models", fill=(255, 255, 255), font=title_font)

    for idx, (name, (img, display_name, category)) in enumerate(renders.items()):
        row = idx // cols
        col = idx % cols
        x = padding + col * (thumb_w + padding)
        y = 60 + padding + row * (thumb_h + padding)

        # Resize to thumbnail
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((thumb_w, thumb_h), Image.LANCZOS)

        gallery.paste(pil_img, (x, y))

    return gallery


def main():
    """Main entry point: render all robots and create gallery."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assets_dir = Path("strands_robots/assets")

    results = {}
    metadata = []
    rendered = 0
    failed = 0
    skipped = 0

    logger.info(f"🤖 Rendering {len(ROBOTS)} robot models...")
    logger.info(f"📁 Output: {OUTPUT_DIR}/")
    logger.info(f"📐 Resolution: {RENDER_WIDTH}×{RENDER_HEIGHT}")
    logger.info("")

    for name, scene_rel, display_name, category in ROBOTS:
        scene_path = assets_dir / scene_rel

        if not scene_path.exists():
            logger.info(f"  ⏭️  {name:<20} — scene not found ({scene_rel}), skipping")
            skipped += 1
            metadata.append({
                "name": name, "display_name": display_name,
                "category": category, "status": "skipped",
                "reason": "scene_not_found",
            })
            continue

        try:
            t0 = time.time()
            img = render_robot(str(scene_path), RENDER_WIDTH, RENDER_HEIGHT, category)
            labeled_img = add_label(img, display_name, category)

            out_path = OUTPUT_DIR / f"{name}.png"
            Image.fromarray(labeled_img).save(out_path, optimize=True)

            elapsed = time.time() - t0
            fsize = out_path.stat().st_size
            rendered += 1

            results[name] = (labeled_img, display_name, category)
            metadata.append({
                "name": name, "display_name": display_name,
                "category": category, "status": "rendered",
                "file": str(out_path), "size_bytes": fsize,
                "render_time_s": round(elapsed, 2),
                "resolution": f"{RENDER_WIDTH}×{RENDER_HEIGHT}",
            })

            logger.info(f"  ✅ {name:<20} — {fsize/1024:.0f} KB ({elapsed:.1f}s)")

        except Exception as e:
            failed += 1
            logger.info(f"  ❌ {name:<20} — {e}")
            metadata.append({
                "name": name, "display_name": display_name,
                "category": category, "status": "failed",
                "error": str(e),
            })

    logger.info("")
    logger.info(f"📊 Results: {rendered} rendered, {skipped} skipped, {failed} failed")

    # Create gallery if we have renders
    if results:
        logger.info("🖼️  Creating gallery grid...")
        gallery = create_gallery(results, cols=6)
        gallery_path = OUTPUT_DIR / "gallery.png"
        gallery.save(gallery_path, optimize=True)
        logger.info(f"  ✅ Gallery saved: {gallery_path} ({gallery_path.stat().st_size / 1024:.0f} KB)")

    # Save metadata
    meta_path = OUTPUT_DIR / "renders.json"
    meta_doc = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "renderer": "MuJoCo OSMesa (headless)",
        "resolution": f"{RENDER_WIDTH}×{RENDER_HEIGHT}",
        "total_robots": len(ROBOTS),
        "rendered": rendered,
        "skipped": skipped,
        "failed": failed,
        "robots": metadata,
    }
    with open(meta_path, "w") as f:
        json.dump(meta_doc, f, indent=2)
    logger.info(f"  ✅ Metadata saved: {meta_path}")

    return rendered, skipped, failed


if __name__ == "__main__":
    rendered, skipped, failed = main()
    sys.exit(1 if failed > 0 and rendered == 0 else 0)
