"""Download robot model assets from MuJoCo Menagerie.

Works as both a CLI tool and a Strands AgentTool:

CLI:
    python -m strands_robots.assets.download
    python -m strands_robots.assets.download so100 panda unitree_g1
    python -m strands_robots.assets.download --category arm
    python -m strands_robots.assets.download --list

Agent:
    from strands_robots.assets.download import download_assets
    agent = Agent(tools=[download_assets])
    agent("Download the SO-100 and Panda robot assets")
    agent("List all available robot models")
    agent("Download all humanoid robots")
"""

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

# Lazy import: strands may not be installed in test/CLI environments
try:
    from strands.tools.decorator import tool
except ImportError:
    # Fallback: no-op decorator when strands is not available
    def tool(f):  # type: ignore[misc]
        return f


from . import _ROBOT_MODELS, format_robot_table, get_assets_dir, resolve_robot_name

logger = logging.getLogger(__name__)

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"


def download_robots(names=None, category=None, force=False) -> Dict[str, Any]:
    """Download robot models from MuJoCo Menagerie.

    Args:
        names: List of robot names to download (None = all)
        category: Filter by category (arm, bimanual, hand, humanoid, mobile, mobile_manip)
        force: Re-download even if already present

    Returns:
        Dict with downloaded/skipped/failed counts and details
    """
    assets_dir = get_assets_dir()

    # Resolve aliases
    if names:
        resolved_names = []
        for n in names:
            canonical = resolve_robot_name(n)
            if canonical in _ROBOT_MODELS:
                resolved_names.append(canonical)
            else:
                logger.warning("Unknown robot: %s (resolved to: %s)", n, canonical)
        robots = {n: _ROBOT_MODELS[n] for n in resolved_names}
    elif category:
        robots = {n: m for n, m in _ROBOT_MODELS.items() if m["category"] == category}
    else:
        robots = dict(_ROBOT_MODELS)

    if not robots:
        return {
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "message": "No matching robots found.",
        }

    # Check which need downloading
    to_download = {}
    skipped = []
    for name, info in robots.items():
        robot_dir = assets_dir / info["dir"]
        model_file = robot_dir / info["model_xml"]
        if force or not model_file.exists():
            to_download[name] = info
        else:
            # Also check if mesh files referenced in the model XML exist
            needs_meshes = False
            try:
                import re

                content = model_file.read_text()
                mesh_files = re.findall(r'file="([^"]+\.(?:stl|STL|obj))"', content)
                if mesh_files:
                    # Resolve meshdir from compiler tag
                    meshdir_match = re.search(r'meshdir="([^"]*)"', content)
                    meshdir = meshdir_match.group(1) if meshdir_match else ""
                    for mf in mesh_files[:3]:  # Check first 3 mesh files
                        mesh_path = robot_dir / meshdir / mf
                        if not mesh_path.exists():
                            needs_meshes = True
                            break
            except Exception:
                pass

            if needs_meshes:
                to_download[name] = info
            else:
                skipped.append(name)

    if not to_download:
        return {
            "downloaded": 0,
            "skipped": len(skipped),
            "failed": 0,
            "skipped_names": skipped,
            "message": f"All {len(robots)} robots already downloaded. Use force=True to re-download.",
        }

    logger.info("Downloading %d robot(s) from MuJoCo Menagerie...", len(to_download))
    logger.info("Downloading %d robots: %s", len(to_download), list(to_download.keys()))

    downloaded = []
    failed = []

    # Clone menagerie to temp dir (shallow)
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "mujoco_menagerie")
        logger.info("Cloning %s...", MENAGERIE_REPO)

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", MENAGERIE_REPO, clone_dir],
                check=True,
                capture_output=True,
                timeout=120,  # 2 minute timeout to avoid CI hangs
            )
        except subprocess.TimeoutExpired:
            return {
                "downloaded": 0,
                "skipped": len(skipped),
                "failed": len(to_download),
                "message": "Git clone timed out after 120s. Check network connectivity.",
            }
        except subprocess.CalledProcessError as e:
            return {
                "downloaded": 0,
                "skipped": len(skipped),
                "failed": len(to_download),
                "message": f"Failed to clone MuJoCo Menagerie: {e.stderr.decode()[:200]}",
            }

        # Copy each robot
        for name, info in to_download.items():
            src = Path(clone_dir) / info["dir"]
            dst = assets_dir / info["dir"]

            if not src.exists():
                logger.warning("%s: source dir not found (%s)", name, info['dir'])
                failed.append(name)
                continue

            try:
                if dst.exists() and force:
                    shutil.rmtree(dst)

                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)

                # Remove non-essential files
                for pattern in ["README.md", "LICENSE", "*.png", "*.jpg"]:
                    for f in dst.glob(pattern):
                        f.unlink()

                downloaded.append(name)
                logger.info("Downloaded %s (%s)", name, info['description'])
            except Exception as e:
                failed.append(name)
                logger.error("Failed to download %s: %s", name, e)

    logger.info(
        "Done! %d downloaded, %d skipped, %d failed.", len(downloaded), len(skipped), len(failed)
    )
    logger.info("Assets saved to: %s", assets_dir)

    return {
        "downloaded": len(downloaded),
        "skipped": len(skipped),
        "failed": len(failed),
        "downloaded_names": downloaded,
        "skipped_names": skipped,
        "failed_names": failed,
        "assets_dir": str(assets_dir),
        "message": f"{len(downloaded)} downloaded, {len(skipped)} already present, {len(failed)} failed.",
    }


@tool
def download_assets(
    action: str = "download",
    robots: str = None,
    category: str = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Download and manage robot model assets from MuJoCo Menagerie.

    32 robots available: arms (SO-100, Panda, UR5e, ...), bimanual (ALOHA),
    hands (Shadow, LEAP), humanoids (Unitree G1, H1, Fourier N1),
    mobile (Spot, Go2), and more.

    Args:
        action: Action to perform:
            - "download": Download robot model files (MJCF XML + meshes)
            - "list": List all available robots with status
            - "status": Show download status of all robots
        robots: Comma-separated robot names to download (e.g. "so100,panda,unitree_g1").
                Supports aliases (e.g. "so100_dualcam" resolves to "so100").
                If not specified, downloads all robots.
        category: Filter by category: arm, bimanual, hand, humanoid, mobile, mobile_manip
        force: Force re-download even if already present

    Returns:
        Dict with status and content

    Examples:
        download_assets(action="list")
        download_assets(action="download", robots="so100,panda")
        download_assets(action="download", category="humanoid")
        download_assets(action="download", force=True)
        download_assets(action="status")
    """
    try:
        if action == "list":
            table = format_robot_table()
            return {
                "status": "success",
                "content": [{"text": f"🤖 Available Robot Models:\n\n{table}"}],
            }

        elif action == "status":
            from . import list_available_robots

            robots_info = list_available_robots()
            available = sum(1 for r in robots_info if r["available"])
            missing = sum(1 for r in robots_info if not r["available"])

            lines = [f"📊 Asset Status: {available} available, {missing} missing\n"]
            for r in robots_info:
                icon = "✅" if r["available"] else "❌"
                lines.append(
                    f"  {icon} {r['name']:<20s} {r['category']:<12s} {r['description']}"
                )

            if missing > 0:
                lines.append(
                    "\n💡 Download missing: download_assets(action='download')"
                )

            return {
                "status": "success",
                "content": [{"text": "\n".join(lines)}],
            }

        elif action == "download":
            # Parse robot names
            robot_names = None
            if robots:
                robot_names = [r.strip() for r in robots.split(",") if r.strip()]

            result = download_robots(
                names=robot_names,
                category=category,
                force=force,
            )

            text = "📦 Download Complete\n\n"
            text += f"✅ Downloaded: {result['downloaded']}\n"
            text += f"⏭️  Skipped (already present): {result['skipped']}\n"
            text += f"❌ Failed: {result['failed']}\n"

            if result.get("downloaded_names"):
                text += f"\nNewly downloaded: {', '.join(result['downloaded_names'])}"
            if result.get("failed_names"):
                text += f"\nFailed: {', '.join(result['failed_names'])}"
            if result.get("assets_dir"):
                text += f"\n\n📁 Assets directory: {result['assets_dir']}"

            return {
                "status": "success",
                "content": [{"text": text}],
            }

        else:
            return {
                "status": "error",
                "content": [
                    {"text": f"Unknown action: {action}. Valid: download, list, status"}
                ],
            }

    except Exception as e:
        logger.error("download_assets error: %s", e)
        return {
            "status": "error",
            "content": [{"text": f"❌ Error: {str(e)}"}],
        }


def main():
    parser = argparse.ArgumentParser(
        description="Download robot assets from MuJoCo Menagerie"
    )
    parser.add_argument(
        "robots", nargs="*", help="Robot names to download (default: all)"
    )
    parser.add_argument(
        "--category",
        "-c",
        choices=["arm", "bimanual", "hand", "humanoid", "mobile", "mobile_manip"],
    )
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available robots"
    )

    args = parser.parse_args()

    if args.list:
        print(format_robot_table())
        return

    download_robots(
        names=args.robots if args.robots else None,
        category=args.category,
        force=args.force,
    )
