"""Download robot model assets - Strands Agent ``@tool`` wrapper.

Thin wrapper around :mod:`strands_robots.assets.download` that exposes
``download_robots()`` as an agent tool.  All download logic lives in the
``assets.download`` module; this file only handles input parsing and
output formatting for the Strands Agent SDK.
"""

from __future__ import annotations

import logging
from typing import Any

from strands.tools.decorator import tool

from strands_robots.assets.download import download_robots, get_user_assets_dir
from strands_robots.assets.manager import list_available_robots
from strands_robots.registry import format_robot_table

logger = logging.getLogger(__name__)


@tool
def download_assets(
    action: str = "download",
    robots: str | None = None,
    category: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Download and manage robot model assets (MJCF XML + meshes).

    Assets are sourced from ``robot_descriptions`` (recommended by MuJoCo
    Menagerie, requires ``pip install strands-robots[sim-mujoco]``).  When
    ``robot_descriptions`` is unavailable, falls back to a shallow
    ``git clone`` of the Menagerie repo.  Robots with a custom GitHub
    source in the registry are cloned from their respective repos.

    Downloaded assets are cached in ``~/.strands_robots/assets/``
    (override with ``STRANDS_ASSETS_DIR``).

    Args:
        action: ``download`` | ``list`` | ``status``.  ``status`` marks each
            robot ``[ok]`` (assets present) or ``[--]`` (missing).
        robots: Comma-separated names (e.g. ``so100,panda``). Omit for all.
        category: Filter: arm, bimanual, hand, humanoid, mobile, mobile_manip
        force: Re-download even if present
    """
    try:
        if action == "list":
            return {
                "status": "success",
                "content": [{"text": f"Available Robots:\n\n{format_robot_table()}"}],
            }

        if action == "status":
            robots_info = list_available_robots()
            available = sum(1 for r in robots_info if r["available"])
            lines = [f"{available} available, {len(robots_info) - available} missing"]
            lines.extend(
                f"{'[ok]' if r['available'] else '[--]'} {r['name']:<20s} {r['category']:<12s} {r['description']}"
                for r in robots_info
            )
            lines.append(f"\nCache: {get_user_assets_dir()}")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        if action == "download":
            robot_names = [r.strip() for r in robots.split(",") if r.strip()] if robots else None
            result = download_robots(names=robot_names, category=category, force=force)
            parts = [
                f"Downloaded: {result['downloaded']}, Skipped: {result['skipped']}, Failed: {result['failed']}",
                f"Method: {result.get('method', '?')}",
            ]
            if result.get("failed_details"):
                parts.extend(f"   {n}: {r}" for n, r in result["failed_details"].items())
            parts.append(f"Assets: {result.get('assets_dir', '?')}")
            return {"status": "success", "content": [{"text": "\n".join(parts)}]}

        return {
            "status": "error",
            "content": [{"text": f"Unknown action: {action}. Valid: download, list, status"}],
        }

    except Exception as exc:
        logger.error("download_assets error: %s", exc)
        return {"status": "error", "content": [{"text": f"Error: {exc}"}]}
