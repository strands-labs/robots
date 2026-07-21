"""Robot Asset Manager for Strands Robots Simulation.

Assets are resolved from ``robot_descriptions`` package or downloaded from
MuJoCo Menagerie GitHub, cached in ``~/.strands_robots/assets/``.
Override with ``STRANDS_ASSETS_DIR`` env var.

Implementation lives in :mod:`~strands_robots.assets.manager` - this file is thin
exports only.
"""

from strands_robots.assets.manager import (
    get_robot_info,
    list_available_robots,
    resolve_model_dir,
    resolve_model_path,
)
from strands_robots.registry import (
    format_robot_table,
    get_robot,
    list_aliases,
    list_robots,
    list_robots_by_category,
)
from strands_robots.registry import (
    resolve_name as resolve_robot_name,
)
from strands_robots.utils import get_assets_dir, get_search_paths

__all__ = [
    "resolve_model_path",
    "resolve_model_dir",
    "resolve_robot_name",
    "get_robot_info",
    "list_available_robots",
    "list_robots_by_category",
    "list_aliases",
    "format_robot_table",
    "get_assets_dir",
    "get_search_paths",
    "get_robot",
    "list_robots",
]
