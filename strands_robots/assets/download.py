"""Download robot model assets via ``robot_descriptions`` or custom GitHub repos.

This module contains the core download logic for robot assets.
The ``strands_robots.tools.download_assets`` tool is a thin ``@tool`` wrapper
that delegates to :func:`download_robots` here.

Strategy (in order of preference):
    1. ``robot_descriptions`` package - recommended by MuJoCo Menagerie.
    2. Shallow ``git clone`` fallback for Menagerie robots.
    3. Custom GitHub repos for non-Menagerie robots.

Assets are cached in ``~/.strands_robots/assets/`` (override with
``STRANDS_ASSETS_DIR``).  Install the optional dependency::

    pip install strands-robots[sim-mujoco]   # includes robot_descriptions
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..registry import get_robot
from ..registry import list_robots as registry_list_robots
from ..registry import resolve_name as resolve_robot_name
from ..utils import get_assets_dir, get_search_paths, safe_join

logger = logging.getLogger(__name__)

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"

# Only HTTPS GitHub URLs are allowed for cloning.
_ALLOWED_CLONE_URL_RE = re.compile(r"^https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+\.git$")


# robot_descriptions integration


def _robot_descriptions_available() -> bool:
    """Check if ``robot_descriptions`` is installed."""
    try:
        import robot_descriptions  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_robot_descriptions_module(name: str, info: dict) -> str | None:
    """Resolve the ``robot_descriptions`` module name for a robot.

    Uses the ``robot_descriptions_module`` field from the registry (O(1)),
    with a lightweight naming-convention fallback for unregistered robots.

    Args:
        name: Canonical robot name.
        info: Robot registry entry.

    Returns:
        Module name (e.g. ``panda_mj_description``) or ``None``.
    """
    asset = info.get("asset", {})

    # Explicit opt-out: robot declares it has no robot_descriptions module
    if asset.get("auto_download") is False:
        return None

    # Primary: explicit registry entry (preferred, O(1))
    module_name: str | None = asset.get("robot_descriptions_module")
    if module_name:
        return str(module_name)

    # Fallback: try common naming conventions (max 3 imports)
    asset_dir = info.get("asset", {}).get("dir", "")
    candidates = [
        f"{asset_dir}_mj_description",
        f"{name}_mj_description",
        f"{name}_description",
    ]
    for candidate in candidates:
        # Allow '+' so robot_descriptions modules like 'tiago++_mj_description'
        # are accepted (still no '/', '.', or whitespace -> no import traversal).
        if not re.match(r"^[a-z0-9_+]+$", candidate):
            continue
        try:
            importlib.import_module(f"robot_descriptions.{candidate}")
            logger.warning(
                "Resolved '%s' via naming heuristic → '%s'. "
                "Consider adding 'robot_descriptions_module' to the registry.",
                name,
                candidate,
            )
            return candidate
        except ImportError:
            continue

    return None


#: Alias for backward compatibility - use :func:`strands_robots.utils.get_assets_dir`.
get_user_assets_dir = get_assets_dir


def _needs_download(name: str, info: dict[str, Any] | None, force: bool = False) -> bool:
    """Return *True* if a robot's mesh files are missing."""
    if info is None:
        return False
    asset = info.get("asset", {})
    if not asset:
        return False

    xml_file, asset_dir = asset["model_xml"], asset["dir"]

    for search_dir in get_search_paths():
        model_path = search_dir / asset_dir / xml_file
        if not model_path.exists():
            continue
        try:
            content = model_path.read_text()
            mesh_files = re.findall(r'file="([^"]+\.(?:stl|STL|obj|OBJ|msh))"', content)
            if not mesh_files:
                return False
            meshdir_match = re.search(r'meshdir="([^"]*)"', content)
            meshdir = meshdir_match.group(1) if meshdir_match else ""
            for mesh in mesh_files:
                if not (model_path.parent / meshdir / mesh).exists():
                    return True
            return force
        except Exception:
            return True

    return True


def _get_source(info: dict[str, Any] | None) -> dict[str, Any]:
    """Get download source for a robot.  Defaults to ``menagerie``."""
    if info is None:
        return {"type": "menagerie"}
    source = info.get("asset", {}).get("source", {})
    return source if source else {"type": "menagerie"}


def _shallow_clone(repo_url: str, dest: str, *, timeout: int = 120) -> None:
    """Shallow-clone *repo_url* into *dest*.

    Only HTTPS ``github.com`` URLs are accepted - ``ssh://``, ``git://``,
    ``file://``, and other schemes are rejected to prevent command-injection
    and SSRF risks.

    Raises:
        ValueError: If *repo_url* does not match the allowed HTTPS GitHub pattern.
        subprocess.CalledProcessError: If the ``git clone`` command fails.
        subprocess.TimeoutExpired: If the clone exceeds *timeout* seconds.
    """
    if not _ALLOWED_CLONE_URL_RE.match(repo_url):
        raise ValueError(f"Blocked clone URL (only HTTPS github.com allowed): {repo_url!r}")
    logger.info("Cloning %s (this may take a moment)...", repo_url)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, dest],
        check=True,
        capture_output=True,
        timeout=timeout,
    )


# Filenames/patterns that are safe to strip from an upstream source tree before
# we copy it into the user's asset cache.  Filtering at *copy* time (rather than
# deleting afterwards) means we never touch files that may already exist in *dst*
# - which matters when the user keeps notes/README alongside assets.
_COPY_CLEAN_SKIP = frozenset({"README.md", "LICENSE", "CHANGELOG.md"})
_COPY_CLEAN_SUFFIX = (".png", ".jpg", ".jpeg")


def _copy_and_clean(src: Path, dst: Path) -> None:
    """Copy *src* tree to *dst*, skipping non-essential files at copy time.

    Previous implementation deleted matching files from *dst* after copytree,
    which meant a user's own ``README.md`` in the destination could be wiped.
    This version filters on read so only files from *src* are dropped.
    """

    def _ignore(_dir: str, names: list[str]) -> list[str]:
        return [
            n for n in names if n in _COPY_CLEAN_SKIP or n.lower().endswith(_COPY_CLEAN_SUFFIX) or n.startswith(".git")
        ]

    shutil.copytree(str(src), str(dst), dirs_exist_ok=True, ignore=_ignore)


def _download_via_robot_descriptions(robots: dict[str, dict], dest_dir: Path) -> dict[str, str]:
    """Download robots using the ``robot_descriptions`` package.

    Imports only the specific module for each robot (O(1) per robot),
    using the ``robot_descriptions_module`` field from the registry.
    The import triggers the upstream clone on first use, then we symlink
    ``PACKAGE_PATH`` into our asset cache.
    """
    results: dict[str, str] = {}
    if not robots:
        return results

    for name, info in robots.items():
        asset_dir = info["asset"]["dir"]
        module_name = _resolve_robot_descriptions_module(name, info)
        if module_name is None:
            results[name] = "skipped: no robot_descriptions module found"
            continue
        # Allow '+' so modules like 'tiago++_mj_description' pass (the upstream
        # robot_descriptions package legitimately uses '++' in some names).
        if not re.match(r"^[a-z0-9_+]+$", module_name):
            results[name] = f"skipped: invalid module name: {module_name}"
            continue

        try:
            mod = importlib.import_module(f"robot_descriptions.{module_name}")
            package_path = Path(mod.PACKAGE_PATH)
            if not package_path.exists():
                results[name] = f"failed: PACKAGE_PATH missing: {package_path}"
                continue

            dst = safe_join(dest_dir, asset_dir)
            if dst.is_symlink() and dst.resolve() == package_path.resolve():
                # Validate existing symlink still has the expected XML
                expected_xml = dst / info["asset"]["model_xml"]
                if expected_xml.exists():
                    results[name] = "downloaded"
                    continue
                # Stale symlink - remove and re-download via git
                dst.unlink()
                results[name] = f"failed: stale symlink - {info['asset']['model_xml']} not found in {package_path}"
                continue
            if dst.exists() or dst.is_symlink():
                dst.unlink() if dst.is_symlink() else shutil.rmtree(str(dst))

            try:
                dst.symlink_to(package_path)
            except OSError:
                shutil.copytree(str(package_path), str(dst), dirs_exist_ok=True)

            # Validate: expected XML must exist in the linked/copied dir
            expected_xml = dst / info["asset"]["model_xml"]
            if not expected_xml.exists():
                logger.warning(
                    "robot_descriptions module '%s' linked for %s but "
                    "expected XML '%s' not found - falling back to git",
                    module_name,
                    name,
                    info["asset"]["model_xml"],
                )
                if dst.is_symlink():
                    dst.unlink()
                else:
                    shutil.rmtree(str(dst), ignore_errors=True)
                results[name] = (
                    f"failed: XML mismatch - module '{module_name}' does not contain {info['asset']['model_xml']}"
                )
                continue

            results[name] = "downloaded"
        except Exception as exc:
            results[name] = f"failed: {exc}"
            logger.warning("robot_descriptions failed for %s: %s", name, exc)

    return results


def _download_via_git(robots: dict[str, dict], dest_dir: Path) -> dict[str, str]:
    """Fallback: shallow-clone Menagerie and copy robot directories."""
    results: dict[str, str] = {}
    if not robots:
        return results

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "mujoco_menagerie")
        try:
            _shallow_clone(MENAGERIE_REPO, clone_dir)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as exc:
            reason = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)[:100]
            return {n: f"failed: git clone {reason}" for n in robots}

        for name, info in robots.items():
            asset_dir = info["asset"]["dir"]
            src = safe_join(Path(clone_dir), asset_dir)
            if not src.exists():
                results[name] = f"failed: {asset_dir} not in menagerie"
                continue
            try:
                _copy_and_clean(src, safe_join(dest_dir, asset_dir))
                results[name] = "downloaded"
            except Exception as exc:
                results[name] = f"failed: {exc}"

    return results


def _download_from_github(name: str, info: dict, dest_dir: Path) -> str:
    """Download a robot from a custom GitHub repo (``asset.source``)."""
    source = info["asset"]["source"]
    repo = source["repo"]
    if not re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", repo):
        return f"failed: invalid repo format: {repo}"

    subdir = source.get("subdir", "")
    asset_dir = info["asset"]["dir"]

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "repo")
        try:
            # URL validation is enforced inside _shallow_clone itself
            _shallow_clone(f"https://github.com/{repo}.git", clone_dir)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as exc:
            reason = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)[:100]
            return f"failed: git clone {reason}"

        src = Path(clone_dir) / subdir if subdir else Path(clone_dir)
        if not src.exists():
            return f"failed: subdir '{subdir}' not found in {repo}"

        dst = safe_join(dest_dir, asset_dir)
        try:
            _copy_and_clean(src, dst)
            return "downloaded"
        except Exception as exc:
            return f"failed: {exc}"


# Orchestrator


def auto_download_robot(name: str, info: dict[str, Any]) -> bool:
    """Auto-download a single robot's assets.

    Called by :func:`strands_robots.assets.manager.resolve_model_path` when
    XML is present but meshes are missing.  Tries ``robot_descriptions``
    first, then custom GitHub source if specified in the registry entry.

    Args:
        name: Robot name (canonical or alias).
        info: Registry entry for the robot.

    Returns:
        ``True`` if a download attempt succeeded, ``False`` otherwise.
    """
    dest_dir = get_assets_dir()
    canonical = resolve_robot_name(name)

    # Try robot_descriptions first (covers most Menagerie robots)
    if _robot_descriptions_available():
        results = _download_via_robot_descriptions({canonical: info}, dest_dir)
        if results.get(canonical, "").startswith("downloaded"):
            logger.info("Auto-downloaded %s via robot_descriptions", canonical)
            return True

    # Fall back to custom GitHub source
    source = info.get("asset", {}).get("source", {})
    if source.get("type") == "github":
        result = _download_from_github(canonical, info, dest_dir)
        if result.startswith("downloaded"):
            logger.info("Auto-downloaded %s from GitHub", canonical)
            return True

    return False


def download_robots(
    names: list[str] | None = None,
    category: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Download robot model assets from their respective sources.

    Strategy (in order of preference):
      1. ``robot_descriptions`` package - recommended by MuJoCo Menagerie.
      2. Shallow ``git clone`` fallback for Menagerie robots.
      3. Custom GitHub repos for non-Menagerie robots.

    Args:
        names: Robot names to download (``None`` = all sim robots).
        category: Filter by category (arm, humanoid, mobile, ...).
        force: Re-download even if present.

    Returns:
        Dict with downloaded/skipped/failed counts, names, and details.
    """
    dest_dir = get_user_assets_dir()
    # Filter None values - get_robot() can return None for unknown names
    all_sim: dict[str, dict[str, Any]] = {
        r["name"]: info for r in registry_list_robots(mode="sim") if (info := get_robot(r["name"])) is not None
    }

    # Resolve requested robots
    if names:
        robots: dict[str, dict[str, Any]] = {}
        for name in names:
            canonical = resolve_robot_name(name)
            if canonical in all_sim:
                robots[canonical] = all_sim[canonical]
            else:
                logger.warning("Unknown robot: %s (resolved: %s)", name, canonical)
    elif category:
        robots = {n: i for n, i in all_sim.items() if i.get("category") == category}
    else:
        robots = dict(all_sim)

    if not robots:
        return {"downloaded": 0, "skipped": 0, "failed": 0, "message": "No matching robots found."}

    # Partition: needs download vs already present
    to_download: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for name, info in robots.items():
        if _needs_download(name, info, force):
            to_download[name] = info
        else:
            skipped.append(name)

    if not to_download:
        return {
            "downloaded": 0,
            "skipped": len(skipped),
            "failed": 0,
            "skipped_names": skipped,
            "message": f"All {len(robots)} robots already have assets. Use force=True to re-download.",
        }

    # Partition by source type
    menagerie_robots: dict[str, Any] = {}
    github_robots: dict[str, Any] = {}
    for name, info in to_download.items():
        source = _get_source(info)
        bucket = github_robots if source["type"] == "github" else menagerie_robots
        bucket[name] = info

    # Download Menagerie robots (robot_descriptions → git fallback)
    results: dict[str, str] = {}
    if menagerie_robots:
        if _robot_descriptions_available():
            results.update(_download_via_robot_descriptions(menagerie_robots, dest_dir))
            # Retry failures with git clone
            retry = {
                n: menagerie_robots[n] for n, r in results.items() if r.startswith("failed") or r.startswith("skipped")
            }
            if retry:
                results.update(_download_via_git(retry, dest_dir))
        else:
            results.update(_download_via_git(menagerie_robots, dest_dir))

    # Download custom GitHub robots
    for name, info in github_robots.items():
        results[name] = _download_from_github(name, info, dest_dir)

    downloaded = [n for n, r in results.items() if r == "downloaded"]
    failed = {n: r for n, r in results.items() if r != "downloaded"}
    method = "robot_descriptions" if _robot_descriptions_available() else "git clone"

    return {
        "downloaded": len(downloaded),
        "skipped": len(skipped),
        "failed": len(failed),
        "downloaded_names": downloaded,
        "skipped_names": skipped,
        "failed_names": list(failed),
        "failed_details": failed,
        "assets_dir": str(dest_dir),
        "method": method,
        "message": (f"{len(downloaded)} downloaded ({method}), {len(skipped)} already present, {len(failed)} failed."),
    }
