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
from ..utils import boolean_flag_error, get_assets_dir, get_search_paths, safe_join

logger = logging.getLogger(__name__)

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"

# Only HTTPS GitHub URLs are allowed for cloning.
_ALLOWED_CLONE_URL_RE = re.compile(r"^https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+\.git\Z")


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
        if not re.match(r"^[a-z0-9_+]+\Z", candidate):
            continue
        try:
            importlib.import_module(f"robot_descriptions.{candidate}")
            logger.warning(
                "Resolved '%s' via naming heuristic -> '%s'. "
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


def _mjcf_mesh_subdir(*contents: str) -> str:
    """Return the mesh search subdirectory declared by MJCF text.

    MuJoCo's ``<compiler>`` element offers two attributes for this. ``meshdir``
    names the mesh directory specifically; ``assetdir`` names the mesh AND
    texture directories at once, and ``meshdir`` overrides it where both appear.
    Both are model-global: they apply across ``<include>``, so the fragment
    declaring the directory need not be the fragment declaring the mesh.

    A reader that knows only ``meshdir`` resolves an ``assetdir`` model against
    the model directory itself, so it reports a mesh that is present as absent.

    Args:
        *contents: MJCF text fragments making up one model, in any order.

    Returns:
        The declared subdirectory, or ``""`` when no fragment declares one.
    """
    for attr in ("meshdir", "assetdir"):
        for content in contents:
            if match := re.search(rf'{attr}="([^"]*)"', content):
                return match.group(1)
    return ""


def _mjcf_mesh_candidates(mesh_ref: str, model_dir: str, mesh_subdir: str, include_dir: str = "") -> list[str]:
    """Return the on-disk locations MuJoCo accepts for one mesh reference.

    MuJoCo resolves a ``<mesh file=...>`` against the MAIN model file's
    directory joined with the model's mesh subdirectory - never against the
    directory of whichever ``<include>``d fragment happened to declare it. When
    the declaring fragment lives in a subdirectory of the model, MuJoCo also
    accepts the reference relative to that fragment's directory, so a reference
    is satisfied by either location.

    Both branches are load-bearing on shipped Menagerie assets: ``skydio_x2``
    and ``stretch3`` place their meshes under the first, ``lekiwi`` under the
    second. Resolving against the declaring fragment's directory instead - a
    location MuJoCo rejects - reports a present mesh as absent.

    Args:
        mesh_ref: The ``file=`` value as authored, e.g. ``meshes/base.stl``.
        model_dir: Directory of the main model file.
        mesh_subdir: Subdirectory from :func:`_mjcf_mesh_subdir`.
        include_dir: Directory of the declaring fragment, relative to
            *model_dir*. Empty when the main file declares the mesh itself.

    Returns:
        Candidate absolute paths; the reference is present if any one exists.
    """
    base = os.path.join(model_dir, mesh_subdir)
    candidates = [os.path.join(base, mesh_ref)]
    if include_dir:
        candidates.append(os.path.join(base, include_dir, mesh_ref))
    return candidates


#: Mesh file extensions an MJCF ``file=`` reference may name. MuJoCo loads
#: ``.stl``, ``.obj`` and ``.msh``; the case variants appear in shipped assets.
_MESH_REF_RE = re.compile(r'file="([^"]+\.(?:stl|STL|obj|OBJ|msh))"')

#: ``<include file="...">`` - the fragments that make up one model.
_INCLUDE_RE = re.compile(r'<include\s+file="([^"]+)"')


def _mjcf_missing_meshes(model_path: str | os.PathLike[str]) -> list[str]:
    """Return the mesh references a model declares that are absent on disk.

    This is the single owner of "are this model's meshes on disk?". Two callers
    ask it - :func:`_needs_download` (should the assets be fetched?) and
    :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine._ensure_meshes`
    (may ``add_robot`` proceed?) - and one owner is what keeps them from judging
    the same model present and absent.

    A model is its main file plus every ``<include>``d fragment, because both
    halves of the rule span fragments:

    * ``<compiler meshdir|assetdir>`` is model-global (see
      :func:`_mjcf_mesh_subdir`), so the fragment declaring the directory need
      not be the one declaring the mesh;
    * a fragment may declare meshes the main file never mentions - shipped
      Menagerie assets do exactly this (``ability_hand``'s scene declares none
      of its 13 meshes; its included hand fragment declares all of them plus
      the ``meshdir``). Reading the main file alone reports such a model as
      declaring no meshes at all, which is indistinguishable from a model whose
      meshes are all present.

    Each reference is resolved the way MuJoCo resolves it, via
    :func:`_mjcf_mesh_candidates`.

    Args:
        model_path: Path to the model's MAIN file (the one the registry names).

    Returns:
        The absent references, as authored. Empty when the model declares no
        meshes or every one of them resolves - the two cases callers may treat
        alike, since neither has anything to fetch.

    Raises:
        OSError: The main file cannot be read.
        UnicodeDecodeError: The main file is not decodable text. An unreadable
            *fragment* is not an error: it declares no reference this scan can
            check, and MuJoCo names it on the load that follows.
    """
    model_dir = os.path.dirname(os.path.abspath(os.fspath(model_path)))
    main = Path(model_path).read_text(encoding="utf-8")

    # (fragment directory relative to model_dir, fragment text)
    fragments: list[tuple[str, str]] = [("", main)]
    for inc in _INCLUDE_RE.findall(main):
        inc_path = os.path.join(model_dir, inc)
        try:
            text = Path(inc_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = os.path.relpath(os.path.dirname(os.path.abspath(inc_path)), model_dir)
        fragments.append(("" if rel == os.curdir else rel, text))

    mesh_subdir = _mjcf_mesh_subdir(*(text for _rel, text in fragments))
    missing: list[str] = []
    for rel_dir, text in fragments:
        for ref in _MESH_REF_RE.findall(text):
            if not any(os.path.exists(c) for c in _mjcf_mesh_candidates(ref, model_dir, mesh_subdir, rel_dir)):
                missing.append(ref)
    return missing


def _needs_download(name: str, info: dict[str, Any] | None, force: bool = False) -> bool:
    """Return *True* if a robot's mesh files are missing.

    ``force`` re-fetches a model whose meshes are all present. A model with
    nothing missing is the only case ``force`` decides: a missing reference is
    fetched either way - which is why that case returns the flag itself, and why
    :func:`download_robots` checks it against the boolean domain before calling
    here rather than letting a non-boolean become this function's verdict.
    """
    if info is None:
        return False
    asset = info.get("asset", {})
    if not asset:
        return False

    xml_file, asset_dir = asset["model_xml"], asset["dir"]

    for search_dir in get_search_paths():
        try:
            model_path = safe_join(search_dir, f"{asset_dir}/{xml_file}")
        except ValueError:
            # The resolver makes this same join
            # (:func:`~strands_robots.assets.manager._resolve_candidates`) and
            # yields no candidate for an entry that escapes its search path, so
            # no file on disk can make this robot present. Fetch rather than
            # report it as already there - the two readings of one entry are
            # what :func:`_mjcf_missing_meshes` exists to keep together.
            logger.warning("assets: path traversal blocked for %s: %r", name, f"{asset_dir}/{xml_file}")
            return True
        if not model_path.exists():
            continue
        try:
            missing = _mjcf_missing_meshes(model_path)
        except Exception:
            # An unreadable model tells us nothing about its meshes, so fetch
            # the assets rather than declare them present.
            return True
        if missing:
            logger.debug("assets: %s declares %d mesh reference(s) not on disk: %s", name, len(missing), missing)
            return True
        return force

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


def _copy_external_tree(src: Path, dst: Path, *, drop_docs: bool = True) -> None:
    """Copy the externally sourced tree *src* into *dst* without following its symlinks.

    Every tree this module copies into the asset cache comes from outside the
    process - a shallow clone, or a ``robot_descriptions`` package that clones
    the same upstream repositories on first import - so a symlinked entry inside
    *src* is always skipped.  ``shutil.copytree`` defaults to ``symlinks=False``,
    which *follows* a nested symlink, so a description carrying
    ``robot_dir/assets -> /home/<user>/.ssh`` would copy host files into the
    cache and the download would still report success.  The skip is
    unconditional because no tree reaching here is trusted; a caller cannot ask
    for the following behaviour.

    Note this covers entries *inside* *src* only: ``shutil.copytree`` follows a
    symlinked *src* root before the ignore callback runs, so callers must
    validate the root separately (see :func:`safe_join` with
    ``resolve_symlinks``).

    Files are filtered on read rather than deleted from *dst* afterwards, so a
    user's own ``README.md`` kept beside the assets is never wiped.

    Args:
        src: Externally sourced tree to copy.
        dst: Destination directory inside the asset cache.
        drop_docs: When ``True`` (the clone routes), also skip the docs, preview
            images and ``.git`` bookkeeping a description repository ships around
            the model.  The ``robot_descriptions`` route passes ``False``: its
            preferred path symlinks the installed package directory whole, and
            the copy fallback taken when the cache cannot hold a symlink must
            expose the same files that symlink would have.
    """

    def _ignore(dir_path: str, names: list[str]) -> list[str]:
        skip = (
            [
                n
                for n in names
                if n in _COPY_CLEAN_SKIP or n.lower().endswith(_COPY_CLEAN_SUFFIX) or n.startswith(".git")
            ]
            if drop_docs
            else []
        )
        parent = Path(dir_path)
        links = [n for n in names if n not in skip and (parent / n).is_symlink()]
        if links:
            # Named rather than dropped quietly: an intra-tree link is skipped by
            # the same rule as an escaping one, so a model that loses a file this
            # way is diagnosable from the log instead of just being incomplete.
            logger.warning(
                "Skipping symlinked entries %s under %s: a symlink in an externally "
                "sourced tree is not followed into the asset cache",
                links,
                dir_path,
            )
            skip.extend(links)
        return skip

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
        if not re.match(r"^[a-z0-9_+]+\Z", module_name):
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
                # Validate existing symlink still has the expected XML.
                # Joined through ``safe_join`` so the validation cannot be
                # satisfied by a file outside the linked directory: raw, an
                # absolute ``model_xml`` discards *dst* and any existing host
                # file passes this check for a link that holds no model at all.
                expected_xml = safe_join(dst, str(info["asset"]["model_xml"]))
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
                # A cache that cannot hold a symlink (a FAT/exFAT card, or
                # Windows without the privilege) takes the copy instead, and the
                # installed package is an externally sourced tree exactly like a
                # fresh clone - so it goes through the same owner, not a bare
                # copytree that would follow a symlink out of the description.
                _copy_external_tree(package_path, dst, drop_docs=False)

            # Validate: expected XML must exist in the linked/copied dir
            expected_xml = safe_join(dst, str(info["asset"]["model_xml"]))
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
            try:
                # clone_dir holds a freshly cloned (untrusted) menagerie tree; resolve
                # symlinks so an asset dir symlinked outside the clone cannot be copied out.
                src = safe_join(Path(clone_dir), asset_dir, resolve_symlinks=True)
                if not src.exists():
                    results[name] = f"failed: {asset_dir} not in menagerie"
                    continue
                _copy_external_tree(src, safe_join(dest_dir, asset_dir))
                results[name] = "downloaded"
            except Exception as exc:
                results[name] = f"failed: {exc}"

    return results


def _download_from_github(name: str, info: dict, dest_dir: Path) -> str:
    """Download a robot from a custom GitHub repo (``asset.source``)."""
    source = info["asset"]["source"]
    repo = source["repo"]
    if not re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+\Z", repo):
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

        try:
            # clone_dir holds a freshly cloned (untrusted) tree and *subdir* comes
            # from the registry entry, so both escape routes must be closed before
            # the copy: a lexical '../' component, and a subdir that is itself a
            # symlink out of the clone.  The nested-symlink filter in
            # _copy_external_tree cannot cover the latter - copytree follows a
            # symlinked *root* before the ignore callback runs.
            src = safe_join(Path(clone_dir), subdir, resolve_symlinks=True) if subdir else Path(clone_dir)
        except ValueError as exc:
            return f"failed: {exc}"
        if not src.exists():
            return f"failed: subdir '{subdir}' not found in {repo}"

        dst = safe_join(dest_dir, asset_dir)
        try:
            _copy_external_tree(src, dst)
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
        names: Robot names to download - a SUBSET of the sim robots the registry
            lists. ``None`` selects all of them; an empty list selects none and
            is refused rather than widened to all (see :exc:`ValueError` below).
        category: Filter by category (arm, humanoid, mobile, ...). Applied only
            when ``names`` is ``None``.
        force: Re-fetch a robot whose assets are already present, replacing the
            cached directory. A posture, so it must be a boolean (see
            :exc:`ValueError` below).

    Returns:
        Dict with downloaded/skipped/failed counts, names, and details.

    Raises:
        ValueError: If ``names`` is an empty selection, which asks for no robot
            and cannot be honored as a request for every robot; or if ``force``
            is not a boolean, which cannot be read as either posture.
    """
    # ``names`` selects a SUBSET of the sim robots the registry already lists, so it
    # is read by membership - the rule ``names`` is read by on the teleoperate path
    # and ``cameras`` on the render path, where an empty selection resolves to no
    # camera rather than to every one. ``None`` is the documented "all sim robots";
    # an explicitly empty selection is the opposite of that, not a spelling of it.
    #
    # Read by truthiness, ``names=[]`` fell through to the branches that do not read
    # it at all. Measured on this registry it downloaded 56 robots on its own, and
    # 13 - the whole ``humanoid`` category - when a ``category`` was passed too,
    # reporting either as the caller's own request. An empty selection is what a
    # filter that matched nothing produces, and the ``download_assets`` tool reaches
    # it from a NON-empty argument: ``robots=","`` parses to no names through that
    # tool's own ``if r.strip()`` filter, so no caller has to write ``[]`` to get here.
    #
    # Refused ahead of ``get_user_assets_dir()``, which creates the cache directory,
    # so a refused selection leaves nothing behind to undo.
    #
    # Only the emptiness verdict is taken here; the shape is deliberately NOT routed
    # through the shared ``name_list_error`` domain. This surface resolves each name
    # by membership into ``robots`` below, so a repeat resolves to its first
    # occurrence and costs nothing - the same carve-out that keeps the WBC
    # provider out of that domain - and a mapping and a one-shot
    # iterator are each read exactly once here. Refusing them would reject calls
    # that are honored as written today.
    if names is not None and not names:
        raise ValueError(
            "download_robots(names=[]) selects no robot, so there is nothing to download. "
            "Pass names=None to download every sim robot, or name the subset to download."
        )

    # ``force`` selects a POSTURE - re-fetch a model whose assets are already present,
    # or leave it alone - so it is held to the shared boolean domain rather than read by
    # truthiness. It is the third flag in this package to sit in front of a
    # ``shutil.rmtree``, and the first two are why the rule exists: ``start_recording``'s
    # ``overwrite`` deleted the caller's dataset and ``restore_calibrations``' overwrote a
    # measurement of the hardware. Here the re-fetch removes the cached directory for a
    # robot whose assets are present, and that directory is where a user's own files sit -
    # ``_copy_external_tree`` filters on read rather than deleting afterwards precisely so
    # a README or notes kept beside the assets are never touched, and this is the one path
    # that does touch them.
    #
    # Measured on the shipped code, ``force="false"`` (also ``"no"``, ``"off"``, ``"0"``,
    # ``1`` and ``math.nan``) was indistinguishable from ``force=True``: the present
    # robot's cache directory was removed and re-fetched, a file kept beside its assets
    # was gone, and the call reported ``downloaded: 1`` to a caller who had spelled the
    # not-re-downloading of it. The falsy non-booleans - ``None``, ``0``, ``[]`` - took the
    # skip branch without ever being a declared spelling of it.
    #
    # Refused here, alongside the selection above and ahead of ``get_user_assets_dir()``,
    # so a refusal cannot arrive after the directory it was refusing to replace is gone.
    # It is also what keeps :func:`_needs_download` honest: that function returns this
    # flag as its own ``bool`` verdict for a model with nothing missing, so an unchecked
    # value was returned from a surface declaring it returns a boolean.
    if text := boolean_flag_error(force, "force", "download_robots"):
        raise ValueError(text)

    dest_dir = get_user_assets_dir()
    # Filter None values - get_robot() can return None for unknown names
    all_sim: dict[str, dict[str, Any]] = {
        r["name"]: info for r in registry_list_robots(mode="sim") if (info := get_robot(r["name"])) is not None
    }

    # Resolve requested robots. Read ``is not None``: an empty selection was
    # refused above, so reaching the ``category``/all branches means the caller
    # named no subset at all.
    # A name the registry does not list is carried in the result as
    # ``unknown_names``, not only logged: the caller asked for it and got nothing,
    # and the ``download_assets`` tool grades its verdict on that list.
    unknown: list[str] = []
    if names is not None:
        robots: dict[str, dict[str, Any]] = {}
        for name in names:
            canonical = resolve_robot_name(name)
            if canonical in all_sim:
                robots[canonical] = all_sim[canonical]
            else:
                logger.warning("Unknown robot: %s (resolved: %s)", name, canonical)
                unknown.append(str(name))
    elif category:
        robots = {n: i for n, i in all_sim.items() if i.get("category") == category}
    else:
        robots = dict(all_sim)

    if not robots:
        return {
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "unknown_names": unknown,
            "message": "No matching robots found.",
        }

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
            "unknown_names": unknown,
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
        "unknown_names": unknown,
        "assets_dir": str(dest_dir),
        "method": method,
        "message": (f"{len(downloaded)} downloaded ({method}), {len(skipped)} already present, {len(failed)} failed."),
    }
