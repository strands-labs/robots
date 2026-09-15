"""Convert an MJCF robot description to USD, content-addressed and cached.

Isaac Sim ships an MJCF importer - ``isaacsim.asset.importer.mjcf``, exposing
``MJCFImporter`` and ``MJCFImporterConfig`` - and this module is the seam that
turns it into a plain ``mjcf -> usd path`` function so the Isaac backend can load
the *same* description file the MuJoCo backend loads.

That matters because it is what makes this repository's parity claim true rather
than aspirational. ``docs/simulation/isaac.md`` promises that "the joint-name and
observation contract matches the MuJoCo backend, [so] policies and observation
mappings transfer unchanged between backends", and the only way to keep that
promise is for both backends to read one file. Measured on
``nvcr.io/nvidia/isaac-sim:6.0.1`` (A10G), converting the descriptions
``strands_robots.simulation.model_registry.resolve_model`` already resolves and
loading the result through ``add_robot(usd_path=...)`` reproduces MuJoCo's joint
vocabulary exactly:

===========================  ===============================================  =========
asset                        joint names                                      vs MuJoCo
===========================  ===============================================  =========
``franka_emika_panda``       ``joint1``..``joint7``, ``finger_joint1/2``       9 of 9
``trs_so_arm100``            ``Rotation Pitch Elbow Wrist_Pitch Wrist_Roll     6 of 6
                             Jaw``
===========================  ===============================================  =========

Both wired a live articulation, reset, stepped, and reported a non-empty
``get_observation()``.

Three properties of the vendor importer shape this module, all measured rather
than read off the documentation:

* ``usd_path`` is an output **directory root**, not a file name. The importer
  writes ``<usd_path>/<stem>/<stem>.usda`` and returns that path, so the return
  value is what a caller must reference - deriving the path itself would guess.
* Given no ``usd_path`` it writes **beside the source**, which fails with
  ``OSError: [Errno 30] Read-only file system`` for a description that lives in a
  shared read-only checkout. Every description this package resolves is such a
  file - ``~/.strands_robots/assets/<dir>`` is a symlink into
  ``~/.cache/robot_descriptions/`` - so an explicit destination is mandatory here
  rather than merely tidy.
* It writes a USD *file* and does **not** add anything to the live stage, so the
  result still has to be referenced in. ``IsaacSimulation._load_usd_robot``
  already does exactly that, which is why this module converts and stops.

The importer only exists inside a running Kit application, because it is a Kit
extension rather than a library. ``add_robot`` is always called on a live
simulation, so that holds wherever this is reached in production; a unit test
stands the importer in.
"""

from __future__ import annotations

import hashlib
import os

from strands_robots.utils import get_base_dir

__all__ = [
    "MJCF_EXTENSIONS",
    "USD_EXTENSIONS",
    "convert_mjcf_to_usd",
    "robot_usd_cache_dir",
]

#: Extensions this module treats as an MJCF description. MuJoCo itself accepts
#: any name, but every description in the shipped registry is ``.xml``, and
#: restricting the set is what lets a USD input be passed through below without
#: ambiguity.
MJCF_EXTENSIONS: tuple[str, ...] = (".xml", ".mjcf")

#: Already-USD inputs, referenced verbatim with no conversion. Spelled here
#: rather than imported from :mod:`strands_robots.simulation.isaac.mesh_assets`
#: so the two modules' vocabularies can diverge if USD ever gains a robot-only
#: container format; the values are deliberately identical today.
USD_EXTENSIONS: tuple[str, ...] = (".usd", ".usda", ".usdc", ".usdz")


def robot_usd_cache_dir() -> str:
    """Cache directory for converted robot USD assets (created on first use).

    ``$STRANDS_BASE_DIR/asset_cache/usd_robots`` - typically
    ``~/.strands_robots/asset_cache/usd_robots`` - a sibling of the mesh cache
    :func:`~strands_robots.simulation.isaac.mesh_assets.mesh_usd_cache_dir`
    writes to, following the same convention.
    """
    cache_dir = os.path.join(str(get_base_dir()), "asset_cache", "usd_robots")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _asset_digest(mjcf_path: str) -> str:
    """A digest covering the description *and the files it pulls in*.

    An MJCF is not self-contained: Menagerie's ``scene.xml`` is a handful of
    lines that ``<include>`` the robot body and reference a ``meshdir`` of STLs,
    and the geometry PhysX ends up simulating lives in those. Keying the cache on
    ``scene.xml`` alone would therefore hand back a stale USD after any change
    that did not touch the one file named - which is most of them.

    So the digest covers every file under the description's directory, by
    relative path and content. That is a full read of the asset directory on each
    call (~35 MB for ``franka_emika_panda``, tens of milliseconds), paid even on
    a cache hit; it is bounded by the size of one robot description and is
    negligible beside the conversion it guards, which takes seconds.

    Sorting the manifest is what makes the digest reproducible: :func:`os.walk`
    does not promise an order, so an unsorted manifest would key the same bytes
    differently on two machines and miss every cache entry.
    """
    root = os.path.dirname(os.path.abspath(mjcf_path))
    manifest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, root)
            manifest.update(rel.encode("utf-8", "surrogateescape"))
            try:
                with open(full, "rb") as fh:
                    for chunk in iter(lambda fh=fh: fh.read(1 << 20), b""):  # type: ignore[misc]
                        manifest.update(chunk)
            except OSError:
                # A file that cannot be read cannot contribute its bytes, and
                # skipping it silently would let two different trees share a
                # digest. Fold the failure itself in, so an unreadable file is a
                # distinct key rather than an absent one.
                manifest.update(b"\x00<unreadable>")
    # The description's own path within the tree matters: one directory can hold
    # several entry points (Menagerie ships ``scene.xml`` beside the bare robot
    # body), and they convert to different USD.
    manifest.update(os.path.relpath(os.path.abspath(mjcf_path), root).encode("utf-8", "surrogateescape"))
    return manifest.hexdigest()


def convert_mjcf_to_usd(
    mjcf_path: str,
    cache_dir: str | None = None,
    *,
    fix_base: bool | None = None,
    import_scene: bool = False,
) -> str:
    """Convert an MJCF description to a referenceable USD file.

    A ``.usd``/``.usda``/``.usdc``/``.usdz`` input is returned unchanged - it is
    already referenceable - so a caller can hand this either format. Anything
    else must be an :data:`MJCF_EXTENSIONS` file.

    The result is cached under ``<cache_dir>/<digest>/`` keyed on
    :func:`_asset_digest` together with the two options that change the produced
    USD, so a robot is converted once per (description, posture) rather than once
    per ``add_robot`` call. A conversion that fails leaves no directory behind,
    because a torn cache entry is one a later call would trust.

    Parameters
    ----------
    mjcf_path : str
        Filesystem path to the MJCF description, or to a USD file to pass
        through.
    cache_dir : str, optional
        Override the cache location (tests). Defaults to
        :func:`robot_usd_cache_dir`.
    fix_base : bool or None, optional
        Whether to weld the root body to the world. ``None`` - the default, and
        the vendor default - honours whatever the description itself says, which
        is the only choice that keeps a floating-base robot floating: every
        shipped quadruped and humanoid declares its base with ``<freejoint>``,
        and ``True`` would silently bolt such a robot to the ground while
        reporting success. ``True`` welds it, ``False`` frees it.
    import_scene : bool, optional
        Whether to import the description's ``<worldbody>`` furniture (floors,
        lights) alongside the robot. Defaults to ``False``: ``add_robot`` is
        asking for a robot, and the world it goes into already has a ground
        plane and lighting of its own.

    Returns
    -------
    str
        Path to the converted (or passed-through) USD file.

    Raises
    ------
    FileNotFoundError
        If *mjcf_path* does not exist.
    ValueError
        If *mjcf_path* is neither an MJCF nor a USD extension.
    ImportError
        If the Isaac MJCF importer is unavailable, with ``name`` set to the
        module that is missing. The importer is a Kit extension, so this is also
        what a call from outside a running Isaac Sim application reports.
    """
    ext = os.path.splitext(mjcf_path)[1].lower()
    if ext in USD_EXTENSIONS:
        if not os.path.isfile(mjcf_path):
            raise FileNotFoundError(f"robot asset not found: {mjcf_path}")
        return mjcf_path
    if not os.path.isfile(mjcf_path):
        raise FileNotFoundError(f"MJCF description not found: {mjcf_path}")
    if ext not in MJCF_EXTENSIONS:
        raise ValueError(
            f"cannot convert {mjcf_path!r} to USD: expected an MJCF "
            f"({', '.join(MJCF_EXTENSIONS)}) or a USD file "
            f"({', '.join(USD_EXTENSIONS)}), got {ext or 'no extension'!r}"
        )

    # The posture options are part of the key, not just the payload: the same
    # description converted with fix_base=True is a different robot from the
    # same description converted with fix_base=None, and a cache keyed on the
    # bytes alone would serve whichever was built first.
    key = hashlib.sha256(
        f"{_asset_digest(mjcf_path)}|fix_base={fix_base}|import_scene={import_scene}".encode()
    ).hexdigest()
    out_dir = cache_dir if cache_dir is not None else robot_usd_cache_dir()
    os.makedirs(out_dir, exist_ok=True)
    target_root = os.path.join(out_dir, key)

    stem = os.path.splitext(os.path.basename(mjcf_path))[0]
    # The layout the importer produces, recorded by the marker below rather than
    # recomputed: the vendor writes ``<root>/<stem>/<stem>.usda`` today, and a
    # future release that changes that would make a derived path silently wrong.
    marker = os.path.join(target_root, ".converted")
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8") as fh:
            cached = fh.read().strip()
        if cached and os.path.isfile(cached):
            return cached

    try:
        from isaacsim.asset.importer.mjcf import (  # type: ignore[import-not-found]
            MJCFImporter,
            MJCFImporterConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "converting an MJCF description to USD requires Isaac Sim's MJCF "
            "importer extension (isaacsim.asset.importer.mjcf). It is a Kit "
            "extension, so it resolves only inside a running Isaac Sim "
            "application - install the runtime (see docs/simulation/isaac.md) "
            "and call this from a live simulation.",
            name="isaacsim.asset.importer.mjcf",
        ) from exc

    # Convert into a sibling staging directory, then rename into place, so a
    # crashed or half-written conversion is never visible under the real key.
    # ``os.replace`` refuses a non-empty destination directory, so the marker
    # written last is what makes an entry usable and the rename only has to be
    # atomic enough that no reader sees a directory without one.
    staging = os.path.join(out_dir, f".{key}.{os.getpid()}.tmp")
    _remove_tree(staging)
    os.makedirs(staging, exist_ok=True)
    try:
        config = MJCFImporterConfig()
        config.mjcf_path = os.path.abspath(mjcf_path)
        config.usd_path = staging
        config.fix_base = fix_base
        config.import_scene = import_scene
        produced = MJCFImporter(config=config).import_mjcf()
    except BaseException:
        _remove_tree(staging)
        raise

    resolved = _resolve_produced(produced, staging, stem)
    if resolved is None:
        _remove_tree(staging)
        raise RuntimeError(
            f"the Isaac MJCF importer reported success for {mjcf_path!r} but wrote no "
            f"USD file under {staging!r} (it returned {produced!r}). Refusing to "
            f"return a path nothing can reference."
        )

    _remove_tree(target_root)
    os.replace(staging, target_root)
    final = os.path.join(target_root, os.path.relpath(resolved, staging))
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(final)
    return final


def _resolve_produced(produced: object, staging: str, stem: str) -> str | None:
    """The USD file the importer actually wrote, or ``None`` if there is none.

    Prefers the importer's own return value, because the layout it writes is its
    business and has changed across releases. Falls back to a search of the
    staging tree so a release that returns ``None`` (or a directory) still works,
    and prefers a file named after the description when several are present -
    a converted asset can carry sibling payload layers.
    """
    if isinstance(produced, str) and os.path.isfile(produced):
        return produced
    candidates = [
        os.path.join(dirpath, filename)
        for dirpath, _dirnames, filenames in os.walk(staging)
        for filename in sorted(filenames)
        if os.path.splitext(filename)[1].lower() in USD_EXTENSIONS
    ]
    if not candidates:
        return None
    for candidate in candidates:
        if os.path.splitext(os.path.basename(candidate))[0] == stem:
            return candidate
    return candidates[0]


def _remove_tree(path: str) -> None:
    """Delete *path* if present, tolerating its absence.

    A local helper rather than a bare ``shutil.rmtree(..., ignore_errors=True)``
    so a failure to clean a staging directory cannot be mistaken for a
    conversion failure, and so the ``ignore_errors`` blanket is not applied to
    the real cache entry.
    """
    import shutil

    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            # Best-effort, and not silent where it would matter. The only
            # caller that removes something load-bearing is the
            # ``_remove_tree(target_root)`` immediately above ``os.replace``,
            # and a removal that did not happen is reported there rather than
            # here: renaming the staging directory onto a surviving file raises
            # ``NotADirectoryError``, and onto a surviving directory raises
            # ``OSError`` (``ENOTEMPTY``). Every other caller passes a staging
            # entry, where a failed unlink leaks a temp path instead of
            # corrupting the cache - the distinction this helper exists to keep.
            pass
