"""LeRobotDataset recorder bridge for strands-robots.

Wraps LeRobotDataset so that both real hardware (:mod:`strands_robots.robot`)
and simulation (:mod:`strands_robots.simulation`) can produce training-ready
datasets with
a single add_frame() call per control step.

Usage:
    recorder = DatasetRecorder.create(
        repo_id="user/my_dataset",
        fps=30,
        robot_features=robot.observation_features,
        action_features=robot.action_features,
        task="pick up the red cube",
    )
    # In control loop:
    recorder.add_frame(observation, action, task="pick up the red cube")
    # End of episode:
    recorder.save_episode()
    # Optionally:
    recorder.push_to_hub()
"""

import difflib
import importlib.util
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.utils import (
    boolean_flag_error,
    camera_schema_key,
    lerobot_version,
    name_list_error,
    non_negative_whole_number_error,
    partial_construction_repr,
    positive_count_error,
    positive_whole_number_error,
    sequence_length,
)

logger = logging.getLogger(__name__)


# Every LeRobot codec surface validates the requested codec against the same
# codec-name allowlist - ``configs.video.VALID_VIDEO_CODECS = {"h264", "hevc",
# "libsvtav1", "libaom-av1", "auto"} | HW_VIDEO_CODECS`` - and *rejects* the
# ffmpeg library names ("libx264"/"libx265"). This holds for the current
# ``rgb_encoder=RGBEncoderConfig(vcodec=...)`` surface (lerobot >=0.6.0,<0.7.0,
# the supported range) exactly as it did for the flat ``vcodec`` kwarg and the
# interim ``camera_encoder=VideoEncoderConfig(...)`` surface the tolerant
# routing below still handles. So there is exactly one correct normalization
# direction: ffmpeg name -> codec name, applied to whichever surface is present.
# Callers may pass either spelling.
_ENCODER_CODEC_NAMES = {"libx264": "h264", "libx265": "hevc"}


def _codec_create_kwargs(sig_params: Any, vcodec: str, *, context: str = "create") -> dict[str, Any]:
    """Map a flat ``vcodec`` onto whatever codec surface this LeRobot exposes.

    LeRobot routes the codec differently across releases. Passing the wrong
    kwarg (or none) silently leaves the encoder on its built-in default, so the
    caller's ``vcodec`` is dropped without error - which is how an AV1 default
    sneaks back in even when the caller asked for H.264:

      * 0.5.0 / 0.5.1: a flat ``create/resume(..., vcodec=...)`` kwarg.
      * an interim build briefly exposed
        ``camera_encoder=VideoEncoderConfig(vcodec=...)``.
      * lerobot >= 0.6 (the supported range) moved the codec into
        ``rgb_encoder=RGBEncoderConfig(vcodec=...)``.

    Every one of these surfaces validates against LeRobot's codec-name allowlist
    (``{"h264", "hevc", "libsvtav1", "libaom-av1", "auto"} | HW_VIDEO_CODECS``)
    and rejects the ffmpeg library names ("libx264"/"libx265"). So the codec is
    normalized to its codec-name spelling once and routed onto whichever surface
    is present. The caller may pass either spelling ("h264" or "libx264"). An
    unknown codec raises loudly (LeRobot's own ValueError) rather than silently
    falling back to the default.

    Args:
        sig_params: ``inspect.Signature.parameters`` of ``create``/``resume``.
        vcodec: Requested codec, as a codec name ("h264", "hevc", "libsvtav1")
            or an ffmpeg encoder name ("libx264", "libx265").
        context: Label for warning messages ("create" or "resume").

    Returns:
        Mapping with exactly one of ``vcodec`` / ``rgb_encoder`` /
        ``camera_encoder``, or an empty dict if no known codec surface is
        present (recorder falls back to the LeRobot default codec).

    Raises:
        ValueError: When the codec surface rejects the requested codec
            (propagated from LeRobot so an unsupported codec fails loudly
            instead of silently reverting to the default).
    """
    # ffmpeg library name -> codec name; codec names (and HW encoders) pass through.
    codec = _ENCODER_CODEC_NAMES.get(vcodec, vcodec)
    if "vcodec" in sig_params:
        return {"vcodec": codec}
    if "rgb_encoder" in sig_params:
        try:
            from lerobot.configs.video import RGBEncoderConfig
        except ImportError as exc:
            logger.warning("RGBEncoderConfig import failed on %s (%s); using default codec", context, exc)
            return {}
        return {"rgb_encoder": RGBEncoderConfig(vcodec=codec)}
    if "camera_encoder" in sig_params:
        try:
            from lerobot.configs.video import VideoEncoderConfig
        except ImportError as exc:
            logger.warning("VideoEncoderConfig import failed on %s (%s); using default codec", context, exc)
            return {}
        return {"camera_encoder": VideoEncoderConfig(vcodec=codec)}
    return {}


# Allowlist patterns for HF Storage Bucket sync targets. Both `bucket` and
# `run_id` reach the `hf` CLI argv and the `hf://buckets/...` URI; they are
# agent-reachable via stop_recording(bucket=, run_id=) dispatched through the
# simulation action layer, so they MUST be validated before any subprocess /
# URI interpolation (AGENTS.md > LLM Input Safety). `bucket` is "name" or
# "org/name"; `run_id` is a single path segment. Neither may contain shell
# metacharacters, path-traversal (".."), or separators beyond the one allowed
# bucket "org/name" slash.
_BUCKET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?\Z")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def sync_dataset_to_bucket(
    root: str | Path,
    bucket: str,
    run_id: str | None = None,
    *,
    create: bool = True,
    private: bool = True,
    delete: bool = False,
) -> dict[str, Any]:
    """Sync an on-disk LeRobotDataset into an HF Storage Bucket (Phase 1/2).

    Lifecycle-independent: needs only a finalized dataset directory on disk
    (``meta/`` present) and the ``hf`` CLI - no live
    :class:`DatasetRecorder`, no sim world. Covers syncing a dataset
    recorded earlier in the process, one recorded on hardware via
    ``lerobot-record``, or a daily re-sync of a directory that grew. Both
    :meth:`DatasetRecorder.sync_to_bucket` and the idle-path bucket sync in
    ``stop_recording`` delegate here so input validation and CLI
    orchestration exist exactly once.

    Mutable, Xet-deduplicated dump target for COLLECTION - avoids git-LFS
    history bloat of push_to_hub during recording. Daily re-sync uploads
    only changed chunks (content-defined chunking). Requires the ``hf`` CLI
    with the ``buckets``/``sync`` subcommands (``huggingface_hub>=1.5``)
    and ``hf auth login``.

    ``bucket`` and ``run_id`` are validated against an allowlist before any
    subprocess or URI interpolation: ``bucket`` must be ``"name"`` or
    ``"org/name"`` and ``run_id`` a single path segment, both restricted to
    ``[A-Za-z0-9._-]`` (no path traversal or shell metacharacters). This
    path is agent-reachable via ``stop_recording(bucket=, run_id=)``. A
    rejected value returns ``{"status": "error", ...}`` without running ``hf``.

    ``create``, ``private`` and ``delete`` select *postures* rather than
    scaling a quantity, so each is checked against
    :func:`~strands_robots.utils.boolean_flag_error` before the ``hf`` CLI is
    even located - the same domain the mesh provisioning entry points apply to
    their own capability flags. Read by truthiness they fail toward the
    permissive posture in *both* directions, because every non-empty string is
    truthy and every falsy non-boolean takes the other branch:
    ``delete="false"`` - the spelling an operator reaches for when opting out -
    appends ``--delete`` and mirror-deletes remote files absent locally, while
    ``private=0`` drops ``--private`` and creates the bucket *public*.

    The shard layout is already Xet/bucket-friendly at lerobot's defaults
    (100 MB data parquet / 200 MB video MP4 shards), and ``meta/`` MUST
    ship or downstream loses normalization stats.

    Args:
        root: Local dataset directory, ``str`` or ``Path`` (must contain
            ``meta/``).
        bucket: Bucket target, ``"name"`` or ``"org/name"``.
        run_id: Subpath inside the bucket; defaults to the dataset directory
            name (``Path(root).name``).
        create: Create the bucket first (pre-existing bucket is not an error).
            Must be a boolean.
        private: Create the bucket as private (only used with ``create=True``).
            Must be a boolean.
        delete: Forward ``--delete`` to ``hf sync`` (mirror semantics -
            remove remote files absent locally). Must be a boolean.

    Returns:
        ``{"status": "success", "bucket_uri": ...}`` or
        ``{"status": "error", "message": ...}``. Never raises on ``hf``
        failure; errors are surfaced in the result dict. A flag outside its
        domain is reported the same way, without locating or running the CLI.
    """
    # Before the CLI probe so the same caller mistake reports identically
    # whether or not `hf` is installed, and so a refused posture flag can
    # never reach `hf buckets create` or `hf sync`.
    for flag_name, flag_value in (("create", create), ("private", private), ("delete", delete)):
        if flag_error := boolean_flag_error(flag_value, flag_name, "sync_dataset_to_bucket"):
            return {"status": "error", "message": flag_error}

    import subprocess

    hf = _hf_executable()
    if hf is None:
        return {
            "status": "error",
            "message": f'`hf` CLI not found. pip install -U "{_HF_BUCKET_CLI_MIN_SPEC}" and run `hf auth login`.',
        }

    # `hf buckets` / `hf sync` need huggingface_hub>=1.5; on every older
    # release (0.36.x, but also 1.0-1.4.x) the CLI exists and rejects those
    # subcommands with usage noise. Gate on the installed package version so
    # users get an upgrade instruction instead.
    version_error = _huggingface_hub_version_error()
    if version_error is not None:
        return {"status": "error", "message": version_error}

    if not _BUCKET_RE.match(bucket):
        return {
            "status": "error",
            "message": f"invalid bucket {bucket!r}: must match "
            "'name' or 'org/name' using [A-Za-z0-9._-] (no path traversal "
            "or shell metacharacters).",
        }

    local_root = str(root)
    # meta/ must ship or downstream loses normalization stats.
    if not (Path(local_root) / "meta").exists():
        return {
            "status": "error",
            "message": f"No meta/ under {local_root}; the dataset was never finalized. "
            "Call finalize() (stop_recording does this) before syncing to a bucket "
            "(stats/info required for streaming/training).",
        }

    run_id = run_id or Path(local_root).name
    if not _RUN_ID_RE.match(run_id):
        return {
            "status": "error",
            "message": f"invalid run_id {run_id!r}: must be a single path "
            "segment using [A-Za-z0-9._-] (no '/', path traversal, or shell "
            "metacharacters).",
        }
    dest = f"hf://buckets/{bucket}/{run_id}"

    if create:
        cp = subprocess.run(
            [hf, "buckets", "create", bucket] + (["--private"] if private else []),
            capture_output=True,
            text=True,
        )
        blob = (cp.stderr + cp.stdout).lower()
        # An already-created bucket is the normal case for a daily re-sync, so it
        # must not fail the sync. The hub reports it as "You already created this
        # bucket repo" with a 409, which does not contain "exists" - match the
        # status code and both phrasings rather than one substring.
        already_exists = "exist" in blob or "409" in blob or "already created" in blob
        if cp.returncode != 0 and not already_exists:
            return {
                "status": "error",
                "message": f"bucket create failed: {cp.stderr.strip()}",
            }

    cmd = [hf, "sync", local_root, dest]
    if delete:
        cmd.append("--delete")
    logger.info("Syncing %s -> %s", local_root, dest)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "status": "error",
            "message": proc.stderr.strip() or proc.stdout.strip(),
        }

    return {"status": "success", "bucket_uri": dest}


def _hf_executable() -> str | None:
    """Resolve the ``hf`` CLI, preferring the one in the running interpreter's
    environment before falling back to PATH.

    ``huggingface_hub`` installs the ``hf`` entry point next to the active
    Python (e.g. inside a virtualenv's ``bin``/``Scripts``). A bare ``hf`` on
    PATH is only found when that environment is also on PATH, which is often not
    the case for a subprocess launched from a venv whose ``bin`` was never
    activated. Checking ``sys.executable``'s directory first makes
    ``sync_to_bucket`` work from any environment where ``huggingface_hub`` is
    installed, not just an activated one. Returns ``None`` if no ``hf`` is found.
    """
    import shutil

    exe_dir = Path(sys.executable).parent
    for name in ("hf", "hf.exe"):
        candidate = exe_dir / name
        if candidate.exists():
            return str(candidate)
    return shutil.which("hf")


#: Oldest ``huggingface_hub`` release whose ``hf`` CLI carries the
#: ``hf buckets`` / ``hf sync`` subcommands :func:`sync_dataset_to_bucket`
#: invokes.
#:
#: They first ship in 1.5.0, as ``huggingface_hub/cli/buckets.py`` registered by
#: ``cli/hf.py`` (``app.add_group(buckets_cli, name="buckets")`` and
#: ``app.command()(sync)``). 1.0-1.4.x install the ``hf`` entry point without
#: that module, so they answer both invocations with
#: ``Error: No such command 'buckets'`` / ``'sync'``.
#:
#: This is the single source of the floor: the version gate below, the upgrade
#: instructions it and :func:`sync_dataset_to_bucket` emit, and the pin the
#: ``[wbc]`` extra declares are all checked against it, so the accepted domain
#: cannot drift from the release that can actually honor a bucket sync.
_HF_BUCKET_CLI_MIN_VERSION = (1, 5)

#: The requirement string every bucket-sync upgrade instruction quotes, derived
#: from :data:`_HF_BUCKET_CLI_MIN_VERSION` so the advice cannot name a release
#: that does not ship the subcommands.
_HF_BUCKET_CLI_MIN_SPEC = "huggingface_hub>=" + ".".join(str(part) for part in _HF_BUCKET_CLI_MIN_VERSION)


def _huggingface_hub_version_error() -> str | None:
    """Return an actionable error message if ``huggingface_hub`` is too old for bucket sync.

    The ``hf buckets`` / ``hf sync`` subcommands ship in
    :data:`_HF_BUCKET_CLI_MIN_VERSION` and later. On any older release - the
    0.36.x stable line, but equally 1.0-1.4.x - the ``hf`` binary exists, so
    :func:`_hf_executable` succeeds, but the subcommands fail with usage noise
    (``Error: No such command 'buckets'``) that gives no hint the fix is an
    upgrade. Version-checking the installed package up front turns that noise
    into a clear upgrade instruction without spawning a subprocess.

    Returns ``None`` (no error) when:

    - the installed version is >= :data:`_HF_BUCKET_CLI_MIN_VERSION`, or
    - ``huggingface_hub`` is not importable in this interpreter (the ``hf``
      binary may come from a different environment on PATH whose version we
      cannot see; the normal subprocess error path still applies), or
    - the version string is unparseable (fail open rather than block a
      possibly-capable CLI on a cosmetic version format).
    """
    try:
        import huggingface_hub
    except ImportError:
        return None

    version = getattr(huggingface_hub, "__version__", "")
    match = re.match(r"(\d+)\.(\d+)", version)
    if match is None:
        return None
    if (int(match.group(1)), int(match.group(2))) >= _HF_BUCKET_CLI_MIN_VERSION:
        return None
    return (
        f"bucket sync requires {_HF_BUCKET_CLI_MIN_SPEC} (`hf buckets`/`hf sync`); "
        f"installed: {version}. pip install -U '{_HF_BUCKET_CLI_MIN_SPEC}'."
    )


# Lazy check for LeRobot availability.
# We must NOT import lerobot at module level because it pulls in
# `datasets` -> `pandas`, which can crash with a numpy ABI mismatch on
# systems where the system pandas was compiled against an older numpy
# (e.g. JetPack / Jetson with system pandas 2.1.4 + pip numpy 2.x).
#
# Only the POSITIVE result is cached: importing lerobot is expensive (it pulls
# in `datasets` -> `pandas`) and worth doing once, but lerobot availability is
# a process capability that can transiently fail to resolve (a slow/locked
# import, or - in tests - a temporarily monkeypatched `sys.modules`). Caching a
# `False` would permanently disable recording for the rest of the process even
# after the condition clears, so a failed probe is re-attempted on the next
# call.
#
# The cache is a single-cell list rather than a module-level bool: a non-empty
# cell means "probed True". Mutating the cell (append/clear) records the result
# without rebinding a module global, so the memoization needs no ``global``
# statement.
_HAS_LEROBOT_DATASET: list[bool] = []


#: The packages ``lerobot[dataset]`` installs and that
#: ``lerobot.datasets.lerobot_dataset`` imports at module scope. Named in the
#: install hint so a single command fixes every one of them at once; the
#: authoritative set is lerobot's own extra, so this is a hint for a human, not
#: a second source of truth strands-robots probes against.
_LEROBOT_DATASET_PACKAGES = "datasets, pandas, pyarrow, av, torchcodec"

#: The module strands-robots imports to record a LeRobotDataset. Named in the
#: drift diagnosis so a caller can check it against their lerobot directly.
_LEROBOT_DATASET_MODULE = "lerobot.datasets.lerobot_dataset"


def _lerobot_installed() -> bool:
    """Whether the ``lerobot`` package itself is present.

    Uses a spec lookup rather than an import so it has no side effects and does
    not pay lerobot's import cost just to answer a question about an error
    message.
    """
    if "lerobot" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("lerobot") is not None
    except (ImportError, ValueError):
        return False


def _describe_lerobot_import_failure(exc: BaseException) -> str:
    """Turn an import failure into the diagnosis a caller can act on.

    Four unrelated failures reach here and they need four different
    instructions, so the message has to say which one happened:

    * lerobot itself is absent -- installing the extra is the fix;
    * lerobot is present but a package its dataset stack needs is not
      (``datasets``, ``pandas``, ``pyarrow``, ``av``, ...). Plain ``pip install
      lerobot`` does not pull those in, so "install lerobot" is not a usable
      instruction here: the package it names is already installed;
    * lerobot is present but does not provide what strands-robots imports -- an
      out-of-range or from-source lerobot that moved or renamed the module;
    * the import failed without any module being missing (``ValueError`` /
      ``RuntimeError``, typically a pandas built against a different numpy).
      Nothing is absent, so no install fixes it.

    Only a non-``ImportError`` may claim that nothing is missing: an
    ``ImportError`` whose ``name`` is unset still reports a failed import, and
    telling that caller to reconcile binary versions would send them after a
    conflict they may not have.

    Args:
        exc: The exception raised while importing ``LeRobotDataset``.

    Returns:
        An actionable diagnosis naming the cause and the install that fixes it.
    """
    detail = f"{type(exc).__name__}: {exc}"

    if not _lerobot_installed():
        return (
            f"lerobot is not installed ({detail}). Install lerobot >= 0.6.0 with: pip install 'strands-robots[lerobot]'"
        )

    version = lerobot_version()

    if not isinstance(exc, ImportError):
        return (
            f"lerobot {version} is installed and no module is missing, but importing its dataset "
            f"stack failed ({detail}). That is a conflict between installed packages -- commonly a "
            f"pandas built against a different numpy -- so no install of lerobot or its extra fixes "
            f"it; reconcile the conflicting packages instead."
        )

    missing = getattr(exc, "name", None) or ""
    if missing and missing.split(".")[0] != "lerobot":
        return (
            f"lerobot {version} is installed, but {missing!r}, which its dataset stack needs, is "
            f"not ({detail}). Install that whole set with: pip install 'lerobot[dataset]' "
            f"-- {_LEROBOT_DATASET_PACKAGES}. Installing lerobot without that extra does not pull "
            f"them in, so reinstalling lerobot alone will not fix this."
        )

    return (
        f"lerobot {version} is installed, but it does not provide "
        f"{_LEROBOT_DATASET_MODULE} ({detail}). strands-robots supports "
        f"lerobot >= 0.6.0,<0.7.0; an out-of-range or from-source lerobot can move or rename "
        f"that module. Install a supported one with: pip install 'strands-robots[lerobot]'"
    )


def lerobot_dataset_import_error() -> str | None:
    """Return None if lerobot's ``LeRobotDataset`` imports, else why it does not.

    This is the probe :func:`has_lerobot_dataset` answers yes/no from, and the
    one a caller should use when it has to tell a human what to do: the reason
    is what distinguishes "install the lerobot extra" from the cases that
    instruction cannot fix.

    A successful probe is cached; a failed probe is intentionally re-attempted
    on the next call so a transient import failure does not permanently disable
    recording for the process.

    Returns:
        ``None`` when ``LeRobotDataset`` is importable, otherwise a
        ready-to-display diagnosis naming the cause and the install that fixes
        it (see :func:`_describe_lerobot_import_failure`).
    """
    if _HAS_LEROBOT_DATASET:
        return None
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except (ImportError, ValueError, RuntimeError) as exc:
        reason = _describe_lerobot_import_failure(exc)
        logger.debug("lerobot dataset stack unavailable: %s", reason)
        return reason
    _HAS_LEROBOT_DATASET.append(True)
    return None


def has_lerobot_dataset() -> bool:
    """Return True if lerobot's ``LeRobotDataset`` can be imported.

    A thin predicate over :func:`lerobot_dataset_import_error` so the two cannot
    disagree about whether recording is available. Callers that must explain the
    unavailability to a human should use that function instead: a bare False
    cannot say which of several unrelated causes applied.
    """
    return lerobot_dataset_import_error() is None


def _get_lerobot_dataset_class():
    """Import and return LeRobotDataset class, or raise ImportError.

    Supports test mocking: if ``strands_robots.dataset_recorder.LeRobotDataset``
    has been set (by a test mock), returns that class directly.
    """
    # Support test mocking: check module-level overrides
    this_module = sys.modules[__name__]

    # If a test injected a mock LeRobotDataset class, use it
    mock_cls = getattr(this_module, "LeRobotDataset", None)
    if mock_cls is not None:
        return mock_cls

    # Actual import
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except (ImportError, ValueError, RuntimeError) as exc:
        raise ImportError(
            f"lerobot not available ({exc}). Install with: pip install lerobot\nRequired for LeRobotDataset recording."
        ) from exc


def _lerobot_home() -> Path:
    """Return LeRobot's on-disk dataset home (``$HF_LEROBOT_HOME``).

    Uses lerobot's own ``HF_LEROBOT_HOME`` constant when importable so the
    resolved path matches exactly where ``LeRobotDataset`` reads/writes
    (honouring the ``HF_LEROBOT_HOME`` environment override). Falls back to the
    documented default ``~/.cache/huggingface/lerobot`` when lerobot is absent.
    """
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        return Path(HF_LEROBOT_HOME)
    except (ImportError, ValueError, RuntimeError):
        return Path.home() / ".cache" / "huggingface" / "lerobot"


def resolve_dataset_dir(repo_id: str, root: str | None = None) -> Path:
    """Resolve the on-disk directory a dataset will live in.

    Mirrors ``LeRobotDataset`` root resolution so callers can inspect the
    target before ``create``/``resume``:

    * explicit ``root`` -> used verbatim;
    * a ``repo_id`` that is itself a path (absolute, ``./`` prefixed, or with no
      ``owner/name`` slash) -> treated as a local directory;
    * otherwise ``$HF_LEROBOT_HOME/{repo_id}``.

    Args:
        repo_id: HuggingFace dataset id (``owner/name``) or a local path.
        root: Explicit local dataset directory, if any.

    Returns:
        The resolved dataset directory as a :class:`~pathlib.Path`.
    """
    if root:
        return Path(root)
    if "/" not in repo_id or repo_id.startswith("/") or repo_id.startswith("./"):
        return Path(repo_id)
    return _lerobot_home() / repo_id


def _prepare_create_target(dataset_dir: Path, *, overwrite: bool) -> None:
    """Make ``dataset_dir`` safe for a fresh ``LeRobotDataset.create()``.

    ``LeRobotDataset.create()`` calls ``mkdir(exist_ok=False)`` and raises a
    bare ``FileExistsError`` whenever its target directory already exists - even
    when empty. This resolves the situation up front with an actionable error:

    * ``overwrite=True``: remove any existing target, then create() fresh.
    * existing dataset (contains a ``meta/`` dir): raise ``FileExistsError``
      naming ``overwrite=True`` (fresh) and :meth:`DatasetRecorder.resume`
      (append) - ``create`` never silently appends or clobbers a real dataset.
    * existing EMPTY dir (e.g. ``tempfile.mkdtemp()``): remove it so create()
      does not trip over its own pre-existing-directory guard.
    * existing NON-empty, non-dataset dir: raise ``ValueError`` instead of
      clobbering unrelated files.

    Args:
        dataset_dir: Resolved on-disk dataset root.
        overwrite: When True, replace any existing target.

    Raises:
        FileExistsError: Target is an existing LeRobotDataset and
            ``overwrite`` is False.
        ValueError: Target exists, is not a LeRobotDataset, is not empty, and
            ``overwrite`` is False.
    """
    import shutil

    if not dataset_dir.exists():
        return
    if overwrite:
        if dataset_dir.is_dir():
            shutil.rmtree(dataset_dir)
        else:
            dataset_dir.unlink()
        logger.info("Removed existing dataset target for overwrite: %s", dataset_dir)
        return
    if not dataset_dir.is_dir():
        raise ValueError(
            f"Recording target {dataset_dir} exists and is not a directory. "
            "Pass a directory path as root=, or overwrite=True to replace it."
        )
    if (dataset_dir / "meta").exists():
        raise FileExistsError(
            f"A LeRobotDataset already exists at {dataset_dir}. Pass overwrite=True "
            "to replace it with a fresh dataset, or use DatasetRecorder.resume() "
            "to append new episodes to it."
        )
    if not any(dataset_dir.iterdir()):
        # Empty dir (e.g. from tempfile.mkdtemp()): clear it so create() does
        # not trip over LeRobot's exist_ok=False guard.
        shutil.rmtree(dataset_dir)
        logger.info("Cleared empty recording target for fresh dataset: %s", dataset_dir)
        return
    raise ValueError(
        f"Recording target {dataset_dir} already exists, is not a LeRobotDataset "
        "(no meta/ directory), and is not empty. Refusing to overwrite unrelated "
        "files. Pass overwrite=True to replace it, or choose a new/empty root=."
    )


def unrecordable_action_columns_error(
    action: Mapping[str, Any],
    declared: Sequence[str],
    required: Sequence[str] | None,
) -> str | None:
    """Reject a frame whose action omits a declared column the caller requires.

    A LeRobot action column must hold the command that was issued at that step.
    When the dataset schema declares a column the frame's action dict does not
    carry, there is no such command, and every candidate placeholder is wrong:

    * ``0.0`` is itself a command wherever the action space is absolute
      position - a LeRobot ``<motor>.pos`` follower, a MuJoCo position actuator
      - so the joint TRAVELS to zero at servo speed on replay rather than
      staying where the un-commanded joint actually stayed.
    * A joint's measured position is in different units from a normalized or
      tendon-driven actuator's command, so substituting it moves those
      actuators too.
    * The command standing on the actuator cannot be read back, because the
      action-to-``ctrl`` mapping is deliberately not injective (a normalized
      open/close fraction and a raw tendon-unit command can share one ``ctrl``).

    So the frame is refused instead of persisted, which keeps the recorded
    action columns equal to what the policy issued and keeps ``replay_episode``
    a faithful round trip.

    ``required`` scopes the check to the columns the caller knows must come from
    this frame - typically the action keys of the robot being driven. Columns
    outside it (in a shared scene, the robots this rollout does not drive) are
    not this frame's to supply and are left alone.

    Args:
        action: The frame's action dict, keyed as the dataset schema spells it.
        declared: Action column names declared by the dataset schema.
        required: Column names this frame must supply, or ``None`` to skip the
            check entirely (the historical behaviour).

    Returns:
        An actionable message naming the missing columns, or ``None`` when every
        required column is present.
    """
    if required is None:
        return None
    declared_set = set(declared)
    missing = [key for key in required if key in declared_set and key not in action]
    if not missing:
        return None
    return (
        f"Recorded action column(s) {missing} have no value in this frame's action, so the "
        "recording would persist a command that was never issued. No placeholder is correct: "
        "0.0 is a 'travel to zero' command for an absolute-position actuator, a joint's measured "
        "position is in different units from a normalized or tendon actuator's command, and the "
        "command standing on the actuator cannot be read back (the action-to-ctrl mapping is not "
        "injective). Record with a policy that produces a value for every declared action column "
        "- an action vector narrower than the actuator list is reported by diagnose_action_dim - "
        "or record a schema covering only the actuators it drives."
    )


def _frame_shape_error(
    camera_dims: Any,
    camera_keys: Sequence[str] | None,
    video_width: Any,
    video_height: Any,
    context: str = "DatasetRecorder.create",
) -> str | None:
    """Error text when the declared camera frame shape cannot be honored.

    ``camera_dims`` and the ``video_width`` / ``video_height`` pair are one
    quantity in two spellings. :meth:`DatasetRecorder._build_features` reads
    ``camera_dims.get(camera, (video_height, video_width))`` once per declared
    camera, so the pair is the shape of every camera the mapping does not
    cover. Both spellings are checked here on the same domain: a value refused
    through one and accepted through the other would let the same unusable
    shape into the schema by choosing where to write it.

    The shape is a declaration rather than a resize - the recorder rescales
    nothing - so it goes straight into the LeRobot feature as ``(3, height,
    width)`` and is not compared against a real frame until the first
    ``add_frame``. Three mistakes could not be honored as written and were
    reported nowhere near the parameter that caused them:

    * A key ``camera_keys`` does not declare is dropped by that ``.get``, so
      the camera it was meant for silently takes the global pair instead and
      the schema declares a shape that camera does not stream. This is the
      quiet one: nothing is logged, the dataset is created, and the mismatch
      surfaces later against ``add_frame``.
    * A component that is not a positive integer is written into the feature
      as given - ``(3, 480, nan)``, ``(3, 480, '640')`` - so the schema
      declares a shape no frame can match.
    * A value that is not a two-element sequence unpacks as a bare
      ``TypeError`` / ``ValueError``, and a non-mapping ``camera_dims`` as a
      bare ``AttributeError`` from the ``.get`` - none of which names this
      parameter.

    :func:`~strands_robots.utils.positive_count_error` owns the component
    domain because a pixel count here is written into ``meta/info.json``: it
    has to be a true ``int``. An integral float lands in the schema as
    ``480.0``, and a ``numpy`` integer raises ``TypeError: Object of type
    int64 is not JSON serializable`` from the metadata write, so the wider
    whole-number domain would accept values this consumer refuses.

    Args:
        camera_dims: Per-camera ``{name: (height, width)}`` mapping, or None.
        camera_keys: The camera names the schema declares. Authoritative: it is
            the only source of camera names, and where it is empty no image
            feature is built, so neither spelling is read.
        video_width: Fallback frame width for every camera not covered above.
        video_height: Fallback frame height for every camera not covered above.
        context: Caller name for the message prefix.

    Returns:
        The first problem found, or None when the declared shape is usable.
    """
    declared = list(camera_keys or ())

    # The pair is read exactly where ``_build_features`` reads it - per declared
    # camera - so with no camera declared it decides nothing and is left alone.
    if declared:
        for value, param in ((video_width, "video_width"), (video_height, "video_height")):
            if text := positive_count_error(value, param, context):
                return text

    if camera_dims is None:
        return None
    if not isinstance(camera_dims, Mapping):
        return (
            f"{context}: camera_dims must be a mapping of camera name to a "
            f"(height, width) pair, got {type(camera_dims).__name__}. Its entries are "
            f"read per declared camera, so a non-mapping cannot be looked up at all."
        )
    if camera_dims and not declared:
        return (
            f"{context}: camera_dims declares a frame shape for "
            f"{sorted(map(str, camera_dims))} but no camera is declared. Pass the same "
            f"names as camera_keys, or drop camera_dims - with no camera_keys no image "
            f"feature is built, so every entry would be ignored."
        )
    for name, dims in camera_dims.items():
        if name not in declared:
            hint = difflib.get_close_matches(str(name), declared, n=1, cutoff=0.6)
            suggestion = f" Did you mean {hint[0]!r}?" if hint else ""
            return (
                f"{context}: camera_dims key {name!r} is not a declared camera."
                f"{suggestion} Declared cameras: {declared}. An entry keyed by any other "
                f"name is never looked up, so that camera would be declared at the global "
                f"video_height/video_width instead of the shape given here."
            )
        if isinstance(dims, str | bytes | Mapping) or sequence_length(dims) != 2:
            return (
                f"{context}: camera_dims[{name!r}] must be a (height, width) pair of positive integers, got {dims!r}."
            )
        for value, axis in zip(dims, ("height", "width"), strict=True):
            if text := positive_count_error(value, f"camera_dims[{name!r}] {axis}", context):
                return text
    return None


class RecordingFrameError(RuntimeError):
    """A frame the dataset recorder could not write, in fail-fast mode.

    Raised by :meth:`DatasetRecorder.add_frame` when the underlying
    ``LeRobotDataset`` write fails and the recorder was constructed with
    ``strict=True`` (the default). The frame is already gone at that point, so
    the episode on disk is shorter than the rollout that produced it and every
    surviving frame is re-timestamped from the declared ``fps`` - the caller has
    to be told.

    A distinct type, rather than the underlying error, so a rollout driver can
    tell a lost recording frame apart from a failure in a caller's telemetry
    hook. The drivers deliberately tolerate a few consecutive telemetry
    failures; granting that tolerance to a lost recording frame truncates the
    dataset while the rollout still reports success. The originating error is
    chained and its text preserved.
    """


class DatasetRecorder:
    """Bridge between strands-robots control loops and LeRobotDataset.

    Handles the full lifecycle:
    1. create() - build LeRobotDataset with correct features
    2. add_frame() - called every control step with obs + action
    3. save_episode() - finalize episode (encodes video, writes parquet)
    4. push_to_hub() - upload to HuggingFace

    Works for both real hardware (:mod:`strands_robots.robot`) and
    simulation (:mod:`strands_robots.simulation`).
    """

    def __init__(
        self,
        dataset,
        task: str = "",
        strict: bool = True,
        camera_key_map: dict[str, str] | None = None,
    ):
        self.dataset = dataset
        self.default_task = task
        self.frame_count = 0
        self.episode_frame_count = 0  # frames in the CURRENT (unsaved) episode
        self.dropped_frame_count = 0
        self.strict = strict
        self.episode_count = 0
        self._closed = False
        self._cached_state_keys: list[str] | None = None
        self._cached_action_keys: list[str] | None = None
        # Ordered SOURCE keys to read observation.state from, when they diverge
        # from the flat schema ``names`` (e.g. a floating base's ``base_quat``
        # source key expands to 4 per-component ``base_quat.*`` schema names).
        # ``None`` -> derive the read order from the schema names (scalar-only).
        self._state_source_keys: list[str] | None = None
        # Optional remap of observed camera stream names -> declared schema
        # names. Keys/values are bare camera names (no "observation.images."
        # prefix); a leading prefix on either side is tolerated and stripped.
        # Lets callers reconcile a policy's declared image_keys (e.g. "image",
        # "wrist_image") with differently-named sim/hardware streams (e.g.
        # "front_camera", "wrist_camera") instead of silently dropping frames.
        self.camera_key_map = self._normalize_camera_key_map(camera_key_map)
        # One-shot guard so the camera-key-mismatch diagnostic is logged once
        # per recorder instead of every control step (50Hz would flood logs).
        self._warned_camera_mismatch = False

    @staticmethod
    def _normalize_camera_key_map(camera_key_map: dict[str, str] | None) -> dict[str, str]:
        """Normalize a camera key remap to bare-name -> bare-name form.

        Accepts entries written either as bare camera names ("front_camera")
        or as fully-qualified feature keys ("observation.images.front_camera")
        on EITHER side, and strips the "observation.images." prefix so the map
        can be applied uniformly against bare camera names in add_frame.

        Args:
            camera_key_map: Caller-supplied remap, or None.

        Returns:
            A dict mapping observed bare camera name -> declared bare camera
            name (empty dict when no map was supplied).
        """
        prefix = "observation.images."
        normalized: dict[str, str] = {}
        for src, dst in (camera_key_map or {}).items():
            src_bare = src[len(prefix) :] if src.startswith(prefix) else src
            dst_bare = dst[len(prefix) :] if dst.startswith(prefix) else dst
            normalized[src_bare] = dst_bare
        return normalized

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int = 30,
        robot_type: str = "unknown",
        robot_features: dict[str, Any] | None = None,
        action_features: dict[str, Any] | None = None,
        camera_keys: list[str] | None = None,
        camera_dims: dict[str, tuple[int, int]] | None = None,
        joint_names: list[str] | None = None,
        action_names: list[str] | None = None,
        extra_state_specs: list[tuple[str, list[str]]] | None = None,
        task: str = "",
        root: str | None = None,
        use_videos: bool = True,
        vcodec: str = "h264",
        streaming_encoding: bool = True,
        image_writer_threads: int = 4,
        video_backend: str | None = None,
        video_width: int = 640,
        video_height: int = 480,
        camera_key_map: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> "DatasetRecorder":
        """Create a new DatasetRecorder with auto-detected features.

        Args:
            repo_id: HuggingFace dataset ID (e.g. "user/my_dataset")
            fps: Recording frame rate, as a positive whole number of frames per
                second. This is the rate the dataset is DECLARED at (it is
                written into ``meta/info.json`` and every frame timestamp is
                derived from it), not a rate anything is throttled to. The domain
                is the shared one every backend's ``start_recording`` applies to
                the ``fps`` it forwards here
                (:func:`~strands_robots.utils.positive_whole_number_error`), so
                the facade and this method cannot disagree about which rates are
                usable.
            robot_type: Robot type string (e.g. "so100", "panda")
            robot_features: Dict of observation feature names → types
                (from robot.observation_features or sim joint names)
            action_features: Dict of action feature names → types
            camera_keys: List of DISTINCT camera names (images become video
                features). One name per camera, in schema order; a single name
                still has to be a one-element list.
            camera_dims: Per-camera declared frame shape as
                ``{camera_key: (height, width)}``, keyed by the same names
                ``camera_keys`` declares. Note the order: ``(height, width)``,
                the reverse of the ``video_width`` / ``video_height`` pair it
                falls back to on this same call. Every backend that supplies it
                reads the camera's real render resolution, so a camera present
                here is declared at its own shape rather than the global one; a
                camera absent from the mapping (or ``None``, the default) takes
                ``(video_height, video_width)``.
                Every key has to be a name ``camera_keys`` declares: an entry
                keyed by anything else is never looked up, so that camera would
                be declared at the global pair instead of the shape given here.
                Each value is a two-element ``(height, width)`` of positive
                integers.
            video_width/video_height: Declared frame shape for every camera
                ``camera_dims`` does not cover. A declaration rather than a
                resize - the recorder rescales nothing - so a camera streaming
                another size is declared at a shape it does not have. This is why
                the backends pass ``camera_dims`` read from the live cameras
                instead of relying on this pair.
                Each is a positive integer - a pixel count is written into
                ``meta/info.json``, so an integral float would be declared as
                ``480.0`` and a ``numpy`` integer is not JSON-serializable.
            joint_names: List of DISTINCT joint names (alternative to
                robot_features for sim). These become the ``observation.state``
                column names, and add_frame reads each one out of the
                observation, so a name the observation does not carry records
                0.0 for the whole episode.
            action_names: Explicit action-column names. Use when the action
                space diverges from the joint names (e.g. actuator short-names
                from SimEngine.robot_action_keys, where a tendon gripper is an
                actuator with no matching joint). Falls back to joint_names.
            extra_state_specs: Optional vector state signals to append to the
                observation.state schema as per-component scalar columns, as
                ``(source_key, [component_suffixes])`` pairs. Each source key is
                read from the observation and flattened in order (e.g. a
                floating base's ``("base_quat", ["w","x","y","z"])`` adds four
                ``base_quat.*`` columns). Scalar joint/action fallbacks ignore
                these; add_frame reads the source keys, not the expanded names.
            task: Default task description
            root: Local directory for dataset storage
            use_videos: Encode camera frames as video (True) or keep as images.
                Selects a posture rather than scaling a quantity, so it must be a
                boolean (:func:`~strands_robots.utils.boolean_flag_error`) - the
                choice is written into the dataset schema as each camera's
                ``dtype`` and is fixed for the life of the dataset.
            vcodec: Video codec for the per-camera MP4 streams. Defaults to
                "h264" (H.264), which is universally decodable - including
                by OpenCV's VideoCapture / FFmpeg build, used by many
                downstream VLM video readers. Use "libsvtav1" (AV1) for
                smaller files in storage-constrained training pipelines;
                LeRobot read-back (torchcodec/pyav) handles AV1, but OpenCV
                wheels commonly cannot decode it and silently yield 0 frames.
            streaming_encoding: Stream-encode video during capture. Must be a
                boolean, on the same domain as ``use_videos``; it is forwarded to
                LeRobot's own boolean parameter.
            image_writer_threads: Threads for writing image frames
            video_backend: LeRobot video *decode* backend for read-back
                ("torchcodec" or "pyav"). Left as None by default so LeRobot
                picks its platform default; only forwarded when explicitly set.
                Encoder selection is controlled by ``vcodec`` (not this param).
            camera_key_map: Optional remap of observed camera stream names to the
                declared schema names (e.g. {"front_camera": "image",
                "wrist_camera": "wrist_image"}). Bare names or fully-qualified
                "observation.images.*" keys are accepted on either side. Use it
                when a policy declares image_keys that differ from the names the
                sim/hardware streams emit, otherwise those frames are dropped.
            overwrite: When the resolved dataset directory already exists,
                ``LeRobotDataset.create`` raises a bare ``FileExistsError`` (its
                ``mkdir`` uses ``exist_ok=False``). With ``overwrite=True`` the
                existing directory is removed first so a fresh dataset is
                created. With ``overwrite=False`` (default) an existing dataset
                (a directory containing ``meta/``) raises a clear
                ``FileExistsError`` naming ``overwrite=True`` (fresh) and
                :meth:`resume` (append) instead of the cryptic LeRobot error; an
                existing EMPTY directory (e.g. from ``tempfile.mkdtemp()``) is
                cleared so ``create`` does not dead-end on its own existence
                guard; a non-empty NON-dataset directory raises ``ValueError``
                rather than clobbering unrelated files.

                This flag is a confirmation gate in front of a delete, so it must
                be a boolean (:func:`~strands_robots.utils.boolean_flag_error`),
                on the same domain every backend's ``start_recording`` applies to
                the ``overwrite`` it forwards here. It is checked before the
                on-disk target is touched, so an unusable value cannot remove a
                dataset on the way to being reported.

        Raises:
            ValueError: ``camera_keys``, ``joint_names`` or ``action_names`` is
                not a list of distinct non-blank names (a bare string, a
                mapping, or a repeated name); or the declared camera frame shape
                cannot be honored - ``camera_dims`` is not a mapping, is keyed by
                a name ``camera_keys`` does not declare, or holds anything but a
                ``(height, width)`` pair of positive integers, or
                ``video_width`` / ``video_height`` is not a positive integer;
                or ``fps`` is not a positive whole number; or ``use_videos``,
                ``streaming_encoding`` or ``overwrite`` is not a boolean.
                Refused before the on-disk target is touched, so an
                ``overwrite=True`` call that is refused leaves an existing
                dataset intact.
            FileExistsError: The resolved dataset directory already holds a
                dataset and ``overwrite`` is False.
        """
        # ``camera_keys`` / ``joint_names`` / ``action_names`` each name an
        # ordered list of DISTINCT schema column names, so each is refused on the
        # shared name-list domain here - before the lerobot extra is probed, so
        # the same caller mistake reports identically on every install, and
        # before the on-disk target is touched, because ``overwrite=True``
        # removes an existing dataset directory and a refusal arriving after that
        # would already have destroyed it.
        #
        # Neither mistake this catches could be honored as written, and both used
        # to be reported nowhere: the dataset was created, every frame was
        # accepted and the episode saved. A single name passed as a bare string
        # is iterable per character, so ``joint_names="gripper"`` declared seven
        # columns (``g``, ``r``, ``i``, ``p``, ``p``, ``e``, ``r``) and every one
        # recorded 0.0 - add_frame reads each declared name out of the
        # observation, and none of those names is in it. A repeated name collapses
        # where it keys a dict (two ``camera_keys`` entries with the same name
        # declare ONE camera column, so the schema has fewer cameras than the
        # caller asked for) and doubles where it indexes a position (a repeated
        # joint name records that joint twice and the joint the caller meant not
        # at all).
        for value, param in (
            (camera_keys, "camera_keys"),
            (joint_names, "joint_names"),
            (action_names, "action_names"),
        ):
            if value and (text := name_list_error(value, param, "DatasetRecorder.create")):
                raise ValueError(text)
        # ``camera_dims`` and the ``video_width`` / ``video_height`` pair are the
        # other half of the same schema declaration: they set the frame shape of
        # every declared camera. Checked on the same shared domain, in the same
        # place and for the same two reasons as the names above - before the
        # lerobot extra is probed, so one caller mistake reports identically on
        # every install, and before the on-disk target is touched, because
        # ``overwrite=True`` removes an existing dataset and a refusal arriving
        # after that would already have destroyed it. Runs after the loop above
        # so ``camera_keys`` is known to be a list of distinct names before it is
        # used as the authoritative set of camera names.
        if text := _frame_shape_error(camera_dims, camera_keys, video_width, video_height):
            raise ValueError(text)
        # ``fps`` is the recording RATE the same declaration fixes - it is written
        # into the dataset metadata and every timestamp is derived from it - so it
        # is refused here too, in the same block and ahead of the same two side
        # effects as the names and the shape above.
        #
        # The domain is deliberately the one the facades already apply, and via the
        # same function rather than a restatement of it: every backend's
        # ``start_recording`` calls ``dataset_recording_option_error``, which is a
        # thin ``{"status": "error"}`` envelope around
        # ``positive_whole_number_error``, and then forwards ``fps`` here
        # unchanged. Any narrower rule at this depth would refuse a value those
        # facades had already reported usable, turning a returned error envelope
        # into a ``ValueError`` raised out of a method that returns one.
        #
        # Direct callers were the only ones left unguarded, and an unusable rate
        # cost them the episode silently: LeRobot rejects only ``fps <= 0``, so a
        # fractional ``2.7``, a ``nan`` or an ``inf`` created the dataset and then
        # saved ZERO frames with ``create``, ``add_frame``, ``save_episode`` and
        # ``finalize`` all returning normally; ``fps=True`` recorded a 1 fps
        # dataset (an ``int`` subclass acting as a 1); and ``fps="30"`` dead-ended
        # in a bare ``TypeError: '<=' not supported between instances of 'str' and
        # 'int'`` naming neither the parameter nor the method.
        if text := positive_whole_number_error(fps, "fps", "DatasetRecorder.create"):
            raise ValueError(text)
        # ``use_videos`` / ``streaming_encoding`` / ``overwrite`` are the posture
        # half of the same guard block: each selects a branch rather than scaling
        # a quantity, so each is checked on
        # :func:`~strands_robots.utils.boolean_flag_error` - the inverse domain to
        # the numeric ones above, which reject ``bool`` because it would pass as a
        # silent ``1``. Checked here for the same two reasons and in the same
        # place: before the lerobot extra is probed, and before the on-disk target
        # is touched.
        #
        # Read by truthiness instead, each failed toward the branch the caller was
        # opting *out* of, because every non-empty string is truthy. ``overwrite``
        # is the destructive one: ``overwrite="false"`` (also ``"no"``, ``"off"``,
        # ``"0"``) reached ``_prepare_create_target`` as True and ``shutil.rmtree``-d
        # the caller's dataset. Measured end to end: a dataset holding one recorded
        # episode came back with zero, and ``create`` returned a working recorder
        # throughout. ``use_videos="false"`` declared the cameras as ``dtype="video"``
        # where ``False`` declares ``"image"``, a schema decision written into
        # ``meta/info.json`` and fixed for the life of the dataset; the same string
        # then reached LeRobot's own boolean parameter unconverted, as
        # ``streaming_encoding="false"`` did on both entry points.
        #
        # ``push_to_hub`` and ``sync_dataset_to_bucket`` in this module, and every
        # backend's ``start_recording`` facade, already check their own posture
        # flags; the documented direct creation API was the surface left reading
        # them by truthiness, so the two disagreed about which values are usable.
        for flag_value, flag_name in (
            (use_videos, "use_videos"),
            (streaming_encoding, "streaming_encoding"),
            (overwrite, "overwrite"),
        ):
            if text := boolean_flag_error(flag_value, flag_name, "DatasetRecorder.create"):
                raise ValueError(text)

        # Lazy import - this is where we actually need lerobot
        LeRobotDatasetCls = _get_lerobot_dataset_class()

        # Build features dict in LeRobot format
        features = cls._build_features(
            robot_features=robot_features,
            action_features=action_features,
            camera_keys=camera_keys,
            camera_dims=camera_dims,
            joint_names=joint_names,
            action_names=action_names,
            extra_state_specs=extra_state_specs,
            use_videos=use_videos,
            video_width=video_width,
            video_height=video_height,
        )

        logger.info(f"Creating LeRobotDataset: {repo_id} @ {fps}fps, {len(features)} features, robot_type={robot_type}")

        # Build kwargs, skip unsupported params for this LeRobot version.
        create_kwargs = dict(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
        )
        import inspect

        create_sig = inspect.signature(LeRobotDatasetCls.create)
        create_params = create_sig.parameters

        # Route the requested codec onto whichever surface this LeRobot version
        # exposes (vcodec / rgb_encoder / camera_encoder). The flat ``vcodec``
        # kwarg and ``camera_encoder`` were both removed in 0.5.2; the codec now
        # lives inside ``rgb_encoder=RGBEncoderConfig(vcodec=...)``. Missing this
        # surface silently leaves the encoder on its built-in default.
        create_kwargs.update(_codec_create_kwargs(create_params, vcodec, context="create"))

        # streaming_encoding / video_backend only in newer LeRobot versions
        if "streaming_encoding" in create_params:
            create_kwargs["streaming_encoding"] = streaming_encoding
        if "video_backend" in create_params and video_backend is not None:
            create_kwargs["video_backend"] = video_backend

        # Resolve create-vs-crash for an existing target BEFORE calling
        # LeRobotDataset.create(), which mkdir()s with exist_ok=False and would
        # otherwise dead-end on a bare FileExistsError. This also keeps the
        # resume() docstring and its no-resume RuntimeError message honest: both
        # point callers at an ``overwrite=`` parameter that now exists here.
        _prepare_create_target(resolve_dataset_dir(repo_id, root), overwrite=overwrite)

        dataset = LeRobotDatasetCls.create(**create_kwargs)

        recorder = cls(dataset=dataset, task=task, camera_key_map=camera_key_map)
        # When vector state specs expand the schema (e.g. a floating base),
        # add_frame must read the SOURCE keys (``base_quat``) rather than the
        # expanded per-component schema names (``base_quat.w``). Record the
        # source order: scalar state keys first, then each vector source key.
        if extra_state_specs:
            scalar_source = list(joint_names) if joint_names else []
            recorder._state_source_keys = scalar_source + [k for k, _ in extra_state_specs]
        logger.info("DatasetRecorder ready: %s", repo_id)
        return recorder

    @classmethod
    def resume(
        cls,
        repo_id: str,
        root: str | None = None,
        task: str = "",
        vcodec: str = "h264",
        streaming_encoding: bool = True,
        image_writer_threads: int = 4,
        video_backend: str | None = None,
        camera_key_map: dict[str, str] | None = None,
    ) -> "DatasetRecorder":
        """Resume recording into an EXISTING LeRobotDataset (append episodes).

        Unlike :meth:`create` (which calls ``LeRobotDataset.create`` and
        hard-fails with ``FileExistsError`` if the dataset dir already
        exists), this opens an on-disk dataset via ``LeRobotDataset.resume``
        so further ``add_frame``/``save_episode`` calls append new episodes.

        This is the multi-episode data-collection path: ``start_recording``
        with ``overwrite=False`` on an existing dataset routes here instead of
        crashing. The plain ``LeRobotDataset(repo_id, root=...)`` constructor
        returns a READ-ONLY dataset (``add_frame`` raises), so ``resume()`` is
        the only correct append entry point in LeRobot 0.5.2+.

        Feature schema is inherited from the existing dataset on disk - the
        caller's joint/camera layout must match what was originally recorded.

        Args:
            repo_id: HuggingFace dataset ID (same as the original recording).
            root: Local dataset directory (same as the original recording).
            task: Default task description for appended frames.
            vcodec: Video codec for the per-camera MP4 streams (default
                "h264"; routed into the version-appropriate encoder
                config). See create() for the H.264-vs-AV1 trade-off.
            streaming_encoding: Stream-encode video during capture. Must be a
                boolean, on the same domain :meth:`create` applies to the flag it
                forwards here (:func:`~strands_robots.utils.boolean_flag_error`).
            image_writer_threads: Threads for writing image frames.
            video_backend: LeRobot video *decode* backend for read-back
                ("torchcodec" or "pyav"); None uses LeRobot's platform default.
                Encoder selection is controlled by ``vcodec`` (not this param).
            camera_key_map: Optional remap of observed camera stream names to
                the declared schema names (see create()).

        Returns:
            A DatasetRecorder wrapping the resumed dataset.

        Raises:
            ValueError: ``streaming_encoding`` is not a boolean. Refused before
                the lerobot extra is probed, so the same caller mistake reports
                identically on every install.
            RuntimeError: The installed LeRobot has no ``LeRobotDataset.resume``
                (append needs ``lerobot>=0.5.2``).
        """
        import inspect

        # Same posture flag as :meth:`create` forwards, on the same domain and
        # ahead of the same lerobot probe, so the two creation entry points cannot
        # disagree about which values are usable. Read by truthiness,
        # ``streaming_encoding="false"`` selected streaming and handed LeRobot's
        # own boolean parameter the string unconverted.
        if text := boolean_flag_error(streaming_encoding, "streaming_encoding", "DatasetRecorder.resume"):
            raise ValueError(text)

        LeRobotDatasetCls = _get_lerobot_dataset_class()

        if not hasattr(LeRobotDatasetCls, "resume"):
            # Older LeRobot (0.5.0/0.5.1) has no resume(); the append workflow
            # is unsupported there. Surface a clear error rather than a cryptic
            # read-only add_frame failure downstream.
            raise RuntimeError(
                "This LeRobot version has no LeRobotDataset.resume(); "
                "multi-episode append requires lerobot>=0.5.2. "
                "Use overwrite=True for a fresh single-session dataset."
            )

        resume_sig = inspect.signature(LeRobotDatasetCls.resume).parameters
        resume_kwargs: dict[str, Any] = dict(repo_id=repo_id, root=root)
        # Mirror create()'s version-tolerant codec routing.
        resume_kwargs.update(_codec_create_kwargs(resume_sig, vcodec, context="resume"))
        if "streaming_encoding" in resume_sig:
            resume_kwargs["streaming_encoding"] = streaming_encoding
        if "image_writer_threads" in resume_sig:
            resume_kwargs["image_writer_threads"] = image_writer_threads
        if "video_backend" in resume_sig and video_backend is not None:
            resume_kwargs["video_backend"] = video_backend

        dataset = LeRobotDatasetCls.resume(**resume_kwargs)
        recorder = cls(dataset=dataset, task=task, camera_key_map=camera_key_map)
        # Seed counters from the existing dataset so reporting reflects totals.
        try:
            recorder.episode_count = int(dataset.meta.total_episodes)
            recorder.frame_count = int(dataset.meta.total_frames)
        except Exception:  # noqa: BLE001 - counters are best-effort
            pass
        logger.info(
            "DatasetRecorder resumed: %s (%d existing episodes)",
            repo_id,
            recorder.episode_count,
        )
        return recorder

    @classmethod
    def _build_features(
        cls,
        robot_features: dict | None = None,
        action_features: dict | None = None,
        camera_keys: list[str] | None = None,
        camera_dims: dict[str, tuple[int, int]] | None = None,
        joint_names: list[str] | None = None,
        action_names: list[str] | None = None,
        extra_state_specs: list[tuple[str, list[str]]] | None = None,
        use_videos: bool = True,
        video_height: int = 480,
        video_width: int = 640,
    ) -> dict[str, Any]:
        """Build LeRobot v3-compatible features dict.

        LeRobot v3 features format:
        {
            "observation.images.camera_name": {"dtype": "video", "shape": (C, H, W), "names": [...]},
            "observation.state": {"dtype": "float32", "shape": (N,), "names": [...]},
            "action": {"dtype": "float32", "shape": (N,), "names": [...]},
        }

        Note: "names" must be a flat list of strings, NOT a dict like {"motors": [...]}.
        """
        features = {}

        # Observation: cameras → video/image features
        if camera_keys:
            camera_dims = camera_dims or {}
            for cam_name in camera_keys:
                key = f"observation.images.{cam_name}"
                dtype = "video" if use_videos else "image"
                # Per-camera (height, width). Falls back to the global
                # video_height/width when a camera has no explicit dims, so
                # callers that don't pass camera_dims keep the old behaviour.
                cam_h, cam_w = camera_dims.get(cam_name, (video_height, video_width))
                features[key] = {
                    "dtype": dtype,
                    "shape": (3, cam_h, cam_w),
                    "names": ["channels", "height", "width"],
                }

        # Observation: state (joint positions)
        state_dim = 0
        state_names = []
        if robot_features:
            # Count scalar features (exclude cameras)
            state_keys = [
                k
                for k, v in robot_features.items()
                if not isinstance(v, dict) or v.get("dtype") not in ("image", "video")
            ]
            state_dim = len(state_keys)
            state_names = state_keys
        elif joint_names:
            state_dim = len(joint_names)
            state_names = list(joint_names)

        # Preserve additional vector state signals (e.g. a floating base's
        # ``base_quat`` / ``base_ang_vel``) as per-component scalar columns so
        # they are not silently dropped from ``observation.state``. Each spec is
        # ``(source_key, [component_suffixes])``; ``add_frame`` reads the source
        # key from the observation and flattens it in this same order. The
        # scalar joint/action fallbacks below use the pre-expansion dims.
        scalar_state_dim = state_dim
        scalar_state_names = list(state_names)
        if extra_state_specs:
            for src_key, comps in extra_state_specs:
                state_names.extend(f"{src_key}.{c}" for c in comps)
                state_dim += len(comps)

        if state_dim > 0:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": state_names,
            }

        # Action. Prefer an explicit action_names list (the actuator keys the
        # rollout loops emit, which can diverge from the joint names - see
        # SimEngine.robot_action_keys) so the recorded action columns match the
        # keys add_frame receives. Fall back to action_features, then to the
        # joint/state names for robots whose actuators mirror their joints.
        action_dim = 0
        action_col_names: list[str] = []
        if action_features:
            action_keys = [
                k
                for k, v in action_features.items()
                if not isinstance(v, dict) or v.get("dtype") not in ("image", "video")
            ]
            action_dim = len(action_keys)
            action_col_names = action_keys
        elif action_names:
            action_dim = len(action_names)
            action_col_names = list(action_names)
        elif joint_names:
            action_dim = len(joint_names)
            action_col_names = list(joint_names)
        elif scalar_state_dim > 0:
            action_dim = scalar_state_dim  # Same dim as scalar state by default
            action_col_names = scalar_state_names[:]

        if action_dim > 0:
            features["action"] = {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": action_col_names[:action_dim],
            }

        return features

    def add_frame(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        task: str | None = None,
        camera_keys: list[str] | None = None,
        required_action_keys: Sequence[str] | None = None,
    ) -> None:
        """Add a single control-loop frame to the dataset.

        This is the key method - called every step in the control loop.

        Args:
            observation: Raw observation dict from robot/sim
                (joint_name → float, camera_name → np.ndarray)
            action: Action dict (joint_name → float)
            task: Task description written to this frame's ``task`` column.
                This argument is the top of a three-level chain and the only
                place it is stated: it falls back to the recorder's
                ``default_task`` (set from ``create(task=...)`` /
                ``resume(task=...)``, which is what a backend's
                ``start_recording(task=...)`` supplies) and then to the literal
                ``"untitled"``. Every simulation rollout hook passes
                ``run_policy(instruction=...)`` here, so a non-empty instruction
                overrides the recording session's task, and a rollout driven with
                neither annotates its frames ``"untitled"`` - a constant
                instruction for any language-conditioned policy trained on the
                result.
            camera_keys: Which keys in observation are camera images
            required_action_keys: Action column names this frame must
                supply a value for - typically the action keys of the robot
                being driven. A declared column in this set that ``action``
                omits raises ``ValueError`` rather than being written as a
                fabricated command; see
                :func:`unrecordable_action_columns_error`. ``None`` skips
                the check.

        Raises:
            ValueError: A column in ``required_action_keys`` is declared by
                the dataset schema but absent from ``action``.
            RecordingFrameError: The dataset write failed and this recorder is
                ``strict`` (the default). With ``strict=False`` the frame is
                counted in ``dropped_frame_count`` and a warning is logged
                instead.
        """
        if self._closed:
            return

        frame = {}

        # Detect camera vs state keys
        if camera_keys is None:
            camera_keys = [k for k, v in observation.items() if isinstance(v, np.ndarray) and v.ndim >= 2]

        state_keys = [k for k in observation.keys() if k not in camera_keys]

        # Camera images → observation.images.{name}
        for cam_key in camera_keys:
            img = observation[cam_key]
            if isinstance(img, np.ndarray):
                # LeRobot expects HWC uint8 for add_frame
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                frame[f"observation.images.{cam_key}"] = img

        # State → observation.state (flattened vector)
        # Use feature schema ordering to match the dataset schema declared in _build_features().
        if state_keys:
            state_vals = []
            if self._cached_state_keys is None:
                if self._state_source_keys is not None:
                    # Vector-expanded schema: read SOURCE keys (e.g. ``base_quat``);
                    # the list/ndarray branch below flattens each in schema order.
                    self._cached_state_keys = list(self._state_source_keys)
                else:
                    feat = self.dataset.features.get("observation.state", {})
                    state_names = feat.get("names", []) if isinstance(feat, dict) else getattr(feat, "names", [])
                    self._cached_state_keys = state_names if state_names else sorted(state_keys)

            for k in self._cached_state_keys:
                v = observation.get(k)
                if v is None:
                    state_vals.append(0.0)
                elif isinstance(v, (int, float)):
                    state_vals.append(float(v))
                elif isinstance(v, (np.generic, np.ndarray)) and v.ndim == 0:
                    # numpy scalars (np.float32/np.int32 from indexing a MuJoCo
                    # qpos/ctrl array) and 0-dim arrays are scalar state values.
                    state_vals.append(float(v))
                elif isinstance(v, (list, np.ndarray)):
                    arr = np.asarray(v, dtype=np.float32).flatten()
                    state_vals.extend(arr.tolist())
            if state_vals:
                frame["observation.state"] = np.array(state_vals, dtype=np.float32)

        # Action → flattened vector
        # Use feature schema ordering for actions too. Resolved before the
        # `if action:` guard so a frame carrying NO action for a column the
        # caller requires is refused rather than silently skipped. The sorted()
        # fallback still only runs for a non-empty action, so an observation-only
        # frame cannot cache an empty key list for the rest of the episode.
        if self._cached_action_keys is None:
            feat = self.dataset.features.get("action", {})
            action_names = feat.get("names", []) if isinstance(feat, dict) else getattr(feat, "names", [])
            if action_names:
                self._cached_action_keys = list(action_names)
            elif action:
                self._cached_action_keys = sorted(action.keys())

        gap = unrecordable_action_columns_error(action, self._cached_action_keys or [], required_action_keys)
        if gap is not None:
            raise ValueError(gap)

        if action:
            action_vals = []
            for k in self._cached_action_keys or []:
                v = action.get(k)
                if v is None:
                    action_vals.append(0.0)
                elif isinstance(v, (int, float)):
                    action_vals.append(float(v))
                elif isinstance(v, (np.generic, np.ndarray)) and v.ndim == 0:
                    # numpy scalars (np.float32/np.int32) and 0-dim arrays are
                    # scalar action values - see state branch above.
                    action_vals.append(float(v))
                elif isinstance(v, (list, np.ndarray)):
                    arr = np.asarray(v, dtype=np.float32).flatten()
                    action_vals.extend(arr.tolist())
            if action_vals:
                frame["action"] = np.array(action_vals, dtype=np.float32)

        # Task (mandatory for LeRobot v3)
        frame["task"] = task or self.default_task or "untitled"

        # Reconcile camera keys between frame and feature schema
        declared_cam_keys = {k for k in self.dataset.features if k.startswith("observation.images.")}

        # Apply the caller-supplied remap FIRST (observed stream name -> declared
        # schema name). This is the explicit escape hatch for the case where a
        # policy declares image_keys (e.g. "image"/"wrist_image") that differ
        # from the names the sim/hardware streams emit (e.g. "front_camera"/
        # "wrist_camera"). Without it those frames are stripped below and the
        # dataset records no image columns.
        if self.camera_key_map:
            prefix = "observation.images."
            for cam_key in [k for k in list(frame.keys()) if k.startswith(prefix)]:
                bare = cam_key[len(prefix) :]
                mapped = self.camera_key_map.get(bare)
                if mapped is not None and mapped != bare:
                    frame[f"{prefix}{mapped}"] = frame.pop(cam_key)

        # Normalize namespaced camera keys (e.g. "arm0/wrist_cam" → "arm0__wrist_cam")
        # to match the schema declared in _build_features. MuJoCo uses "/" as a
        # namespace separator for multi-robot cameras, but LeRobot feature names
        # cannot contain "/" (reserved for nested-feature addressing).
        frame_cam_keys = {k for k in list(frame.keys()) if k.startswith("observation.images.")}
        for cam_key in frame_cam_keys:
            normalized = camera_schema_key(cam_key)
            if normalized != cam_key and normalized in declared_cam_keys:
                frame[normalized] = frame.pop(cam_key)

        # Strip undeclared cameras (keys present in obs but not registered in
        # _build_features). This avoids LeRobot's "Extra features" error.
        # Declared-but-missing cameras (e.g. when a render fails) are left alone -
        # LeRobot tolerates absent columns and the episode simply won't have that
        # camera's data.
        frame_cam_keys_final = {k for k in frame if k.startswith("observation.images.")}
        stripped_cam_keys = frame_cam_keys_final - declared_cam_keys
        for extra in stripped_cam_keys:
            del frame[extra]

        # Surface the silent data-loss case: camera frames arrived but NONE of
        # them matched a declared schema key, so every image is being dropped
        # and the dataset will record zero image columns. This is the
        # "image_keys never match the streams" failure mode that otherwise
        # produces episodes with no video and no error. Warn once per recorder
        # (50Hz would flood) with the observed-vs-declared keys and the
        # camera_key_map remedy. A PARTIAL strip (some cameras matched) is left
        # quiet - that is the normal "ignore an extra debug camera" path.
        if (
            not self._warned_camera_mismatch
            and stripped_cam_keys
            and declared_cam_keys
            and not (frame_cam_keys_final & declared_cam_keys)
        ):
            self._warned_camera_mismatch = True
            logger.warning(
                "DatasetRecorder: none of the observed camera streams %s match the "
                "declared image features %s - all image frames are being dropped and "
                "this dataset will have no video. Pass camera_key_map={observed: declared} "
                "to remap (e.g. {%r: %r}), or declare cameras with names matching the streams.",
                sorted(k[len("observation.images.") :] for k in stripped_cam_keys),
                sorted(k[len("observation.images.") :] for k in declared_cam_keys),
                next(iter(sorted(stripped_cam_keys)))[len("observation.images.") :],
                next(iter(sorted(declared_cam_keys)))[len("observation.images.") :],
            )

        # Add to dataset
        try:
            self.dataset.add_frame(frame)
            self.frame_count += 1
            self.episode_frame_count += 1
        except Exception as e:
            if self.strict:
                # Fail-fast per AGENTS.md convention #5. Raised as
                # RecordingFrameError so a rollout driver does not absorb it
                # into the tolerance it grants a caller's telemetry hook - the
                # frame is lost, so silently continuing writes a short episode
                # and reports success. The original error is chained.
                raise RecordingFrameError(
                    f"dataset add_frame failed after {self.frame_count} frame(s) written; "
                    f"the recording is incomplete from this frame on "
                    f"(strict=True, so it is not dropped silently): {e}"
                ) from e
            self.dropped_frame_count += 1
            n = self.dropped_frame_count
            # Log at 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, then every 1000
            if (n & (n - 1)) == 0 or n % 1000 == 0:
                logger.warning(
                    "add_frame failed (frame %d, dropped %d): %s",
                    self.frame_count,
                    self.dropped_frame_count,
                    e,
                )

    def save_episode(self) -> dict[str, Any]:
        """Finalize current episode - writes parquet, encodes video, computes stats.

        LeRobot v3: save_episode() takes no task argument. Tasks are stored
        per-frame in the episode buffer via add_frame().

        Returns:
            Dict with episode info
        """
        if self._closed:
            return {"status": "error", "message": "Recorder closed"}

        try:
            self.dataset.save_episode()
            self.episode_count += 1
            # Report frames in THIS episode, not the cumulative total.
            # frame_count is monotonic across all episodes; episode_frame_count
            # is the count since the last save. Reset it after reporting.
            ep_frames = self.episode_frame_count
            total_frames = self.frame_count
            self.episode_frame_count = 0
            logger.info(
                "Episode %d saved: %d frames (%d total across dataset)",
                self.episode_count,
                ep_frames,
                total_frames,
            )
            return {
                "status": "success",
                "episode": self.episode_count,
                "episode_frames": ep_frames,
                "total_frames": total_frames,
            }
        except Exception as e:
            logger.error("save_episode failed: %s", e)
            # Mark recorder as poisoned - the LeRobot episode buffer is in
            # undefined state after a failed save. Subsequent add_frame calls
            # would silently corrupt the dataset. Close to prevent drift.
            self._closed = True
            return {"status": "error", "message": f"save_episode failed (recorder closed): {e}"}

    def clear_episode_buffer(self) -> bool:
        """Discard frames buffered for the current (unsaved) episode.

        After an aborted recording (e.g. a policy returned an empty action
        chunk mid-loop) the open episode buffer still holds the frames written
        so far. Without discarding them, the next ``add_frame`` appends to the
        half-episode and the eventual ``save_episode`` flushes a Frankenstein
        episode that mixes two runs. Call this to start the next episode at
        frame 0.

        LeRobot's buffer-reset surface drifted across 0.5.x, so this routes
        version-tolerantly:
          * ``LeRobotDataset.clear_episode_buffer()`` if exposed (preferred), else
          * reset via ``create_episode_buffer()`` if exposed, else
          * leave the buffer in place and warn (caller must ``stop_recording`` /
            ``save_episode`` to drain it before recording again).

        Discarded frames are un-counted from both ``frame_count`` and
        ``episode_frame_count``, because ``add_frame`` counts at buffer time
        while frames only reach disk on ``save_episode``. Both counters
        therefore describe frames that reached, or will reach, disk - never
        frames that were thrown away. When no clear surface is available the
        frames are still buffered and will be written by the
        ``stop_recording``/``save_episode`` drain the warning recommends, so
        neither counter changes in that case.

        Returns:
            True if the buffer was actively cleared; False if no clear surface
            was available (a warning is logged in that case).
        """
        cleared = False
        try:
            if hasattr(self.dataset, "clear_episode_buffer"):
                self.dataset.clear_episode_buffer()
                cleared = True
            elif hasattr(self.dataset, "create_episode_buffer"):
                self.dataset.episode_buffer = self.dataset.create_episode_buffer()
                cleared = True
        except Exception as e:  # noqa: BLE001 - best-effort discard; never mask the original abort
            logger.warning("clear_episode_buffer failed: %s", e)
            cleared = False

        if cleared:
            # The buffered frames are gone. They were counted optimistically by
            # add_frame (which counts at buffer time) but never reached disk,
            # since frames are only written by save_episode - so un-count them
            # from the cumulative total as well as the per-episode one. Both
            # counters describe frames that reached, or will reach, disk:
            # resume() seeds frame_count from dataset.meta.total_frames, and
            # stop_recording refuses an empty dataset by asking frame_count
            # whether anything was ever captured. Leaving the discarded frames
            # counted makes the recorder report a total no parquet row backs,
            # and blinds that refusal.
            self.frame_count -= self.episode_frame_count
            self.episode_frame_count = 0
        # When the discard did NOT happen the frames are still buffered, and the
        # warning below tells the caller to drain them with
        # stop_recording()/save_episode() - which writes them to disk. Nothing
        # was thrown away, so both counters are left as they are: the eventual
        # flush then reports the frames it really wrote, and a later successful
        # clear can still un-count them.

        if not cleared:
            logger.warning(
                "Could not auto-discard the partial episode buffer on this "
                "LeRobot version; call stop_recording()/save_episode() to drain "
                "it before the next recording to avoid a mixed episode."
            )
        return cleared

    def finalize(self) -> None:
        """Finalize the dataset (close parquet writers, flush metadata)."""
        if self._closed:
            return
        try:
            self.dataset.finalize()
        except Exception as e:
            logger.warning("finalize warning: %s", e)
        self._closed = True

    def push_to_hub(
        self,
        tags: list[str] | None = None,
        private: bool = False,
    ) -> dict[str, Any]:
        """Push dataset to HuggingFace Hub.

        Args:
            tags: Optional tags for the dataset
            private: Upload as private dataset. Must be a boolean.

        Refuses to publish an empty dataset (no frames written or no episode
        saved). Pushing then would create a Hub repo containing only
        ``meta/info.json`` (no parquet, no video) and silently pollute the
        namespace. The ``stop_recording`` facade has its own empty-dataset
        guard; this protects the direct-API path (and any caller that reaches
        ``push_to_hub`` after a rollout that never fed the recorder).

        Returns:
            Dict with push status. ``status="error"`` (no Hub call made) when
            the dataset is empty.

        ``private`` selects the published repository's visibility, so it is
        checked against :func:`~strands_robots.utils.boolean_flag_error`
        ahead of the empty-dataset state check and any Hub call: the flag is
        a property of this call rather than of the recorder, so the same
        mistake reports identically whether or not the dataset happens to be
        empty. It is otherwise forwarded verbatim to LeRobot, whose own
        parameter is ``bool | None`` where ``None`` means *use the namespace
        default* - a third visibility this signature's ``bool`` does not
        describe, and one a caller reading ``private: bool = False`` would not
        expect to select.
        """
        if flag_error := boolean_flag_error(private, "private", "push_to_hub"):
            return {"status": "error", "message": flag_error}
        if self.frame_count == 0 or self.episode_count == 0:
            msg = (
                f"refusing to push empty dataset {self.dataset.repo_id} "
                f"({self.frame_count} frames, {self.episode_count} episodes) - "
                "would create a Hub repo with only meta/info.json. Record frames "
                "with add_frame and flush at least one episode with save_episode "
                "before push_to_hub."
            )
            logger.error("push_to_hub aborted: %s", msg)
            return {"status": "error", "message": msg}
        try:
            self.dataset.push_to_hub(tags=tags, private=private)
            logger.info("Dataset pushed to hub: %s", self.dataset.repo_id)
            return {
                "status": "success",
                "repo_id": self.dataset.repo_id,
                "episodes": self.episode_count,
                "frames": self.frame_count,
            }
        except Exception as e:
            logger.error("push_to_hub failed: %s", e)
            return {"status": "error", "message": str(e)}

    def sync_to_bucket(
        self,
        bucket: str,  # "my-org/robot-fave"
        run_id: str | None = None,  # subpath; defaults to dataset name
        *,
        create: bool = True,
        private: bool = True,
        delete: bool = False,
    ) -> dict[str, Any]:
        """Sync the on-disk LeRobotDataset into an HF Storage Bucket (Phase 1/2).

        Thin delegate to :func:`sync_dataset_to_bucket` (the lifecycle-
        independent module-level helper, which holds the input validation and
        ``hf`` CLI orchestration), passing this recorder's dataset root and
        defaulting ``run_id`` to the dataset name (the last segment of
        ``repo_id``). On success the result is augmented with this recorder's
        ``episodes`` and ``frames`` counts.
        """
        result = sync_dataset_to_bucket(
            str(self.dataset.root),
            bucket,
            run_id=run_id or self.dataset.repo_id.split("/")[-1],
            create=create,
            private=private,
            delete=delete,
        )
        if result.get("status") == "success":
            result["episodes"] = self.episode_count
            result["frames"] = self.frame_count
        return result

    @property
    def repo_id(self) -> str:
        """The Hugging Face ``repo_id`` of the dataset being recorded."""
        return self.dataset.repo_id

    @property
    def root(self) -> str:
        """Filesystem path to the dataset's on-disk root directory, as a string."""
        return str(self.dataset.root)

    def __repr__(self) -> str:
        try:
            return f"DatasetRecorder(repo_id={self.repo_id}, episodes={self.episode_count}, frames={self.frame_count})"
        except AttributeError:
            return partial_construction_repr(self)


# Shared replay-episode helpers


def load_lerobot_episode(repo_id: str, episode: int = 0, root: str | None = None):
    """Load a LeRobotDataset and resolve the frame range for an episode.

    Args:
        repo_id: HuggingFace dataset id.
        episode: Episode index, a non-negative whole number. Any real scalar
            with an integral value is accepted (a ``2.0`` from a config, a
            ``np.int64`` from arithmetic); the value is coerced with ``int()``
            once the shared guard has round-tripped it, so an accepted index
            reaches the O(1) episode-row lookup rather than the last-resort
            frame scan a float index falls through to.
        root: Optional local dataset root override.

    Returns:
        Tuple of (dataset, episode_start, episode_length) on success.

    Raises:
        ImportError: If lerobot is not installed.
        ValueError: If the episode index is not a usable non-negative whole
            number, is out of range, or the resolved episode has no frames.
    """
    # The domain is the shared non-negative whole-number rule, not a bare
    # ``< 0`` test. That test gave a verdict to three classes of value it
    # could not actually honor:
    #
    # * ``bool`` passed it (``True < 0`` is False) and then indexed the
    #   episode table as an int, so ``episode=True`` resolved **episode 1**
    #   and returned it as a success - a different episode than any caller
    #   passing a flag could have meant.
    # * A non-integral or non-finite value passed it too and was blamed on
    #   the dataset after a full-length boundary scan ("Episode 2.5 has no
    #   frames"), naming the data rather than the index.
    # * A str/list/None reached the comparison itself and raised
    #   ``TypeError``, which is not the ``ValueError`` this function
    #   documents as its refusal channel.
    #
    # Shared with the ``replay_episode`` teleop knob rather than restated:
    # that parameter is the same quantity on a neighbouring surface, and
    # ``non_negative_whole_number_error`` already names it.
    if msg := non_negative_whole_number_error(episode, "episode", "load_lerobot_episode"):
        raise ValueError(msg)
    # Safe because the guard performed this coercion and compared the result
    # back; see its docstring on why the two steps are ordered this way.
    episode = int(episode)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=root)

    num_episodes = ds.meta.total_episodes if hasattr(ds.meta, "total_episodes") else len(ds.meta.episodes)
    if episode >= num_episodes:
        raise ValueError(f"Episode {episode} out of range (0-{num_episodes - 1})")

    episode_start = 0
    episode_length = 0
    try:
        ep_info = ds.meta.episodes[episode] if hasattr(ds.meta, "episodes") else {}
        if "dataset_from_index" in ep_info:
            # LeRobot 0.6 records the range on the episode's own metadata row,
            # so this is one row read whatever the index is. Every lerobot in
            # the declared range writes those columns, which is why this rung
            # leads: the two below it are compatibility fallbacks, and the
            # ``length`` accumulation reads one row per *preceding* episode to
            # recompute a number this row already states.
            episode_start = int(ep_info["dataset_from_index"])
            episode_length = int(ep_info["dataset_to_index"]) - episode_start
        elif hasattr(ds, "episode_data_index"):
            from_idx = ds.episode_data_index["from"][episode].item()
            to_idx = ds.episode_data_index["to"][episode].item()
            episode_start = from_idx
            episode_length = to_idx - from_idx
        else:
            for i in range(episode):
                prior_info = ds.meta.episodes[i] if hasattr(ds.meta, "episodes") else {}
                episode_start += prior_info.get("length", 0)
            episode_length = ep_info.get("length", 0)
    except Exception:
        # Last resort: scan frames to find episode boundaries
        for idx in range(len(ds)):
            frame = ds[idx]
            frame_ep = frame.get("episode_index", -1) if hasattr(frame, "get") else -1
            if hasattr(frame_ep, "item"):
                frame_ep = frame_ep.item()
            if frame_ep == episode:
                if episode_length == 0:
                    episode_start = idx
                episode_length += 1
            elif episode_length > 0:
                break

    if episode_length == 0:
        raise ValueError(f"Episode {episode} has no frames")

    return ds, episode_start, episode_length


def read_dataset_episode_indices(root: str | Path) -> dict[str, Any]:
    """Read episode-level ground truth from a LeRobot v3 dataset on disk.

    Parses every ``meta/episodes/**/*.parquet`` file under ``root`` and returns
    the recorded episode index set plus per-episode frame counts. This is the
    parquet source of truth used by :meth:`SimEngine.verify_dataset_episodes`
    to confirm a recording session produced the number of distinct episodes the
    caller intended (rather than one merged ``episode_index=0`` mega-episode).

    Pure ``pyarrow`` read - it does NOT import ``lerobot`` or instantiate a
    ``LeRobotDataset`` (which would re-validate/scan the whole dataset). Reads
    only the lightweight episode metadata parquet.

    Args:
        root: Dataset root directory (the dir that contains ``meta/``).

    Returns:
        Dict with:
          - ``episode_indices``: sorted list of distinct ``episode_index`` values.
          - ``total_episodes``: number of distinct episodes (``len`` of above).
          - ``total_frames``: sum of per-episode ``length`` (0 if unavailable).
          - ``frames_per_episode``: per-episode frame counts aligned to
            ``episode_indices`` (empty list if the ``length`` column is absent).
          - ``info_total_episodes``: the ``total_episodes`` recorded in
            ``meta/info.json`` (``None`` if that file is absent or unreadable).
            Returned alongside the parquet truth so callers can cross-check the
            two metadata sources for agreement - a healthy dataset has
            ``info_total_episodes == total_episodes``.
          - ``unreadable_files``: ``"<path relative to root>: <error>"`` for
            every ``meta/episodes`` parquet that could not be read (empty list
            for a healthy dataset). A partially-corrupt dataset - one truncated
            file out of twenty, the usual outcome of an interrupted sync or hub
            download - still yields the episode truth of the readable files, so
            callers can localise the damage instead of seeing zero episodes.
            The episode counts above cover ONLY the readable files, so any
            non-empty ``unreadable_files`` means the totals are a lower bound
            and the dataset must not be certified as complete.

    Raises:
        ImportError: If ``pyarrow`` is not installed.
        FileNotFoundError: If no ``meta/episodes`` parquet exists under ``root``
            (no episode was ever flushed - the dataset is empty/unfinalized).
        ValueError: If every ``meta/episodes`` parquet is unreadable, so there
            is no episode ground truth at all. The message lists each file and
            its read error.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - pyarrow ships with lerobot
        raise ImportError("read_dataset_episode_indices requires pyarrow (installed with the lerobot extra).") from e

    root_path = Path(root)
    parquet_files = sorted((root_path / "meta" / "episodes").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No meta/episodes parquet under {root_path}. The dataset is empty or was "
            "never finalized (episodes are flushed to parquet at stop_recording/finalize)."
        )

    pairs: list[tuple[int, int]] = []
    seen: set[int] = set()
    unreadable_files: list[str] = []
    readable_files = 0
    for pf in parquet_files:
        # A corrupt / truncated / foreign parquet raises ArrowInvalid (a
        # ValueError subclass); an unreadable one raises OSError. Damage is
        # usually confined to a few files (interrupted rsync, partial hub
        # download), so record which file failed and keep reading the rest -
        # aborting the whole read here would report zero episodes for a dataset
        # that is mostly intact and hide which file is actually broken.
        try:
            table = pq.read_table(pf)
        except (ValueError, OSError) as e:
            unreadable_files.append(f"{pf.relative_to(root_path)}: {e}")
            continue
        readable_files += 1
        cols = table.column_names
        if "episode_index" not in cols:
            continue
        data = table.to_pydict()
        ep_indices = data["episode_index"]
        lengths = data.get("length")
        for i, ep in enumerate(ep_indices):
            ep_int = int(ep)
            if ep_int in seen:
                continue
            seen.add(ep_int)
            length = int(lengths[i]) if lengths is not None and lengths[i] is not None else 0
            pairs.append((ep_int, length))

    if unreadable_files and readable_files == 0:
        # Nothing readable at all: there is no ground truth to return, so this
        # is a hard read failure rather than a partial one.
        detail = "; ".join(unreadable_files)
        raise ValueError(f"No readable meta/episodes parquet under {root_path}: {detail}")

    pairs.sort(key=lambda p: p[0])
    episode_indices = [p[0] for p in pairs]
    frames_per_episode = [p[1] for p in pairs]
    has_lengths = any(f > 0 for f in frames_per_episode)

    # Read meta/info.json total_episodes as a second, independent metadata
    # source. A healthy LeRobot dataset has info.json.total_episodes equal to
    # the distinct episode count in the parquet; a mismatch means the dataset
    # is internally inconsistent (e.g. an interrupted finalize), which
    # verify_dataset_episodes surfaces. Absent/corrupt info.json -> None (the
    # parquet remains the ground truth and is still reported).
    info_total_episodes: int | None = None
    info_path = root_path / "meta" / "info.json"
    if info_path.is_file():
        try:
            with info_path.open() as f:
                info_total_episodes = int(json.load(f)["total_episodes"])
        except (OSError, ValueError, KeyError, TypeError):
            info_total_episodes = None

    return {
        "episode_indices": episode_indices,
        "total_episodes": len(episode_indices),
        "total_frames": sum(frames_per_episode) if has_lengths else 0,
        "frames_per_episode": frames_per_episode if has_lengths else [],
        "info_total_episodes": info_total_episodes,
        "unreadable_files": unreadable_files,
    }
