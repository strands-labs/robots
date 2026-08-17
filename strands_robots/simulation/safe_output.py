"""Sandboxed validation for LLM-supplied simulation output paths.

Several simulation entry points persist an artifact to a caller-chosen
filesystem path that originates from an untrusted (LLM tool-call) source:

* ``render(output_path=...)`` - a PNG still,
* ``export_xml(output_path=...)`` - the scene as canonical MJCF,
* ``run_policy(video={"path": ...})`` - a rollout MP4,
* ``start_cameras_recording(output_dir=..., name=...)`` - per-camera MP4s.

Left unchecked, such a path is an arbitrary-write vector: ``..`` traversal,
a symlinked target, shell metacharacters interpolated into a later command, or
a ``name`` tag carrying path separators can all escape the intended location.
This module centralizes the guards so every sink shares one implementation
instead of each re-deriving a partial blacklist.

Two confinement policies are supported, selected per sink:

* **Sandbox (default-on)** - the resolved path must live under a sandbox root
  (used by ``render``, whose ``output_path`` is a newer, sandboxed-by-design
  feature). Pass a non-``None`` ``sandbox_root`` with ``allow_abs=False``.
* **Guards-only (opt-in sandbox)** - absolute paths are permitted (the historic
  video/recording contract, and ``export_xml``'s) but the metacharacter,
  backslash, symlink, and
  name-traversal guards still apply. Pass ``sandbox_root=None`` (or
  ``allow_abs=True``); callers opt in to confinement by supplying a root.
"""

import contextlib
import os
import tempfile
from pathlib import Path

# Shell metacharacters / control bytes that must never appear in an LLM-supplied
# path: they enable command injection if the path is later interpolated into a
# shell and serve no purpose in a legitimate filesystem path.
PATH_BAD_CHARS = frozenset({";", "|", "$", "`", ">", "<", "\n", "\r", "\x00"})

# Environment variable that re-permits absolute paths for the video / camera-
# recording sinks. Named here so :func:`video_sandbox_args` can hand the exact
# spelling to :func:`validate_output_path`, which quotes it in the refusal: a
# caller told only that "a *_ALLOW_ABS variable" exists has to grep the package
# to act on the message.
_VIDEO_ALLOW_ABS_ENV = "STRANDS_ROBOTS_VIDEO_ALLOW_ABS"


def env_flag(name: str) -> bool:
    """Return True when env var ``name`` is a truthy opt-in (``1``/``true``/``yes``)."""
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes")


def resolve_sandbox_root(env_var: str, default_subdir: str) -> Path:
    """Resolve a sandbox root from ``env_var`` (read at call time).

    Falls back to ``~/.strands_robots/<default_subdir>`` when ``env_var`` is
    unset. The result is fully resolved (``..`` normalized, symlinks expanded)
    so confinement checks compare true on-disk locations.
    """
    raw = os.getenv(env_var) or str(Path.home() / ".strands_robots" / default_subdir)
    return Path(raw).expanduser().resolve(strict=False)


def _reject_unsafe_chars(value: str, *, label: str) -> None:
    """Reject empty strings, shell metacharacters, and backslash separators."""
    if not value or not value.strip():
        raise ValueError(f"unsafe {label}: empty")
    if any(b in value for b in PATH_BAD_CHARS):
        raise ValueError(f"unsafe {label}: shell metacharacters")
    # Backslash is not a POSIX separator, so "..\\..\\etc" would survive a
    # "/"-only traversal check. Reject it outright rather than guess intent.
    if "\\" in value:
        raise ValueError(f"unsafe {label}: backslash separators not allowed")


def _confinement_refusal(
    output_path: str,
    *,
    resolved: Path,
    sandbox_root: Path,
    label: str,
    opt_in: str,
) -> str:
    """Build the confinement refusal, matched to the class of path the caller gave.

    An absolute destination outside the sandbox and a *relative* one are the same
    ``relative_to`` failure but not the same caller mistake, so one shared
    sentence can only fit the absolute case. A relative path is resolved against
    the process CWD, so the refusal for one reported a CWD-absolute path the
    caller never supplied and then offered an absolute-path opt-in - a remedy
    for a class of input that was not given. Following it does lift the refusal,
    but it disables confinement and leaves the artifact under the CWD, i.e.
    anywhere except the sandbox the caller was told the sink writes to.

    The relative wording therefore states where the path was resolved from and
    quotes the sandbox-anchored spelling to pass instead, keeping the opt-in as
    the last resort it is. This is the same rule the traversal / metacharacter
    refusals already follow: only offer a remedy that achieves what was asked.

    Args:
        output_path: The caller's original, unresolved destination string.
        resolved: The fully-resolved destination that failed confinement.
        sandbox_root: Root the destination had to live under.
        label: Noun for the parameter (``"output_path"`` / ``"output_dir"``).
        opt_in: Pre-rendered instruction naming this sink's absolute-path opt-in.

    Returns:
        The refusal message.
    """
    raw = Path(output_path).expanduser()
    if raw.is_absolute():
        return (
            f"{label} {resolved} is outside the sandbox {sandbox_root} "
            f"({opt_in} to permit absolute paths, or pass a path under the sandbox)"
        )
    remedies = []
    # The sandbox-anchored spelling of exactly what the caller asked for. Omitted
    # when it is the destination that just failed: a one-component name is
    # already anchored there, so re-suggesting it would loop the caller.
    suggested = (sandbox_root / raw).resolve(strict=False)
    if suggested != resolved:
        remedies.append(f"pass {str(suggested)!r} to write it into the sandbox")
    remedies.append(f"or a bare {label} with no directory part, which is written into the sandbox")
    remedies.append(f"or {opt_in} to write outside the sandbox instead")
    return (
        f"{label} {output_path!r} is relative, so it resolved against the current "
        f"directory {Path.cwd()} to {resolved}, which is outside the sandbox "
        f"{sandbox_root} (" + ", ".join(remedies) + ")"
    )


def validate_output_path(
    output_path: str,
    *,
    sandbox_root: Path | None,
    allow_abs: bool,
    label: str = "output_path",
    allow_abs_env: str | None = None,
) -> Path:
    """Validate and resolve an LLM-supplied output path (file or directory).

    Always rejects shell metacharacters, backslash separators, and a symlinked
    target. Normalizes ``..`` via ``resolve()``. When ``sandbox_root`` is given
    and ``allow_abs`` is False, the resolved path must live under it.

    A bare filename (one component, no separator, e.g. ``"frame.png"``) is
    anchored to ``sandbox_root`` rather than the process CWD whenever
    confinement is active. Without that, the most natural call a caller can
    make is refused for naming a CWD path it never supplied. The anchoring
    cannot widen the sandbox: it applies only to a single-component name, and
    every guard above still runs against the anchored destination. In
    guards-only mode (``sandbox_root=None`` or ``allow_abs=True``) a relative
    name stays CWD-relative, preserving the historic video/recording contract.

    Args:
        output_path: Caller-supplied destination path.
        sandbox_root: Directory the path must resolve under, or ``None`` to skip
            confinement (guards-only mode).
        allow_abs: When True, skip the sandbox confinement check even if
            ``sandbox_root`` is provided (explicit absolute-path opt-in).
        label: Noun used in error messages (e.g. ``"output_path"``/``"output_dir"``).
        allow_abs_env: Name of the environment variable that sets ``allow_abs``
            for this sink, quoted verbatim in the confinement refusal so the
            caller can act on the message without searching for the spelling.
            Every sink that establishes confinement supplies it; ``None`` (a
            sink with no opt-in variable) falls back to naming the convention.

    Returns:
        The fully-resolved destination ``Path``.

    Raises:
        ValueError: If the path is unsafe (callers map this to a tool error).
    """
    _reject_unsafe_chars(output_path, label=label)

    raw = Path(output_path).expanduser()
    # Reject ".." traversal segments outright (defense-in-depth): even in
    # guards-only mode (no sandbox root) this blocks escaping the intended
    # directory, while plain absolute paths without ".." remain permitted.
    if ".." in raw.parts:
        raise ValueError(f"unsafe {label}: path traversal")
    # A bare filename ("frame.png") names no directory, so resolving it against
    # the process CWD is a guess - and under confinement it is a guess that
    # always fails, refusing the call with a CWD-absolute path the caller never
    # supplied. Anchor it to the sandbox root instead: a one-component name
    # cannot escape it, and ".." is refused above by a check that scans every
    # part (so it holds for the joined path too). Done BEFORE the symlink probe
    # below because that guard must inspect the destination actually opened -
    # after it, a link planted inside the sandbox would be probed at the CWD
    # path instead and followed.
    if sandbox_root is not None and not allow_abs and not raw.is_absolute() and len(raw.parts) == 1:
        raw = sandbox_root / raw
    # Refuse to follow a symlink planted at the target (arbitrary-write vector).
    if raw.is_symlink():
        raise ValueError(f"{label} {output_path!r} is a symlink - refusing to follow")

    # resolve() normalizes "..", expands the chain, and follows any intermediate
    # symlinks, so the confinement check below sees the true on-disk destination.
    resolved = raw.resolve(strict=False)

    if sandbox_root is not None and not allow_abs:
        try:
            resolved.relative_to(sandbox_root)
        except ValueError as e:
            # Name the variable that lifts this refusal. The glob this replaced
            # ("set the corresponding *_ALLOW_ABS env var") told the caller a
            # pattern rather than a name, so acting on the message meant
            # grepping the package for the sink's actual spelling -- a dead end
            # in the one place the caller most needs a next step. Every
            # confining sink knows its own variable, so the message can state
            # it, matching the sibling env-var refusal in
            # ``strands_robots.simulation.isaac.config`` (which interpolates the
            # variable name it read).
            opt_in = f"set {allow_abs_env}=1" if allow_abs_env else "set this sink's *_ALLOW_ABS env var"
            raise ValueError(
                _confinement_refusal(
                    output_path,
                    resolved=resolved,
                    sandbox_root=sandbox_root,
                    label=label,
                    opt_in=opt_in,
                )
            ) from e
    return resolved


def video_sandbox_args() -> tuple[Path | None, bool, str]:
    """Return ``(sandbox_root, allow_abs, allow_abs_env)`` for video / recording paths.

    Unlike ``render``, the video and camera-recording sinks have historically
    accepted arbitrary absolute paths, so confinement is OPT-IN: when
    ``STRANDS_ROBOTS_VIDEO_ROOT`` is set, writes are confined to it (and
    ``STRANDS_ROBOTS_VIDEO_ALLOW_ABS`` re-permits absolute paths inside that
    mode); otherwise absolute paths are allowed. The metacharacter, backslash,
    symlink, and traversal guards in :func:`validate_output_path` apply in
    either mode. The third element is the opt-in variable's name, passed
    through to :func:`validate_output_path` so a confinement refusal quotes the
    exact spelling instead of a ``*_ALLOW_ABS`` pattern.
    """
    if os.getenv("STRANDS_ROBOTS_VIDEO_ROOT"):
        return (
            resolve_sandbox_root("STRANDS_ROBOTS_VIDEO_ROOT", "videos"),
            env_flag(_VIDEO_ALLOW_ABS_ENV),
            _VIDEO_ALLOW_ABS_ENV,
        )
    return None, True, _VIDEO_ALLOW_ABS_ENV


def sanitize_name_component(name: str, *, label: str = "name") -> str:
    """Validate a filename tag that will be interpolated into an output path.

    A ``name`` that carries path separators or ``..`` can escape the directory
    it is joined into (e.g. ``name="../../etc/x"`` -> a write outside
    ``output_dir``). Reject separators, traversal, metacharacters, and leading
    dots rather than silently sanitizing, so the caller sees the rejection.

    Args:
        name: Caller-supplied filename tag.
        label: Noun used in error messages.

    Returns:
        The validated ``name`` unchanged.

    Raises:
        ValueError: If ``name`` is unsafe as a single path component.
    """
    _reject_unsafe_chars(name, label=label)
    if "/" in name or "\\" in name:
        raise ValueError(f"unsafe {label}: path separators not allowed")
    if name in (".", "..") or name.startswith(".."):
        raise ValueError(f"unsafe {label}: path traversal")
    return name


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically with owner-only permissions.

    Writes to a temp file in the destination directory then ``os.replace``s it
    into place, so a crash mid-write cannot truncate or corrupt an existing file
    at ``path``. A newly created parent directory is ``0o700`` and the final
    file is ``0o600`` (the sim output roots are private to the running user).
    """
    parent = path.parent
    parent_existed = parent.exists()
    parent.mkdir(parents=True, exist_ok=True)
    if not parent_existed:
        with contextlib.suppress(OSError):
            os.chmod(parent, 0o700)

    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    os.chmod(path, 0o600)
