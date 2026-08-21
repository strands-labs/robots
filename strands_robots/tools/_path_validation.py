"""Shared path validation utilities for tools that write to the filesystem.

Provides two helpers all tool modules can import to reject dangerous path values
before any I/O occurs, one per half of a write target:

* :func:`validate_save_path` validates the *directory* a tool writes into.
* :func:`resolve_output_path` resolves the *file name* a caller asks for inside
  that directory, and refuses a name that leaves it.

Both are needed because a tool composes its write target from the two, and
validating only the directory leaves the composition unchecked - the defect
:func:`resolve_output_path` exists to close.

Cross-platform: blocks sensitive directories on Linux, macOS, and Windows.
"""

import os
import re
import sys

# Characters that have no business appearing in file paths supplied by tool callers.
_DANGEROUS_CHARS = re.compile(r"[\x00]")

# Well-known sensitive system directories that tool callers should never write to.
# Each entry ends with '/' (or '\' on Windows) so ``str.startswith`` only matches
# paths *inside* the directory, not unrelated paths that share a common prefix
# (e.g. "/var/spool/crondata" should NOT match "/var/spool/cron/").
_LINUX_BLOCKED_PREFIXES = (
    "/etc/",
    "/usr/",
    "/bin/",
    "/sbin/",
    "/boot/",
    "/dev/",
    "/proc/",
    "/sys/",
    "/var/spool/cron/",
    "/var/spool/at/",
)

_MACOS_BLOCKED_PREFIXES = (
    "/System/",
    "/Library/LaunchDaemons/",
    "/Library/LaunchAgents/",
)

_WINDOWS_BLOCKED_PREFIXES = (
    "C:\\Windows\\",
    "C:\\Program Files\\",
    "C:\\Program Files (x86)\\",
)


def _get_blocked_prefixes() -> tuple[str, ...]:
    """Return blocked prefixes for the current platform.

    On macOS, many system directories (``/etc``, ``/var``, ``/tmp``) are
    symlinks into ``/private/``. Since :func:`validate_save_path` compares
    against ``os.path.realpath`` output, we must include the ``/private/``-
    prefixed variants so that ``/etc/passwd`` (which resolves to
    ``/private/etc/passwd``) is still rejected.
    """
    if sys.platform == "win32":
        return _WINDOWS_BLOCKED_PREFIXES
    elif sys.platform == "darwin":
        private_variants = tuple("/private" + p for p in _LINUX_BLOCKED_PREFIXES)
        return _LINUX_BLOCKED_PREFIXES + private_variants + _MACOS_BLOCKED_PREFIXES
    else:
        return _LINUX_BLOCKED_PREFIXES


BLOCKED_PREFIXES = _get_blocked_prefixes()


def validate_save_path(path: str, *, label: str = "path") -> str:
    """Validate and resolve a user-supplied file-system path.

    Rejects paths that contain:
    - Null bytes (``\\x00``)
    - ``..`` traversal components

    Then resolves the path to an absolute form via ``os.path.realpath``
    and ensures it does **not** escape into well-known sensitive directories.

    Cross-platform: validates against OS-specific blocked directories on
    Linux, macOS, and Windows.

    Args:
        path: The raw path string from the tool caller.
        label: A human-readable name for error messages (e.g. ``"save_path"``).

    Returns:
        The validated, resolved absolute path.

    Raises:
        ValueError: If the path fails any validation check.
    """
    if not path:
        raise ValueError(f"{label} must not be empty")

    if _DANGEROUS_CHARS.search(path):
        raise ValueError(f"{label} contains invalid characters")

    # Reject explicit '..' components (before resolution to catch intent)
    parts = path.replace("\\", "/").split("/")
    if ".." in parts:
        raise ValueError(f"{label} must not contain '..' path traversal components")

    # Resolve to absolute path (follows symlinks)
    resolved = os.path.realpath(os.path.expanduser(path))

    # Ensure resolved path ends with separator for directory-prefix matching
    sep = "\\" if sys.platform == "win32" else "/"
    check_path = resolved if resolved.endswith(sep) else resolved + sep

    for prefix in BLOCKED_PREFIXES:
        if check_path.startswith(prefix):
            raise ValueError(f"{label} resolves to a protected system directory ({prefix}): {resolved}")

    return resolved


def resolve_output_path(directory: str, name: str, *, label: str = "filename") -> str:
    """Resolve ``name`` as a file inside the already-validated ``directory``.

    :func:`validate_save_path` validates the directory a tool was told to write
    into. It cannot validate the *file*, because the file name is composed
    afterwards from separate caller-supplied parts - a ``filename`` and often an
    extension - and ``os.path.join`` happily walks back out of the directory it
    was given. So a tool that validated its ``save_path`` and then joined a name
    onto it wrote wherever the name pointed:

    ``os.path.join("/captures", "../../etc/x" + ".jpg")`` -> ``/etc/x.jpg``

    which is the traversal ``validate_save_path`` refuses in its own argument,
    reached through the part that was never checked. This helper is the second
    half of that pair, and it is deliberately a *containment* check rather than a
    character allowlist: an allowlist has to guess which spellings are dangerous
    on which platform, and it would refuse working requests. What is actually
    required is narrower and exactly statable - the resolved write target must lie
    inside the resolved directory - so that is what is asserted, after resolution,
    on the composed path. Resolving with :func:`os.path.realpath` also means a
    symlink inside the directory cannot be used to step outside it.

    A name that stays inside the directory is accepted whatever it is spelled
    with, including one that names a subdirectory (``"a/b.jpg"``). That is not an
    oversight: such a name is contained, so it is not this function's concern, and
    if the subdirectory does not exist the write fails loudly at the call site
    rather than silently landing somewhere else.

    Args:
        directory: The directory to resolve within. Pass the value returned by
            :func:`validate_save_path`, not the raw caller-supplied one.
        name: The file name to resolve, composed of whatever caller-supplied
            parts the tool builds it from.
        label: A human-readable name for error messages (e.g. ``"filename"``).

    Returns:
        The resolved absolute path of the file, guaranteed to be inside
        ``directory``.

    Raises:
        ValueError: If ``name`` is empty, contains a null byte, or resolves to a
            location that is not inside ``directory``.
    """
    if not name:
        raise ValueError(f"{label} must not be empty")

    if _DANGEROUS_CHARS.search(name):
        raise ValueError(f"{label} contains invalid characters")

    root = os.path.realpath(directory)
    resolved = os.path.realpath(os.path.join(root, name))

    # ``commonpath`` rather than ``startswith``: the latter matches a sibling
    # directory that merely shares a prefix ("/captures2" against "/captures").
    # Both operands are absolute here, so it cannot raise on mixed forms.
    if os.path.commonpath([root, resolved]) != root:
        raise ValueError(
            f"{label}: {name!r} resolves to {resolved}, which is outside the directory it must be written into ({root})"
        )

    # Reported separately: this one *is* inside the directory, so the message
    # above would have to say it is outside it, which is a contradiction rather
    # than a diagnosis.
    if resolved == root:
        raise ValueError(f"{label}: {name!r} names the directory {root} itself, not a file inside it")

    return resolved
