"""Repo hygiene: block host-specific absolute paths from being committed.

History: PR #85 shipped a hardcoded ``/Users/cagatay/robots/...`` in
``tests/simulation/mujoco/test_agenttool_contract.py`` that passed on the
author's laptop, got committed, and was only caught by CI because CI happens
to not live at that path.

This test is a cheap regex sweep over ``strands_robots/`` and ``tests/`` that
fails fast if anyone re-introduces a ``/Users/<name>/``, ``/home/<name>/`` or
``C:\\Users\\`` string. Prefer module-relative paths, ``pathlib.Path`` +
``__file__``, ``importlib.resources``, or fixtures.

Allowlist patterns live below - keep it narrow.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories to scan (source + tests; not docs, not third-party).
SCAN_DIRS = ("strands_robots", "tests", "tests_integ")

# Patterns that indicate a hardcoded host-specific user path.
HOST_PATH_PATTERNS = [
    # POSIX home directories with a specific user segment
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    # Windows user profile
    re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+\\\\"),
    re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+\\"),
]

# Explicit allowlist - files or string occurrences that are ABOUT these patterns
# (documentation, validators themselves, regex sources).
ALLOWED_FILES = {
    # This test itself defines the patterns above.
    "tests/test_no_host_paths.py",
    # Path validation logic *contains* Windows system paths as blocklist entries;
    # those are C:\Windows\, C:\Program Files\ - not user profiles.
    "strands_robots/tools/_path_validation.py",
    "tests/tools/test_path_validation.py",
    # Container volume-safety tests contain protected host paths as test data
    # (the test asserts that the production code REJECTS these paths).
    "tests/tools/test_gr00t_container_hardening.py",
    # Protected host paths (incl. //home/<u>/.aws as a
    # leading-double-slash bypass vector) as attack input; each assertion proves
    # the production guard REJECTS the path rather than using it.
    "tests/tools/test_gr00t_pentest_regressions.py",
}


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            # Skip bytecode caches and anything inside .venv / build dirs
            if "__pycache__" in p.parts or ".venv" in p.parts:
                continue
            files.append(p)
    return files


def test_no_host_specific_absolute_paths() -> None:
    """Fail if any .py file contains ``/Users/<name>/`` or ``/home/<name>/``.

    If you need a path in a test, use module-relative resolution:

        Path(__file__).parent / "fixture.json"

    or the existing module constants:

        from strands_robots.simulation.mujoco import simulation
        simulation._TOOL_SPEC_PATH
    """
    offenders: list[tuple[str, int, str]] = []

    for path in _iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED_FILES:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for pat in HOST_PATH_PATTERNS:
                if pat.search(line):
                    offenders.append((rel, lineno, line.strip()[:120]))
                    break

    if offenders:
        msg = ["Host-specific absolute paths detected (use Path(__file__) or fixtures instead):"]
        for rel, lineno, snippet in offenders:
            msg.append(f"  {rel}:{lineno}: {snippet}")
        raise AssertionError("\n".join(msg))
