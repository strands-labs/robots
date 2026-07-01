#!/usr/bin/env python3
"""Supply-chain audit for the project's declared dependencies.

Guards against three classes of packaging defect:

1. Dependency confusion / name hijacking -- a dependency whose distribution
   name is known to be unaffiliated with the upstream it claims to be (for
   example, the ``mimicgen`` name on PyPI is not NVlabs MimicGen, which has
   never published a release there). Such names are listed in ``DENYLIST`` and
   must never appear as a PyPI-sourced dependency in ``pyproject.toml``.

2. Nonexistent / typosquat names -- a PyPI-sourced dependency whose name does
   not resolve on PyPI at all. Catching these early stops a typo from becoming
   a future confusion target. This check hits the network and is only run when
   ``--check-pypi`` is passed.

3. Direct references -- a PEP 508 ``name @ <url>`` requirement (git/URL/file).
   Such a dependency lets the wheel build locally and pass ``twine check``, but
   the PyPI upload endpoint hard-rejects any distribution whose metadata carries
   a direct reference, so it silently breaks the release at the final upload
   step. A git-only dependency must be documented as a manual install, never
   declared as a dependency or extra of a distribution that ships to PyPI.

Git-sourced dependencies (``pkg @ git+https://...``) and self-references
(``strands-robots[...]``) are intentionally excluded: they never resolve from
PyPI, so neither check applies to them.

Exit code is 0 when the audit passes and 1 when any finding is reported, so the
script can gate CI directly.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

# Distribution names that must never be pinned as a PyPI dependency because the
# PyPI name is not controlled by the upstream project it appears to reference.
# Map name -> reason (surfaced in the failure message).
DENYLIST: dict[str, str] = {
    "mimicgen": (
        "NVlabs MimicGen has no PyPI release; the 'mimicgen' name on PyPI is an "
        "unaffiliated package (dependency-confusion risk). Install from source: "
        'pip install "mimicgen @ git+https://github.com/NVlabs/mimicgen.git"'
    ),
}

# Leading distribution name in a PEP 508 requirement string.
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _canonical(name: str) -> str:
    """Return the PEP 503 canonical (lowercased, dash-normalized) name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def collect_pypi_dependencies(pyproject_path: Path) -> dict[str, str]:
    """Return {canonical_name: original_requirement} for PyPI-sourced deps.

    Git/URL-sourced requirements (``name @ <url>``) and self-references to
    ``strands-robots[...]`` extras are excluded because they never resolve from
    PyPI.
    """
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    own_name = _canonical(project.get("name", ""))
    requirements: list[str] = list(project.get("dependencies", []))
    for deps in project.get("optional-dependencies", {}).values():
        requirements.extend(deps)

    found: dict[str, str] = {}
    for req in requirements:
        stripped = req.strip()
        name_match = _NAME_RE.match(stripped)
        if name_match and own_name and _canonical(name_match.group(1)) == own_name:
            continue  # self-reference to another extra (e.g. pkg[all])
        if "@" in stripped:
            continue  # direct URL / git reference, not from PyPI
        match = _NAME_RE.match(stripped)
        if not match:
            continue
        found.setdefault(_canonical(match.group(1)), stripped)
    return found


def check_denylist(deps: dict[str, str]) -> list[str]:
    """Return a list of failure messages for any denylisted dependency."""
    findings = []
    for name, reason in DENYLIST.items():
        canonical = _canonical(name)
        if canonical in deps:
            findings.append(f"DENYLISTED '{deps[canonical]}': {reason}")
    return findings


def _exists_on_pypi(canonical_name: str, retries: int = 3) -> bool:
    """Return True if the name resolves on PyPI (404 -> False).

    Transient network / server errors are retried and, if still failing, treated
    as inconclusive (True) so the audit never fails the build on flakiness -- a
    definitive 404 is the only signal that fails it.
    """
    url = f"https://pypi.org/pypi/{canonical_name}/json"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=15) as resp:
                return bool(resp.status == 200)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            # 5xx / rate limit: retry, then treat as inconclusive.
        except (urllib.error.URLError, TimeoutError, OSError):
            pass  # network hiccup: retry, then inconclusive
    return True


def check_pypi_existence(deps: dict[str, str]) -> list[str]:
    """Return failure messages for PyPI-sourced deps that 404 on PyPI."""
    findings = []
    for name, req in sorted(deps.items()):
        if not _exists_on_pypi(name):
            findings.append(f"NOT ON PYPI '{req}': name '{name}' returns 404 (typo or unregistered?)")
    return findings


def collect_all_requirements(pyproject_path: Path) -> list[str]:
    """Return every declared requirement string (core deps + all extras)."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    requirements: list[str] = list(project.get("dependencies", []))
    for deps in project.get("optional-dependencies", {}).values():
        requirements.extend(deps)
    return requirements


def check_direct_references(pyproject_path: Path) -> list[str]:
    """Return failure messages for any direct-reference (URL/VCS/file) dependency.

    A PEP 508 direct reference (``name @ <url>``, e.g. ``pkg @ git+https://...``)
    is the only place ``@`` may appear in a requirement string -- version
    specifiers never use it -- so its presence unambiguously marks a direct
    reference. PyPI rejects any uploaded distribution whose metadata carries one,
    which is why declaring such a dependency (even behind ``allow-direct-references``)
    breaks the publish at the upload step while passing ``twine check``.
    """
    findings = []
    for req in collect_all_requirements(pyproject_path):
        stripped = req.strip()
        if "@" in stripped:
            findings.append(
                f"DIRECT REFERENCE '{stripped}': PyPI rejects distributions whose "
                "metadata carries a VCS/URL reference, so this breaks the publish "
                "at the upload step. Document a manual git install instead of "
                "declaring it as a dependency or extra."
            )
    return findings


def audit(pyproject_path: Path, check_pypi: bool = False) -> list[str]:
    """Run all enabled checks and return the combined findings list."""
    deps = collect_pypi_dependencies(pyproject_path)
    findings = check_denylist(deps)
    findings.extend(check_direct_references(pyproject_path))
    if check_pypi:
        findings.extend(check_pypi_existence(deps))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "pyproject.toml",
        help="Path to pyproject.toml (default: repo root).",
    )
    parser.add_argument(
        "--check-pypi",
        action="store_true",
        help="Also verify each PyPI-sourced dependency name resolves on PyPI (network).",
    )
    args = parser.parse_args(argv)

    findings = audit(args.pyproject, check_pypi=args.check_pypi)
    if findings:
        print("Dependency audit FAILED:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print("Dependency audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
