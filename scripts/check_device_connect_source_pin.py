#!/usr/bin/env python3
"""Report where the device-connect packages in *this* interpreter came from.

Why this exists
---------------
``test-lint.yml`` redirects ``device-connect-edge`` and
``device-connect-agent-tools`` to a git source when a pull request touches
``strands_robots/device_connect/``, by writing a ``uv`` override file and
exporting ``UV_OVERRIDE`` before the hatch test environment is created. The
step then prints ``Pinned device-connect packages to <repo>@<ref>``.

That line is an announcement, not a measurement. Nothing in the job log said
which distribution the suite actually imported: hatch creates its environment
silently, so no ``Resolved``/``Installed`` lines appear for it, while the
*outer* ``pip install -e ".[all,dev]"`` into the runner interpreter - which
reads no ``UV_OVERRIDE`` and installs nothing the suite imports - prints a
``Downloading device_connect_edge-<version>-py3-none-any.whl`` line in full
view. Issue #3222 read that line as proof the pin was never consulted. It was
consulted (measured: a hatch ``installer = "uv"`` environment created under
``UV_OVERRIDE`` records the git origin in the distribution's
``direct_url.json``), but the log carried no evidence either way, and a claim
CI cannot support is one a reviewer has to take on trust in both directions.

This script is that evidence. Run it *inside* the environment under test::

    hatch run python scripts/check_device_connect_source_pin.py \\
        --repo arm/device-connect --ref main

It reads each distribution's ``direct_url.json`` - the PEP 610 record every
installer writes for a non-registry install and omits for a registry wheel - and
exits 1 unless every distribution names the requested repository and revision.
Run under the runner interpreter instead, it reports the wheel the outer
``pip install`` fetched, and fails: that interpreter is not the one under test,
which is exactly the misreading above. The workflow step therefore goes through
``hatch run``, and ``tests/test_device_connect_source_pin_is_verified.py`` pins
that it does.

Outcomes
--------
``pinned-to-source``
    ``direct_url.json`` is present, ``vcs`` is ``git``, the URL names
    ``https://github.com/<repo>`` (with or without ``.git``) and
    ``requested_revision`` is the requested ref.
``registry-wheel``
    The distribution is installed and carries no ``direct_url.json``, which is
    what a wheel resolved from an index looks like.
``other-source``
    ``direct_url.json`` is present but names a different URL, revision or VCS.
``not-installed``
    No distribution of that name is importable from this interpreter.

Only ``pinned-to-source`` passes. The report names the origin that was found so
a red step says what *was* loaded rather than only that it was not the pin.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from dataclasses import dataclass

#: The two distributions the workflow's override file redirects. The default for
#: ``--distribution`` rather than a constant the caller cannot change, so the
#: probe in the test module can point the same verdict at a small stand-in.
DEFAULT_DISTRIBUTIONS = ("device-connect-edge", "device-connect-agent-tools")

PINNED = "pinned-to-source"
REGISTRY = "registry-wheel"
OTHER = "other-source"
MISSING = "not-installed"


@dataclass(frozen=True)
class Origin:
    """Where one installed distribution came from, as its installer recorded it."""

    distribution: str
    outcome: str
    version: str
    detail: str


def expected_urls(repo: str) -> frozenset[str]:
    """The URL spellings that name ``repo`` on GitHub.

    ``uv`` records the URL as the override spelled it, and the workflow spells it
    with ``.git``; the bare form is accepted too so a later edit to the override
    that drops the suffix is not read as a different repository.
    """
    base = f"https://github.com/{repo.strip('/')}"
    return frozenset({base, f"{base}.git"})


def classify(
    distribution: str,
    *,
    repo: str,
    ref: str,
    version: str | None,
    direct_url_text: str | None,
) -> Origin:
    """Classify one distribution from its version and ``direct_url.json`` text.

    ``version`` is ``None`` when the distribution is not installed at all;
    ``direct_url_text`` is ``None`` when it is installed with no PEP 610 record.
    Pure so the test module can drive every outcome without an installer.
    """
    if version is None:
        return Origin(distribution, MISSING, "", "no distribution of this name is importable")
    if direct_url_text is None:
        return Origin(distribution, REGISTRY, version, "no direct_url.json: a wheel resolved from an index")
    try:
        record = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        return Origin(distribution, OTHER, version, f"direct_url.json is not JSON: {exc}")
    url = str(record.get("url", ""))
    vcs_info = record.get("vcs_info") or {}
    vcs = str(vcs_info.get("vcs", ""))
    requested = str(vcs_info.get("requested_revision", ""))
    commit = str(vcs_info.get("commit_id", ""))
    found = f"{vcs or 'non-vcs'} {url}@{requested or '<no requested_revision>'}"
    if commit:
        found += f" ({commit[:12]})"
    if vcs == "git" and url in expected_urls(repo) and requested == ref:
        return Origin(distribution, PINNED, version, found)
    return Origin(distribution, OTHER, version, found)


def inspect(distribution: str, *, repo: str, ref: str) -> Origin:
    """Classify ``distribution`` as installed in the running interpreter."""
    try:
        dist = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return classify(distribution, repo=repo, ref=ref, version=None, direct_url_text=None)
    return classify(
        distribution,
        repo=repo,
        ref=ref,
        version=dist.version,
        direct_url_text=dist.read_text("direct_url.json"),
    )


def render(origins: list[Origin], *, repo: str, ref: str) -> str:
    """Render the report as markdown, the shape the step summary renders."""
    lines = [
        "## Device-connect source pin",
        "",
        f"Expected every distribution below to come from `https://github.com/{repo}.git@{ref}`.",
        f"Interpreter: `{sys.executable}`",
        "",
        "| distribution | outcome | version | origin found |",
        "|---|---|---|---|",
    ]
    for origin in origins:
        lines.append(f"| `{origin.distribution}` | {origin.outcome} | {origin.version or '-'} | {origin.detail} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="GitHub repository the pin names, e.g. arm/device-connect")
    parser.add_argument("--ref", required=True, help="git ref the pin names, e.g. main")
    parser.add_argument(
        "--distribution",
        action="append",
        dest="distributions",
        metavar="NAME",
        help="distribution to inspect (repeatable; default: the two device-connect packages)",
    )
    args = parser.parse_args(argv)
    names = tuple(args.distributions) if args.distributions else DEFAULT_DISTRIBUTIONS

    origins = [inspect(name, repo=args.repo, ref=args.ref) for name in names]
    print(render(origins, repo=args.repo, ref=args.ref))

    failed = [origin for origin in origins if origin.outcome != PINNED]
    for origin in failed:
        print(
            f"::error title=device-connect is not the pinned source::{origin.distribution} is "
            f"{origin.outcome} ({origin.detail}); expected https://github.com/{args.repo}.git@{args.ref}. "
            "If this ran outside `hatch run`, it measured the runner interpreter rather than the test env."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
