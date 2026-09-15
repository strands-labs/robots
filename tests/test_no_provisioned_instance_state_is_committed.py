"""No branch commits ``examples/isaac_on_aws/.instance.json``.

That file is written by ``provision.sh`` and holds the instance id, region and
security-group id of a **live, billing** AWS instance. ``run_smoke.sh`` and
``teardown.sh`` both read it, and ``teardown.sh`` terminates whatever it names.

Committing it is worse than an information leak. A user who clones the repo and
follows the README - which says "``./teardown.sh`` - terminate + clean up. Do not
skip this." - reaches ``teardown.sh``'s guard:

    [ -f "$STATE_FILE" ] || { echo "no .instance.json - run ./provision.sh first"; exit 1; }

With the file committed, that guard passes on a machine that provisioned nothing,
and the script issues ``terminate-instances`` against an id from someone else's
account.

The per-directory ``examples/isaac_on_aws/.gitignore`` covers this only on
branches that carry the example. A fix branch cut from ``main`` has no such
directory, so the file sits untracked-and-unignored in a shared working tree and a
single ``git add -A`` commits it - which is exactly how it happened once, and why
this guard is a test rather than another ignore rule.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Paths that must never be tracked, and why.
#:
#: ``.instance.json`` names a live, billing AWS instance and ``teardown.sh``
#: terminates whatever it names.
#:
#: ``ISAAC_FILED_FINDINGS.md`` is a working handover note, not repository content -
#: a scratch artifact in a PR is review noise.
#:
#: Both reached a commit through ``git add -A`` on a branch where the per-directory
#: ignore rule does not apply, which is precisely why this is a test and not another
#: ignore line: it fails wherever the file is staged, on any branch.
FORBIDDEN = (
    "examples/isaac_on_aws/.instance.json",
    "ISAAC_FILED_FINDINGS.md",
)


def _tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(out.split())


def test_no_live_instance_state_is_tracked() -> None:
    tracked = _tracked_files()
    committed = [p for p in FORBIDDEN if p in tracked]
    assert not committed, (
        f"{committed} is tracked and must not be. Run: git rm --cached <path>. "
        f".instance.json names a live billing instance teardown.sh would terminate; "
        f"ISAAC_FILED_FINDINGS.md is a working note, not repository content."
    )


def test_the_ignore_rule_is_present_where_the_example_lives() -> None:
    """Belt and braces: the rule that stops it on THIS branch is still there."""
    ignore = REPO_ROOT / "examples" / "isaac_on_aws" / ".gitignore"
    assert ignore.is_file(), "the example's .gitignore is gone"
    assert ".instance.json" in ignore.read_text()


def test_teardown_reads_the_file_this_guard_protects() -> None:
    """The premise. If teardown stopped reading it, this guard would be stale."""
    teardown = (REPO_ROOT / "examples" / "isaac_on_aws" / "teardown.sh").read_text()
    assert ".instance.json" in teardown
    assert "terminate-instances" in teardown
