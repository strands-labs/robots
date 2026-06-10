"""Pin: examples/mesh_acl_example.json5 references must resolve to live test files.

Background: in the prior fix the test files under ``tests/mesh/`` were renamed away from
methodology-style names (``test_redteam_zenoh.py``, ``test_pentest_findings.py``)
to subject-under-test names (``test_zenoh_transport_security.py``,
``test_application_security.py``). Five direct references were updated, but
``examples/mesh_acl_example.json5:8`` retained the stale name and was caught
re-flagged earlier review.

This test pins the example-file references against the actual test tree so a
future rename / move surfaces here, not in a 5-rounds-later reviewer comment.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _REPO_ROOT / "examples" / "mesh_acl_example.json5"


def _extract_test_path_refs(text: str) -> list[str]:
    """Return every ``tests/.../test_*.py`` substring in *text*."""
    pattern = re.compile(r"tests/[A-Za-z0-9_/]*test_[A-Za-z0-9_]+\.py")
    return list(dict.fromkeys(pattern.findall(text)))


def test_acl_example_test_refs_resolve() -> None:
    """Every ``tests/.../test_*.py`` reference in the example file must exist."""
    body = _EXAMPLE.read_text(encoding="utf-8")
    refs = _extract_test_path_refs(body)
    assert refs, "expected at least one test-file reference in the example header"
    missing = [ref for ref in refs if not (_REPO_ROOT / ref).is_file()]
    assert not missing, (
        f"examples/mesh_acl_example.json5 references missing test files: {missing}. "
        "Update the example header or rename the test back."
    )


def test_acl_example_does_not_reference_renamed_files() -> None:
    """Stale references from the R16 rename must not creep back."""
    body = _EXAMPLE.read_text(encoding="utf-8")
    stale_names = {
        "test_redteam_zenoh.py": "test_zenoh_transport_security.py",
        "test_pentest_findings.py": "test_application_security.py",
    }
    found = {old: new for old, new in stale_names.items() if old in body}
    assert not found, (
        f"examples/mesh_acl_example.json5 contains stale R16-renamed references: {found}. "
        "Replace each old name with its new name."
    )
