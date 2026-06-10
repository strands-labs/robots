"""Pin test for the prior design-thread fix: ACL-file shape validation.

this PR review thread PRRT_kwDORUMiZs6ER6__ flagged that the ACL loader
only validated 4 top-level keys + ``enabled: true``; everything below
was unchecked. A typo like ``interface:`` (singular) or a missing
``cert_common_names`` field silently degrades a role-separated ACL to
"match nothing" at the Zenoh layer -- a silent total outage operators
must debug from Zenoh logs.

the prior fix adds ``_validate_acl_shape`` which raises ``ValueError`` at parse
time on each footgun.

Pin: each test crafts a minimally-broken ACL file and asserts a clear
ValueError with the expected substring. Pre-fix HEAD silently accepts
all of these and returns the ACL dict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.mesh import _acl_config


def _write(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "acl.json5"
    p.write_text(json.dumps(doc))
    return p


def _valid_skeleton() -> dict:
    """A minimal valid ACL — used as the base for negative-shape tests."""
    return {
        "enabled": True,
        "default_permission": "deny",
        "subjects": [
            {
                "id": "robot",
                "interfaces": ["lo", "eth0"],
                "cert_common_names": ["robot-1"],
            }
        ],
        "rules": [
            {
                "id": "r1",
                "key_exprs": ["**/cmd"],
                "messages": ["put"],
                "flows": ["ingress"],
                "permission": "allow",
            }
        ],
        "policies": [
            {"rules": ["r1"], "subjects": ["robot"]},
        ],
    }


def test_valid_skeleton_loads_clean(tmp_path: Path) -> None:
    p = _write(tmp_path, _valid_skeleton())
    parsed = _acl_config._load_acl_file(p)
    assert parsed["default_permission"] == "deny"
    assert len(parsed["subjects"]) == 1


def test_subject_omitted_interfaces_accepted(tmp_path: Path) -> None:
    """``interfaces`` is OPTIONAL per Zenoh schema.

    Per ``zenoh-config/src/lib.rs`` AclConfigSubjects.interfaces is
    ``Option<NEVec<...>>``; ``authorization.rs:446-454`` maps
    ``None`` to ``SubjectProperty::Wildcard`` (matches every link).
    The CN-only ACL pattern (used by Zenoh's own
    ``tests/authentication.rs``) requires this; older revisions of
    ``_validate_acl_shape`` rejected it, blocking the cleanest
    deployment shape.
    """
    doc = _valid_skeleton()
    doc["subjects"][0].pop("interfaces")
    p = _write(tmp_path, doc)
    parsed = _acl_config._load_acl_file(p)
    assert "interfaces" not in parsed["subjects"][0]
    assert parsed["subjects"][0]["cert_common_names"] == ["robot-1"]


def test_subject_empty_interfaces_rejected(tmp_path: Path) -> None:
    """Empty list is still rejected -- Zenoh raises ``Found empty
    interface value`` server-side, and the silent-total-deny failure
    mode is real (prior footgun). Either omit the field or enumerate.
    """
    doc = _valid_skeleton()
    doc["subjects"][0]["interfaces"] = []
    p = _write(tmp_path, doc)
    with pytest.raises(ValueError, match=r"interfaces is an empty list"):
        _acl_config._load_acl_file(p)


def test_subject_interfaces_non_list_rejected(tmp_path: Path) -> None:
    """Catches the scalar-string typo ``interfaces: "lo"`` (Rust deny_unknown_fields-style)."""
    doc = _valid_skeleton()
    doc["subjects"][0]["interfaces"] = "lo"  # should be a list
    p = _write(tmp_path, doc)
    with pytest.raises(ValueError, match=r"interfaces must be a list"):
        _acl_config._load_acl_file(p)


def test_subject_interfaces_with_empty_string_rejected(tmp_path: Path) -> None:
    """Empty-string entries inside the list are still rejected."""
    doc = _valid_skeleton()
    doc["subjects"][0]["interfaces"] = ["lo", ""]
    p = _write(tmp_path, doc)
    with pytest.raises(ValueError, match=r"non-empty strings"):
        _acl_config._load_acl_file(p)


def test_subject_cert_common_names_typo_rejected(tmp_path: Path) -> None:
    """Reject ``cert_common_names`` if not a list (catches scalar string typo)."""
    doc = _valid_skeleton()
    doc["subjects"][0]["cert_common_names"] = "robot-1"  # should be list
    p = _write(tmp_path, doc)
    with pytest.raises(ValueError, match=r"cert_common_names.*list"):
        _acl_config._load_acl_file(p)


def test_rule_missing_key_exprs_rejected(tmp_path: Path) -> None:
    doc = _valid_skeleton()
    doc["rules"][0]["key_exprs"] = []
    p = _write(tmp_path, doc)
    with pytest.raises(ValueError, match=r"key_exprs.*non-empty"):
        _acl_config._load_acl_file(p)


def test_rule_invalid_permission_rejected(tmp_path: Path) -> None:
    doc = _valid_skeleton()
    doc["rules"][0]["permission"] = "permit"  # wrong word
    p = _write(tmp_path, doc)
    with pytest.raises(ValueError, match=r"permission.*allow.*deny"):
        _acl_config._load_acl_file(p)


def test_policy_references_unknown_rule_rejected(tmp_path: Path) -> None:
    doc = _valid_skeleton()
    doc["policies"][0]["rules"] = ["r1", "r99-typo"]
    p = _write(tmp_path, doc)
    with pytest.raises(ValueError, match=r"unknown rule id.*r99-typo"):
        _acl_config._load_acl_file(p)


def test_policy_references_unknown_subject_rejected(tmp_path: Path) -> None:
    doc = _valid_skeleton()
    doc["policies"][0]["subjects"] = ["robtot"]  # typo
    p = _write(tmp_path, doc)
    with pytest.raises(ValueError, match=r"unknown subject id.*robtot"):
        _acl_config._load_acl_file(p)


def test_subjects_not_a_list_rejected(tmp_path: Path) -> None:
    doc = _valid_skeleton()
    doc["subjects"] = {"id": "robot"}  # dict not list
    p = _write(tmp_path, doc)
    with pytest.raises(ValueError, match=r"subjects.*list"):
        _acl_config._load_acl_file(p)


def test_shipped_example_still_loads(tmp_path: Path) -> None:
    """Anti-regression: the shipped example file must still pass shape validation."""
    example_path = Path(__file__).parent.parent.parent / "examples" / "mesh_acl_example.json5"
    if not example_path.exists():
        pytest.skip(f"example not found at {example_path}")
    parsed = _acl_config._load_acl_file(example_path)
    assert "subjects" in parsed
    assert "rules" in parsed
    assert "policies" in parsed
