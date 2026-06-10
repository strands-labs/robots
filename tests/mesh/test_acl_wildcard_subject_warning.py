"""Pin test for wildcard-subject HARD REJECT in ACL validator.

Review threads PRRT_kwDORUMiZs6GTwcv and _acl_config.py:279/293 flagged
that a subject with neither ``interfaces`` nor ``cert_common_names``
silently match every peer on every link
(``SubjectProperty::Wildcard`` on both dimensions). Combined with
``default_permission: "deny"`` and an ``allow`` rule, this produces
an effectively-permissive ACL the operator did not intend.

Pin: the validator now HARD-REJECTS wildcard subjects (raises
``ValueError`` at parse time). This is symmetric with the empty-list
rejection elsewhere in the validator: "match nothing" and "match
everything" are both refused loudly. Pre-fix HEAD only WARNED on the
unbounded case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.mesh import _acl_config


def _minimal_acl(subjects: list[dict]) -> dict:
    """Build a minimal valid ACL structure with the given subjects."""
    return {
        "enabled": True,
        "default_permission": "deny",
        "subjects": subjects,
        "rules": [
            {
                "id": "allow-all",
                "key_exprs": ["**"],
                "messages": ["put"],
                "flows": ["ingress"],
                "permission": "allow",
            }
        ],
        "policies": [
            {
                "rules": ["allow-all"],
                "subjects": [s["id"] for s in subjects],
            }
        ],
    }


def _write(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "acl.json5"
    p.write_text(json.dumps(doc))
    return p


class TestWildcardSubjectHardReject:
    """Validate that subjects with neither interfaces nor cert_common_names
    are HARD-REJECTED at parse time (was a soft WARNING pre-fix)."""

    def test_subject_without_interfaces_or_cns_rejected(self, tmp_path: Path) -> None:
        """A subject with only an ``id`` is rejected with ValueError."""
        doc = _minimal_acl([{"id": "wide-open"}])
        with pytest.raises(ValueError, match="(?i)match every peer"):
            _acl_config._validate_acl_shape(doc, _write(tmp_path, doc))

    def test_subject_with_interfaces_only_accepts(self, tmp_path: Path) -> None:
        """A subject with ``interfaces`` set parses cleanly."""
        doc = _minimal_acl([{"id": "nic-bound", "interfaces": ["eth0"]}])
        _acl_config._validate_acl_shape(doc, _write(tmp_path, doc))  # no raise

    def test_subject_with_cns_only_accepts(self, tmp_path: Path) -> None:
        """A subject with ``cert_common_names`` set parses cleanly."""
        doc = _minimal_acl([{"id": "cn-bound", "cert_common_names": ["robot-1"]}])
        _acl_config._validate_acl_shape(doc, _write(tmp_path, doc))  # no raise

    def test_subject_with_both_accepts(self, tmp_path: Path) -> None:
        """A subject with both fields set parses cleanly."""
        doc = _minimal_acl([{"id": "fully-bound", "interfaces": ["wlan0"], "cert_common_names": ["op-1"]}])
        _acl_config._validate_acl_shape(doc, _write(tmp_path, doc))  # no raise

    def test_mixed_subjects_rejects_first_unbounded(self, tmp_path: Path) -> None:
        """Mixed list with one wildcard subject is rejected (first match)."""
        doc = _minimal_acl(
            [
                {"id": "bounded", "cert_common_names": ["robot-1"]},
                {"id": "unbounded"},
            ]
        )
        # Fix policies to reference both subjects
        doc["policies"][0]["subjects"] = ["bounded", "unbounded"]
        with pytest.raises(ValueError, match=r"unbounded.*match every peer"):
            _acl_config._validate_acl_shape(doc, _write(tmp_path, doc))

    def test_subject_with_empty_cns_list_rejected(self, tmp_path: Path) -> None:
        """A subject with ``cert_common_names: []`` is rejected (empty
        list = no constraint, same as None)."""
        doc = _minimal_acl([{"id": "wide-open", "cert_common_names": []}])
        with pytest.raises(ValueError, match="(?i)match every peer"):
            _acl_config._validate_acl_shape(doc, _write(tmp_path, doc))
