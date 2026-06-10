"""Pin: ACL loader requires ``enabled: true`` (literal boolean).

The ``_load_acl_file`` gate uses an identity check
(``data.get("enabled") is not True``) rather than a truthy test. Any
truthy non-bool such as ``enabled: 1`` (JSON5 int), ``enabled: "true"``
(string typo), or ``enabled: [false]`` (non-empty list) would otherwise
pass ``not False`` but fail the downstream Zenoh deserializer with an
opaque "expected boolean" several frames deeper. Failing closed at the
loader keeps misconfig debuggable.

These tests pin the rejection of common operator typos and the happy
path for the canonical ``enabled: true`` shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.mesh import _acl_config as ac


def _write_acl(tmp_path: Path, enabled_value: str) -> Path:
    """Write a minimal ACL file with the given JSON5 ``enabled:`` value."""
    body = (
        "{\n"
        f"  enabled: {enabled_value},\n"
        '  default_permission: "deny",\n'
        "  rules: [],\n"
        "  subjects: [],\n"
        "  policies: [],\n"
        "}\n"
    )
    p = tmp_path / "acl.json5"
    p.write_text(body)
    return p


@pytest.mark.parametrize(
    "bad_value",
    [
        "1",  # JSON5 int truthy
        '"true"',  # string typo (would be parsed as quoted string)
        '"yes"',  # string truthy
        "[false]",  # non-empty list -- truthy
        '{"x":1}',  # non-empty dict -- truthy
        "false",  # explicit false (must still reject)
        "0",  # falsy int
    ],
)
def test_acl_enabled_must_be_literal_true(tmp_path, bad_value):
    """Every truthy-or-falsy non-True value must be rejected with a clear message."""
    p = _write_acl(tmp_path, bad_value)
    with pytest.raises(ValueError, match="literal boolean"):
        ac._load_acl_file(p)


def test_acl_enabled_true_accepted(tmp_path):
    """Sanity: literal ``true`` passes the gate."""
    p = _write_acl(tmp_path, "true")
    data = ac._load_acl_file(p)
    assert data["enabled"] is True


def test_acl_enabled_missing_rejected(tmp_path):
    """Pre-existing behaviour: missing ``enabled:`` is still rejected."""
    body = '{\n  default_permission: "deny",\n  rules: [],\n  subjects: [],\n  policies: [],\n}\n'
    p = tmp_path / "acl.json5"
    p.write_text(body)
    with pytest.raises(ValueError, match="must set ``enabled: true``"):
        ac._load_acl_file(p)
