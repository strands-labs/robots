"""Tests for the CA pin rotation grace-period contract (issue #250).

Issue #250 (Option B) ships a documented rotation runbook for the Amazon
Root CA1 pin instead of a signed-manifest fetch. The operational guarantee
the runbook depends on is that the accepted-pin set is a collection, so a
rotation can keep both the old and the new pin valid during fleet uptake:

* The built-in pin tuple plus a staged second pin from
  ``STRANDS_MESH_CA_PINS`` both resolve as accepted (dual-pin overlap).
* Verification accepts a certificate matching either pin in the set.
"""

from __future__ import annotations

import hashlib

import pytest

from strands_robots.mesh.iot import provision


def test_resolve_ca_pins_accepts_both_builtin_and_staged_pin_for_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """During a rotation, the built-in pin and a staged STRANDS_MESH_CA_PINS pin coexist."""
    builtin = set(provision._AMAZON_ROOT_CA1_PINS)
    assert builtin, "expected at least one built-in pin"
    staged = "a" * 64  # stand-in for the next root CA1 pin staged out-of-band
    assert staged not in builtin

    monkeypatch.setenv("STRANDS_MESH_CA_PINS", staged)
    resolved = provision._resolve_ca_pins()

    assert builtin.issubset(resolved), "old pin must stay valid during overlap"
    assert staged in resolved, "newly staged pin must also be accepted"
    assert len(resolved) >= len(builtin) + 1


def test_resolve_ca_pins_returns_collection_not_scalar() -> None:
    """The accepted-pin surface is a set so the dual-pin grace period is expressible."""
    resolved = provision._resolve_ca_pins()
    assert isinstance(resolved, frozenset)


def test_verify_ca_bytes_accepts_either_pin_during_dual_pin_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cert matching the staged pin verifies even when it is not the built-in pin."""
    rogue = b"-----BEGIN CERTIFICATE-----\nstaged-root-bytes\n-----END CERTIFICATE-----\n"
    staged = hashlib.sha256(rogue).hexdigest()
    assert staged not in set(provision._AMAZON_ROOT_CA1_PINS)

    # Without the staged pin, the rogue cert is rejected.
    assert provision._verify_ca_bytes(rogue) is False

    # Once staged out-of-band, the same cert is accepted (grace-period overlap).
    monkeypatch.setenv("STRANDS_MESH_CA_PINS", staged)
    assert provision._verify_ca_bytes(rogue) is True
