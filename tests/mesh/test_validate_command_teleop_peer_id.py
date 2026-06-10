"""Pin: ``teleop_receive.source_peer_id`` and ``device_name`` are bounded.

``teleop_receive`` validates both ``source_peer_id`` and the optional
``device_name`` against ``_PEER_ID_RE`` (printable ASCII identifier,
no shell metacharacters, no whitespace, no NULs) plus a
``MAX_PEER_ID_LEN`` length cap. Both fields flow into
``r.start_teleop_receive(source, dev)`` and into log messages, and
``device_name`` becomes a key in the per-device state mapping.

An authenticated peer publishing arbitrary unicode / control
characters / NUL bytes / shell metacharacters in either field has no
business reaching downstream code, regardless of whether today's
downstream consumers happen to be safe. The validator's job is to
enforce the contract at the wire (AGENTS.md > Review Learnings #92,
"Validate every agent-boundary string against an explicit shape").

These tests pin the rejection paths and the canonical accepted shapes
for both fields.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh.security import MAX_PEER_ID_LEN, ValidationError, validate_command


@pytest.mark.parametrize(
    "bad_source",
    [
        "evil; rm -rf /",  # shell metacharacter
        "evil$(whoami)",
        "evil`whoami`",
        "evil\nwith newline",
        "evil\twith tab",
        "evil with space",
        "evil\x00null",
        "evil/with/slashes",  # not a peer-id
        "evil>redirect",
        "evil|pipe",
        "evil`bt",
        "ünicode-emoji-",
    ],
)
def test_teleop_source_peer_id_rejects_bad_charset(bad_source):
    """Every shell-metacharacter / whitespace / control-byte source is rejected."""
    cmd = {"action": "teleop_receive", "source_peer_id": bad_source}
    with pytest.raises(ValidationError, match="must match"):
        validate_command(cmd)


def test_teleop_source_peer_id_rejects_oversized():
    """Over-length source_peer_id is rejected with a clear message."""
    cmd = {"action": "teleop_receive", "source_peer_id": "a" * (MAX_PEER_ID_LEN + 1)}
    with pytest.raises(ValidationError, match="MAX_PEER_ID_LEN"):
        validate_command(cmd)


def test_teleop_source_peer_id_at_length_limit_accepted():
    """Sanity: exactly ``MAX_PEER_ID_LEN`` chars is accepted."""
    cmd = {"action": "teleop_receive", "source_peer_id": "a" * MAX_PEER_ID_LEN}
    out = validate_command(cmd)
    assert out["source_peer_id"] == "a" * MAX_PEER_ID_LEN


@pytest.mark.parametrize(
    "good_source",
    [
        "operator-1",
        "robot_99",
        "peer.with.dots",
        "MixedCase-Allowed",
        "a",  # single char ok
        "abc-123_def.99",
    ],
)
def test_teleop_source_peer_id_canonical_shapes_accepted(good_source):
    """Real-shaped peer IDs still pass."""
    cmd = {"action": "teleop_receive", "source_peer_id": good_source}
    out = validate_command(cmd)
    assert out["source_peer_id"] == good_source


def test_teleop_device_name_charset_enforced():
    """``device_name`` carries the same charset+length discipline."""
    cmd = {
        "action": "teleop_receive",
        "source_peer_id": "operator-1",
        "device_name": "evil; rm -rf /",
    }
    with pytest.raises(ValidationError, match="must match"):
        validate_command(cmd)


def test_teleop_device_name_oversized_rejected():
    """``device_name`` length cap is enforced."""
    cmd = {
        "action": "teleop_receive",
        "source_peer_id": "operator-1",
        "device_name": "x" * (MAX_PEER_ID_LEN + 1),
    }
    with pytest.raises(ValidationError, match="MAX_PEER_ID_LEN"):
        validate_command(cmd)


def test_teleop_empty_source_still_rejected():
    """Pre-existing: empty ``source_peer_id`` is rejected with the original message."""
    cmd = {"action": "teleop_receive", "source_peer_id": ""}
    with pytest.raises(ValidationError, match="non-empty"):
        validate_command(cmd)


def test_teleop_non_str_source_rejected():
    """Pre-existing: non-string ``source_peer_id`` is rejected."""
    cmd = {"action": "teleop_receive", "source_peer_id": 12345}
    with pytest.raises(ValidationError):
        validate_command(cmd)
