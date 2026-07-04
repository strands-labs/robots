"""Mesh identifier validators must reject a trailing newline.

Both mesh identifier allowlists anchor with ``$``:

* ``init_mesh``'s ``peer_id`` regex (rejects MQTT-unsafe / reserved names),
* ``_extract_sample_source_zid``'s wire-level Zenoh-ZID hex shape.

Python's ``$`` matches at the end of the string *or immediately before a
single trailing newline*, so ``re.match(r"...$", "value\n")`` succeeds.
That let an otherwise-valid identifier smuggle a trailing ``\n`` past the
allowlist. A newline in ``peer_id`` is interpolated straight into MQTT
topics (``strands/{peer_id}/cmd``) and AWS Thing-names -- exactly the topic
structure the allowlist exists to protect. A newline in the wire ZID lets a
malformed sample be accepted as a valid safety source identity.

These tests pin that both validators anchor to the absolute end of the
string (``\\Z``), so a trailing newline is rejected.
"""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import core


def _zid_obj(zid_str: str) -> Any:
    """Stand-in for ``zenoh.ZenohId`` whose ``str()`` returns the digest."""

    class _Zid:
        def __str__(self) -> str:
            return zid_str

    return _Zid()


def _make_sample(source_zid: str) -> SimpleNamespace:
    """Minimal Zenoh sample whose ``source_info.source_id.zid`` is *source_zid*."""
    body = json.dumps({}).encode("utf-8")
    return SimpleNamespace(
        payload=SimpleNamespace(to_bytes=lambda: body),
        source_info=SimpleNamespace(
            source_id=SimpleNamespace(zid=_zid_obj(source_zid)),
            source_sn=1,
        ),
    )


@pytest.fixture(autouse=True)
def _mesh_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure STRANDS_MESH does not short-circuit init_mesh before validation."""
    monkeypatch.delenv("STRANDS_MESH", raising=False)


@pytest.mark.parametrize("peer_id", ["robot-1\n", "arm\r", "valid.peer\n"])
def test_init_mesh_rejects_peer_id_with_trailing_newline(peer_id: str) -> None:
    """A peer_id that is valid except for a trailing newline/CR is rejected.

    Without ``\\Z`` anchoring the ``$`` regex accepts ``"robot-1\\n"`` and the
    newline lands in ``strands/{peer_id}/cmd`` MQTT topics.
    """
    with pytest.raises(ValueError, match="invalid characters"):
        core.init_mesh(MagicMock(), peer_id=peer_id)


def test_init_mesh_accepts_clean_peer_id_is_not_broken_by_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tightening to ``\\Z`` must not reject a well-formed peer_id.

    The validation runs before any transport is constructed; monkeypatch the
    Mesh constructor to a no-op so the test exercises only the allowlist and
    never touches Zenoh.
    """
    monkeypatch.setattr(core, "Mesh", lambda *a, **k: MagicMock(alive=False))
    # Should not raise: a clean id passes the tightened anchor.
    core.init_mesh(MagicMock(), peer_id="robot-1.arm_2")


def test_init_mesh_still_rejects_reserved_and_embedded_unsafe_chars() -> None:
    """Pin the surrounding allowlist contract (reserved names + MQTT chars)."""
    for reserved in ("broadcast", "safety"):
        with pytest.raises(ValueError, match="reserved"):
            core.init_mesh(MagicMock(), peer_id=reserved)
    for bad in ("a/b", "a+b", "a#b", "a\x00b", ".leadingdot", "a" * 130):
        with pytest.raises(ValueError, match="invalid characters"):
            core.init_mesh(MagicMock(), peer_id=bad)


@pytest.mark.parametrize(
    "zid",
    ["0123456789abcdef0123456789abcdef\n", "deadbeef\n", "abc\r"],
)
def test_extract_source_zid_rejects_trailing_newline(zid: str) -> None:
    """A wire ZID that differs from the hex shape only by a trailing newline
    is rejected (returns None) instead of being accepted as a valid identity."""
    assert core._extract_sample_source_zid(_make_sample(zid)) is None


def test_extract_source_zid_accepts_clean_hex_is_not_broken_by_the_fix() -> None:
    """The tightened anchor must still accept a well-formed 32-hex ZID."""
    clean = "0123456789abcdef0123456789abcdef"
    assert core._extract_sample_source_zid(_make_sample(clean)) == clean
