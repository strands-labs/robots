"""the prior fix pin tests for safety-topic rate / size caps in zenoh transport.

Yin's review on _zenoh_config.py:251 -- downsampling and low_pass_filter
only target **/cmd / **/broadcast / **/camera/**. Neither glob covers
**/safety/**. Under permissive default ACL, any cert-holding peer can
flood safety/estop with novel-`t` envelopes (bypassing the receiver
replay cache) and consume CPU on freshness checks at line rate.

the prior fix fix: extend both blocks to cover **/safety/** with their own
(lower) caps -- 2 Hz frequency, 4 KiB size. Configurable via
``STRANDS_MESH_SAFETY_RATE_HZ`` / ``STRANDS_MESH_MAX_SAFETY_BYTES``.

Per AGENTS.md > Review Learnings (#85) > "Pin regression tests for
reviewed fixes."
"""

from __future__ import annotations

import json

from strands_robots.mesh import _zenoh_config as zc


def _downsampling_rules() -> list[dict]:
    """Parse the JSON5 downsampling block back into a rules list."""
    _, body = zc.downsampling_block()
    parsed = json.loads(body)
    return parsed[0]["rules"]


def _low_pass_filter_blocks() -> list[dict]:
    _, body = zc.low_pass_filter_block()
    return json.loads(body)


def test_downsampling_covers_safety_topics():
    """R21: ``**/safety/**`` is rate-capped via downsampling."""
    rules = _downsampling_rules()
    safety_rules = [r for r in rules if r["key_expr"] == "**/safety/**"]
    assert len(safety_rules) == 1, f"exactly one safety/** downsampling rule expected; got {rules!r}"
    assert safety_rules[0]["freq"] == zc.DEFAULT_SAFETY_RATE_HZ
    assert safety_rules[0]["freq"] > 0


def test_low_pass_filter_covers_safety_topics():
    """R21: ``**/safety/**`` is byte-capped via low_pass_filter."""
    blocks = _low_pass_filter_blocks()
    safety_blocks = [b for b in blocks if "**/safety/**" in b["key_exprs"]]
    assert len(safety_blocks) == 1, "exactly one safety/** low_pass_filter block expected"
    assert safety_blocks[0]["size_limit"] == zc.DEFAULT_MAX_SAFETY_BYTES
    assert safety_blocks[0]["size_limit"] > 0


def test_safety_rate_hz_env_override(monkeypatch):
    """``STRANDS_MESH_SAFETY_RATE_HZ`` overrides the default safety rate."""
    monkeypatch.setenv("STRANDS_MESH_SAFETY_RATE_HZ", "0.5")
    rules = _downsampling_rules()
    safety_rule = next(r for r in rules if r["key_expr"] == "**/safety/**")
    assert safety_rule["freq"] == 0.5


def test_max_safety_bytes_env_override(monkeypatch):
    """``STRANDS_MESH_MAX_SAFETY_BYTES`` overrides the default safety size."""
    monkeypatch.setenv("STRANDS_MESH_MAX_SAFETY_BYTES", "8192")
    blocks = _low_pass_filter_blocks()
    safety_block = next(b for b in blocks if "**/safety/**" in b["key_exprs"])
    assert safety_block["size_limit"] == 8192


def test_safety_rate_below_cmd_rate_by_design():
    """Safety legitimately is far below cmd traffic; the default cap
    must reflect that or it lets through traffic the design considers
    flood."""
    assert zc.DEFAULT_SAFETY_RATE_HZ < zc.DEFAULT_CMD_RATE_HZ


def test_safety_size_cap_below_camera_cap_by_design():
    """Safety envelopes are small JSON dicts; camera frames are MiB.
    Caps must differ accordingly."""
    assert zc.DEFAULT_MAX_SAFETY_BYTES < zc.DEFAULT_MAX_CAMERA_BYTES


def test_existing_cmd_and_broadcast_rules_unaffected():
    """the prior fix must not regress the prior/the prior fix rate caps on cmd / broadcast --
    those are pinned by other tests but we double-check here."""
    rules = _downsampling_rules()
    cmd = next((r for r in rules if r["key_expr"] == "**/cmd"), None)
    broadcast = next((r for r in rules if r["key_expr"] == "**/broadcast"), None)
    assert cmd is not None and cmd["freq"] == zc.DEFAULT_CMD_RATE_HZ
    assert broadcast is not None and broadcast["freq"] == zc.DEFAULT_CMD_RATE_HZ


def test_existing_camera_size_cap_unaffected():
    blocks = _low_pass_filter_blocks()
    camera = next((b for b in blocks if "**/camera/**" in b["key_exprs"]), None)
    assert camera is not None
    assert camera["size_limit"] == zc.DEFAULT_MAX_CAMERA_BYTES
