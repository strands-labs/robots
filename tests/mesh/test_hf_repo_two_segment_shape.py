"""Pin: ``is_safe_model_path(hf_only=True)`` requires exactly ``<org>/<repo>``.

Real HuggingFace repo IDs are exactly two slash-separated segments
(``<org>/<repo>``). Without the length-2 shape check, paths like
``nvidia/etc/passwd`` (3 segments, traversal-shaped, ``nvidia`` matches
the org allowlist) would pass. They would then 404 at HF resolution,
but the validator's job is to enforce the wire contract, not rely on
downstream rejection -- a future loader that accepts deeper paths
(e.g. ``org/repo/blob/sha`` revision pinning) would silently inherit
the gap.

These tests pin the shape gate, the canonical happy path, and the
pre-existing traversal / leading-slash defences.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh.security import is_safe_model_path


@pytest.mark.parametrize(
    "deep_path",
    [
        "nvidia/etc/passwd",  # original H3 example
        "nvidia/foo/bar",
        "nvidia/a/b/c/d",
        "huggingface/x/y",
        "lerobot/repo/branch/sha",  # would be revision-pin shape
    ],
)
def test_deep_paths_rejected_under_hf_only(deep_path):
    """Any path with more than 2 segments must be rejected."""
    assert not is_safe_model_path(deep_path, hf_only=True), f"deep path {deep_path!r} must NOT pass hf_only gate"


@pytest.mark.parametrize(
    "single_segment",
    [
        "nvidia",
        "huggingface",
        "lerobot",
        "nvidia/",  # trailing slash -- still 1 non-empty segment
    ],
)
def test_single_segment_rejected_under_hf_only(single_segment):
    """``<org>`` alone (no repo) must be rejected."""
    assert not is_safe_model_path(single_segment, hf_only=True), (
        f"single-segment {single_segment!r} must NOT pass hf_only gate"
    )


@pytest.mark.parametrize(
    "two_segment",
    [
        "nvidia/gr00t-n1.5",
        "huggingface/my-model",
        "lerobot/act-base",
        "nvidia/Eagle-VLA-7B",
    ],
)
def test_canonical_two_segment_accepted(two_segment):
    """Sanity: real ``<org>/<repo>`` shape still works."""
    assert is_safe_model_path(two_segment, hf_only=True), f"canonical {two_segment!r} must pass hf_only gate"


def test_traversal_dotdot_still_rejected():
    """Pre-existing defence: any ``..`` segment is rejected before shape check."""
    assert not is_safe_model_path("nvidia/../etc/passwd", hf_only=True)
    assert not is_safe_model_path("../nvidia/foo", hf_only=True)


def test_local_path_branch_unchanged():
    """``hf_only=False`` (local-path) branch still accepts deep relative paths."""
    # The local-path branch is unchanged; it accepts arbitrary relative
    # depths because ``model_path`` callers expect to point at on-disk
    # checkpoint dirs.
    assert is_safe_model_path("./local/checkpoint/dir/model", hf_only=False)


def test_absolute_path_rejected_under_hf_only():
    """Pre-existing: leading ``/`` rejects under hf_only."""
    assert not is_safe_model_path("/nvidia/foo", hf_only=True)
