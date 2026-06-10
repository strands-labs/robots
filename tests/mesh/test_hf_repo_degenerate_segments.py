"""R3 review pin: HF two-segment shape gate must reject degenerate
empty / ``.`` / ``..`` segments.

The reviewer (PR #223 R3, thread on ``security.py:354``) flagged that
the prior ``non_empty_parts``-based length check accepted three
degenerate inputs while still passing the regex gate above:

* ``nvidia//repo``  -- parts=['nvidia','','repo'], non_empty_parts=2.
* ``nvidia/.``      -- regex passes, ``.`` is not ``..``.
* ``nvidia/./repo`` -- 3 non-empty segments, but ``.`` is not ``..``.

HuggingFace would 404 on all three at resolution time, but the
validator's job is to enforce the ``<org>/<repo>`` wire contract at
the boundary -- relying on downstream rejection is the same
``etc/passwd``-shaped anti-pattern that R1 rejected for the
``nvidia/etc/passwd`` case. The reviewer-suggested one-line fix
(scoped to ``hf_only=True`` so legitimate local relative paths like
``./local/checkpoint`` still pass under ``hf_only=False``):

    if any(seg in ("", ".") for seg in parts):
        return False

closes all three cases at the cost of one extra membership check.

These tests fail on pre-fix ``is_safe_model_path`` (which used a
``non_empty_parts`` length-2 gate) and pass on the post-fix code.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh.security import is_safe_model_path


@pytest.mark.parametrize(
    "degenerate",
    [
        # Doubled separator -- parts contains ''
        "nvidia//repo",
        "huggingface//repo",
        "lerobot//foo",
        # Current-dir segment in the tail -- regex passes
        "nvidia/.",
        "huggingface/.",
        # Both current-dir and double-separator
        "nvidia/./repo",
        "huggingface/./model",
        # Trailing slash -- parts ends with ''
        "nvidia/repo/",
        # Multiple doubled separators
        "nvidia///repo",
    ],
)
def test_hf_only_rejects_degenerate_segments(degenerate):
    """Empty / ``.`` segments must NOT pass the hf_only shape gate.

    Pre-R3-fix: the ``non_empty_parts`` length-2 gate accepted
    ``nvidia//repo`` and ``nvidia/.`` (regex passed, ``.`` was not
    ``..``). Post-fix: the ``any(seg in ("", "."))`` reject in the
    ``hf_only`` branch closes all three shapes.
    """
    assert not is_safe_model_path(degenerate, hf_only=True), (
        f"degenerate path {degenerate!r} must NOT pass hf_only gate "
        "(pre-R3-fix gap: non_empty_parts gate accepted these)"
    )


@pytest.mark.parametrize(
    "canonical",
    [
        "nvidia/repo",
        "nvidia/cosmos",
        "huggingface/some-model",
        "lerobot/foo-bar",
        "nvidia/GR00T-N1.7-3B",
    ],
)
def test_canonical_shape_still_passes(canonical):
    """Pin: the legitimate ``<org>/<repo>`` happy path is unchanged.

    Regression guard against an over-correction that would also reject
    valid two-segment paths.
    """
    assert is_safe_model_path(canonical, hf_only=True), f"canonical hf path {canonical!r} must still pass"


@pytest.mark.parametrize(
    "relative",
    [
        # Legitimate local relative paths -- ``.`` is a current-dir
        # segment, not a wire-contract violation. The ``hf_only`` reject
        # MUST stay scoped to the ``<org>/<repo>`` branch so these still
        # pass for ``model_path=`` callers that point at on-disk dirs.
        "./local/checkpoint",
        "./local/checkpoint/dir/model",
        "checkpoints/lerobot/aloha-sim",
    ],
)
def test_local_path_branch_keeps_dot_segments(relative):
    """Pin: ``hf_only=False`` (local-path) branch still accepts ``.`` segments.

    The R3 fix moves the empty / ``.`` reject into the ``hf_only=True``
    branch deliberately. ``model_path`` callers expect to point at
    arbitrary on-disk relative paths; rejecting ``./...`` here would
    break legitimate `lerobot_local` consumers without buying any
    additional security beyond what `_MODEL_PATH_RE` already provides.
    """
    assert is_safe_model_path(relative, hf_only=False), f"local-path {relative!r} must still pass hf_only=False"


def test_traversal_still_rejected_in_both_modes():
    """``..`` remains a traversal red flag regardless of mode.

    Sanity guard: the R3 fix scoped the empty / ``.`` reject to
    ``hf_only=True``, but ``..`` (traversal) MUST stay rejected in
    both branches.
    """
    assert not is_safe_model_path("nvidia/..", hf_only=True)
    assert not is_safe_model_path("nvidia/..", hf_only=False)
    assert not is_safe_model_path("../etc/passwd", hf_only=False)
    assert not is_safe_model_path("./local/../escape", hf_only=False)
