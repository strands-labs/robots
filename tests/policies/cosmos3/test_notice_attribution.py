"""Pin test: NOTICE file must attribute vendored Apache-2.0 code.

Apache-2.0 section 4(d) requires redistributors to carry forward
attribution for derivative works. The vendored _msgpack_numpy.py
(from openpi-client / msgpack-numpy) is Apache-2.0 licensed and
MUST be attributed in the repo-level NOTICE file.

Regression: review thread on PR #317 (2026-06-07) flagged this as
a compliance gap that blocks merge.
"""

from pathlib import Path


def test_notice_attributes_vendored_msgpack_numpy():
    """NOTICE must mention openpi-client attribution for _msgpack_numpy.py."""
    repo_root = Path(__file__).resolve().parents[3]
    notice_path = repo_root / "NOTICE"

    assert notice_path.exists(), (
        f"NOTICE file not found at {notice_path}. Apache-2.0 section 4(d) requires attribution for vendored code."
    )

    content = notice_path.read_text(encoding="utf-8")

    # Must reference the vendored file
    assert "_msgpack_numpy" in content or "msgpack_numpy" in content, (
        "NOTICE must reference the vendored _msgpack_numpy.py file"
    )

    # Must reference the upstream source (openpi-client or Physical Intelligence)
    assert (
        "openpi" in content.lower()
        or "physical-intelligence" in content.lower()
        or "physical intelligence" in content.lower()
    ), "NOTICE must attribute openpi-client (Physical Intelligence) as upstream source"

    # Must mention Apache-2.0
    assert "apache" in content.lower(), "NOTICE must reference the Apache License"
