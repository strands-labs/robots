# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sandbox refusal must name the variable that lifts it, not a glob.

``validate_output_path`` confines LLM-supplied artifact destinations to a
sandbox root. When it refuses, the caller's only next step is the environment
variable that re-permits absolute paths -- and each sink has its own spelling
(``STRANDS_ROBOTS_RENDER_ALLOW_ABS`` for ``render(output_path=...)``,
``STRANDS_ROBOTS_VIDEO_ALLOW_ABS`` for the video / camera-recording sinks).
Reporting the pattern ``*_ALLOW_ABS`` instead of the name left the caller to
grep the package for it, in the one message that exists to say what to do next.

The load-bearing assertion here is a round trip: the name is parsed back OUT of
the refusal, set, and the same call retried. That fails for a missing name and
for a wrong one, so it pins the remedy rather than the wording.
"""

from __future__ import annotations

import re

import pytest

from strands_robots.simulation.mujoco.rendering import _validate_render_output_path
from strands_robots.simulation.safe_output import validate_output_path, video_sandbox_args

# The refusal states its opt-in as "set <NAME>=1". Parsing it is how a caller
# (human or agent) acts on the message, so the test consumes it the same way.
_OPT_IN_RE = re.compile(r"set ([A-Z][A-Z0-9_]+)=1")


def _named_opt_in(message: str) -> str | None:
    """Return the env var the refusal tells the caller to set, if it names one."""
    match = _OPT_IN_RE.search(message)
    return match.group(1) if match else None


def test_render_refusal_names_its_own_opt_in_variable(monkeypatch, tmp_path):
    """The render sink's refusal quotes STRANDS_ROBOTS_RENDER_ALLOW_ABS by name."""
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ROOT", str(tmp_path / "renders"))
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_ALLOW_ABS", raising=False)

    with pytest.raises(ValueError) as excinfo:
        _validate_render_output_path(str(tmp_path / "elsewhere" / "frame.png"))

    message = str(excinfo.value)
    assert _named_opt_in(message) == "STRANDS_ROBOTS_RENDER_ALLOW_ABS"
    # The glob is what sent the caller grepping; it must not survive as the
    # only actionable token in the message.
    assert "*_ALLOW_ABS" not in message


def test_the_variable_the_render_refusal_names_is_the_one_that_lifts_it(monkeypatch, tmp_path):
    """Following the refusal's own instruction permits the write it refused.

    Round-trip: parse the name out of the message, set it, retry. A refusal that
    named nothing (or named a variable the sink does not read) fails here.
    """
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ROOT", str(tmp_path / "renders"))
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_ALLOW_ABS", raising=False)
    outside = tmp_path / "elsewhere" / "frame.png"

    with pytest.raises(ValueError) as excinfo:
        _validate_render_output_path(str(outside))

    named = _named_opt_in(str(excinfo.value))
    assert named is not None, f"refusal named no variable to set: {excinfo.value}"

    monkeypatch.setenv(named, "1")
    assert _validate_render_output_path(str(outside)) == outside.resolve()


def test_the_variable_the_video_refusal_names_is_the_one_that_lifts_it(monkeypatch, tmp_path):
    """Same round trip for the video / recording sink, which has its own spelling."""
    monkeypatch.setenv("STRANDS_ROBOTS_VIDEO_ROOT", str(tmp_path / "vids"))
    monkeypatch.delenv("STRANDS_ROBOTS_VIDEO_ALLOW_ABS", raising=False)
    outside = tmp_path / "elsewhere" / "rollout.mp4"

    root, allow_abs, allow_abs_env = video_sandbox_args()
    with pytest.raises(ValueError) as excinfo:
        validate_output_path(
            str(outside),
            sandbox_root=root,
            allow_abs=allow_abs,
            label="video path",
            allow_abs_env=allow_abs_env,
        )

    named = _named_opt_in(str(excinfo.value))
    assert named == "STRANDS_ROBOTS_VIDEO_ALLOW_ABS"

    monkeypatch.setenv(named, "1")
    root, allow_abs, allow_abs_env = video_sandbox_args()
    resolved = validate_output_path(
        str(outside),
        sandbox_root=root,
        allow_abs=allow_abs,
        label="video path",
        allow_abs_env=allow_abs_env,
    )
    assert resolved == outside.resolve()


def test_the_two_sinks_name_different_variables(monkeypatch, tmp_path):
    """Each sink names ITS OWN opt-in, so one message cannot misdirect the other.

    A single shared string would be wrong for whichever sink it did not
    describe; that is the failure mode the glob papered over.
    """
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ROOT", str(tmp_path / "renders"))
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_ALLOW_ABS", raising=False)
    monkeypatch.setenv("STRANDS_ROBOTS_VIDEO_ROOT", str(tmp_path / "vids"))
    monkeypatch.delenv("STRANDS_ROBOTS_VIDEO_ALLOW_ABS", raising=False)
    outside = tmp_path / "elsewhere" / "artifact.bin"

    with pytest.raises(ValueError) as render_exc:
        _validate_render_output_path(str(outside))

    root, allow_abs, allow_abs_env = video_sandbox_args()
    with pytest.raises(ValueError) as video_exc:
        validate_output_path(str(outside), sandbox_root=root, allow_abs=allow_abs, allow_abs_env=allow_abs_env)

    assert _named_opt_in(str(render_exc.value)) == "STRANDS_ROBOTS_RENDER_ALLOW_ABS"
    assert _named_opt_in(str(video_exc.value)) == "STRANDS_ROBOTS_VIDEO_ALLOW_ABS"


def test_a_sink_that_names_no_variable_still_refuses(tmp_path):
    """``allow_abs_env=None`` keeps the confinement check and stays a clean refusal.

    The name is message-only: omitting it must not widen the sandbox or raise
    something other than the documented ``ValueError``.
    """
    with pytest.raises(ValueError, match="outside the sandbox"):
        validate_output_path(
            str(tmp_path / "elsewhere" / "x.png"),
            sandbox_root=(tmp_path / "box").resolve(),
            allow_abs=False,
        )


@pytest.mark.parametrize(
    "bad_path,reason",
    [
        ("../escape.png", "path traversal"),
        ("frame;rm -rf /.png", "shell metacharacters"),
        ("dir\\frame.png", "backslash separators"),
    ],
)
def test_refusals_the_opt_in_cannot_fix_do_not_advertise_it(monkeypatch, tmp_path, bad_path, reason):
    """Traversal / metacharacter / backslash refusals must not suggest the opt-in.

    Those guards run in every mode, so ``*_ALLOW_ABS=1`` would not lift them;
    offering it there would send the caller after a remedy that cannot work.
    """
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ROOT", str(tmp_path / "renders"))
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_ALLOW_ABS", raising=False)

    with pytest.raises(ValueError) as excinfo:
        _validate_render_output_path(bad_path)

    message = str(excinfo.value)
    assert reason in message
    assert _named_opt_in(message) is None


def test_a_path_inside_the_sandbox_is_still_accepted(monkeypatch, tmp_path):
    """Control: the confinement decision itself is unchanged by naming the variable."""
    root = tmp_path / "renders"
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ROOT", str(root))
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_ALLOW_ABS", raising=False)

    inside = root / "frame.png"
    assert _validate_render_output_path(str(inside)) == inside.resolve()
    # A bare filename is anchored into the sandbox, not the CWD.
    assert _validate_render_output_path("bare.png") == (root / "bare.png").resolve()
