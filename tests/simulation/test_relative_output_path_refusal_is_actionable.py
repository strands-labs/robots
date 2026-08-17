# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A confinement refusal must fit the class of path the caller actually gave.

``validate_output_path`` confines LLM-supplied artifact destinations to a sandbox
root. A *relative* destination is resolved against the process CWD, so under
confinement it can only be refused -- and the refusal reported a CWD-absolute
path the caller never supplied, then offered the sink's absolute-path opt-in as
the way forward. That is a remedy for a class of input that was not given:
setting it does stop the refusal, but it disables confinement and leaves the
artifact under the CWD, i.e. anywhere except the sandbox the caller was told the
sink writes to. Meanwhile the spelling that does work -- the same path anchored
under the sandbox root, or a bare name, which ``validate_output_path`` itself
writes into the sandbox -- went unmentioned.

The load-bearing assertions here are round trips: the suggested path is parsed
back OUT of the refusal, passed in, and required to be accepted inside the
sandbox. That fails for a missing suggestion and for a wrong one, so it pins the
remedy rather than the wording.

The narrow anchoring contract is deliberately NOT changed: a multi-component
relative path is still refused (see
``test_sandbox_still_rejects_relative_path_with_separator``). Only the message
changes, which the controls below pin from both directions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from strands_robots.simulation.mujoco.rendering import _validate_render_output_path
from strands_robots.simulation.safe_output import validate_output_path

# The refusal offers the sandbox-anchored destination to pass instead. A caller
# (human or agent) acts on the message by lifting that path out of it, so the
# test consumes it the same way.
_SUGGESTED_RE = re.compile(r"pass '([^']+)' to write it into the sandbox")
# The opt-in instruction pinned by tests/simulation/test_sandbox_refusal_names_the_opt_in_variable.py.
_OPT_IN_RE = re.compile(r"set ([A-Z][A-Z0-9_]+)=1")

_TOOL_SPEC = Path(__file__).resolve().parents[1] / ".." / "strands_robots" / "simulation" / "mujoco" / "tool_spec.json"


def _sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox"
    root.mkdir()
    return root.resolve()


def _refusal(output_path: str, root: Path, *, label: str = "output_path") -> str:
    with pytest.raises(ValueError) as excinfo:
        validate_output_path(
            output_path,
            sandbox_root=root,
            allow_abs=False,
            label=label,
            allow_abs_env="STRANDS_ROBOTS_RENDER_ALLOW_ABS",
        )
    return str(excinfo.value)


def test_the_path_the_relative_refusal_suggests_is_accepted(tmp_path, monkeypatch):
    """Round trip: parse the suggested destination out of the refusal and pass it.

    A refusal that suggests nothing, or suggests a path the sink then rejects,
    fails here - so this pins the remedy, not the phrasing.
    """
    root = _sandbox(tmp_path)
    monkeypatch.chdir(tmp_path)

    message = _refusal("sub/frame.png", root)

    match = _SUGGESTED_RE.search(message)
    assert match is not None, f"refusal suggested no path to pass instead: {message}"

    resolved = validate_output_path(match.group(1), sandbox_root=root, allow_abs=False)
    assert resolved == (root / "sub" / "frame.png").resolve()
    assert resolved.is_relative_to(root), f"the suggested path landed outside the sandbox: {resolved}"


def test_the_relative_refusal_quotes_the_spelling_the_caller_gave(tmp_path, monkeypatch):
    """The message must echo the caller's own string, not only the CWD path.

    Reporting solely the CWD-absolute destination described a path the caller
    never typed, which is what made the refusal read as unrelated to the call.
    """
    root = _sandbox(tmp_path)
    monkeypatch.chdir(tmp_path)

    message = _refusal("sub/frame.png", root)

    assert "'sub/frame.png'" in message, message


def test_the_relative_refusal_says_where_the_path_was_resolved_from(tmp_path, monkeypatch):
    """Naming the CWD explains why a relative path landed outside the sandbox.

    Asserted as whole phrases, not as the bare words "relative" / the CWD:
    pytest derives ``tmp_path`` from the test name, so both appear inside every
    interpolated path here and a substring check on either passes vacuously.
    """
    root = _sandbox(tmp_path)
    monkeypatch.chdir(tmp_path)

    message = _refusal("sub/frame.png", root)

    assert "is relative, so it resolved against the current directory" in message, message
    assert f"current directory {Path.cwd()} to" in message, message


def test_the_relative_refusal_offers_the_bare_name_rule_this_function_implements(tmp_path, monkeypatch):
    """The cheapest working remedy is the anchoring rule 20 lines above the raise.

    ``validate_output_path`` writes a one-component name into the sandbox
    itself, so a refusal that never mentions it withholds its own behaviour.
    """
    root = _sandbox(tmp_path)
    monkeypatch.chdir(tmp_path)

    message = _refusal("sub/frame.png", root)

    assert "bare output_path" in message, message
    # And the rule the message states is real.
    assert validate_output_path("frame.png", sandbox_root=root, allow_abs=False) == (root / "frame.png").resolve()


def test_the_relative_refusal_does_not_present_the_opt_in_as_an_absolute_path_remedy(tmp_path, monkeypatch):
    """The caller passed no absolute path, so the opt-in cannot be about one.

    It is still offered - disabling confinement is a legitimate last resort -
    but described by what it does here: write outside the sandbox.
    """
    root = _sandbox(tmp_path)
    monkeypatch.chdir(tmp_path)

    message = _refusal("sub/frame.png", root)

    assert "to permit absolute paths" not in message, message
    assert "to write outside the sandbox" in message, message


def test_an_output_dir_refusal_uses_its_own_label(tmp_path, monkeypatch):
    """The recording sinks validate a directory, so the noun must follow ``label``."""
    root = _sandbox(tmp_path)
    monkeypatch.chdir(tmp_path)

    message = _refusal("clips/today", root, label="output_dir")

    assert "bare output_dir" in message, message
    assert _SUGGESTED_RE.search(message) is not None, message


# --- controls: what this change must NOT touch --------------------------------


def test_an_absolute_refusal_keeps_its_wording(tmp_path, monkeypatch):
    """Control: the absolute case was already right, so its message is unchanged.

    Passing an absolute path outside the sandbox IS an absolute-path mistake, so
    "permit absolute paths" fits it. Widening the relative wording to both would
    have regressed the case ``test_sandbox_refusal_names_the_opt_in_variable``
    pins.
    """
    root = _sandbox(tmp_path)
    monkeypatch.chdir(tmp_path)
    outside = (tmp_path / "elsewhere" / "frame.png").resolve()

    message = _refusal(str(outside), root)

    assert message == (
        f"output_path {outside} is outside the sandbox {root} "
        f"(set STRANDS_ROBOTS_RENDER_ALLOW_ABS=1 to permit absolute paths, "
        f"or pass a path under the sandbox)"
    )


def test_a_multi_component_relative_path_is_still_refused(tmp_path, monkeypatch):
    """Control: this changes the message, not which destinations are accepted.

    The narrow anchoring rule (only a one-component name is anchored) is a
    deliberate contract; making a relative subpath succeed would be a different
    change.
    """
    root = _sandbox(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="outside the sandbox"):
        validate_output_path("sub/frame.png", sandbox_root=root, allow_abs=False)


def test_the_relative_refusal_still_names_the_variable_that_lifts_it(tmp_path, monkeypatch):
    """Control: the opt-in name survives, so its own round-trip guard still holds."""
    root = _sandbox(tmp_path)
    monkeypatch.chdir(tmp_path)

    message = _refusal("sub/frame.png", root)

    assert _OPT_IN_RE.search(message) is not None, message
    assert "*_ALLOW_ABS" not in message, message


@pytest.mark.parametrize("bad", ["../escape.png", "frame;rm.png", "dir\\frame.png"])
def test_refusals_that_run_in_every_mode_do_not_suggest_a_sandbox_path(tmp_path, bad):
    """Control: traversal / metacharacter / backslash guards are not confinement.

    They fire regardless of the sandbox, so offering a sandbox-anchored spelling
    there would send the caller after a remedy that cannot work - the rule
    ``test_refusals_the_opt_in_cannot_fix_do_not_advertise_it`` already pins for
    the opt-in.
    """
    root = _sandbox(tmp_path)

    message = _refusal(bad, root)

    assert _SUGGESTED_RE.search(message) is None, message


def test_an_already_anchored_name_is_not_suggested_back_to_the_caller(tmp_path, monkeypatch):
    """A bare name is anchored before the check, so re-suggesting it would loop.

    Reached when the sandbox root itself resolves elsewhere (a symlinked root):
    the anchored destination is the one that just failed, so the suggestion is
    withheld rather than repeated.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    monkeypatch.chdir(tmp_path)

    # sandbox_root is the unresolved symlink, so (root / name).resolve() leaves it.
    message = _refusal("frame.png", link)

    assert _SUGGESTED_RE.search(message) is None, message
    assert "bare output_path" in message, message


def test_the_agent_schema_publishes_the_bare_filename_rule(tmp_path):
    """The schema is the only spec a schema-constrained caller reads.

    It described the sandbox and the absolute-path opt-in, so the documented way
    to save a render was to know and spell the sandbox path. The rule that makes
    that unnecessary was implemented but never published, which is graded here
    against the behaviour rather than trusted as prose.
    """
    schema = json.loads(_TOOL_SPEC.read_text())
    description = schema["properties"]["output_path"]["description"]

    assert "bare filename" in description, description
    # The published claim must be true of the code.
    root = _sandbox(tmp_path)
    assert validate_output_path("frame.png", sandbox_root=root, allow_abs=False) == (root / "frame.png").resolve()


def test_the_render_sink_binding_reports_the_same_actionable_refusal(tmp_path, monkeypatch):
    """End to end through ``render``'s own validator, not just the shared helper."""
    root = tmp_path / "renders"
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ROOT", str(root))
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_ALLOW_ABS", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        _validate_render_output_path("sub/frame.png")

    match = _SUGGESTED_RE.search(str(excinfo.value))
    assert match is not None, str(excinfo.value)
    assert _validate_render_output_path(match.group(1)) == (root / "sub" / "frame.png").resolve()
