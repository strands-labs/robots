"""The publish workflow's Validate version step takes a pre-release, and only under its own tag.

M2 publishes 1.0.0rc1 from dev (#3818). A release runs the workflow file of the tagged commit, so this step has to
take it on dev before the tag. The step runs here as GitHub runs it (`bash -e`), with a stand-in `hatch` that prints
the tree's version. Parsing is line-based, as on main: PyYAML is not a dev dependency.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_WF = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pypi-publish-on-release.yml"


def _validate_step_body() -> str:
    lines = _WF.read_text().splitlines()
    step = next(i for i, line in enumerate(lines) if line.strip() == "- name: Validate version")
    run = next(i for i in range(step, len(lines)) if lines[i].strip() == "run: |")
    indent = len(lines[run + 1]) - len(lines[run + 1].lstrip())
    body = []
    for line in lines[run + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) < indent:
            break
        body.append(line[indent:])
    return "\n".join(body) + "\n"


@pytest.mark.skipif(shutil.which("bash") is None, reason="the step is a bash script")
@pytest.mark.parametrize(
    ("version", "tag", "published"),
    [
        ("1.0.0rc1", "v1.0.0rc1", True),  # M2
        ("1.0.0", "v1.0.0", True),  # M3
        ("0.5.3", "v0.5.3", True),  # a v0.5 tag re-published by workflow_dispatch
        ("1.0.0.dev0", "v1.0.0rc1", False),  # the release commit forgot to set the version
        ("1.0.0rc1", "v1.0.0", False),  # a tag the tree's version is not
        ("1.0.0", "v1.0.1", False),
    ],
)
def test_validate_version_takes_a_pre_release_and_only_its_own_tag(tmp_path, version, tag, published) -> None:
    hatch = tmp_path / "hatch"
    hatch.write_text(f"#!/bin/sh\necho {version}\n")
    hatch.chmod(0o755)
    script = tmp_path / "validate.sh"
    script.write_text(_validate_step_body())
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}", "TAG": tag}
    done = subprocess.run(["bash", "-e", str(script)], env=env, capture_output=True, text=True, timeout=30)
    assert (done.returncode == 0) is published, done.stdout + done.stderr


def test_validate_version_holds_the_version_to_the_tag_the_build_checks_out() -> None:
    # A release names its tag in github.ref_name (github.ref is refs/tags/<it>), a dispatch in inputs.tag.
    lines = [line.strip() for line in _WF.read_text().splitlines()]
    assert "ref: ${{ inputs.tag || github.ref }}" in lines  # the build job's checkout
    assert "TAG: ${{ inputs.tag || github.ref_name }}" in lines  # what Validate version compares the version with
