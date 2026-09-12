# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every camera name this library documents or requires is one ``add_camera`` accepts.

A sim camera's name is the key its frames travel under, so ``add_camera`` holds it
to a bare token optionally scoped to one robot
(:func:`~strands_robots.utils.scoped_camera_name_error`). Three populations name
cameras on the reader's behalf, and each has to stay inside that alphabet:

* **The examples.** A fence showing ``add_camera(name="wrist cam")`` does not fail
  at the line the reader copied - it returns ``status="error"``, and every line
  below it runs against a scene missing the camera the example is about.
* **The camera-naming table.** ``docs/policies/camera-naming.md`` tells the reader
  to name their sim cameras after a policy's expected source key, so a key in that
  column that ``add_camera`` refuses is advice that cannot be followed.
* **The embodiments.** Those source keys come from ``obs_rename`` in
  ``strands_robots/policies/lerobot_local/embodiments.json``. A key that is not a
  nameable camera makes the embodiment unsatisfiable from sim: the pre-flight
  check would name a camera the caller has no way to create.

The names are graded rather than the prose, for the reason the sibling
``tests/test_docs_add_robot_names_are_free_in_the_fence.py`` gives: a rule stated
in one paragraph and shown in twenty places drifts in the twenty. Only literal
names are read; one computed at runtime is not something this file can resolve.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import strands_robots
from strands_robots.utils import scoped_camera_name_error

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_EMBODIMENTS = _REPO_ROOT / "strands_robots" / "policies" / "lerobot_local" / "embodiments.json"
_CAMERA_NAMING_DOC = _REPO_ROOT / "docs" / "policies" / "camera-naming.md"

#: A literal name claimed at an ``add_camera`` call, keyword or positional. Read
#: with a pattern rather than through :mod:`ast`, because the documented fences
#: use ``...`` as a placeholder argument and do not all parse.
_ADD_CAMERA_NAME = re.compile(r"""add_camera\(\s*(?:name\s*=\s*)?(['"])([^'"]*)\1""")


def _documentation_files() -> list[Path]:
    files = sorted((_REPO_ROOT / "docs").rglob("*.md"))
    readme = _REPO_ROOT / "README.md"
    return [*files, readme] if readme.is_file() else files


def _documented_camera_names() -> list[tuple[str, str]]:
    """Every ``(where, name)`` an example claims through ``add_camera``."""
    found: list[tuple[str, str]] = []
    for path in _documentation_files():
        text = path.read_text(encoding="utf-8")
        for match in _ADD_CAMERA_NAME.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            found.append((f"{path.relative_to(_REPO_ROOT)}:{line}", match.group(2)))
    return found


def _table_source_keys() -> list[str]:
    """The ``add_camera`` name column of the camera-naming translation table."""
    keys: list[str] = []
    for row in _CAMERA_NAMING_DOC.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in row.split("|")]
        if len(cells) < 5 or not cells[4].startswith("`observation.images."):
            continue
        for cell in cells[3].split("/"):
            name = cell.strip().strip("`")
            if name:
                keys.append(name)
    return keys


def _obs_rename_source_keys() -> list[str]:
    """Every camera name an embodiment's ``obs_rename`` expects the runtime to hold."""
    keys: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "obs_rename" and isinstance(value, dict):
                    keys.extend(str(k) for k in value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(json.loads(_EMBODIMENTS.read_text(encoding="utf-8")))
    return keys


def test_no_documented_example_claims_a_name_the_door_refuses() -> None:
    names = _documented_camera_names()
    assert len(names) >= 9, f"the reader found no examples to grade: {names}"
    offenders = [f"{where} adds a camera named {name!r}" for where, name in names if _refused(name)]
    assert not offenders, (
        "add_camera refuses a name no frame consumer could key (status=error, not an "
        "exception), so the example silently continues without that camera:\n  " + "\n  ".join(offenders)
    )


def test_every_camera_naming_table_key_can_be_created() -> None:
    keys = _table_source_keys()
    assert len(keys) >= 6, f"the table reader resolved nothing: {keys}"
    assert [k for k in keys if _refused(k)] == []


def test_every_embodiment_expects_a_camera_name_that_can_exist() -> None:
    keys = _obs_rename_source_keys()
    assert len(keys) >= 9, f"no obs_rename key was read: {keys}"
    assert [k for k in keys if _refused(k)] == [], (
        "an embodiment expects a camera the sim door cannot create, so its pre-flight "
        "check would name a source key the caller has no way to produce"
    )


def test_the_documented_alphabet_agrees_with_the_live_rule() -> None:
    """The prose in ``docs/recording.md`` and the door agree on both lists.

    Accepted shapes and refused ones, so the paragraph cannot drift into
    describing a rule the door does not apply.
    """
    text = (_REPO_ROOT / "docs" / "recording.md").read_text(encoding="utf-8")
    accepted = ("wrist", "front_cam", "cam-2", "arm0/wrist_cam")
    refused = ("a b", "wrist.rgb", "*", "..", "sub/../etc", "a//b")
    for name in accepted + refused:
        assert f"`{name}`" in text, f"docs/recording.md no longer shows {name!r}"
    assert [n for n in accepted if _refused(n)] == []
    assert [n for n in refused if not _refused(n)] == []


def test_the_readers_resolve_the_shapes_the_sources_are_written_in() -> None:
    """An empty offender list and a reader that resolves nothing are the same list."""
    assert [n for _, n in _documented_camera_names()] != []
    planted = _ADD_CAMERA_NAME.findall(
        'sim.add_camera(name="wrist cam", ...)\nsim.add_camera("../etc")\nsim.add_camera(name=chosen)'
    )
    assert [name for _, name in planted] == ["wrist cam", "../etc"]
    assert all(_refused(name) for _, name in planted)


def _refused(name: str) -> bool:
    return scoped_camera_name_error("add_camera", "name", name) is not None
