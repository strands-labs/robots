"""The mesh guide's opening block joins a mesh under the environment it sets.

``Mesh.start`` refuses under the default posture - mTLS auth, the built-in
permissive ACL, no acknowledgement - and logs ``PERMISSIVE_ACL_REFUSAL``, which
names three environment variables as the ways out. ``docs/mesh.md`` opens with
two processes calling ``Robot(..., mesh=True)``; on a fresh install that block
must not end in ``Mesh did NOT start``. This test replays every ``export`` the
page issues before its first ``mesh=True`` fence into a clean environment and
asks the gate ``Mesh.start`` asks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots
from strands_robots.mesh import core

_GUIDE = Path(strands_robots.__file__).resolve().parent.parent / "docs" / "mesh.md"
_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.S)
_EXPORT = re.compile(r"^\s*export\s+([A-Z_][A-Z0-9_]*)=(\S+)", re.M)


def _exports_before_the_first_mesh_true_fence() -> dict[str, str]:
    env: dict[str, str] = {}
    for lang, body in _FENCE.findall(_GUIDE.read_text(encoding="utf-8")):
        if "mesh=True" in body:
            return env
        if lang in ("bash", "sh", "shell", ""):
            env.update({name: value.strip("'\"") for name, value in _EXPORT.findall(body)})
    pytest.fail(f"{_GUIDE.name} has no fence calling Robot(..., mesh=True)")


def test_the_guide_exports_a_posture_the_start_gate_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [n for n in os.environ if n.startswith("STRANDS_MESH")]:
        monkeypatch.delenv(name, raising=False)
    exports = _exports_before_the_first_mesh_true_fence()
    for name, value in exports.items():
        monkeypatch.setenv(name, value)

    mesh: Any = core.Mesh.__new__(core.Mesh)
    core.Mesh.__init__(mesh, MagicMock(), "docs-mesh-guide")
    refused = mesh._refuse_under_permissive_default_acl()

    assert not refused, (
        f"under the environment {exports or '{}'} the guide sets before its first mesh=True fence, "
        f"Mesh.start refuses: {core.PERMISSIVE_ACL_REFUSAL.splitlines()[0]}"
    )
