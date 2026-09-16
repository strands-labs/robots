"""The Training overview's data-loop fence must open the gate its step 4 hits.

``create_policy(ckpt)`` resolves a lerobot checkpoint directory to
``lerobot_local``, which ``_check_trust_remote_code`` refuses unless
``STRANDS_TRUST_REMOTE_CODE`` is set - a local, freshly trained directory
included. ``examples/07_post_tune_any_policy.py`` sets the opt-in at that step;
the ``docs/training/overview.md`` fence copied from it is what a reader runs, so
it must carry the same line or step 4 raises ``UntrustedRemoteCodeError`` on a
clean install.

This execs only the fence's ``os.environ`` statements under a cleared
environment and then calls the gate exactly as ``create_policy`` does.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import strands_robots
from strands_robots.policies.factory import _check_trust_remote_code

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PAGE = _REPO_ROOT / "docs" / "training" / "overview.md"
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _data_loop_fence() -> str:
    fences = [f for f in _PYTHON_FENCE.findall(_PAGE.read_text(encoding="utf-8")) if "create_policy(ckpt" in f]
    assert len(fences) == 1, "the data-loop fence is the one that loads the trained checkpoint"
    return fences[0]


def _environ_statements(source: str) -> list[ast.stmt]:
    """Top-level imports plus every statement that touches ``os.environ``."""
    picked: list[ast.stmt] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            picked.append(node)
        elif isinstance(node, ast.stmt) and "os.environ" in ast.unparse(node):
            picked.append(node)
    return picked


def test_the_fence_sets_the_opt_in_before_it_loads_the_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
    fence = _data_loop_fence()

    env_source = "\n".join(ast.unparse(stmt) for stmt in _environ_statements(fence))
    exec(env_source, {})  # noqa: S102 - the docs fence is the artefact under test

    _check_trust_remote_code("lerobot_local")  # raises UntrustedRemoteCodeError when the fence did not opt in

    body = fence.split("if __name__", 1)[1]
    assert body.index("STRANDS_TRUST_REMOTE_CODE") < body.index("create_policy(ckpt"), "the opt-in precedes the load"
