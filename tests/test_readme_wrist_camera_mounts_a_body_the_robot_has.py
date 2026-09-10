"""The README's wrist-camera example names a body that exists on the robot it adds.

``add_camera(parent_body="<robot>/<body>")`` mounts a camera on a MuJoCo body
and refuses a name the model does not carry. The body vocabulary is the MJCF's,
not ours: the SO-101 names its gripper body ``gripper`` while the SO-100 names
its jaws ``Fixed_Jaw`` and ``Moving_Jaw``. A README fence that adds one robot
and mounts on the other's body name returns ``status: error`` on the first copy.

Each ``python`` fence in ``README.md`` is read for ``add_robot(name=...,
data_config=...)`` and ``add_camera(..., parent_body="<name>/<body>")`` calls;
where the prefix matches a robot the same fence added, ``<body>`` must be a body
of that robot's shipped model.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

_README = Path(__file__).resolve().parent.parent / "README.md"
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _keyword_strings(call: ast.Call) -> dict[str, str]:
    return {
        kw.arg: kw.value.value
        for kw in call.keywords
        if kw.arg and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)
    }


def _method_name(call: ast.Call) -> str | None:
    return call.func.attr if isinstance(call.func, ast.Attribute) else None


def _mounts() -> list[tuple[str, str]]:
    """Return ``(data_config, body)`` for every mount on a robot the same fence added."""
    pairs: list[tuple[str, str]] = []
    for fence in _PYTHON_FENCE.findall(_README.read_text(encoding="utf-8")):
        try:
            tree = ast.parse(fence)
        except SyntaxError:
            continue
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        robots = {
            kws["name"]: kws["data_config"]
            for call in calls
            if _method_name(call) == "add_robot"
            for kws in [_keyword_strings(call)]
            if "name" in kws and "data_config" in kws
        }
        for call in calls:
            if _method_name(call) != "add_camera":
                continue
            parent = _keyword_strings(call).get("parent_body")
            if parent is None or "/" not in parent:
                continue
            robot, body = parent.split("/", 1)
            if robot in robots:
                pairs.append((robots[robot], body))
    return pairs


def _body_names(data_config: str) -> list[str]:
    from strands_robots.simulation.model_registry import resolve_model_path

    model = mujoco.MjModel.from_xml_path(str(resolve_model_path(data_config)))
    return [name for i in range(model.nbody) if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))]


def test_the_readme_mounts_at_least_one_wrist_camera() -> None:
    """A fence that grades nothing would pass silently; the corpus is asserted."""
    assert _mounts(), "README.md no longer adds a robot and mounts a camera on it in one fence"


@pytest.mark.parametrize(("data_config", "body"), _mounts())
def test_the_mounted_body_exists_on_the_robot_the_fence_added(data_config: str, body: str) -> None:
    bodies = _body_names(data_config)
    assert body in bodies, f"README mounts on {data_config!r} body {body!r}; that model has {bodies}"
