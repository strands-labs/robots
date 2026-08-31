"""The humanoid camera-mount recipe must describe the asset it names.

``docs/robots/humanoids.md`` carries the only mounting recipe written for a
humanoid. Every other one in the tree is arm-shaped - ``world-building.md``
reads the mount from ``list_bodies(...)["gripper_body"]``, ``camera-naming.md``
and ``annotation.md`` both mount on ``<arm>/gripper`` - and a humanoid in this
catalog reports ``gripper_body: None``, because that hint set (``gripper``,
``hand``, ``jaw``, ``ee``, ``tool``) is arm-shaped. So the humanoid recipe names
a body directly, and naming a body directly is what makes it rot: the recipe is
prose about a third-party asset this repository downloads rather than vendors.

Two things could rot independently, so both are graded here:

* The recipe's *premise* - that the asset has no head link, which is the whole
  reason it mounts on the torso. If a future asset revision adds one, the
  premise cell fails and the recipe should name the head link instead of an
  offset.
* The recipe's *mechanics* - that the body it names exists, that the offset it
  documents is applied in that body's local frame rather than the world frame,
  and that a camera mounted there renders.

The expectations are parsed out of the recipe rather than restated, so the doc
is the single source of truth: editing the offset in the prose edits what these
cells assert. ``tests/test_docs_python_examples_are_callable.py`` already grades
the block's *keywords* against ``add_camera``'s signature; it does not resolve
the body name, the frame or the render, which is the gap this file closes.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

from strands_robots.registry import get_robot, resolve_name
from strands_robots.simulation.ik import GRIPPER_BODY_HINTS, hint_matches_name
from strands_robots.simulation.model_registry import get_search_paths

_DOC = Path(__file__).resolve().parents[2] / "docs" / "robots" / "humanoids.md"
_HEADING = "## Mounting a camera on a humanoid"

# Body-name components that would mean the asset had grown a head mount, making
# the torso offset the recipe documents the wrong advice.
_HEAD_WORDS = ("head", "neck", "eye")


def _recipe_call() -> dict[str, Any]:
    """Parse the recipe's ``add_camera`` keywords out of the humanoid page.

    Returns:
        The call's keyword arguments, with literal values evaluated - e.g.
        ``{"name": "head", "parent_body": "g1/torso_link", "position": [...],
        "target": [...]}``.
    """
    text = _DOC.read_text(encoding="utf-8")
    assert _HEADING in text, f"{_DOC.name} no longer has a {_HEADING!r} section"
    section = text.split(_HEADING, 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    assert len(blocks) == 1, f"expected exactly one python fence in the recipe, found {len(blocks)}"
    calls = [
        node
        for node in ast.walk(ast.parse(blocks[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_camera"
    ]
    assert len(calls) == 1, f"expected one add_camera call in the recipe, found {len(calls)}"
    return {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords if kw.arg}


def _asset_path() -> Path:
    """Locate the recipe's robot asset without reaching the network.

    ``resolve_model_path`` would download a missing asset; this walks the
    configured search paths instead so a host with no asset cache skips rather
    than fetching one.

    Returns:
        The model XML path.
    """
    alias = _robot_alias()
    canonical = resolve_name(alias) or alias
    info = get_robot(canonical)
    # An unregistered name is a defect in the recipe, not a reason to skip: a
    # skip here would let a typo in the documented body name quietly stop every
    # cell below from grading anything. Only a genuinely absent asset file - a
    # host with no download cache - is a skip.
    assert info is not None, f"recipe names robot {alias!r}, which the registry does not know"
    asset = info.get("asset") or {}
    assert asset.get("dir") and asset.get("model_xml"), f"{canonical} declares no MuJoCo asset to mount on"
    # One explicit return: a loop that returns and then falls through to
    # ``pytest.skip`` reads as an implicit ``return None`` to a static
    # analyser, even though ``skip`` is annotated ``NoReturn``.  Resolving
    # the candidate first keeps the single exit this function documents.
    present = next(
        (
            candidate
            for root in get_search_paths()
            if (candidate := Path(root) / asset["dir"] / asset["model_xml"]).is_file()
        ),
        None,
    )
    if present is None:
        pytest.skip(f"{canonical} asset is not present in any search path")
    return present


def _robot_alias() -> str:
    """Return the robot alias the recipe's ``parent_body`` is namespaced with."""
    parent = _recipe_call()["parent_body"]
    alias, _, _body = str(parent).partition("/")
    return alias


def _mount_suffix() -> str:
    """Return the recipe's mount body with its robot namespace stripped."""
    parent = _recipe_call()["parent_body"]
    _alias, _, body = str(parent).partition("/")
    return body


def _body_names(model: Any) -> list[str]:
    """Return every named body in a compiled model."""
    import mujoco

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    return [name for name in names if name]


class TestTheRecipeIsReadable:
    """Non-vacuity: the cells below grade a recipe that is actually there."""

    def test_the_recipe_names_a_namespaced_body_and_a_three_vector(self) -> None:
        call = _recipe_call()
        parent = str(call["parent_body"])
        assert "/" in parent, f"recipe mount {parent!r} is not namespaced <robot>/<body>"
        assert _robot_alias(), "recipe mount carries no robot namespace"
        assert _mount_suffix(), "recipe mount carries no body name"
        assert len(call["position"]) == 3, f"recipe position is not a 3-vector: {call['position']!r}"

    def test_the_recipe_mounts_above_its_parent(self) -> None:
        """The documented offset must be the head-height lift the prose claims."""
        vertical = float(_recipe_call()["position"][2])
        floor = 0.0
        assert vertical > floor, f"recipe offset does not lift the camera: {vertical}"

    def test_the_recipe_robot_alias_names_a_registered_robot(self) -> None:
        """``resolve_name`` normalises rather than validates, so read the entry.

        For an unknown name ``resolve_name`` returns that name lowercased rather
        than ``None``, so ``resolve_name(x) is not None`` accepts anything. The
        registry entry is the existence oracle.
        """
        alias = _robot_alias()
        assert get_robot(resolve_name(alias) or alias) is not None, (
            f"recipe names robot {alias!r}, which the registry does not know"
        )


class TestTheRecipePremiseHolds:
    """Why the recipe mounts on the torso rather than on a head link."""

    def test_the_asset_has_no_head_link_to_mount_on(self) -> None:
        """A head body would make the documented torso offset the wrong advice.

        This is the recipe's premise, not an aspiration: if a future revision of
        the third-party asset adds a head, neck or eye body, this cell fails and
        the recipe should name that body instead of an offset from the torso.
        """
        mujoco = pytest.importorskip("mujoco")
        model = mujoco.MjModel.from_xml_path(str(_asset_path()))
        heads = [name for name in _body_names(model) if any(word in name.lower() for word in _HEAD_WORDS)]
        assert heads == [], (
            f"{_robot_alias()} now has head-like bodies {heads}; the recipe in "
            f"{_DOC.name} mounts on a torso offset because it had none"
        )

    def test_the_arm_shaped_mount_guess_declines_for_this_humanoid(self) -> None:
        """``gripper_body`` must stay ``None`` here - and not name a leg.

        The recipe exists because the arm-shaped guess has no answer for this
        robot. A bare-substring match would have answered ``left_knee_link``,
        because ``ee`` occurs in ``knee``; hints match name components instead,
        so the guess declines rather than naming a leg.
        """
        mujoco = pytest.importorskip("mujoco")
        model = mujoco.MjModel.from_xml_path(str(_asset_path()))
        matched = [
            name
            for name in _body_names(model)
            if any(hint_matches_name(hint, name.rsplit("/", 1)[-1]) for hint in GRIPPER_BODY_HINTS)
        ]
        assert matched == [], f"{_robot_alias()} now has a gripper-like body {matched}; the recipe can name it"
        knee = [name for name in _body_names(model) if "knee" in name.lower()]
        assert knee, "expected this humanoid to have a knee link for the word-boundary check to be meaningful"
        for name in knee:
            hits = [hint for hint in GRIPPER_BODY_HINTS if hint_matches_name(hint, name)]
            assert hits == [], f"a leg link {name!r} matched end-effector hints {hits}"


class TestTheRecipeMechanicsHold:
    """The recipe's body name, frame and render, driven as written."""

    @pytest.fixture
    def mounted(self) -> Any:
        """Add the recipe's robot and camera to a world, exactly as documented."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation import create_simulation

        call = _recipe_call()
        sim: Any = create_simulation("mujoco")
        sim.create_world()
        added = sim.add_robot(name=_robot_alias(), urdf_path=str(_asset_path()))
        assert added["status"] == "success", added
        result = sim.add_camera(
            name=call["name"],
            parent_body=call["parent_body"],
            position=call["position"],
            target=call["target"],
        )
        yield sim, call, result
        sim.cleanup()

    def test_the_documented_mount_is_accepted(self, mounted: Any) -> None:
        _sim, call, result = mounted
        assert result["status"] == "success", (
            f"{_DOC.name} documents parent_body={call['parent_body']!r}, which add_camera refused: {result}"
        )

    def test_the_offset_is_applied_in_the_body_local_frame(self, mounted: Any) -> None:
        """The prose says LOCAL frame; a world-frame read would not ride along."""
        import mujoco

        sim, call, _result = mounted
        model = sim._world._model
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, call["name"])
        assert camera_id >= 0, f"camera {call['name']!r} is not in the compiled model"
        documented = [float(value) for value in call["position"]]
        stored = [float(value) for value in model.cam_pos[camera_id]]
        assert stored == pytest.approx(documented), (
            f"documented offset {documented} did not land in the body frame (cam_pos={stored})"
        )
        parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.cam_bodyid[camera_id]))
        assert parent == call["parent_body"], f"camera is parented to {parent!r}, not {call['parent_body']!r}"

    def test_the_camera_rides_above_its_parent_by_the_documented_lift(self, mounted: Any) -> None:
        """The world pose must show the documented vertical lift once posed."""
        import mujoco

        sim, call, _result = mounted
        model, data = sim._world._model, sim._world._data
        mujoco.mj_forward(model, data)
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, call["name"])
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, call["parent_body"])
        lift = float(data.cam_xpos[camera_id][2] - data.xpos[body_id][2])
        documented_lift = float(call["position"][2])
        assert lift == pytest.approx(documented_lift, abs=1e-6), (
            f"camera sits {lift} m above the torso, the recipe documents {documented_lift} m"
        )

    def test_the_mounted_camera_renders(self, mounted: Any) -> None:
        sim, call, _result = mounted
        frame = sim.render(camera_name=call["name"], width=320, height=240)
        assert frame["status"] == "success", frame
        blocks = [key for block in frame["content"] for key in block]
        assert "image" in blocks, f"render returned no image block: {blocks}"
