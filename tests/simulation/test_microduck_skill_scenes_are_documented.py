"""Every Microduck skill the page advertises names the scene it needs.

``docs/policies/microduck.md`` opens by listing the nine shipped Pollen weights
:class:`~strands_robots.policies.microduck.MicroduckPolicy` wraps and says they
drive the biped "through the standard ``Robot(...).run_policy`` seam - in MuJoCo
or on hardware". Five of those nine run on the scene the registry entry declares.
Four do not: ``roller`` and ``roller_crouch`` need the four passive ankle wheels
that only ``scene_rollers.xml`` carries, and ``ball_kick_left`` /
``ball_kick_right`` need the ball prop that only ``scene_ball.xml`` places.

Running one of the four on the default scene is not an error. The policy writes
the same fourteen control targets, the rollout reports success, and the physics
simply has nothing to roll on or nothing to kick - so a reader who follows the
page gets a duck standing still and no indication why. ``render_video.py`` builds
a bare ``Robot("microduck")``, which is why the page used to say any shipped
weight "drops straight in": true of the five, false of the four.

The scenes are not missing. All three ship in the one asset directory the entry
already downloads, and the entry keeps naming the fourteen-hinge model on purpose
- ``tests/simulation/test_microduck_asset_matches_the_declared_shape.py`` pins
that, and its ``TestTheEntryPointsAtTheDocumentedLayout`` records the reason ("a
caller can load it by path"). So the gap was never reachability; it was that no
page said which scene a skill needs, and nothing graded the claim.

This file grades it in two layers. The first reads the page: every skill named in
the opening paragraph must appear in the Skill scenes table, so a tenth weight
cannot be advertised without naming its scene. The second reads the compiled
models, so the table's claims are true rather than asserted - the scene a row
names really does carry the wheels or the ball, and the default scene really does
not. The second layer skips when the asset is absent, which is the case on a
clean checkout; the first holds on any install, with no MuJoCo and no network.

A weight and its stance are a pair in the same way, so the stance section below
is graded here too. Every shipped weight bakes the stance it was trained in into
its ONNX metadata, the asset ships the same stance as its ``STAND`` keyframe, and
``add_robot(keyframe=...)`` seats a robot there and keeps it across resets. What
was missing was any page saying so, and any check that the exported values and
the shipped keyframe still agree - the asset has revised this pose once already,
which is why the page names the constant instead of repeating the numbers.

A registry ``variant=`` spelling would be a nicer front door and is deliberately
not invented here: no entry among the seventy-three declares one, so the schema
is a public-API decision rather than a docs fix.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.microduck import MICRODUCK_DEFAULT_POSE, MICRODUCK_JOINT_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAGE = REPO_ROOT / "docs" / "policies" / "microduck.md"

#: The directory the ``microduck`` entry downloads into, and the scene it names.
ASSET_DIR = "microduck"
DEFAULT_SCENE = "scene.xml"

#: A skill list shorter than this means the opening paragraph stopped listing
#: weights, so the cross-reference below would pass by having nothing to check.
MINIMUM_SKILLS = 9

#: Fewer rows than this means the table stopped covering the scenes, so the
#: per-scene cells would pass by never reaching a non-default scene.
MINIMUM_TABLE_ROWS = 3

#: The offset ``reset_ball_in_front_of_foot`` trained the two kick weights on, read
#: off Pollen's reference runtime (``scripts/infer_policy.py``, ``BALL_OFFSET_X`` and
#: ``BALL_OFFSET_ABS_Y``). The scene's own declared position is not this, which is
#: what the page's placement paragraph exists to say.
TRAINED_BALL_OFFSET_X = 0.09
TRAINED_BALL_ABS_Y = 0.042


#: The subsection that documents the stance, and the keyframe name it teaches.
#: The asset's own comment calls the current values "STAND2" because they
#: supersede an earlier ``STAND`` commented out beside them; the live keyframe
#: kept the name, so ``keyframe="STAND2"`` is refused.
STANCE_HEADING = "### The stance every weight was trained in"
STANCE_KEYFRAME = "STAND"

#: The one scene a ball kick needs, and the one that declares no keyframe.
BALL_SCENE = "scene_ball.xml"

#: The shipped keyframe rounds the exported stance to four decimals, so the two
#: agree to about 4e-05. This sits far under the smallest revision the asset has
#: already shipped - the superseded ``STAND`` is 0.066 rad away at the hip and
#: flips the sign of ``head_pitch`` - so a retune fails a cell here rather than
#: leaving the page quietly describing a pose the asset no longer declares.
STANCE_TOLERANCE = 1e-3


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """Return one ``##`` section's body, so a rule reads only its own prose."""
    start = text.index(heading)
    rest = text[start + len(heading) :]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def _advertised_skills(text: str) -> set[str]:
    """The weights named in the opening sentence's parenthetical.

    The list is read from the parenthetical that follows "policies", rather than
    from the whole opening: the paragraph below it names metadata fields
    (``joint_names``, ``action_scale``) in the same backticked style, and those
    are not skills. Anchoring on the structure keeps the rule derived - a tenth
    weight added to the list is graded - where a list of names to ignore would
    have to be edited every time the prose grew.

    A pair is spelled ``ball_kick_left``/``ball_kick_right`` inside one span
    pair, so each backticked run is split on ``/`` rather than taken whole.
    """
    opening = text[: text.index("## Walking in MuJoCo")]
    start = opening.index("policies (") + len("policies (")
    listed = opening[start : opening.index(")", start)]
    return {
        token.strip()
        for span in re.findall(r"`([^`]+)`", listed)
        for token in span.split("/")
        if re.fullmatch(r"[a-z][a-z0-9_]*", token.strip())
    }


def _scene_table(text: str) -> dict[str, str]:
    """Map every skill named in the Skill scenes table to the scene it needs."""
    rows: dict[str, str] = {}
    for line in _section(text, "## Skill scenes").splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2 or cells[0] == "skill":
            continue
        scene = re.search(r"`([A-Za-z0-9_]+\.xml)`", cells[1])
        if scene is None:
            continue
        for skill in re.findall(r"`([a-z][a-z0-9_]*)`", cells[0]):
            rows[skill] = scene.group(1)
    return rows


def _scene_path(scene: str) -> Path:
    """Locate a scene the asset directory carries, or skip.

    Reads the search paths rather than resolving through the asset manager,
    which downloads a missing asset: a test that clones a third-party
    repository fails on a host with no network instead of skipping.
    """
    from strands_robots.utils import get_search_paths

    present = next(
        (candidate for root in get_search_paths() if (candidate := Path(root) / ASSET_DIR / scene).exists()),
        None,
    )
    if present is None:
        pytest.skip(f"microduck {scene} is not downloaded, so its contents cannot be read")
    return present


def _scene_model(scene: str):
    """Compile a scene the asset directory carries, or skip."""
    mujoco = pytest.importorskip("mujoco")
    return mujoco, mujoco.MjModel.from_xml_path(str(_scene_path(scene)))


def _subsection(text: str, heading: str) -> str:
    """Return one ``###`` subsection's body, so a rule reads only its own prose."""
    start = text.index(heading)
    rest = text[start + len(heading) :]
    ends = [offset for offset in (rest.find("\n### "), rest.find("\n## ")) if offset != -1]
    return rest if not ends else rest[: min(ends)]


@contextlib.contextmanager
def _spawned(scene: str, keyframe: str | None = None):
    """Spawn the duck on ``scene``, optionally at a named keyframe."""
    mujoco = pytest.importorskip("mujoco")
    path = _scene_path(scene)
    from strands_robots.simulation import create_simulation

    sim = create_simulation("mujoco")
    assert sim.create_world().get("status") == "success"
    # Annotated ``Any`` because ``add_robot`` takes mixed types: a homogeneous
    # ``dict[str, str]`` splat reads as the ``position`` list to a type checker.
    extra: dict[str, Any] = {"keyframe": keyframe} if keyframe is not None else {}
    try:
        yield mujoco, sim, sim.add_robot(name="duck", urdf_path=str(path), **extra)
    finally:
        sim.destroy()


def _stance_in_sim(mujoco, model, data) -> np.ndarray:
    """Read the fourteen actuated joints by NAME, not by a flat qpos slice.

    The rollers scene renumbers that slice, so a name-indexed read is the only
    one that means the same thing on every scene the page names.
    """
    out = []
    for short in MICRODUCK_JOINT_NAMES:
        joint = next(
            (
                i
                for i in range(model.njnt)
                if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)) and name.split("/")[-1] == short
            ),
            None,
        )
        assert joint is not None, f"{short} is not a joint of the compiled model"
        out.append(float(data.qpos[model.jnt_qposadr[joint]]))
    return np.asarray(out)


def _joint_names(mujoco, model) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


def _wheel_joints(mujoco, model) -> list[str]:
    return [name for name in _joint_names(mujoco, model) if name and "wheel" in name]


def _ball_bodies(mujoco, model) -> list[str]:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    return [name for name in names if name and "ball" in name]


#: The subsection the placement invariant lives in.
PLACEMENT_HEADING = "### The ball scene carries the ball, not the kick geometry"


def _placement_section(text: str) -> str:
    """The placement paragraph, refusing by name when the heading is gone.

    ``_section`` resolves with ``str.index``, so a renamed heading would surface
    as ``ValueError: substring not found`` and name nothing. Asserting presence
    first means a rename fails saying which heading it could not find.
    """
    assert PLACEMENT_HEADING in text, f"the page no longer carries {PLACEMENT_HEADING!r}"
    return _section(text, PLACEMENT_HEADING)


def _declared_ball_position(mujoco, model) -> list[float]:
    """Where the scene puts the ball, read off the compiled model.

    ``body_pos`` is the declared placement, so this needs no ``MjData`` and no
    stepping: the question is where the file puts the prop, not where physics
    carries it.
    """
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    assert body >= 0, "the ball scene must carry a body named 'ball'"
    return [float(value) for value in model.body_pos[body]]


class TestThePageNamesASceneForEverySkill:
    """Read from the page alone, so it holds with no asset and no MuJoCo."""

    def test_the_opening_paragraph_still_lists_the_shipped_weights(self) -> None:
        """Non-vacuity: the cross-reference needs a skill list to check."""
        skills = _advertised_skills(_page())
        assert len(skills) >= MINIMUM_SKILLS, f"expected at least {MINIMUM_SKILLS} skills, found {sorted(skills)}"

    def test_the_table_still_covers_more_than_the_default_scene(self) -> None:
        """Non-vacuity: a one-row table would name no variant to verify."""
        scenes = set(_scene_table(_page()).values())
        assert len(scenes) >= MINIMUM_TABLE_ROWS, f"expected at least {MINIMUM_TABLE_ROWS} scenes, found {scenes}"

    def test_every_advertised_skill_appears_in_the_scene_table(self) -> None:
        """A weight advertised without a scene is the gap this file closes."""
        text = _page()
        unnamed = sorted(_advertised_skills(text) - set(_scene_table(text)))
        assert not unnamed, f"advertised with no scene named: {unnamed}"

    def test_the_table_names_no_skill_the_page_does_not_advertise(self) -> None:
        """Over-reach guard: the table describes the page, not a wider set."""
        text = _page()
        unadvertised = sorted(set(_scene_table(text)) - _advertised_skills(text))
        assert not unadvertised, f"in the table but never advertised: {unadvertised}"


class TestTheTablesClaimsAreTrueOfTheAssets:
    """Re-derive each row from the model it names, so the table cannot drift."""

    def test_every_scene_the_table_names_is_a_scene_that_compiles(self) -> None:
        """A row pointing at a file that is not there is a dead instruction."""
        for scene in sorted(set(_scene_table(_page()).values())):
            mujoco, model = _scene_model(scene)
            assert model.nu > 0, f"{scene} compiled with no actuators"

    def test_the_default_scene_carries_neither_a_wheel_nor_a_ball(self) -> None:
        """This is why four of the nine skills need a different scene."""
        mujoco, model = _scene_model(DEFAULT_SCENE)
        assert _wheel_joints(mujoco, model) == []
        assert _ball_bodies(mujoco, model) == []

    def test_a_roller_skill_is_pointed_at_a_scene_that_has_wheels(self) -> None:
        """The rollers scene adds the four passive ankle wheels."""
        scene = _scene_table(_page())["roller"]
        assert scene != DEFAULT_SCENE, "roller must not be pointed at the wheel-less default"
        mujoco, model = _scene_model(scene)
        assert len(_wheel_joints(mujoco, model)) == 4

    def test_a_ball_kick_skill_is_pointed_at_a_scene_that_has_a_ball(self) -> None:
        """The ball scene carries the prop. Where it sits is graded separately."""
        scene = _scene_table(_page())["ball_kick_left"]
        assert scene != DEFAULT_SCENE, "ball_kick must not be pointed at the ball-less default"
        mujoco, model = _scene_model(scene)
        assert _ball_bodies(mujoco, model) == ["ball"]

    def test_the_two_roller_skills_share_one_scene(self) -> None:
        """Both roller weights want the same wheels; one row covers them."""
        table = _scene_table(_page())
        assert table["roller"] == table["roller_crouch"]

    def test_the_two_ball_kick_skills_share_one_scene(self) -> None:
        """One scene carries the prop for both; the side is the policy's."""
        table = _scene_table(_page())
        assert table["ball_kick_left"] == table["ball_kick_right"]


class TestTheJointLayoutClaimsHold:
    """The page tells a raw ``qpos`` reader which scene renumbers the slice."""

    def test_the_actuator_order_is_the_same_on_every_documented_scene(self) -> None:
        """Why a policy writing ``ctrl`` is unaffected by the scene choice."""
        mujoco, default = _scene_model(DEFAULT_SCENE)

        def actuators(model) -> list[str]:
            return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

        expected = actuators(default)
        assert len(expected) == 14
        for scene in sorted(set(_scene_table(_page()).values())):
            _, model = _scene_model(scene)
            assert actuators(model) == expected, f"{scene} permutes the actuator order"

    def test_the_ball_scene_leaves_the_robots_position_slice_alone(self) -> None:
        """The ball's free joint is appended, so ``qpos[7:21]`` still holds."""
        mujoco, default = _scene_model(DEFAULT_SCENE)
        _, ball = _scene_model(_scene_table(_page())["ball_kick_left"])

        def layout(model) -> dict[str, int]:
            return {
                name: int(model.jnt_qposadr[i])
                for i, name in enumerate(_joint_names(mujoco, model))
                if name is not None
            }

        base, with_ball = layout(default), layout(ball)
        assert {name: adr for name, adr in with_ball.items() if name in base} == base

    def test_the_rollers_scene_moves_nine_of_the_fourteen_joints(self) -> None:
        """The count the page and the asset-shape guard both state."""
        mujoco, default = _scene_model(DEFAULT_SCENE)
        _, rollers = _scene_model(_scene_table(_page())["roller"])

        def layout(model) -> dict[str, int]:
            return {
                name: int(model.jnt_qposadr[i])
                for i, name in enumerate(_joint_names(mujoco, model))
                if name is not None
            }

        base, moved_layout = layout(default), layout(rollers)
        moved = [name for name, adr in base.items() if name in moved_layout and moved_layout[name] != adr]
        assert len(moved) == 9, f"expected nine renumbered joints, got {sorted(moved)}"

    def test_the_wheels_land_where_the_page_says_the_head_joints_sit(self) -> None:
        """The concrete mis-read the page warns a slice reader about."""
        mujoco, default = _scene_model(DEFAULT_SCENE)
        _, rollers = _scene_model(_scene_table(_page())["roller"])

        def at(model, addresses: set[int]) -> list[str]:
            return [
                name
                for i, name in enumerate(_joint_names(mujoco, model))
                if name is not None and int(model.jnt_qposadr[i]) in addresses
            ]

        assert at(default, {12, 13}) == ["neck_pitch", "head_pitch"]
        assert at(rollers, {12, 13}) == ["passive_LF_wheel", "passive_LR_wheel"]


class TestThePageSaysTheSceneDoesNotPlaceTheBallWhereTrainingDid:
    """Naming the scene is necessary and not sufficient, and the page says so.

    ``TestTheTablesClaimsAreTrueOfTheAssets`` grades the ball's **presence**: the row
    points at a scene, and that scene carries a body named ``ball``. It says nothing
    about **where**, and the position is the half that decides whether a kick connects.
    The scene declares the ball far enough ahead that no robot body reaches it, so a
    reader who follows the table alone still gets an air-swing - the outcome the table
    was added to remove for the roller skills.

    The first cell is a premise about the asset and holds either way - it is what
    makes the paragraph worth writing. The other two read the page and fail without
    it.
    """

    def test_the_scene_does_not_declare_the_ball_at_the_trained_offset(self) -> None:
        """The premise: naming the scene leaves the geometry wrong.

        Both axes differ. Forward, the scene is more than twice the trained offset;
        laterally it is centred where training offset the ball to the kicking foot.
        """
        mujoco, model = _scene_model(_scene_table(_page())["ball_kick_left"])
        forward, lateral, _height = _declared_ball_position(mujoco, model)
        assert forward > 2 * TRAINED_BALL_OFFSET_X, (
            f"the scene declares the ball {forward} m ahead, which is no longer far "
            f"enough past the trained {TRAINED_BALL_OFFSET_X} m for this page's "
            "placement paragraph to be the reason a kick misses"
        )
        assert lateral == 0.0, f"the scene now offsets the ball laterally to {lateral} m"

    def test_the_page_records_the_trained_offset_and_the_scenes_own_position(self) -> None:
        """Both numbers, so a reader can see the gap rather than take it on trust."""
        section = _placement_section(_page())
        mujoco, model = _scene_model(_scene_table(_page())["ball_kick_left"])
        forward = _declared_ball_position(mujoco, model)[0]
        for number in (f"{forward} m", f"{TRAINED_BALL_OFFSET_X} m", f"{TRAINED_BALL_ABS_Y} m"):
            assert number in section, f"the placement paragraph does not state {number}"

    def test_the_page_tells_the_reader_the_joint_name_is_not_fixed(self) -> None:
        """``add_robot`` prefixes every joint, so a fixed default resolves for one caller."""
        section = _placement_section(_page())
        assert "ball_free" in section, "the paragraph does not name the joint to write"
        assert "add_robot" in section, "the paragraph does not say the name is prefixed"


class TestThePageDocumentsTheTrainedStance:
    """Read the page and the package, so these hold with no asset and no MuJoCo."""

    def test_the_exported_stance_is_public_and_covers_every_joint(self) -> None:
        """Non-vacuity: every rule below compares against this constant."""
        from strands_robots.policies import microduck

        assert "MICRODUCK_DEFAULT_POSE" in microduck.__all__
        assert len(MICRODUCK_DEFAULT_POSE) == len(MICRODUCK_JOINT_NAMES)

    def test_the_section_names_the_keyframe_that_reaches_the_stance(self) -> None:
        """A reader who cannot name the keyframe cannot reach the stance."""
        body = _subsection(_page(), STANCE_HEADING)
        assert f'keyframe="{STANCE_KEYFRAME}"' in body

    def test_the_section_names_the_constant_rather_than_repeating_the_pose(self) -> None:
        """A copied pose drifts: the asset has already revised this one."""
        body = _subsection(_page(), STANCE_HEADING)
        assert "MICRODUCK_DEFAULT_POSE" in body

    def test_the_section_names_the_scene_where_the_route_is_unavailable(self) -> None:
        """The one scene a ball kick needs declares no keyframe at all."""
        assert BALL_SCENE in _subsection(_page(), STANCE_HEADING)


class TestTheStanceClaimsAreTrueOfTheAssets:
    """Re-derive the stance from the shipped models, so the page cannot drift."""

    def test_the_shipped_keyframe_is_the_stance_the_package_exports(self) -> None:
        """The drift guard: a revised keyframe fails here instead of diverging."""
        mujoco, model = _scene_model(DEFAULT_SCENE)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
        assert STANCE_KEYFRAME in names, f"{DEFAULT_SCENE} declares {names}"
        keyed = model.key_qpos[names.index(STANCE_KEYFRAME)][7 : 7 + len(MICRODUCK_DEFAULT_POSE)]
        gap = float(np.max(np.abs(np.asarray(keyed) - np.asarray(MICRODUCK_DEFAULT_POSE))))
        assert gap <= STANCE_TOLERANCE, f"{STANCE_KEYFRAME} is {gap:.6g} rad from MICRODUCK_DEFAULT_POSE"

    def test_the_keyframe_route_reaches_the_stance_and_survives_a_reset(self) -> None:
        """The route the section documents, and the stickiness it promises."""
        with _spawned(DEFAULT_SCENE, STANCE_KEYFRAME) as (mujoco, sim, result):
            assert result.get("status") == "success", result
            spawned = _stance_in_sim(mujoco, sim._world._model, sim._world._data)
            assert np.allclose(spawned, MICRODUCK_DEFAULT_POSE, atol=STANCE_TOLERANCE)
            assert sim.reset().get("status") == "success"
            after = _stance_in_sim(mujoco, sim._world._model, sim._world._data)
            assert np.allclose(after, MICRODUCK_DEFAULT_POSE, atol=STANCE_TOLERANCE), (
                "a keyframe spawn must survive reset, or every episode after the first starts elsewhere"
            )

    def test_a_spawn_without_a_keyframe_starts_at_the_zero_configuration(self) -> None:
        """The counterfactual: no refusal, and the wrong origin for the decode."""
        with _spawned(DEFAULT_SCENE, None) as (mujoco, sim, result):
            assert result.get("status") == "success", result
            spawned = _stance_in_sim(mujoco, sim._world._model, sim._world._data)
            assert np.allclose(spawned, 0.0)
            gap = float(np.max(np.abs(spawned - np.asarray(MICRODUCK_DEFAULT_POSE))))
            assert gap > STANCE_TOLERANCE, "the zero configuration must not already be the trained stance"

    def test_the_ball_scene_declares_no_keyframe_so_the_route_refuses_there(self) -> None:
        """Why the section names the ball scene as the exception."""
        _mujoco, model = _scene_model(BALL_SCENE)
        assert model.nkey == 0, f"{BALL_SCENE} now declares {model.nkey} keyframe(s)"
        with _spawned(BALL_SCENE, STANCE_KEYFRAME) as (_mj, _sim, result):
            assert result.get("status") == "error"
            assert "no <keyframe>" in result["content"][0]["text"]
