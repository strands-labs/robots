"""Every backend answers ``gripper_body`` from one gripper-hint vocabulary.

``list_bodies(robot_name=...)`` advertises a best-guess gripper/end-effector
mount in its ``gripper_body`` field - the surface a caller reads to resolve
``add_camera(parent_body=...)`` for a wrist view. The backends matched on that
field through one shared rule
(:func:`~strands_robots.simulation.ik.hint_matches_name`) and each kept a hint
vocabulary of its own, and the two vocabularies had drifted: only one of them
carried ``jaw``. So the SO-100 - whose gripper bodies are named ``Fixed_Jaw``
and ``Moving_Jaw``, and which is a shipped registry robot - reported its mount
on one backend and reported ``gripper_body: None`` on the other, while that same
payload listed both jaw bodies.

Sharing the *matcher* is not enough for two surfaces to agree; they agree only
where they also read the same vocabulary. ``gripper_body`` is one question about
one robot, so the vocabulary behind it has one owner:
:data:`~strands_robots.simulation.ik.GRIPPER_BODY_HINTS`.

:func:`~strands_robots.simulation.ik.discover_ee_frame` deliberately keeps its
own words. It answers a different question - which frame to solve IK *to* - and
may prefer a wrist to a jaw on the very same arm. The cells here pin that
difference rather than erasing it.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation import ik
from strands_robots.simulation.factory import _BUILTIN_BACKENDS
from strands_robots.simulation.ik import GRIPPER_BODY_HINTS, hint_matches_name
from strands_robots.simulation.models import SimRobot, SimWorld

mujoco = pytest.importorskip("mujoco")

# The vocabulary as this file states it, independently of the module under test.
# Only one cell compares the two, so a mutation to the shipped constant cannot
# quietly move every expectation here with it.
EXPECTED_HINTS = ("gripper", "hand", "jaw", "ee", "tool")

# The word whose absence was the defect, and a body name spelled with it. The
# SO-100 family names its gripper bodies this way.
JAW_HINT = "jaw"
JAW_BODY = "Moving_Jaw"

# A body-name set whose only end-effector-like member is a jaw. This is what
# makes the parity cells able to see the drift at all: a fixture whose mount is
# named ``gripper`` matches every candidate vocabulary and so cannot distinguish
# them.
DISTINGUISHING_BODIES = ("left_knee_link", "wheel_hub_back_link", JAW_BODY)


def _chain(links: str) -> str:
    """A serial chain of hinge-jointed bodies named by ``links``."""
    names = links.split()
    xml = '<mujoco><worldbody><body name="base">'
    xml += '<joint name="j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>'
    for i, name in enumerate(names):
        xml += f'<body name="{name}">'
        xml += f'<joint name="j{i + 1}" type="hinge"/><geom type="box" size=".05 .05 .05"/>'
    xml += "</body>" * len(names)
    xml += "</body></worldbody></mujoco>"
    return xml


def _mujoco_mount(tmp_path: Path, bodies: tuple[str, ...]) -> str | None:
    """``gripper_body`` from the real MuJoCo backend for an inline chain."""
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    path = tmp_path / "arm.xml"
    path.write_text(_chain(" ".join(bodies)))
    sim: Any = MuJoCoSimEngine()
    try:
        sim.create_world(ground_plane=False)
        added = sim.add_robot(name="bot", urdf_path=str(path))
        assert added["status"] == "success", added["content"][0]["text"]
        payload = sim.list_bodies(robot_name="bot")["content"][1]["json"]
        assert payload["bodies"], "the chain must contribute bodies to search"
        return payload["gripper_body"]
    finally:
        sim.destroy()


def _newton_mount(bodies: tuple[str, ...]) -> str | None:
    """``gripper_body`` from the real Newton ``list_bodies``, no Newton runtime.

    The method reads only the robot registry and the per-robot body map, so a
    stand-in ``self`` carrying those drives the shipped code - the pattern the
    sibling word-boundary suite uses.
    """
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    labels = [f"bot/base/{name}" for name in bodies]
    world = SimWorld()
    world.robots["bot"] = SimRobot(name="bot", urdf_path="bot.xml", data_config="bot", joint_names=[])
    engine: Any = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = types.SimpleNamespace(body_label=list(labels))
    engine._robot_body_map = {"bot": list(labels)}
    result = NewtonSimEngine.list_bodies(engine, "bot")
    assert result["status"] == "success"
    return result["content"][1]["json"]["gripper_body"]


# Backends whose ``gripper_body`` this file drives end to end. A premise cell
# below asserts this covers every backend that answers the question at all, so a
# backend added later is either driven here or fails that cell.
MOUNT_DRIVERS = ("mujoco", "newton")


def _backend_classes() -> dict[str, type]:
    """Every registered backend class, by the name ``create_simulation`` takes."""
    resolved: dict[str, type] = {}
    for name, (module_path, class_name) in _BUILTIN_BACKENDS.items():
        resolved[name] = getattr(importlib.import_module(module_path), class_name)
    return resolved


def _backends_answering_the_mount_question() -> set[str]:
    """Backends whose ``list_bodies`` puts a ``gripper_body`` in its payload."""
    answering = set()
    for name, cls in _backend_classes().items():
        method = getattr(cls, "list_bodies", None)
        if method is None:
            continue
        if '"gripper_body"' in inspect.getsource(method):
            answering.add(name)
    return answering


def _vocabularies_read_for_the_mount() -> dict[str, set[str]]:
    """The hint-vocabulary name each backend's ``list_bodies`` reads.

    Read off the source rather than the value so two backends that happen to
    hold equal tuples today still register as two vocabularies. Equal contents
    are what the drift looked like before it drifted.
    """
    read: dict[str, set[str]] = {}
    for name in sorted(_backends_answering_the_mount_question()):
        cls = _backend_classes()[name]
        source = textwrap.dedent(inspect.getsource(cls.list_bodies))  # type: ignore[attr-defined]
        tree = ast.parse(source)
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id.upper().endswith("GRIPPER_BODY_HINTS")
        }
        read[name] = names
    return read


class TestThePremiseIsReal:
    """The drift was reachable through a shipped asset, not only a fixture."""

    def test_the_shipped_so100_names_its_gripper_bodies_with_the_jaw_word(self) -> None:
        """SO-100's gripper bodies carry ``jaw`` and no other hint word.

        This is why the missing word cost the mount outright rather than merely
        picking a different body: nothing else in the arm matches at all.
        """
        from strands_robots.simulation.model_registry import resolve_model_path

        model = mujoco.MjModel.from_xml_path(str(resolve_model_path("so100")))
        bodies = [
            name for index in range(model.nbody) if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index))
        ]
        jaws = [name for name in bodies if hint_matches_name(JAW_HINT, name)]
        assert jaws, bodies
        others = [
            name for name in bodies for hint in EXPECTED_HINTS if hint != JAW_HINT and hint_matches_name(hint, name)
        ]
        assert others == [], others

    def test_the_matcher_is_shared_by_both_surfaces(self) -> None:
        """One matcher; the vocabularies are what this file is about."""
        assert callable(hint_matches_name)
        assert "hint_matches_name" in inspect.getsource(ik.discover_ee_frame)

    def test_the_driven_backends_are_every_backend_that_answers(self) -> None:
        """A backend that later advertises a mount is driven here, or fails.

        Without this the parity cells could silently cover a shrinking share of
        the backends that answer the question.
        """
        assert _backends_answering_the_mount_question() == set(MOUNT_DRIVERS)

    def test_the_distinguishing_fixture_really_distinguishes(self) -> None:
        """The fixture's only mount candidate is the contested word.

        A fixture named ``gripper`` matches every candidate vocabulary, so it
        cannot see a vocabulary drift - which is how the drift survived.
        """
        matched = [hint for hint in EXPECTED_HINTS for name in DISTINGUISHING_BODIES if hint_matches_name(hint, name)]
        assert matched == [JAW_HINT], matched


class TestBothBackendsAdvertiseTheSameMount:
    """The same robot reports the same mount whichever backend answers."""

    def test_the_two_backends_agree_on_a_jaw_only_arm(self, tmp_path: Path) -> None:
        """Pre-fix MuJoCo answered ``None`` here and Newton answered the jaw."""
        assert _mujoco_mount(tmp_path, DISTINGUISHING_BODIES) == f"bot/{JAW_BODY}"
        assert _newton_mount(DISTINGUISHING_BODIES) == f"bot/base/{JAW_BODY}"

    def test_neither_backend_reports_no_mount_for_a_jaw_only_arm(self, tmp_path: Path) -> None:
        """Stated as the absence, because the pre-fix answer was an absence."""
        assert _mujoco_mount(tmp_path, DISTINGUISHING_BODIES) is not None
        assert _newton_mount(DISTINGUISHING_BODIES) is not None

    @pytest.mark.parametrize("hint", EXPECTED_HINTS)
    def test_every_hint_resolves_a_mount_on_both_backends(self, tmp_path: Path, hint: str) -> None:
        """No entry of the shared set is honoured by only one backend."""
        bodies = ("shoulder_link", f"{hint}_link")
        assert _mujoco_mount(tmp_path, bodies) == f"bot/{hint}_link"
        assert _newton_mount(bodies) == f"bot/base/{hint}_link"


class TestTheShippedAssetResolvesItsMount:
    """The registry robot the drift affected now reports its mount."""

    def test_so100_advertises_a_jaw_as_its_mount(self) -> None:
        """The payload listed both jaws while saying there was no mount."""
        from strands_robots.simulation import create_simulation
        from strands_robots.simulation.model_registry import resolve_model_path

        sim: Any = create_simulation("mujoco")
        try:
            sim.create_world()
            added = sim.add_robot(name="arm", urdf_path=str(resolve_model_path("so100")))
            assert added["status"] == "success", added["content"][0]["text"]
            result = sim.list_bodies(robot_name="arm")
            payload = next(block["json"] for block in result["content"] if "json" in block)
            text = next(block["text"] for block in result["content"] if "text" in block)
        finally:
            sim.destroy()

        mount = payload["gripper_body"]
        assert mount is not None, payload["bodies"]
        assert hint_matches_name(JAW_HINT, mount.rsplit("/", 1)[-1]), mount
        assert mount in payload["bodies"], (mount, payload["bodies"])
        assert "Gripper/EEF mount" in text


class TestTheVocabularyHasOneOwner:
    """One question, one vocabulary - derived, so a third backend is graded."""

    def test_every_answering_backend_reads_the_shared_name(self) -> None:
        read = _vocabularies_read_for_the_mount()
        assert read, "no backend was found answering the mount question"
        for backend, names in read.items():
            assert names == {"GRIPPER_BODY_HINTS"}, (backend, names)

    def test_no_backend_defines_a_gripper_vocabulary_of_its_own(self) -> None:
        """A per-backend copy is the shape that drifted; refuse a second one."""
        owners = []
        package = Path(inspect.getfile(ik)).parent
        for path in sorted(package.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.upper().endswith("GRIPPER_BODY_HINTS"):
                        owners.append(f"{path.relative_to(package)}:{node.lineno} {target.id}")
        assert len(owners) == 1, owners
        assert owners[0].startswith("ik.py"), owners

    def test_the_shipped_vocabulary_is_the_one_this_file_expects(self) -> None:
        """The single cell that couples this file to the shipped constant."""
        assert GRIPPER_BODY_HINTS == EXPECTED_HINTS


class TestWhatMustNotChange:
    """Sharing the vocabulary must not widen it or move a settled answer."""

    def test_a_knee_or_a_wheel_is_still_not_a_mount(self, tmp_path: Path) -> None:
        legs = ("left_knee_link", "wheel_hub_back_link")
        assert _mujoco_mount(tmp_path, legs) is None
        assert _newton_mount(legs) is None

    def test_so101_reports_the_mount_it_always_reported(self) -> None:
        """Its ``gripper`` body precedes its jaw, so the answer is unchanged."""
        from strands_robots.simulation import create_simulation
        from strands_robots.simulation.model_registry import resolve_model_path

        sim: Any = create_simulation("mujoco")
        try:
            sim.create_world()
            assert sim.add_robot(name="arm", urdf_path=str(resolve_model_path("so101")))["status"] == "success"
            payload = next(block["json"] for block in sim.list_bodies(robot_name="arm")["content"] if "json" in block)
        finally:
            sim.destroy()
        assert payload["gripper_body"] == "arm/gripper"

    def test_ik_still_prefers_a_wrist_to_a_jaw_on_the_same_arm(self) -> None:
        """The IK surface answers a different question and keeps its words.

        Its target frame for the SO-100 is the wrist, not a jaw. Folding the two
        vocabularies together would move it.
        """
        from strands_robots.simulation.model_registry import resolve_model_path

        model = mujoco.MjModel.from_xml_path(str(resolve_model_path("so100")))
        discovered = ik.discover_ee_frame(model)
        assert discovered is not None
        frame, kind = discovered
        assert kind == "body"
        assert hint_matches_name("wrist", frame), frame
        assert not hint_matches_name(JAW_HINT, frame), frame

    def test_the_ik_vocabulary_is_not_the_mount_vocabulary(self) -> None:
        """Pinned as a difference so neither can be quietly made the other."""
        assert JAW_HINT not in ik._BODY_HINTS
        assert JAW_HINT in GRIPPER_BODY_HINTS
