"""A refusal that offers a joint-discovery action names one an agent can call.

Two ``set_joint_positions`` / ``set_joint_velocities`` refusals - an empty
mapping, and a mapping whose keys are not joints of the model - closed with
``use action='robot_joint_names' to see one robot's joints.``
``robot_joint_names`` is a deliberate Python-only capability: it is absent from
``tool_spec.json``'s ``action`` enum, and the agent-facing entry points refuse a
non-enum action outright ("is not available to an agent ... reachable from
Python only"). So an agent that read the remedy and followed it verbatim got a
second refusal, and that one names no alternative either - a closed loop with no
way out of it, on the discovery step of writing a pose.

``_unknown_mj_entity_msg`` had the answer one branch above: its ``Body`` arm
offers ``action='list_bodies'``, which *is* published. The ``Joint`` arm is the
same sentence about a different entity kind, so the two only had to agree.
``get_robot_state`` is published, is accepted with or without ``robot_name``,
and reports every joint of a robot by name - which is what the sentence
promises - so the hint now names it.

What this pins, in the order it matters:

* Following the remedy works. The remedy is parsed back *out* of the live
  refusal and invoked, so this fails for a missing name, a wrong name, or a name
  the boundary refuses - not for a wording change.
* Nothing in ``physics.py`` offers an unpublished action.
* Across the MuJoCo backend the only unpublished action still named in a refusal
  is ``send_action``, whose published equivalent is an open question (#2378):
  ``set_joint_positions`` bypasses the actuators and ``actuate_robot`` rewrites
  an actuator-less arm, so neither is a drop-in. Pinning the set here does not
  settle that; it stops a *new* one arriving in silence, the same job
  ``_PYTHON_ONLY_ACTIONS`` does for the dispatch surface.

The controls are the other half. ``robot_joint_names`` stays Python-only and
stays dispatchable: publishing it would widen the agent surface to settle a
wording bug, and deleting it would break the Python callers and
``policy_runner``, so the fix must do neither.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import (  # noqa: E402
    _PUBLISHED_ACTIONS,
    Simulation,
)
from tests.simulation.mujoco.test_tool_spec import _PYTHON_ONLY_ACTIONS  # noqa: E402

#: The one action still named in a MuJoCo refusal that an agent cannot call.
#: Not an exemption - the reason it is still here is recorded in #2378, and this
#: set failing is the signal that the list changed rather than that it is fine.
_KNOWN_UNPUBLISHED_IN_REFUSALS = {"send_action"}

_MUJOCO_SRC = Path(__file__).resolve().parents[3] / "strands_robots" / "simulation" / "mujoco"

#: ``action='<name>'`` as a refusal writes it when offering the caller a remedy.
_OFFERED_ACTION_RE = re.compile(r"action='([a-z_]+)'")

_NOT_FOR_AGENTS = "is not available to an agent"


@pytest.fixture
def sim():
    s = Simulation(tool_name="joint_discovery_hint_test", mesh=False)
    s.create_world()
    s.add_robot(name="so101")
    yield s
    s.cleanup()


def _text(result: dict) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []))


def _offered_actions(text: str) -> list[str]:
    return _OFFERED_ACTION_RE.findall(text)


# --- the remedy is followable ------------------------------------------------

# Both refusals reach an agent through the published ``set_joint_*`` actions, so
# each is a remedy a model is handed in practice rather than a Python-only path.
_REFUSALS = {
    "empty_mapping": {"positions": {}},
    "unresolved_joint": {"positions": {"not_a_joint": 0.1}},
    "empty_velocities": {"velocities": {}},
    "unresolved_velocity_joint": {"velocities": {"not_a_joint": 0.1}},
}


@pytest.mark.parametrize("case", sorted(_REFUSALS))
def test_following_the_offered_joint_discovery_action_succeeds(sim, case):
    """Parse the remedy out of the refusal, call it, and require it to work."""
    action = "set_joint_positions" if "positions" in _REFUSALS[case] else "set_joint_velocities"
    refusal = sim(action=action, robot_name="so101", **_REFUSALS[case])
    assert refusal["status"] == "error", refusal

    offered = _offered_actions(_text(refusal))
    assert offered, f"{case}: the refusal offers no action at all: {_text(refusal)!r}"

    for name in offered:
        assert name in _PUBLISHED_ACTIONS, (
            f"{case}: the refusal offers action={name!r}, which tool_spec does not publish, "
            f"so an agent that follows it is refused again. Refusal was: {_text(refusal)!r}"
        )
        followed = sim(action=name, robot_name="so101")
        assert followed["status"] == "success", (
            f"{case}: following the offered action={name!r} did not succeed: {_text(followed)!r}"
        )
        assert _NOT_FOR_AGENTS not in _text(followed)


def test_the_offered_action_actually_reports_the_joints_by_name(sim):
    """The sentence promises "one robot's joints", so the remedy must name them."""
    refusal = sim(action="set_joint_positions", robot_name="so101", positions={})
    offered = _offered_actions(_text(refusal))
    assert offered

    reported = _text(sim(action=offered[0], robot_name="so101"))
    joints = sim.robot_joint_names("so101")
    assert joints, "premise: so101 has joints to report"
    for joint in joints:
        # The state report keys joints by their short name, as the write path does.
        short = joint.split("/")[-1]
        assert short in reported, f"joint {joint!r} is absent from the offered action's report: {reported!r}"


# --- no refusal offers an action an agent cannot call ------------------------


def test_no_physics_refusal_offers_an_unpublished_action():
    """``physics.py`` owns both joint-discovery hints; every action it offers is published."""
    offered = _offered_actions((_MUJOCO_SRC / "physics.py").read_text())
    assert offered, "premise: physics.py offers at least one action"
    assert set(offered) <= _PUBLISHED_ACTIONS, {a for a in offered if a not in _PUBLISHED_ACTIONS}


def test_the_unpublished_actions_offered_by_mujoco_refusals_are_the_known_set():
    """A *new* unfollowable remedy cannot ship in silence."""
    offered: dict[str, set[str]] = {}
    for path in sorted(_MUJOCO_SRC.glob("*.py")):
        for name in _offered_actions(path.read_text()):
            offered.setdefault(name, set()).add(path.name)
    assert offered, "premise: the backend offers actions in its refusals"

    unpublished = {name: sorted(files) for name, files in offered.items() if name not in _PUBLISHED_ACTIONS}
    assert set(unpublished) == _KNOWN_UNPUBLISHED_IN_REFUSALS, (
        "the set of unpublished actions offered by a MuJoCo refusal changed: "
        f"{unpublished}. An agent cannot call these; either offer a published "
        "action or record the decision before adding to the set."
    )


# --- controls: true before and after the fix ---------------------------------


def test_the_body_branch_still_offers_list_bodies():
    """The precedent branch this fix follows is left alone."""
    src = (_MUJOCO_SRC / "physics.py").read_text()
    assert "action='list_bodies'" in src
    assert "list_bodies" in _PUBLISHED_ACTIONS


def test_robot_joint_names_is_still_python_only(sim):
    """Publishing it would widen the agent surface to settle a wording bug."""
    assert "robot_joint_names" in _PYTHON_ONLY_ACTIONS
    assert "robot_joint_names" not in _PUBLISHED_ACTIONS
    refused = sim(action="robot_joint_names", robot_name="so101")
    assert refused["status"] == "error"
    assert _NOT_FOR_AGENTS in _text(refused)


def test_robot_joint_names_is_still_dispatchable_from_python(sim):
    """Deleting it would break the Python callers and ``policy_runner``."""
    names = sim.robot_joint_names("so101")
    assert names and all(isinstance(n, str) for n in names)


def test_the_unresolved_joint_refusal_still_names_the_key_and_the_joints(sim):
    """Redirecting the hint must not cost the caller the rest of the diagnosis."""
    text = _text(sim(action="set_joint_positions", robot_name="so101", positions={"not_a_joint": 0.1}))
    assert "not_a_joint" in text
    assert "Available joints" in text
    for joint in sim.robot_joint_names("so101"):
        assert joint in text
