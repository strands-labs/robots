"""``success_fn="contact"`` must read the payload the engine actually returns.

The string is one of exactly two values
:meth:`~strands_robots.simulation.policy_runner.PolicyRunner.evaluate` accepts
for sparse success, and ``eval_policy``'s own docstring recommends it. The
reader behind it indexed the ``get_contacts`` result as if the result *were*
the payload::

    if result.get("n_contacts", 0) > 0: ...
    contacts = result.get("contacts")

Every in-tree backend returns the agent-tool envelope, whose only top-level
keys are ``status`` and ``content``, so both lookups missed and the check
returned ``False`` for every episode regardless of what the arm did. The bare
mapping a test double returns *did* satisfy that reader, which is why the
divergence never showed up in the suite: the code worked for the stub and
failed for the engine.

The mode now shares the predicate DSL's ``contact_any`` -- whose docstring
already claimed to "match the legacy ``success_fn='contact'`` path" -- so
there is one reader instead of two. These tests pin:

* the engine consequence: a resting pair scores a success and a separated
  pair does not, with the scene's premise measured through
  ``get_contact_forces`` so it cannot silently drift into a no-contact scene;
* the envelope premise: the contact list is reachable only through a ``json``
  content block, which is why the replaced reader could not work;
* parity: the string mode and ``make_predicate("contact_any")`` agree for
  every payload shape either one accepts, so they cannot drift apart again;
* the shape tolerance: an envelope and a bare payload mapping read alike, so
  a minimal engine that returns a plain reading is still understood.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner
from strands_robots.simulation.predicates import _extract_json, make_predicate
from tests.simulation.test_policy_runner import FakeSim

# A one-hinge arm so ``evaluate`` has a robot to drive. The base geom is offset
# well clear of the link capsule: a self-contact inside the arm would make the
# separated control case report a contact and the test would pass vacuously.
ARM_XML = """<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.40">
      <geom name="post" type="box" size="0.03 0.03 0.03" pos="0 0 -0.14" rgba="0.3 0.3 0.35 1"/>
      <body name="link1" pos="0 0 0">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2 2" limited="true" damping="3"/>
        <geom name="l1" type="capsule" fromto="0 0 0 0.18 0 0" size="0.025" rgba="0.2 0.45 0.85 1"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="a1" joint="j1" kp="30" ctrlrange="-2 2"/></actuator>
</mujoco>
"""

CUBE_MASS = 0.5
_SETTLE_STEPS = 120


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    """Read a tool result's ``json`` content block the way a caller must."""
    return next((b["json"] for b in result.get("content", []) if "json" in b), {})


def _build(tmp_path, *, touching: bool):
    """A cube either resting on a static plate or floating well above it."""
    arm = tmp_path / "arm.xml"
    arm.write_text(ARM_XML)
    sim = Simulation(backend="mujoco", mesh=False)
    # No ground plane: the only contact the scene can report is the pair under
    # test, so a verdict cannot come from the robot standing on the floor.
    assert sim.create_world(ground_plane=False)["status"] == "success"
    assert sim.add_robot(name="arm", urdf_path=str(arm))["status"] == "success"
    assert (
        sim.add_object(
            "plate",
            shape="box",
            position=[0.42, 0.0, 0.10],
            size=[0.30, 0.30, 0.04],
            color=[0.85, 0.85, 0.88, 1.0],
            is_static=True,
        )["status"]
        == "success"
    )
    assert (
        sim.add_object(
            "cube",
            shape="box",
            position=[0.42, 0.0, 0.19 if touching else 0.62],
            size=[0.12, 0.12, 0.12],
            color=[0.95, 0.45, 0.12, 1.0],
            mass=CUBE_MASS,
        )["status"]
        == "success"
    )
    sim.step(_SETTLE_STEPS)
    return sim


def _normal_force(sim) -> float:
    """Total normal force the solver reports, in newtons."""
    contacts = _payload(sim.get_contact_forces()).get("contacts", [])
    return sum(abs(float(c.get("normal_force", 0.0))) for c in contacts)


def _evaluate(sim) -> dict[str, Any]:
    result = sim.eval_policy(
        robot_name="arm",
        policy_provider="mock",
        n_episodes=2,
        max_steps=12,
        success_fn="contact",
        control_frequency=50.0,
    )
    assert result["status"] == "success"
    return _payload(result)


class TestAgainstTheRealEngine:
    """The consequence a caller sees: an evaluated success rate."""

    def test_a_resting_pair_scores_the_episode_a_success(self, tmp_path):
        """A cube at rest on a plate is a contact, so every episode succeeds.

        The premise is measured rather than assumed: the solver carries the
        cube's full weight, so a scene that silently stopped touching would
        fail here instead of turning this into a vacuous pass.
        """
        sim = _build(tmp_path, touching=True)
        try:
            assert _normal_force(sim) == pytest.approx(CUBE_MASS * 9.81, rel=0.05)
            report = _evaluate(sim)
            assert report["success_measured"] is True
            assert report["success_rate"] == 1.0
        finally:
            sim.cleanup()

    def test_a_separated_pair_scores_no_success(self, tmp_path):
        """The same scene with the cube in free fall must score nothing.

        Without this the fix could be "always report success" and the suite
        would not notice.
        """
        sim = _build(tmp_path, touching=False)
        try:
            assert _payload(sim.get_contacts())["contacts"] == []
            assert _normal_force(sim) == 0.0
            assert _evaluate(sim)["success_rate"] == 0.0
        finally:
            sim.cleanup()

    def test_the_contact_list_is_only_reachable_through_a_content_block(self, tmp_path):
        """Why the replaced reader could not work, pinned as a premise.

        ``get_contacts`` carries its records inside the envelope's ``json``
        block; the envelope itself has neither ``contacts`` nor ``n_contacts``.
        Any reader indexing the result directly misses both.
        """
        sim = _build(tmp_path, touching=True)
        try:
            result = sim.get_contacts()
            assert sorted(result) == ["content", "status"]
            assert "contacts" not in result
            assert "n_contacts" not in result
            assert len(_payload(result)["contacts"]) > 0
        finally:
            sim.cleanup()


class _ContactSim(FakeSim):
    """A sim whose ``get_contacts`` returns a caller-supplied result."""

    def __init__(self, result: Any):
        super().__init__()
        self._result = result

    def get_contacts(self) -> Any:
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


ENVELOPE_TOUCHING = {"status": "success", "content": [{"json": {"contacts": [{"geom1": "a", "geom2": "b"}]}}]}
ENVELOPE_EMPTY = {"status": "success", "content": [{"text": "No contacts."}, {"json": {"contacts": []}}]}
ENVELOPE_ERROR = {"status": "error", "content": [{"text": "No world."}]}

# Payload shapes, and whether "any contact" holds for each.
PAYLOAD_SHAPES: list[tuple[str, Any]] = [
    ("envelope_with_contacts", ENVELOPE_TOUCHING),
    ("envelope_empty_list", ENVELOPE_EMPTY),
    ("envelope_error", ENVELOPE_ERROR),
    ("bare_positive_count", {"n_contacts": 2}),
    ("bare_zero_count", {"n_contacts": 0}),
    ("bare_contacts_list", {"contacts": [{"geom1": "hand", "geom2": "cube"}]}),
    ("bare_empty_mapping", {}),
    ("malformed_contacts", {"contacts": "not-a-list", "n_contacts": 0}),
    ("not_a_mapping", "unexpected"),
    ("raises", NotImplementedError("backend has no contacts")),
    ("raises_unexpectedly", RuntimeError("boom")),
]


class TestOneReaderForBothSurfaces:
    """The string mode and the DSL predicate must not drift apart again."""

    @pytest.mark.parametrize("label, result", PAYLOAD_SHAPES, ids=[label for label, _ in PAYLOAD_SHAPES])
    def test_the_string_mode_agrees_with_the_contact_any_predicate(self, label, result):
        """Both surfaces read one payload the same way, for every shape."""
        sim = _ContactSim(result)
        success_fn = PolicyRunner(sim)._resolve_success_fn("contact")
        assert success_fn is not None
        assert success_fn({}) is make_predicate("contact_any")(sim), label

    def test_the_engine_envelope_is_understood_by_both(self):
        """The shape that used to score nothing now scores a success."""
        sim = _ContactSim(ENVELOPE_TOUCHING)
        assert PolicyRunner(sim)._resolve_success_fn("contact")({}) is True
        assert make_predicate("contact_any")(sim) is True

    def test_a_missing_get_contacts_is_not_a_success(self):
        """A backend with no contact list scores nothing rather than raising."""

        class _NoContacts(FakeSim):
            get_contacts = None  # type: ignore[assignment]

        sim = _NoContacts()
        assert PolicyRunner(sim)._resolve_success_fn("contact")({}) is False


class TestPayloadShapeTolerance:
    """``_extract_json`` reads an envelope and a bare payload mapping alike."""

    def test_an_envelope_yields_its_json_block(self):
        assert _extract_json(ENVELOPE_TOUCHING) == {"contacts": [{"geom1": "a", "geom2": "b"}]}

    def test_an_envelope_without_a_json_block_yields_nothing(self):
        assert _extract_json(ENVELOPE_ERROR) == {}

    def test_a_bare_mapping_is_the_payload(self):
        """A result with no ``content`` at all is not an envelope."""
        assert _extract_json({"n_contacts": 2}) == {"n_contacts": 2}

    def test_a_non_mapping_yields_nothing(self):
        assert _extract_json(None) == {}
