"""The Isaac backend applies the shared action-value domain.

``SimEngine._coerce_action`` is the one place the action-value domain lives: a
value must coerce to a finite scalar number, a boolean is refused rather than
written as a ``1.0``/``0.0`` target, a single-element sequence -- the
``list[float]`` a 1-DoF key carries under the ``Policy.get_actions -> list[dict]``
contract -- is unwrapped to its scalar, and an ordered vector is bound
positionally to ``robot_action_keys`` with its width required to match that list
exactly. ``TestEveryBackendInheritsTheGuard`` in
``tests/simulation/test_send_action_rejects_a_boolean_command.py`` states the
invariant this file measures: the guard lives in the shared coercion so no
backend can ship without it.

The Isaac backend shipped its own conversion instead, so on this backend alone:

  * a boolean reached the articulation as a ``1.0`` / ``0.0`` PD target;
  * ``nan`` / ``inf`` reached it as a target no solver can honor;
  * a vector whose width did not match the robot was handed to the articulation
    anyway, addressed at every DOF -- so a caller error surfaced, at best, as a
    shape complaint from the runtime rather than as the width mismatch it is;
  * a multi-element value, a non-numeric value and ``None`` raised ``TypeError``
    / ``ValueError`` straight past ``send_action``'s structured envelope; and
  * the single-element row a 1-DoF key carries raised ``TypeError`` as well, so
    a policy emitting the documented shape could not drive this backend at all.

Every value is checked against the shared coercion's own verdict rather than
against a hand-written table, so the two cannot drift apart. None of it requires
NVIDIA Isaac Sim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState

from .test_backend_parity import _FakeArticulation, fake_isaacsim_types  # noqa: F401 - fixture

JOINTS = ["shoulder", "elbow", "wrist", "gripper"]


class _CountingWorld:
    """Isaac World stand-in that records how many physics steps were run."""

    def __init__(self) -> None:
        self.steps = 0

    def step(self, render: bool = False) -> None:  # noqa: ARG002 - signature parity
        self.steps += 1


def _running_sim(articulation: Any) -> IsaacSimulation:
    sim = IsaacSimulation()
    sim._world = _CountingWorld()
    sim._world_created = True
    sim._robots = {
        "arm": _RobotState(
            name="arm",
            prim_path="/World/Robots/arm",
            joint_names=list(JOINTS),
            articulation=articulation,
        )
    }
    return sim


def _call(sim: IsaacSimulation, action: Any) -> dict[str, Any]:
    """Invoke send_action with a deliberately off-domain value.

    Routed through one funnel so the off-type values this file exists to probe
    are not each a separate typing suppression at the call site.
    """
    return sim.send_action(action, robot_name="arm")


# Values that must be refused. Each is reachable from a policy or an agent tool
# rather than being a typo: a boolean is the conventional binary-gripper action,
# a non-finite value is what a diverging policy emits, and a width mismatch is
# what an action head sized from the wrong key list produces.
REFUSED: list[tuple[str, Any]] = [
    ("python bool True", {"gripper": True}),
    ("python bool False", {"gripper": False}),
    ("numpy bool", {"gripper": np.bool_(True)}),
    ("nan", {"gripper": float("nan")}),
    ("inf", {"gripper": float("inf")}),
    ("multi-element value", {"gripper": [1.0, 2.0, 3.0]}),
    ("non-numeric value", {"gripper": "not-a-number"}),
    ("None value", {"gripper": None}),
    ("vector too short", [0.1, 0.2]),
    ("vector too long", [0.1, 0.2, 0.3, 0.4, 0.5]),
    ("ndarray too short", np.array([0.1, 0.2])),
    ("nan in vector", [0.1, float("nan"), 0.3, 0.04]),
    ("bool in vector", [0.1, True, 0.3, 0.04]),
    ("non-numeric in vector", [0.1, "x", 0.3, 0.04]),
]

# Values that were already accepted and must stay accepted.
ACCEPTED: list[tuple[str, Any, list[float]]] = [
    ("plain float", {"gripper": 0.04}, [0.04]),
    ("numeric string", {"gripper": "0.04"}, [0.04]),
    ("negative float", {"gripper": -0.04}, [-0.04]),
    ("full-width vector", [0.1, 0.2, 0.3, 0.04], [0.1, 0.2, 0.3, 0.04]),
    ("single-element row", {"gripper": [0.04]}, [0.04]),
    ("single-element ndarray row", {"gripper": np.array([0.04])}, [0.04]),
]


class TestTheDomainMatchesTheSharedContract:
    """Isaac's verdict for a value is the shared coercion's verdict for it."""

    @pytest.mark.parametrize(("label", "action"), REFUSED + [(lbl, a) for lbl, a, _ in ACCEPTED])
    def test_the_verdicts_agree(self, fake_isaacsim_types, label, action) -> None:  # noqa: F811, ARG002
        sim = _running_sim(_FakeArticulation())
        _, contract_error = SimEngine._coerce_action(sim, action, "arm")
        contract_refuses = contract_error is not None

        result = _call(sim, action)
        backend_refuses = result["status"] == "error"

        assert backend_refuses is contract_refuses, (
            f"{label}: backend says {'error' if backend_refuses else 'success'} while the shared "
            f"contract says {'error' if contract_refuses else 'accepted'}"
        )

    @pytest.mark.parametrize(("label", "action"), REFUSED)
    def test_a_refused_value_reports_the_contract_reason(self, fake_isaacsim_types, label, action) -> None:  # noqa: F811, ARG002
        sim = _running_sim(_FakeArticulation())
        _, contract_error = SimEngine._coerce_action(sim, action, "arm")
        assert contract_error is not None, f"{label} must be off-domain for this test to mean anything"

        result = _call(sim, action)
        assert result["content"][0]["text"] == contract_error["content"][0]["text"], label


class TestARefusalReachesNoActuator:
    """A refusal is reported before anything is written or advanced."""

    @pytest.mark.parametrize(("label", "action"), REFUSED)
    def test_no_target_is_applied_and_no_step_is_run(self, fake_isaacsim_types, label, action) -> None:  # noqa: F811, ARG002
        art = _FakeArticulation()
        sim = _running_sim(art)

        assert _call(sim, action)["status"] == "error", label
        assert art.last_action is None, f"{label} reached the articulation"
        assert sim._world.steps == 0, f"{label} advanced physics"

    def test_no_off_domain_value_escapes_the_structured_envelope(self, fake_isaacsim_types) -> None:  # noqa: F811
        """Not one of them raises: the envelope is send_action's whole contract.

        A raise here is worse than a wrong value. ``PolicyRunner`` counts an
        error result to decide whether a rollout is degrading; an exception
        instead unwinds the rollout mid-episode.
        """
        for label, action in REFUSED:
            sim = _running_sim(_FakeArticulation())
            try:
                result = _call(sim, action)
            except Exception as exc:  # noqa: BLE001 - an escape past the envelope IS the finding
                pytest.fail(f"{label} raised {type(exc).__name__} past the envelope: {exc}")
            assert result["status"] == "error", label


class TestTheSingleElementRowIsHonored:
    """The shape the get_actions contract emits for a 1-DoF key is applied."""

    def test_a_one_element_row_applies_its_scalar(self, fake_isaacsim_types) -> None:  # noqa: F811
        art = _FakeArticulation()
        sim = _running_sim(art)

        assert _call(sim, {"gripper": [0.04]})["status"] == "success"
        assert list(np.asarray(art.last_action.joint_positions)) == pytest.approx([0.04])
        assert list(np.asarray(art.last_action.joint_indices)) == [JOINTS.index("gripper")]

    def test_a_one_element_row_matches_the_bare_scalar(self, fake_isaacsim_types) -> None:  # noqa: F811
        """The unwrap is an unwrap: the two spellings command the same target."""
        wrapped_art, bare_art = _FakeArticulation(), _FakeArticulation()
        assert _call(_running_sim(wrapped_art), {"gripper": [0.04]})["status"] == "success"
        assert _call(_running_sim(bare_art), {"gripper": 0.04})["status"] == "success"

        assert list(np.asarray(wrapped_art.last_action.joint_positions)) == pytest.approx(
            list(np.asarray(bare_art.last_action.joint_positions))
        )


class TestAVectorWidthMustMatchTheActionKeys:
    """A width that does not match is refused rather than partly applied."""

    def test_the_fixture_makes_truncation_detectable(self, fake_isaacsim_types) -> None:  # noqa: F811
        sim = _running_sim(_FakeArticulation())
        assert len(sim.robot_action_keys("arm")) == len(JOINTS) > 2, (
            "the robot needs several action keys or a dropped trailing command is invisible"
        )

    def test_a_short_vector_does_not_command_the_joints_it_covers(self, fake_isaacsim_types) -> None:  # noqa: F811
        art = _FakeArticulation()
        sim = _running_sim(art)

        result = _call(sim, [0.1, 0.2])
        assert result["status"] == "error"
        assert art.last_action is None, "a partial command reached the articulation"

    @pytest.mark.parametrize("action", [[0.1, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5]])
    def test_the_refusal_names_both_widths(self, fake_isaacsim_types, action) -> None:  # noqa: F811
        text = _call(_running_sim(_FakeArticulation()), action)["content"][0]["text"]
        assert f"length {len(action)}" in text
        assert f"count {len(JOINTS)}" in text
        assert "gripper" in text, "the message must show the action keys the caller should target"

    def test_an_exact_width_vector_commands_every_joint_in_order(self, fake_isaacsim_types) -> None:  # noqa: F811
        art = _FakeArticulation()
        sim = _running_sim(art)

        assert _call(sim, [0.1, 0.2, 0.3, 0.04])["status"] == "success"
        assert list(np.asarray(art.last_action.joint_indices)) == list(range(len(JOINTS)))
        assert list(np.asarray(art.last_action.joint_positions)) == pytest.approx([0.1, 0.2, 0.3, 0.04])


class TestUsableValuesStayUsable:
    """The over-reach control: nothing that worked is now refused."""

    @pytest.mark.parametrize(("label", "action", "expected"), ACCEPTED)
    def test_an_on_domain_value_is_applied_and_advances_physics(
        self,
        fake_isaacsim_types,  # noqa: F811
        label,
        action,
        expected,
    ) -> None:
        art = _FakeArticulation()
        sim = _running_sim(art)

        assert _call(sim, action)["status"] == "success", label
        assert list(np.asarray(art.last_action.joint_positions)) == pytest.approx(expected), label
        assert sim._world.steps == 1, label

    def test_a_partial_dict_still_commands_only_the_named_joint(self, fake_isaacsim_types) -> None:  # noqa: F811
        """The coercion returns a mapping unchanged, so subset targeting survives."""
        art = _FakeArticulation()
        sim = _running_sim(art)

        assert _call(sim, {"gripper": 0.04})["status"] == "success"
        assert list(np.asarray(art.last_action.joint_indices)) == [JOINTS.index("gripper")]

    def test_an_unresolved_key_is_still_reported_with_the_keys_it_applied(self, fake_isaacsim_types) -> None:  # noqa: F811
        """Name resolution is a separate question from the value domain."""
        sim = _running_sim(_FakeArticulation())

        result = _call(sim, {"gripper": 0.04, "nosuchjoint": 0.1})
        assert result["status"] == "error"
        payload = next(block["json"] for block in result["content"] if "json" in block)
        assert payload["unresolved_keys"] == ["nosuchjoint"]
        assert payload["applied"] == ["gripper"], "the report must name keys, not the vector's floats"
