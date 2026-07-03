# mypy: disable-error-code="attr-defined,arg-type"
"""Tests for the declarative ``staged_reward`` phase-machine primitive.

``staged_reward`` is the one generic stateful reward term: it composes EXISTING
registry predicates into a monotonic Reach->Grasp->Transport->Place curriculum.
The task (e.g. pick-and-place) is authored as DATA, never shipped as code, and
never via ``eval``. These tests drive a fake SimEngine through the phases and
pin: phase advance, one-time bonus, monotonicity (no regression), reset(), and
that a full pick-place reward compiles from a spec dict alone.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    StatefulRewardTerm,
    make_predicate,
)


class _ScriptedEngine:
    """Fake SimEngine exposing body positions we mutate to script a trajectory.

    Provides ``get_body_state(body_name=...)`` in the MuJoCo-backend result
    shape so the stock ``distance_neg`` / ``distance_less_than`` /
    ``body_above_z`` predicates work unmodified against it.
    """

    def __init__(self) -> None:
        self.positions: dict[str, list[float]] = {
            "gripper": [0.0, 0.0, 0.0],
            "cube": [1.0, 0.0, 0.0],
            "target": [1.0, 0.0, 0.5],
        }

    def set(self, body: str, pos: list[float]) -> None:
        self.positions[body] = list(pos)

    def get_body_state(self, body_name: str) -> dict:
        pos = self.positions.get(body_name)
        if pos is None:
            return {"status": "error", "content": []}
        return {"status": "success", "content": [{"json": {"position": list(pos)}}]}


# A 3-stage pick-place-style reward, authored as PURE DATA (no Python class).
PICK_PLACE_STAGES = [
    {
        # Reach: pull gripper toward cube; advance when close.
        "reward": {"predicate": "distance_neg", "body_a": "gripper", "body_b": "cube", "weight": 1.0},
        "advance_when": {"predicate": "distance_less_than", "body_a": "gripper", "body_b": "cube", "threshold": 0.1},
        "bonus": 5.0,
    },
    {
        # Lift/Transport: reward cube height; advance when cube is high enough.
        "reward": {"predicate": "distance_neg", "body_a": "cube", "body_b": "target", "weight": 1.0},
        "advance_when": {"predicate": "body_above_z", "body": "cube", "z": 0.4},
        "bonus": 10.0,
    },
    {
        # Place: final dense term, no gate (terminal handled by success predicate).
        "reward": {"predicate": "distance_neg", "body_a": "cube", "body_b": "target", "weight": 2.0},
    },
]


def test_staged_reward_registered() -> None:
    assert "staged_reward" in PREDICATE_REGISTRY


def test_staged_reward_is_stateful_term() -> None:
    term = make_predicate("staged_reward", stages=PICK_PLACE_STAGES)
    assert isinstance(term, StatefulRewardTerm)
    assert hasattr(term, "reset") and callable(term.reset)
    assert term.phase == 0


def test_phase_advance_and_one_time_bonus() -> None:
    eng = _ScriptedEngine()
    term = make_predicate("staged_reward", stages=PICK_PLACE_STAGES)

    # Phase 0 (Reach): gripper far from cube -> negative distance, no advance.
    eng.set("gripper", [0.0, 0.0, 0.0])  # dist to cube (1,0,0) = 1.0
    r0 = term(eng)
    assert term.phase == 0
    assert r0 == pytest.approx(-1.0)

    # Move gripper onto the cube -> within threshold -> advance + 5.0 bonus once.
    eng.set("gripper", [1.0, 0.0, 0.0])  # dist 0 < 0.1
    r1 = term(eng)
    assert term.phase == 1
    # reward = distance_neg(gripper,cube)=0.0 at transition + bonus 5.0
    assert r1 == pytest.approx(5.0)

    # Next call is now in phase 1 (Transport), cube at (1,0,0) vs target (1,0,0.5)=0.5.
    r2 = term(eng)
    assert term.phase == 1
    assert r2 == pytest.approx(-0.5)  # no bonus re-award
    # The bonus is truly one-time: staying in-phase never re-awards.
    assert r2 != pytest.approx(5.0)


def test_full_progression_to_terminal_stage() -> None:
    eng = _ScriptedEngine()
    term = make_predicate("staged_reward", stages=PICK_PLACE_STAGES)

    eng.set("gripper", [1.0, 0.0, 0.0])  # advance 0->1 (+5)
    term(eng)
    assert term.phase == 1

    eng.set("cube", [1.0, 0.0, 0.45])  # above z=0.4 -> advance 1->2 (+10)
    r = term(eng)
    assert term.phase == 2
    # transition reward = distance_neg(cube,target) at cube z=0.45, target z=0.5 -> -0.05, +10 bonus
    assert r == pytest.approx(-0.05 + 10.0)

    # Final stage has no gate: phase stays at 2, weight=2.0 applies.
    eng.set("cube", [1.0, 0.0, 0.5])  # exactly on target -> dist 0
    r_final = term(eng)
    assert term.phase == 2
    assert r_final == pytest.approx(0.0)


def test_monotonic_no_regression() -> None:
    eng = _ScriptedEngine()
    term = make_predicate("staged_reward", stages=PICK_PLACE_STAGES)
    eng.set("gripper", [1.0, 0.0, 0.0])
    term(eng)
    assert term.phase == 1
    # Move gripper BACK far from cube: must NOT drop back to phase 0.
    eng.set("gripper", [0.0, 0.0, 0.0])
    term(eng)
    assert term.phase == 1


def test_reset_clears_phase() -> None:
    eng = _ScriptedEngine()
    term = make_predicate("staged_reward", stages=PICK_PLACE_STAGES)
    eng.set("gripper", [1.0, 0.0, 0.0])
    term(eng)
    assert term.phase == 1
    term.reset()
    assert term.phase == 0


# --- validation / safety ---


def test_rejects_empty_stages() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        make_predicate("staged_reward", stages=[])


def test_rejects_non_final_stage_without_advance_when() -> None:
    bad = [
        {"reward": {"predicate": "constant", "value": 1.0}},  # no advance_when, not final
        {"reward": {"predicate": "constant", "value": 1.0}},
    ]
    with pytest.raises(ValueError, match="advance_when"):
        make_predicate("staged_reward", stages=bad)


def test_rejects_unknown_sub_predicate() -> None:
    bad = [{"reward": {"predicate": "no_such_predicate"}}]
    with pytest.raises(ValueError, match="Unknown predicate"):
        make_predicate("staged_reward", stages=bad)


def test_rejects_unknown_stage_key() -> None:
    bad = [{"reward": {"predicate": "constant", "value": 1.0}, "bogus": 1}]
    with pytest.raises(ValueError, match="unknown keys"):
        make_predicate("staged_reward", stages=bad)


def test_rejects_non_numeric_bonus() -> None:
    bad = [
        {
            "reward": {"predicate": "constant", "value": 1.0},
            "advance_when": {"predicate": "contact_any"},
            "bonus": "lots",
        },
        {"reward": {"predicate": "constant", "value": 1.0}},
    ]
    with pytest.raises(ValueError, match="bonus must be a number"):
        make_predicate("staged_reward", stages=bad)


def test_rejects_non_dict_stage() -> None:
    # A stage that is not a mapping cannot declare reward/advance_when, so the
    # factory must reject it up front rather than raising a cryptic AttributeError
    # deeper in compilation.
    with pytest.raises(ValueError, match=r"stage\[0\] must be a dict, got str"):
        make_predicate("staged_reward", stages=["reach then grasp"])


def test_rejects_stage_with_missing_reward() -> None:
    # ``reward`` is mandatory and must be a predicate-call dict. A stage that
    # omits it (or supplies a non-dict) is malformed.
    with pytest.raises(ValueError, match=r"stage\[0\]\.reward must be a predicate-call dict"):
        make_predicate("staged_reward", stages=[{"advance_when": {"predicate": "contact_any"}}])


def test_rejects_non_dict_reward_call() -> None:
    with pytest.raises(ValueError, match=r"stage\[0\]\.reward must be a predicate-call dict"):
        make_predicate("staged_reward", stages=[{"reward": "distance_neg"}])


def test_rejects_non_dict_advance_when() -> None:
    # When ``advance_when`` is present it must itself be a predicate-call dict;
    # a bare string is not a valid gate.
    bad = [
        {"reward": {"predicate": "constant", "value": 1.0}, "advance_when": "contact_any"},
        {"reward": {"predicate": "constant", "value": 1.0}},
    ]
    with pytest.raises(ValueError, match=r"stage\[0\]\.advance_when must be a predicate-call dict"):
        make_predicate("staged_reward", stages=bad)


def test_empty_compiled_stages_reward_is_zero() -> None:
    # The factory rejects empty stages, but the compiled term itself must stay
    # safe if ever handed an empty stage list directly: it scores 0.0 rather
    # than indexing out of range.
    from strands_robots.simulation.predicates import _StagedReward

    term = _StagedReward([])
    assert term(_ScriptedEngine()) == 0.0
    assert term.phase == 0
