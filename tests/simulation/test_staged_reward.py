# mypy: disable-error-code="attr-defined,arg-type"
"""Tests for the declarative ``staged_reward`` phase-machine primitive.

``staged_reward`` is the one generic stateful reward term: it composes EXISTING
registry predicates into a monotonic Reach->Grasp->Transport->Place curriculum.
The task (e.g. pick-and-place) is authored as DATA, never shipped as code, and
never via ``eval``. These tests drive a fake SimEngine through the phases and
pin: phase advance, one-time bonus, monotonicity (no regression), reset(), and
that a full pick-place reward compiles from a spec dict alone.

``staged_reward`` is itself registered, and a stage's ``reward`` is compiled by
calling back into ``make_predicate``, so a stage may hold another phase machine.
A nested machine is per-episode state exactly like the outer one, and the outer
phase resets correctly whether or not the inner one does - so the reset pins
below drive a nested machine and not only a flat one.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    StatefulRewardTerm,
    make_predicate,
    register_predicate,
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


def test_advances_at_most_one_stage_per_step() -> None:
    # A single ``__call__`` advances the phase machine by AT MOST one stage,
    # even when a later stage's ``advance_when`` gate is already satisfiable on
    # the same step. Here one step simultaneously satisfies BOTH the stage-0
    # gate (gripper on cube: distance < 0.1) AND the stage-1 gate (cube above
    # z=0.4). The machine must land in phase 1 (not skip straight to phase 2),
    # award only stage-0's one-time bonus (5.0, not 5.0 + 10.0), and emit
    # stage-0's dense reward. This is the curriculum-monotonicity contract: a
    # phase is never skipped and its bonus never awarded before the machine has
    # actually spent a step in it, so a fast trajectory that trips two gates in
    # one control period cannot double-award or emit the wrong stage's reward.
    # Guards against a regression to a per-step ``while`` advance loop.
    eng = _ScriptedEngine()
    term = make_predicate("staged_reward", stages=PICK_PLACE_STAGES)
    eng.set("cube", [1.0, 0.0, 0.45])  # stage-1 gate would also fire: cube z=0.45 > 0.4
    eng.set("gripper", [1.0, 0.0, 0.45])  # stage-0 gate fires too: dist(gripper, cube) = 0 < 0.1
    r = term(eng)
    # Advanced exactly one stage (0 -> 1), NOT two (0 -> 2).
    assert term.phase == 1
    # Emitted stage-0 reward (distance_neg gripper<->cube = 0.0) + stage-0 bonus
    # only. The stage-1 bonus (10.0) is NOT awarded this step.
    assert r == pytest.approx(5.0)


def test_reset_clears_phase() -> None:
    """The flat case: every sub-predicate is stateless, so the phase IS the state."""
    eng = _ScriptedEngine()
    term = make_predicate("staged_reward", stages=PICK_PLACE_STAGES)
    eng.set("gripper", [1.0, 0.0, 0.0])
    term(eng)
    assert term.phase == 1
    term.reset()
    assert term.phase == 0


def test_reset_does_not_disturb_the_stages_themselves() -> None:
    """Control: a machine still advances and still pays its bonus after a reset."""
    eng = _ScriptedEngine()
    term = make_predicate("staged_reward", stages=PICK_PLACE_STAGES)
    term.reset()
    eng.set("gripper", [1.0, 0.0, 0.0])
    assert term(eng) == pytest.approx(5.0)
    assert term.phase == 1


# --- a nested machine is per-episode state too ---
#
# The inner machine below pays a large one-time bonus on its single transition,
# so an episode's score says which stage it opened in. The OUTER gate asks for
# z=5.0, which nothing in these episodes reaches, so the outer machine stays in
# stage 0 for the whole episode and every step is scored by the inner one.


def _nested_machine_call(depth: int) -> dict[str, Any]:
    """A ``staged_reward`` predicate call holding ``depth`` machines in total."""
    call: dict[str, Any] = {
        "predicate": "staged_reward",
        "stages": [
            {
                "reward": {"predicate": "constant", "value": 1.0},
                "advance_when": {"predicate": "body_above_z", "body": "cube", "z": 0.2},
                "bonus": 100.0,
            },
            {"reward": {"predicate": "constant", "value": 7.0}},
        ],
    }
    for _ in range(depth - 1):
        call = {
            "predicate": "staged_reward",
            "stages": [
                {"reward": call, "advance_when": {"predicate": "body_above_z", "body": "cube", "z": 5.0}},
                {"reward": {"predicate": "constant", "value": 0.0}},
            ],
        }
    return call


# What one episode scores when the innermost gate fires on its third step: two
# steps of the inner stage-0 shaping term, the transition paying that term plus
# the one-time bonus, then the inner terminal stage.
FIRST_EPISODE = [1.0, 1.0, 101.0, 7.0, 7.0]


def _score_episode(term: Any, eng: _ScriptedEngine, *, steps: int = 5) -> list[float]:
    """Score ``steps`` steps from the pose every episode starts at, lifting on step 2."""
    eng.set("cube", [1.0, 0.0, 0.0])
    scores = []
    for step in range(steps):
        if step == 2:
            eng.set("cube", [1.0, 0.0, 0.6])
        scores.append(round(float(term(eng)), 4))
    return scores


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_a_second_episode_replays_a_nested_curriculum(depth: int) -> None:
    """Episodes are independent, however deep the machine holding the state is."""
    eng = _ScriptedEngine()
    term = make_predicate("staged_reward", stages=_nested_machine_call(depth)["stages"])
    assert _score_episode(term, eng) == FIRST_EPISODE
    term.reset()
    assert _score_episode(term, eng) == FIRST_EPISODE


def test_a_nested_one_time_bonus_is_awarded_once_per_episode() -> None:
    """Once per episode, not once per process: three episodes pay it three times."""
    eng = _ScriptedEngine()
    term = make_predicate("staged_reward", stages=_nested_machine_call(2)["stages"])
    awarded = 0
    for _ in range(3):
        term.reset()
        awarded += sum(1 for score in _score_episode(term, eng) if score > 50.0)
    assert awarded == 3


def test_a_nested_machine_reports_stage_zero_after_a_reset() -> None:
    """The inner phase directly, so a failure names the state and not the score."""
    outer = make_predicate("staged_reward", stages=_nested_machine_call(2)["stages"])
    inner = outer._stages[0][0]
    assert isinstance(inner, StatefulRewardTerm)
    eng = _ScriptedEngine()
    _score_episode(outer, eng)
    assert (outer.phase, inner.phase) == (0, 1)
    outer.reset()
    assert (outer.phase, inner.phase) == (0, 0)


@pytest.mark.parametrize("slot", ["reward", "advance_when"])
def test_reset_clears_a_stateful_sub_term_in_whichever_slot_holds_it(slot: str) -> None:
    """The rule is the sub-term's, not the slot's: anything resettable is cleared.

    No shipped predicate reaches ``advance_when`` statefully - the kind guard
    refuses a reward term there - but one registered through
    :func:`register_predicate` can, and a latched gate is exactly the "latched
    successes" the :class:`StatefulRewardTerm` contract names.
    """

    class _Latch:
        def __init__(self) -> None:
            self.resets = 0

        def __call__(self, _sim: Any) -> Any:
            return True if slot == "advance_when" else 1.0

        def reset(self) -> None:
            self.resets += 1

    latch = _Latch()
    name = "test_staged_reward_stateful_sub_term"
    register_predicate(name, lambda: latch)
    try:
        stage: dict[str, Any] = {
            "reward": {"predicate": "constant", "value": 1.0},
            "advance_when": {"predicate": "body_above_z", "body": "cube", "z": 0.2},
        }
        stage[slot] = {"predicate": name}
        term = make_predicate(
            "staged_reward",
            stages=[stage, {"reward": {"predicate": "constant", "value": 0.0}}],
        )
        assert latch.resets == 0
        term.reset()
        assert latch.resets == 1
    finally:
        PREDICATE_REGISTRY.pop(name, None)


def test_a_spec_authored_nested_machine_is_cleared_by_the_consumers_rule() -> None:
    """The path a real task takes: authored as data, compiled by the spec
    compiler, and cleared by the duck-typed call both consumers make on the
    terms they hold (``SimEnv.reset`` / ``on_episode_start``)."""
    bench = DeclarativeBenchmark.from_dict(
        {
            "name": "nested_curriculum",
            "default_robot": "so101",
            "supported_robots": ["so101"],
            "dense_reward": [_nested_machine_call(2)],
        }
    )
    eng = _ScriptedEngine()

    def episode() -> list[float]:
        eng.set("cube", [1.0, 0.0, 0.0])
        scores = []
        for step in range(5):
            if step == 2:
                eng.set("cube", [1.0, 0.0, 0.6])
            scores.append(round(bench.on_step(eng, {}, {}).reward, 4))
        return scores

    assert episode() == FIRST_EPISODE
    for term in bench._reward_terms:
        term_reset = getattr(term, "reset", None)
        if callable(term_reset):
            term_reset()
    assert episode() == FIRST_EPISODE


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
    with pytest.raises(ValueError, match="bonus must be a finite number"):
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
