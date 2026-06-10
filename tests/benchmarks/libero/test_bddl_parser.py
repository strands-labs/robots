"""Tests for the LIBERO BDDL parser.

Covers:

* Tokenizer handling of comments, quoted strings, nested parens.
* S-expression parsing - depth, arity, EOF errors.
* Top-level ``(define ...)`` structure + section extraction
  (``:domain``, ``:objects``, ``:init``, ``:goal``, ``:language``).
* Predicate compilation for every entry in ``PREDICATE_VOCABULARY``.
* Boolean combinators (``and`` / ``or`` / ``not``) with short-circuit behaviour.
* Rejection of unknown predicates / wrong arities.
* Round-trip on a curated 5-task subset covering each predicate family.

The compiled callables are executed against the same fake sims used by
``tests/simulation/test_benchmark_predicates.py`` - no LIBERO / MuJoCo
dependency required.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.benchmarks.libero.bddl_parser import (
    PREDICATE_VOCABULARY,
    And,
    BDDLParseError,
    Not,
    Or,
    Pred,
    _tokenize,
    compile_goal,
    parse_bddl,
    parse_bddl_file,
)

# Fake sims


class _BodyStateSim:
    def __init__(self, bodies: dict[str, dict[str, Any]]):
        self._bodies = bodies

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._bodies:
            return {"status": "error", "content": [{"text": "missing"}]}
        return {
            "status": "success",
            "content": [
                {"text": body_name},
                {
                    "json": {
                        "position": self._bodies[body_name].get("position", [0, 0, 0]),
                        "quaternion": self._bodies[body_name].get("quaternion", [1, 0, 0, 0]),
                        "mass": 1.0,
                    }
                },
            ],
        }

    def get_observation(self, *_, **__) -> dict[str, Any]:
        return self._bodies.get("_joints", {})


class _ContactSim:
    def __init__(self, contacts: list[dict[str, str]]):
        self._contacts = contacts

    def get_contacts(self) -> dict[str, Any]:
        return {
            "status": "success",
            "content": [
                {"text": f"{len(self._contacts)} contacts"},
                {"json": {"contacts": self._contacts, "n_contacts": len(self._contacts)}},
            ],
        }


class _CombinedSim(_BodyStateSim, _ContactSim):
    """Both body state and contacts for multi-predicate goals."""

    def __init__(
        self,
        bodies: dict[str, dict[str, Any]] | None = None,
        contacts: list[dict[str, str]] | None = None,
    ):
        _BodyStateSim.__init__(self, bodies or {})
        _ContactSim.__init__(self, contacts or [])


# Tokenizer


class TestTokenize:
    def test_basic(self):
        assert _tokenize("(and a b)") == ["(", "and", "a", "b", ")"]

    def test_comments_stripped(self):
        assert _tokenize("(foo) ; trailing comment\n(bar)") == ["(", "foo", ")", "(", "bar", ")"]

    def test_quoted_strings_preserved(self):
        toks = _tokenize('(:language "pick the red cube")')
        # The quoted region is a single token, including the quotes.
        assert '"pick the red cube"' in toks

    def test_unterminated_quote_errors(self):
        with pytest.raises(BDDLParseError, match="unterminated quoted string"):
            _tokenize('(:language "unterminated')


# Top-level parser


class TestParseBDDL:
    def test_minimal(self):
        text = """
            (define (problem libero_pick)
              (:domain kitchen)
              (:goal (on cube_1 plate_1)))
        """
        problem = parse_bddl(text)
        assert problem.name == "libero_pick"
        assert problem.domain == "kitchen"
        assert isinstance(problem.goal, Pred)
        assert problem.goal.name == "on"
        assert problem.goal.args == ("cube_1", "plate_1")

    def test_extracts_language(self):
        text = """
            (define (problem p1)
              (:language "pick up the red cube")
              (:goal (grasped cube_1)))
        """
        problem = parse_bddl(text)
        assert problem.language == "pick up the red cube"

    def test_extracts_objects_flattening_typed_syntax(self):
        """PDDL-style ``obj1 obj2 - type`` annotations are flattened to symbols."""
        text = """
            (define (problem p)
              (:objects cube_1 plate_1 - object table_1 - fixture)
              (:goal (on cube_1 plate_1)))
        """
        problem = parse_bddl(text)
        assert problem.objects == ["cube_1", "plate_1", "object", "table_1", "fixture"]

    def test_extracts_init_clauses(self):
        text = """
            (define (problem p)
              (:init (on cube_1 table_1) (upright bottle_1))
              (:goal (on cube_1 plate_1)))
        """
        problem = parse_bddl(text)
        assert len(problem.init) == 2
        # Each init clause is a compiled Pred.
        assert all(isinstance(n, Pred) for n in problem.init)

    def test_goal_with_and(self):
        text = """
            (define (problem p)
              (:goal (and (on cube_1 plate_1) (upright cube_1))))
        """
        problem = parse_bddl(text)
        assert isinstance(problem.goal, And)
        assert len(problem.goal.clauses) == 2

    def test_goal_with_or_and_not(self):
        text = """
            (define (problem p)
              (:goal (or (grasped cube_1) (not (on cube_1 table_1)))))
        """
        problem = parse_bddl(text)
        assert isinstance(problem.goal, Or)
        inner = problem.goal.clauses[1]
        assert isinstance(inner, Not)

    def test_missing_define_rejected(self):
        with pytest.raises(BDDLParseError, match="top-level"):
            parse_bddl("(problem foo)")

    def test_empty_input_rejected(self):
        with pytest.raises(BDDLParseError):
            parse_bddl("")

    def test_missing_paren_rejected(self):
        with pytest.raises(BDDLParseError, match="closing"):
            parse_bddl("(define (problem p)")

    def test_trailing_tokens_rejected(self):
        with pytest.raises(BDDLParseError, match="trailing"):
            parse_bddl("(define (problem p) (:goal (on a b))) (extra)")


# Predicate vocabulary


class TestPredicateVocabulary:
    @pytest.mark.parametrize("bddl_name", sorted(PREDICATE_VOCABULARY.keys()))
    def test_every_predicate_compiles(self, bddl_name: str):
        """Each BDDL predicate must produce a compilable goal with a valid argc."""
        sample_args = {
            "on": "cube_1 table_1",
            "near": "cube_1 gripper_1",
            "inside": "cube_1 basket_1",
            "open": "drawer_joint",
            "closed": "drawer_joint",
            "grasped": "cube_1",
            "upright": "bottle_1",
        }
        args = sample_args[bddl_name]
        text = f"(define (problem p) (:goal ({bddl_name} {args})))"
        problem = parse_bddl(text)
        # Must compile without error.
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        assert callable(fn)

    def test_unknown_predicate_rejected_with_list(self):
        text = "(define (problem p) (:goal (telekinesis cube_1)))"
        with pytest.raises(BDDLParseError) as exc:
            parse_bddl(text)
        assert "unknown predicate" in str(exc.value).lower()
        # Error must list the valid vocabulary so the author can fix it.
        for expected in ("on", "grasped", "upright"):
            assert expected in str(exc.value)

    @pytest.mark.parametrize(
        "expr,reason",
        [
            ("(on cube_1)", "wrong arity"),
            ("(on cube_1 plate_1 extra)", "extra arg"),
            ("(grasped)", "no arg"),
            ("(upright a b)", "extra arg"),
        ],
    )
    def test_wrong_arity_rejected(self, expr: str, reason: str):
        with pytest.raises(BDDLParseError):
            parse_bddl(f"(define (problem p) (:goal {expr}))")

    def test_not_with_wrong_arity(self):
        with pytest.raises(BDDLParseError, match="not"):
            parse_bddl("(define (problem p) (:goal (not (on a b) (on c d))))")


# Compiled goal evaluation


class TestCompileGoal:
    def test_and_short_circuits(self):
        """``and`` must evaluate to False as soon as one clause fails."""
        text = """
            (define (problem p)
              (:goal (and
                (on cube_1 table_1)
                (upright bottle_1))))
        """
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        sim_hit = _BodyStateSim(
            {
                "cube_1": {"position": [0, 0, 0.2]},
                "table_1": {"position": [0, 0, 0.0]},
                "bottle_1": {"quaternion": [1.0, 0.0, 0.0, 0.0]},
            }
        )
        sim_miss_upright = _BodyStateSim(
            {
                "cube_1": {"position": [0, 0, 0.2]},
                "table_1": {"position": [0, 0, 0.0]},
                "bottle_1": {"quaternion": [0.707, 0.707, 0, 0]},
            }
        )
        assert fn(sim_hit) is True
        assert fn(sim_miss_upright) is False

    def test_or(self):
        text = "(define (problem p) (:goal (or (upright a) (upright b))))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        only_b = _BodyStateSim(
            {
                "a": {"quaternion": [0.707, 0.707, 0, 0]},  # tipped
                "b": {"quaternion": [1.0, 0, 0, 0]},  # upright
            }
        )
        assert fn(only_b) is True

    def test_not(self):
        text = "(define (problem p) (:goal (not (grasped cube_1))))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        # Without any contacts, grasped is False, so (not grasped) is True.
        no_contacts = _ContactSim([])
        assert fn(no_contacts) is True
        # With a gripper contact, grasped is True, so (not grasped) is False.
        with_grip = _ContactSim([{"geom1": "robot0_gripper_finger_r", "geom2": "cube_1_geom"}])
        assert fn(with_grip) is False

    def test_on_libero_tight_thresholds_real_success_state(self):
        """#170: The ``on`` predicate must accept LIBERO's actual at-success
        geometry — empirically, mug.z is only ~4 mm above plate.z and
        mug.xy is ~1 cm off plate.xy at the moment ``env.check_success()``
        returns True on ``libero-10/SCENE5``.

        Pre-#170 ``_on_kwargs`` left ``z_offset=0.02`` and ``xy_tol=0.15``
        as the ``_body_on`` defaults. Those tolerances are too LOOSE on
        xy (15 cm — over-permissive) but too TIGHT on z (2 cm — rejects
        a 4 mm gap). At the actual success state, ``z_offset=0.02``
        rejected the predicate as False even though ``env.check_success``
        was True — the silent counter bug PR #168 round 44 found.

        #170 changes ``_on_kwargs`` to pass ``z_offset=0.0,
        xy_tol=0.03`` matching upstream LIBERO's ``ObjectState.check_ontop``
        geometric semantics. This pins those thresholds via the actual
        post-success body positions captured from a real eval run."""
        text = "(define (problem p) (:goal (on porcelain_mug_1 plate_1)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]

        # Real positions captured at the env-success step on
        # libero-10/SCENE5 seed=42 ep 0 (via the #170 BDDL ↔
        # ``env.check_success`` diagnostic, since removed in #178).
        success_state = _BodyStateSim(
            {
                "porcelain_mug_1": {"position": [-0.003, -0.311, 0.443]},
                "plate_1": {"position": [-0.004, -0.323, 0.439]},
            }
        )
        # Post-#170: True (mug.z=0.443 > plate.z=0.439 by 4 mm; xy 1.2 cm).
        assert fn(success_state) is True, (
            "Post-#170 _on_kwargs must accept the real LIBERO success state "
            "(4 mm z, 1.2 cm xy). If this fails, the z_offset / xy_tol "
            "thresholds may have regressed back to the pre-#170 loose values."
        )

    def test_on_libero_rejects_above_plate_off_xy(self):
        """#170 sentinel: with the tighter ``xy_tol=0.03`` from
        ``_on_kwargs``, an ``on`` predicate must REJECT a state where
        the mug is positioned 5 cm to the side of the plate (no longer
        on it).

        Pre-#170 the loose 15-cm tolerance accepted this state as True
        (false positive); upstream LIBERO's 3-cm tolerance correctly
        rejects it. This test pins the tightening so a future
        wider-tolerance regression is caught."""
        text = "(define (problem p) (:goal (on cube_1 plate_1)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]

        # Cube 5 cm to the side of plate, but at correct z (above).
        off_to_side = _BodyStateSim(
            {
                "cube_1": {"position": [0.05, 0.0, 0.10]},  # 5 cm off-center
                "plate_1": {"position": [0.0, 0.0, 0.05]},
            }
        )
        assert fn(off_to_side) is False, (
            "5 cm off-center is outside the 3 cm xy tolerance — must reject. "
            "If this passes, _on_kwargs may have regressed to the pre-#170 "
            "loose 15 cm xy_tol."
        )

        # Cube 2 cm to the side — should pass (within 3 cm).
        slightly_off = _BodyStateSim(
            {
                "cube_1": {"position": [0.02, 0.0, 0.10]},
                "plate_1": {"position": [0.0, 0.0, 0.05]},
            }
        )
        assert fn(slightly_off) is True

    def test_on_libero_z_above_required(self):
        """#170: ``on(A, B)`` requires A.z >= B.z (mug above or at plate
        level). Pin the directional contract — a body BELOW the
        reference body is not "on" it regardless of xy alignment."""
        text = "(define (problem p) (:goal (on cube_1 plate_1)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]

        below = _BodyStateSim(
            {
                "cube_1": {"position": [0.0, 0.0, 0.04]},  # below plate
                "plate_1": {"position": [0.0, 0.0, 0.05]},
            }
        )
        assert fn(below) is False

        same_z = _BodyStateSim(
            {
                "cube_1": {"position": [0.0, 0.0, 0.05]},  # exactly at plate level
                "plate_1": {"position": [0.0, 0.0, 0.05]},
            }
        )
        # z_offset=0.0 means same-z is acceptable (matches upstream's <=).
        # Reading the predicate body: ``pos_a[2] > pos_b[2] + z_offset``
        # ⇒ 0.05 > 0.05 + 0 = 0.05 is False. So same-z is REJECTED by
        # ours, while upstream (using <=) would accept it.
        # This is a minor difference; in practice, when a body is
        # actually resting on another, mj_step's collision response
        # always pushes them slightly apart so positions are never
        # exactly equal. Pin the current strict-> behaviour so a future
        # change that "fixes" this asymmetry doesn't silently shift
        # near-success cases.
        assert fn(same_z) is False

    def test_on_libero_requires_contact_when_get_contacts_available(self):
        """#171 sub-task 3e: ``on(A, B)`` now requires physics contact
        between A and B when ``sim.get_contacts`` is available. Matches
        upstream LIBERO's ``ObjectState.check_ontop`` which combines
        geometric thresholds with ``check_contact``.

        Pre-#171 the geometric check alone fired True at "mug suspended
        5cm above plate, xy-aligned" transient states; this is now
        rejected because there's no contact.

        Sentinel for the false-positive elimination: a sim with the
        right geometry but NO contact must reject the predicate.
        """
        text = "(define (problem p) (:goal (on cube_1 plate_1)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]

        # Geometry passes (cube above plate, xy aligned within 3cm).
        # But contacts list is EMPTY — engine reports no contact.
        no_contact = _CombinedSim(
            bodies={
                "cube_1": {"position": [0.0, 0.0, 0.443]},
                "plate_1": {"position": [0.0, 0.0, 0.439]},
            },
            contacts=[],  # ← no contact = no "on"
        )
        assert fn(no_contact) is False, (
            "Without contact, on() should return False even when geometry passes. "
            "If True, the contact check (require_contact=True from _on_kwargs) "
            "may have regressed."
        )

        # Same geometry + a contact between cube_g0 and plate_g0 → True.
        with_contact = _CombinedSim(
            bodies={
                "cube_1": {"position": [0.0, 0.0, 0.443]},
                "plate_1": {"position": [0.0, 0.0, 0.439]},
            },
            contacts=[{"geom1": "cube_1_g0", "geom2": "plate_1_g0"}],
        )
        assert fn(with_contact) is True

    def test_on_libero_contact_check_matches_either_direction(self):
        """#171 sub-task 3e: contact records may list (cube_geom,
        plate_geom) or (plate_geom, cube_geom) — both must match the
        same predicate. Pin the order-insensitivity so a robosuite
        contact-record convention change doesn't silently break us.
        """
        text = "(define (problem p) (:goal (on cube_1 plate_1)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]

        bodies = {
            "cube_1": {"position": [0.0, 0.0, 0.443]},
            "plate_1": {"position": [0.0, 0.0, 0.439]},
        }
        # Direction 1: cube first.
        forward = _CombinedSim(
            bodies=bodies,
            contacts=[{"geom1": "cube_1_g3", "geom2": "plate_1_g7"}],
        )
        assert fn(forward) is True
        # Direction 2: plate first.
        reverse = _CombinedSim(
            bodies=bodies,
            contacts=[{"geom1": "plate_1_g7", "geom2": "cube_1_g3"}],
        )
        assert fn(reverse) is True

    def test_on_libero_contact_check_gracefully_degrades_without_get_contacts(self):
        """#171 sub-task 3e: when the sim engine doesn't expose
        ``get_contacts`` (e.g. a custom backend, test stub), the
        contact check is SKIPPED rather than failing. The geometric
        check alone determines the verdict — preserves pre-#171
        behaviour for engines without contact support.

        Without this graceful degradation, every non-mujoco backend
        would suddenly fail every ``on`` predicate."""
        text = "(define (problem p) (:goal (on cube_1 plate_1)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]

        # _BodyStateSim has NO get_contacts → graceful degradation.
        no_contacts_method = _BodyStateSim(
            {
                "cube_1": {"position": [0.0, 0.0, 0.443]},
                "plate_1": {"position": [0.0, 0.0, 0.439]},
            }
        )
        # Geometric check passes, contact check skipped → True.
        assert fn(no_contacts_method) is True


# Representative LIBERO-style round-trip


class TestRoundTrip:
    """One example per predicate family so a regression in any predicate is caught here."""

    def test_pick_task_on(self):
        text = """
            (define (problem libero_spatial_pick_up_the_red_cube)
              (:language "pick up the red cube and place it on the plate")
              (:objects cube_1 plate_1 table_1 - object)
              (:init (on cube_1 table_1))
              (:goal (on cube_1 plate_1)))
        """
        problem = parse_bddl(text)
        assert problem.language == "pick up the red cube and place it on the plate"
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        sim_success = _BodyStateSim({"cube_1": {"position": [0, 0, 0.25]}, "plate_1": {"position": [0, 0, 0.1]}})
        assert fn(sim_success) is True

    def test_open_task(self):
        text = "(define (problem libero_open_drawer) (:goal (open drawer_slide)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        sim = _BodyStateSim({"_joints": {"drawer_slide": 0.2}})
        assert fn(sim) is True
        sim2 = _BodyStateSim({"_joints": {"drawer_slide": 0.02}})
        assert fn(sim2) is False

    def test_grasp_task(self):
        text = "(define (problem libero_grasp_cube) (:goal (grasped cube_1)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        sim = _ContactSim([{"geom1": "robot0_gripper_finger_l", "geom2": "cube_1"}])
        assert fn(sim) is True

    def test_upright_task(self):
        text = "(define (problem libero_keep_upright) (:goal (upright bottle_1)))"
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        sim = _BodyStateSim({"bottle_1": {"quaternion": [1.0, 0, 0, 0]}})
        assert fn(sim) is True

    def test_inside_task(self):
        text = """
            (define (problem libero_put_inside)
              (:goal (inside cube_1 basket_1)))
        """
        problem = parse_bddl(text)
        fn = compile_goal(problem.goal)  # type: ignore[arg-type]
        # Approximate-inside uses default tolerances (0.15, 0.15).
        sim = _BodyStateSim({"cube_1": {"position": [0.05, 0.02, 0.1]}, "basket_1": {"position": [0, 0, 0.1]}})
        assert fn(sim) is True


# File loader


class TestParseBDDLFile:
    def test_happy_path(self, tmp_path):
        p = tmp_path / "task.bddl"
        p.write_text("(define (problem p) (:goal (grasped cube_1)))")
        problem = parse_bddl_file(p)
        assert problem.name == "p"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_bddl_file(tmp_path / "nope.bddl")

    def test_not_a_file(self, tmp_path):
        with pytest.raises(ValueError):
            parse_bddl_file(tmp_path)


# Case-insensitivity (PDDL grammar is case-insensitive)
#
# Real LIBERO BDDL files mix lowercase predicate vocabulary keys (`on`,
# `grasped`, …) with capital-initial spelling in the source files
# (`(And (On cube_1 plate_1))`). The parser needs to accept both forms;
# this section pins the contract.


class TestCaseInsensitivity:
    def test_capital_and_connective(self):
        """``(And ...)`` parses identically to ``(and ...)``."""
        text = """
            (define (problem p)
              (:goal (And (on cube_1 plate_1) (upright cube_1))))
        """
        problem = parse_bddl(text)
        assert isinstance(problem.goal, And)
        assert len(problem.goal.clauses) == 2

    def test_capital_or_and_not_connectives(self):
        text = """
            (define (problem p)
              (:goal (Or (grasped cube_1) (Not (on cube_1 table_1)))))
        """
        problem = parse_bddl(text)
        assert isinstance(problem.goal, Or)
        assert isinstance(problem.goal.clauses[1], Not)

    def test_capital_predicate_normalises_to_lowercase(self):
        """Leaf predicate names are normalised so ``compile_goal`` can look
        them up against the lowercase ``PREDICATE_VOCABULARY``."""
        text = """
            (define (problem p)
              (:goal (On cube_1 plate_1)))
        """
        problem = parse_bddl(text)
        assert isinstance(problem.goal, Pred)
        assert problem.goal.name == "on"  # normalised, not "On"
        assert problem.goal.args == ("cube_1", "plate_1")

    def test_real_libero_spatial_shape_round_trips(self):
        """End-to-end shape mirroring real LIBERO ``libero_spatial`` BDDLs:
        capital connectives, capital predicate, typed objects, init clauses.
        Every spatial / object / goal task in upstream LIBERO follows this
        layout, so this is the regression that gates ``load_libero_suite``
        actually registering tasks instead of skipping all of them."""
        text = """
            (define (problem libero_pick_bowl)
              (:domain kitchen)
              (:language "pick up the black bowl and place it on the plate")
              (:objects akita_black_bowl_1 plate_1 - object)
              (:init
                (On akita_black_bowl_1 main_table_region)
                (On plate_1 main_table_plate_region))
              (:goal
                (And (On akita_black_bowl_1 plate_1))))
        """
        problem = parse_bddl(text)
        assert problem.name == "libero_pick_bowl"
        assert problem.language == "pick up the black bowl and place it on the plate"
        assert isinstance(problem.goal, And)
        assert len(problem.goal.clauses) == 1
        inner = problem.goal.clauses[0]
        assert isinstance(inner, Pred)
        assert inner.name == "on"
        assert inner.args == ("akita_black_bowl_1", "plate_1")

    def test_unknown_predicate_error_preserves_source_casing(self):
        """When a predicate is genuinely unknown (not just mis-cased), the
        error message should show the original casing so users see exactly
        what they wrote in the BDDL file."""
        with pytest.raises(BDDLParseError, match="'NotAPredicate'"):
            parse_bddl("(define (problem p) (:goal (NotAPredicate cube_1)))")

    def test_compile_goal_works_after_capital_parse(self):
        """A capital-cased BDDL should compile to a working goal callable."""
        text = """
            (define (problem p)
              (:goal (Grasped cube_1)))
        """
        problem = parse_bddl(text)
        callable_ = compile_goal(problem.goal)
        assert callable(callable_)
