"""Tests for ``strands_robots.simulation.predicates``.

Each predicate is tested against a lightweight fake sim that implements
only the methods the predicate exercises. Real MuJoCo integration is out
of scope here - those predicates are covered end-to-end in the dispatch
tests under ``tests/simulation/mujoco/``.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    StatefulRewardTerm,
    _extract_json,
    _reset_resolution_warnings,
    _StagedReward,
    make_predicate,
    register_predicate,
)

# Fake sim helpers


class _BodyStateSim:
    """Sim that exposes ``get_body_state`` with caller-provided positions."""

    def __init__(self, positions: dict[str, list[float]]):
        self._pos = positions

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._pos:
            return {"status": "error", "content": [{"text": f"Body '{body_name}' not found."}]}
        return {
            "status": "success",
            "content": [
                {"text": f"body {body_name}"},
                {
                    "json": {
                        "position": self._pos[body_name],
                        "quaternion": [1, 0, 0, 0],
                        "mass": 1.0,
                    }
                },
            ],
        }

    # Predicates that probe `get_observation` for joint state need this stub.
    def get_observation(self, *_, **__) -> dict[str, Any]:
        return {}


class _JointObsSim:
    """Sim that exposes joint positions via ``get_observation``."""

    def __init__(self, joints: dict[str, float]):
        self._joints = joints

    def get_observation(self, *_, **__) -> dict[str, float]:
        return dict(self._joints)

    def get_body_state(self, body_name: str) -> dict[str, Any]:  # pragma: no cover
        return {"status": "error", "content": [{"text": "no bodies"}]}


class _ContactSim:
    """Sim that exposes ``get_contacts`` in the MuJoCo-backend shape."""

    def __init__(self, contacts: list[dict[str, Any]]):
        self._contacts = contacts

    def get_contacts(self) -> dict[str, Any]:
        return {
            "status": "success",
            "content": [
                {"text": f"{len(self._contacts)} contacts"},
                {
                    "json": {
                        "contacts": self._contacts,
                        "n_contacts": len(self._contacts),
                    }
                },
            ],
        }


class _NoHelpersSim:
    """Sim missing get_body_state / get_contacts entirely (e.g. future backend)."""

    def get_observation(self, *_, **__) -> dict[str, Any]:
        return {}


# Registry


class TestRegistry:
    def test_builtin_predicates_registered(self):
        required = {
            "body_above_z",
            "body_below_z",
            "joint_above",
            "joint_below",
            "distance_less_than",
            "inside_region",
            "contact_between",
            "contact_any",
            "body_on",
            "body_inside",
            "body_upright",
            "grasped",
            "distance_neg",
            "joint_progress",
            "constant",
        }
        assert required.issubset(PREDICATE_REGISTRY.keys())

    def test_make_predicate_unknown_raises(self):
        with pytest.raises(ValueError) as exc:
            make_predicate("totally_made_up")
        assert "Unknown predicate" in str(exc.value)
        # Error message should list valid names so the user can fix the spec.
        assert "body_above_z" in str(exc.value)

    def test_register_predicate_rejects_shadow(self):
        with pytest.raises(ValueError):
            register_predicate("body_above_z", lambda **_: lambda _sim: True)

    def test_register_predicate_rejects_non_callable(self):
        with pytest.raises(TypeError):
            register_predicate("my_pred", "not a callable")  # type: ignore[arg-type]

    def test_register_predicate_custom(self):
        try:

            def factory(value: float):
                return lambda _sim: value > 0

            register_predicate("positive_constant", factory)
            pred = make_predicate("positive_constant", value=1.5)
            assert pred(None) is True
        finally:
            PREDICATE_REGISTRY.pop("positive_constant", None)


# Body-position predicates


class TestBodyPositionPredicates:
    def test_body_above_z_true(self):
        sim = _BodyStateSim({"cube": [0.1, 0.0, 0.25]})
        pred = make_predicate("body_above_z", body="cube", z=0.2)
        assert pred(sim) is True

    def test_body_above_z_false(self):
        sim = _BodyStateSim({"cube": [0.1, 0.0, 0.15]})
        pred = make_predicate("body_above_z", body="cube", z=0.2)
        assert pred(sim) is False

    def test_body_above_z_missing_body_returns_false(self):
        sim = _BodyStateSim({"other": [0, 0, 1]})
        pred = make_predicate("body_above_z", body="cube", z=0.2)
        assert pred(sim) is False

    def test_body_below_z(self):
        sim = _BodyStateSim({"cube": [0.0, 0.0, -0.05]})
        pred = make_predicate("body_below_z", body="cube", z=0.0)
        assert pred(sim) is True

    def test_distance_less_than_true(self):
        sim = _BodyStateSim({"a": [0, 0, 0], "b": [0.05, 0, 0]})
        pred = make_predicate("distance_less_than", body_a="a", body_b="b", threshold=0.1)
        assert pred(sim) is True

    def test_distance_less_than_false(self):
        sim = _BodyStateSim({"a": [0, 0, 0], "b": [1.0, 0, 0]})
        pred = make_predicate("distance_less_than", body_a="a", body_b="b", threshold=0.1)
        assert pred(sim) is False

    def test_inside_region_matches(self):
        sim = _BodyStateSim({"cube": [0.1, 0.2, 0.3]})
        pred = make_predicate("inside_region", body="cube", min=[-0.5, 0.0, 0.0], max=[0.5, 0.5, 1.0])
        assert pred(sim) is True

    def test_inside_region_outside(self):
        sim = _BodyStateSim({"cube": [0.6, 0.0, 0.0]})
        pred = make_predicate("inside_region", body="cube", min=[0, 0, 0], max=[0.5, 0.5, 0.5])
        assert pred(sim) is False

    def test_inside_region_rejects_malformed_args(self):
        with pytest.raises(ValueError):
            make_predicate("inside_region", body="cube", min=[0, 0], max=[1, 1, 1])
        with pytest.raises(ValueError):
            # min > max should error up front, not silently always return False.
            make_predicate("inside_region", body="cube", min=[1, 1, 1], max=[0, 0, 0])

    def test_body_predicate_without_get_body_state_returns_false(self):
        sim = _NoHelpersSim()
        pred = make_predicate("body_above_z", body="cube", z=0)
        assert pred(sim) is False

    def test_body_position_libero_main_suffix_fallback(self):
        """Round 46 (#176 sub-task 3d) - LIBERO objects' BDDL names
        (``porcelain_mug_1``) map to MJCF root bodies suffixed with
        ``_main`` (``porcelain_mug_1_main``). The predicate evaluator
        must transparently retry with the suffix when the bare name
        misses, mirroring upstream LIBERO's
        ``env.objects_dict[name].root_body`` resolution. Without this,
        BDDL goal predicates like ``(On porcelain_mug_1 plate_1)``
        silently resolve to ``False`` even when physics has the mug
        on the plate.

        Pin: a sim that only exposes ``porcelain_mug_1_main`` (NOT
        ``porcelain_mug_1``) must still resolve via the predicate as
        if the bare name worked.
        """
        # Sim only knows the suffixed name (mimics MJCF body naming).
        sim = _BodyStateSim({"porcelain_mug_1_main": [0.0, 0.0, 0.5]})
        pred = make_predicate("body_above_z", body="porcelain_mug_1", z=0.4)
        assert pred(sim) is True, (
            "body_above_z with bare BDDL name should fall back to ``<name>_main`` "
            "for LIBERO scenes; round-46 fix may have regressed."
        )

    def test_body_position_main_suffix_no_double_suffix(self):
        """Already-suffixed names must not double-suffix on retry.
        Round 46 (#176 sub-task 3d).
        """
        # Sim only knows the suffixed name; caller passes already-suffixed.
        sim = _BodyStateSim({"plate_1_main": [0.0, 0.0, 0.4]})
        pred = make_predicate("body_above_z", body="plate_1_main", z=0.3)
        assert pred(sim) is True

    def test_body_position_bare_name_wins_over_suffix(self):
        """When BOTH ``<name>`` and ``<name>_main`` exist, prefer the
        bare lookup. This preserves the contract for fixtures /
        explicit-named bodies (e.g. ``living_room_table``) which don't
        use the LIBERO suffix.

        Round 46 (#176 sub-task 3d).
        """
        sim = _BodyStateSim(
            {
                "living_room_table": [0.0, 0.0, 0.46],
                "living_room_table_main": [99.0, 99.0, 99.0],  # decoy
            }
        )
        pred = make_predicate("body_above_z", body="living_room_table", z=0.4)
        assert pred(sim) is True
        # Decoy at 99.0 should not be reached if bare lookup wins.
        pred2 = make_predicate("body_above_z", body="living_room_table", z=98.0)
        assert pred2(sim) is False, "bare name should win over _main suffix; double-resolve detected"


# Joint predicates


class TestJointPredicates:
    def test_joint_above(self):
        sim = _JointObsSim({"drawer_slide": 0.18})
        assert make_predicate("joint_above", joint="drawer_slide", value=0.15)(sim) is True
        assert make_predicate("joint_above", joint="drawer_slide", value=0.2)(sim) is False

    def test_joint_below(self):
        sim = _JointObsSim({"gripper": 0.02})
        assert make_predicate("joint_below", joint="gripper", value=0.05)(sim) is True

    def test_joint_missing_returns_false(self):
        sim = _JointObsSim({"other_joint": 1.0})
        assert make_predicate("joint_above", joint="missing", value=0.0)(sim) is False

    def test_joint_progress_reward(self):
        sim = _JointObsSim({"drawer": 0.1})
        term = make_predicate("joint_progress", joint="drawer", target=0.2, weight=10.0)
        # -weight * |q - target| = -10 * 0.1 = -1.0
        assert term(sim) == pytest.approx(-1.0)

    def test_joint_progress_at_target_gives_zero_reward(self):
        sim = _JointObsSim({"drawer": 0.2})
        term = make_predicate("joint_progress", joint="drawer", target=0.2, weight=1.0)
        assert term(sim) == pytest.approx(0.0)


# Contact predicates


class TestContactPredicates:
    def test_contact_between_matches_either_order(self):
        sim = _ContactSim([{"geom1": "cube", "geom2": "gripper", "dist": -0.001}])
        assert make_predicate("contact_between", geom_a="cube", geom_b="gripper")(sim) is True
        assert make_predicate("contact_between", geom_a="gripper", geom_b="cube")(sim) is True

    def test_contact_between_no_match(self):
        sim = _ContactSim([{"geom1": "cube", "geom2": "ground"}])
        assert make_predicate("contact_between", geom_a="cube", geom_b="gripper")(sim) is False

    def test_contact_any(self):
        assert make_predicate("contact_any")(_ContactSim([{"geom1": "a", "geom2": "b"}])) is True
        assert make_predicate("contact_any")(_ContactSim([])) is False

    def test_contact_predicate_without_get_contacts(self):
        sim = _NoHelpersSim()
        assert make_predicate("contact_any")(sim) is False
        assert make_predicate("contact_between", geom_a="a", geom_b="b")(sim) is False


# Reward terms


class TestRewardTerms:
    def test_distance_neg_monotonic(self):
        far = _BodyStateSim({"a": [0, 0, 0], "b": [1, 0, 0]})
        near = _BodyStateSim({"a": [0, 0, 0], "b": [0.1, 0, 0]})
        term = make_predicate("distance_neg", body_a="a", body_b="b", weight=1.0)
        # Closer is greater (less negative).
        assert term(near) > term(far)

    def test_distance_neg_weight(self):
        sim = _BodyStateSim({"a": [0, 0, 0], "b": [1, 0, 0]})
        weighted = make_predicate("distance_neg", body_a="a", body_b="b", weight=5.0)
        assert weighted(sim) == pytest.approx(-5.0)

    def test_distance_neg_missing_body_returns_zero(self):
        """Missing bodies should not crash or reward heavily - return 0.0."""
        sim = _BodyStateSim({"a": [0, 0, 0]})
        term = make_predicate("distance_neg", body_a="a", body_b="ghost", weight=1.0)
        assert term(sim) == 0.0

    def test_constant(self):
        term = make_predicate("constant", value=-0.01)
        assert term(None) == pytest.approx(-0.01)


# LIBERO / #110 predicates


class _BodyStateWithQuatSim:
    """Extends _BodyStateSim with quaternion in the body-state payload."""

    def __init__(self, bodies: dict[str, dict[str, Any]]):
        self._bodies = bodies

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._bodies:
            return {"status": "error", "content": [{"text": "missing"}]}
        payload = {
            "position": self._bodies[body_name].get("position", [0, 0, 0]),
            "quaternion": self._bodies[body_name].get("quaternion", [1, 0, 0, 0]),
            "mass": 1.0,
        }
        return {"status": "success", "content": [{"text": body_name}, {"json": payload}]}

    def get_observation(self, *_, **__) -> dict[str, Any]:
        return {}


class TestBodyOn:
    def test_true_when_above_and_aligned(self):
        sim = _BodyStateSim({"cube": [0.0, 0.0, 0.22], "table": [0.0, 0.0, 0.05]})
        pred = make_predicate("body_on", body_a="cube", body_b="table", z_offset=0.1)
        assert pred(sim) is True

    def test_false_when_not_above(self):
        sim = _BodyStateSim({"cube": [0.0, 0.0, 0.04], "table": [0.0, 0.0, 0.05]})
        pred = make_predicate("body_on", body_a="cube", body_b="table", z_offset=0.01)
        assert pred(sim) is False

    def test_false_when_too_far_horizontally(self):
        sim = _BodyStateSim({"cube": [1.0, 0.0, 0.2], "table": [0.0, 0.0, 0.05]})
        pred = make_predicate("body_on", body_a="cube", body_b="table", xy_tol=0.1)
        assert pred(sim) is False

    def test_missing_body_returns_false(self):
        sim = _BodyStateSim({"table": [0, 0, 0.05]})
        pred = make_predicate("body_on", body_a="cube", body_b="table")
        assert pred(sim) is False


class TestBodyInside:
    def test_true_inside_box(self):
        sim = _BodyStateSim({"cube": [0.02, 0.01, 0.03], "basket": [0, 0, 0]})
        pred = make_predicate("body_inside", body="cube", container="basket", xy_tol=0.1, z_tol=0.1)
        assert pred(sim) is True

    def test_false_outside_xy(self):
        sim = _BodyStateSim({"cube": [0.5, 0.0, 0.0], "basket": [0, 0, 0]})
        pred = make_predicate("body_inside", body="cube", container="basket", xy_tol=0.1, z_tol=0.1)
        assert pred(sim) is False

    def test_false_outside_z(self):
        sim = _BodyStateSim({"cube": [0.0, 0.0, 0.5], "basket": [0, 0, 0]})
        pred = make_predicate("body_inside", body="cube", container="basket", xy_tol=0.2, z_tol=0.1)
        assert pred(sim) is False


class TestBodyUpright:
    def test_identity_quat_is_upright(self):
        sim = _BodyStateWithQuatSim({"bottle": {"quaternion": [1.0, 0.0, 0.0, 0.0]}})
        pred = make_predicate("body_upright", body="bottle")
        assert pred(sim) is True

    def test_tipped_on_side_is_not_upright(self):
        # 90-deg rotation about x-axis: quat = (cos(pi/4), sin(pi/4), 0, 0) ≈ (0.707, 0.707, 0, 0)
        sim = _BodyStateWithQuatSim({"bottle": {"quaternion": [0.7071, 0.7071, 0.0, 0.0]}})
        pred = make_predicate("body_upright", body="bottle", tol=0.15)
        assert pred(sim) is False

    def test_small_tilt_within_tolerance(self):
        # Small rotation about x-axis - x component ~= 0.1, so 2*(x²+y²) ~= 0.02 < default tol 0.15.
        sim = _BodyStateWithQuatSim({"bottle": {"quaternion": [0.995, 0.1, 0.0, 0.0]}})
        pred = make_predicate("body_upright", body="bottle", tol=0.15)
        assert pred(sim) is True

    def test_missing_body_returns_false(self):
        sim = _BodyStateWithQuatSim({})
        pred = make_predicate("body_upright", body="bottle")
        assert pred(sim) is False

    def test_negative_tol_rejected(self):
        with pytest.raises(ValueError):
            make_predicate("body_upright", body="bottle", tol=-0.1)


class TestGrasped:
    def test_detects_gripper_contact_by_prefix(self):
        sim = _ContactSim(
            [
                {"geom1": "robot0_gripper_finger_r", "geom2": "cube_geom"},
            ]
        )
        pred = make_predicate("grasped", body="cube", gripper_prefix="robot0_gripper")
        assert pred(sim) is True

    def test_contact_without_gripper_prefix_is_not_grasp(self):
        sim = _ContactSim([{"geom1": "table", "geom2": "cube_geom"}])
        pred = make_predicate("grasped", body="cube", gripper_prefix="robot0_gripper")
        assert pred(sim) is False

    def test_matches_either_ordering(self):
        sim = _ContactSim(
            [
                {"geom1": "cube_geom", "geom2": "robot0_gripper_finger_l"},
            ]
        )
        pred = make_predicate("grasped", body="cube", gripper_prefix="robot0_gripper")
        assert pred(sim) is True

    def test_no_contacts_returns_false(self):
        sim = _ContactSim([])
        pred = make_predicate("grasped", body="cube", gripper_prefix="robot0_gripper")
        assert pred(sim) is False

    def test_without_get_contacts_returns_false(self):
        sim = _NoHelpersSim()
        pred = make_predicate("grasped", body="cube", gripper_prefix="robot0_gripper")
        assert pred(sim) is False

    def test_detects_libero_multi_geom_object(self):
        # LIBERO / robosuite name a BDDL object ``cube_1``'s collision
        # geoms ``cube_1_g0`` / ``cube_1_g1`` ... (the same ``<body>_g<idx>``
        # convention _body_contact matches). ``(grasped cube_1)`` must fire
        # when a gripper geom touches such a geom - previously it silently
        # returned False because only exact ``cube_1`` / ``cube_1_geom``
        # names were matched.
        sim = _ContactSim(
            [
                {"geom1": "robot0_gripper_finger_l", "geom2": "cube_1_g0"},
            ]
        )
        pred = make_predicate("grasped", body="cube_1", gripper_prefix="robot0_gripper")
        assert pred(sim) is True

    def test_detects_libero_multi_geom_object_higher_index(self):
        sim = _ContactSim(
            [
                {"geom1": "cube_1_g3", "geom2": "robot0_gripper_finger_r"},
            ]
        )
        pred = make_predicate("grasped", body="cube_1", gripper_prefix="robot0_gripper")
        assert pred(sim) is True

    def test_detects_bare_body_geom_name(self):
        # A single-geom scene whose geom is named exactly after the body.
        sim = _ContactSim([{"geom1": "cube", "geom2": "robot0_gripper_pad"}])
        pred = make_predicate("grasped", body="cube", gripper_prefix="robot0_gripper")
        assert pred(sim) is True

    def test_prefix_does_not_bleed_across_distinct_bodies(self):
        # ``cube_1`` must not match ``cube_10``'s geoms - the ``_g`` boundary
        # keeps distinct names apart.
        sim = _ContactSim(
            [
                {"geom1": "robot0_gripper_finger_l", "geom2": "cube_10_g0"},
            ]
        )
        pred = make_predicate("grasped", body="cube_1", gripper_prefix="robot0_gripper")
        assert pred(sim) is False


# Degradation / defensive contract
#
# The module docstring promises predicates "should never crash the eval loop"
# and that they "degrade gracefully" when a backend method raises, returns an
# error stub, or returns a malformed payload. These fakes exercise exactly
# those failure modes - the happy paths above already cover well-formed
# backends.


class _RaisingBodyStateSim:
    """Sim whose ``get_body_state`` always raises (e.g. backend died mid-eval)."""

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        raise RuntimeError(f"backend exploded reading {body_name!r}")

    def get_observation(self, *_, **__) -> dict[str, Any]:
        raise RuntimeError("backend exploded reading observation")


class _RaisingContactSim:
    """Sim whose ``get_contacts`` always raises."""

    def get_contacts(self) -> dict[str, Any]:
        raise RuntimeError("contact solver crashed")


class _MalformedBodyStateSim:
    """Sim that returns success but with a malformed / wrong-shape payload."""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        return {"status": "success", "content": [{"json": self._payload}]}

    def get_observation(self, *_, **__) -> dict[str, Any]:
        return {}


class _MalformedContactSim:
    """Sim whose ``get_contacts`` returns a non-list ``contacts`` payload."""

    def __init__(self, payload: dict[str, Any], status: str = "success"):
        self._payload = payload
        self._status = status

    def get_contacts(self) -> dict[str, Any]:
        return {"status": self._status, "content": [{"json": self._payload}]}


class TestDegradationContract:
    """A backend that raises / errors / returns garbage must yield a safe
    verdict (``False`` for bool predicates, ``0.0`` for reward terms) rather
    than propagating the exception up through the eval loop."""

    def test_body_predicate_swallows_get_body_state_exception(self):
        sim = _RaisingBodyStateSim()
        pred = make_predicate("body_above_z", body="cube", z=0.1)
        assert pred(sim) is False

    def test_distance_predicate_swallows_get_body_state_exception(self):
        sim = _RaisingBodyStateSim()
        pred = make_predicate("distance_less_than", body_a="a", body_b="b", threshold=1.0)
        assert pred(sim) is False

    def test_body_upright_swallows_get_body_state_exception(self):
        sim = _RaisingBodyStateSim()
        pred = make_predicate("body_upright", body="cube")
        assert pred(sim) is False

    def test_joint_predicate_swallows_get_observation_exception(self):
        sim = _RaisingBodyStateSim()  # get_observation raises here too
        pred = make_predicate("joint_above", joint="elbow", value=0.0)
        assert pred(sim) is False

    def test_joint_progress_reward_zero_when_observation_raises(self):
        sim = _RaisingBodyStateSim()
        term = make_predicate("joint_progress", joint="elbow", target=1.0)
        assert term(sim) == 0.0

    def test_contact_between_swallows_get_contacts_exception(self):
        sim = _RaisingContactSim()
        pred = make_predicate("contact_between", geom_a="g1", geom_b="g2")
        assert pred(sim) is False

    def test_contact_any_swallows_get_contacts_exception(self):
        sim = _RaisingContactSim()
        pred = make_predicate("contact_any")
        assert pred(sim) is False

    def test_grasped_swallows_get_contacts_exception(self):
        sim = _RaisingContactSim()
        pred = make_predicate("grasped", body="cube", gripper_prefix="grip")
        assert pred(sim) is False

    def test_body_position_rejects_wrong_length_payload(self):
        sim = _MalformedBodyStateSim({"position": [0.0, 1.0]})  # only 2 coords
        pred = make_predicate("body_above_z", body="cube", z=-1.0)
        assert pred(sim) is False

    def test_body_position_rejects_non_numeric_payload(self):
        sim = _MalformedBodyStateSim({"position": ["x", "y", "z"]})
        pred = make_predicate("body_above_z", body="cube", z=-1.0)
        assert pred(sim) is False

    def test_body_upright_rejects_wrong_length_quaternion(self):
        sim = _MalformedBodyStateSim({"quaternion": [1.0, 0.0, 0.0]})  # only 3
        pred = make_predicate("body_upright", body="cube")
        assert pred(sim) is False

    def test_joint_position_rejects_bool_value(self):
        # bool is a subclass of int; the lookup must NOT treat True as 1.0.
        sim = _JointObsSim({})
        sim._joints = {"gripper": True}  # type: ignore[dict-item]
        pred = make_predicate("joint_above", joint="gripper", value=0.5)
        assert pred(sim) is False

    def test_contact_between_handles_non_list_contacts(self):
        sim = _MalformedContactSim({"contacts": "not-a-list", "n_contacts": 0})
        pred = make_predicate("contact_between", geom_a="g1", geom_b="g2")
        assert pred(sim) is False

    def test_grasped_handles_non_list_contacts(self):
        sim = _MalformedContactSim({"contacts": 42})
        pred = make_predicate("grasped", body="cube", gripper_prefix="grip")
        assert pred(sim) is False


class _GeomContactSim:
    """Sim exposing both body positions and geom-prefix contacts, for the
    ``body_on(require_contact=True)`` LIBERO-style combined check."""

    def __init__(self, positions: dict[str, list[float]], contacts: list[dict[str, Any]], status: str = "success"):
        self._pos = positions
        self._contacts = contacts
        self._status = status

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        if body_name not in self._pos:
            return {"status": "error", "content": [{"text": "missing"}]}
        return {"status": "success", "content": [{"json": {"position": self._pos[body_name]}}]}

    def get_contacts(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "content": [{"json": {"contacts": self._contacts, "n_contacts": len(self._contacts)}}],
        }

    def get_observation(self, *_, **__) -> dict[str, Any]:
        return {}


class TestBodyOnRequireContact:
    """``body_on(require_contact=True)`` combines the geometric check with a
    physics-contact check, degrading to geometric-only when the engine cannot
    report contacts (pre-#171 behaviour)."""

    _STACKED = {"cube": [0.0, 0.0, 0.10], "table": [0.0, 0.0, 0.0]}

    def test_true_when_geometric_and_contact_agree(self):
        sim = _GeomContactSim(
            self._STACKED,
            [{"geom1": "cube_g0", "geom2": "table_g1"}],
        )
        pred = make_predicate("body_on", body_a="cube", body_b="table", require_contact=True)
        assert pred(sim) is True

    def test_false_when_geometry_passes_but_no_contact(self):
        sim = _GeomContactSim(self._STACKED, [])
        pred = make_predicate("body_on", body_a="cube", body_b="table", require_contact=True)
        assert pred(sim) is False

    def test_geometric_only_when_engine_lacks_get_contacts(self):
        # _BodyStateSim has no get_contacts -> _body_contact returns None ->
        # body_on falls back to the geometric verdict (True here).
        sim = _BodyStateSim(self._STACKED)
        pred = make_predicate("body_on", body_a="cube", body_b="table", require_contact=True)
        assert pred(sim) is True

    def test_geometric_only_when_contacts_error_stub(self):
        # Engine returns an error stub -> _body_contact returns None -> fall
        # back to geometric-only verdict rather than a false negative.
        sim = _GeomContactSim(self._STACKED, [], status="error")
        pred = make_predicate("body_on", body_a="cube", body_b="table", require_contact=True)
        assert pred(sim) is True


class _RaisingContactGeomSim(_GeomContactSim):
    """body_on(require_contact) target whose get_contacts raises -> degrade."""

    def get_contacts(self) -> dict[str, Any]:
        raise RuntimeError("contact solver crashed")


class TestExtractJsonAndEntryGuards:
    """Lower-level guards: malformed top-level results and non-dict contact
    entries must be skipped, not crash."""

    def test_body_on_require_contact_degrades_when_get_contacts_raises(self):
        sim = _RaisingContactGeomSim({"cube": [0.0, 0.0, 0.10], "table": [0.0, 0.0, 0.0]}, [])
        pred = make_predicate("body_on", body_a="cube", body_b="table", require_contact=True)
        # get_contacts raises -> _body_contact returns None -> geometric-only True
        assert pred(sim) is True

    def test_contact_between_skips_non_dict_entries(self):
        sim = _ContactSim(["not-a-dict", {"geom1": "g1", "geom2": "g2"}])  # type: ignore[list-item]
        pred = make_predicate("contact_between", geom_a="g1", geom_b="g2")
        assert pred(sim) is True

    def test_grasped_skips_non_dict_entries(self):
        sim = _ContactSim(["junk", {"geom1": "cube_geom", "geom2": "grip_l"}])  # type: ignore[list-item]
        pred = make_predicate("grasped", body="cube", gripper_prefix="grip")
        assert pred(sim) is True

    def test_inside_region_missing_body_returns_false(self):
        sim = _BodyStateSim({})  # cube absent
        pred = make_predicate("inside_region", body="cube", min=[0, 0, 0], max=[1, 1, 1])
        assert pred(sim) is False


# Remaining degradation guards
#
# The module docstring promises predicates "should never crash the eval loop"
# and "degrade gracefully" against malformed backend payloads. The cases below
# pin the last few guards that the happy-path and exception-path suites above do
# not exercise: a success result that carries no ``json`` content block, a
# ``get_observation`` that returns a non-dict, a quaternion lookup on a backend
# without ``get_body_state``, contact payloads whose ``contacts`` is not a list
# or contains non-dict entries, and a containment check against an absent body.


class _NoJsonBlockSim:
    """``get_body_state`` succeeds but its content carries no ``json`` block.

    A backend could legitimately return only a human-readable ``text`` block;
    the position lookup must then resolve to ``None`` (predicate ``False``)
    rather than crash hunting for a missing payload.
    """

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": f"body {body_name}"}]}

    def get_observation(self, *_, **__) -> dict[str, Any]:
        return {}


class _NonDictObsSim:
    """``get_observation`` returns a non-dict (e.g. a bare list)."""

    def get_observation(self, *_, **__) -> Any:
        return ["not", "a", "dict"]

    def get_body_state(self, body_name: str) -> dict[str, Any]:  # pragma: no cover
        return {"status": "error", "content": [{"text": "no bodies"}]}


class TestPayloadShapeGuards:
    """Malformed-but-non-raising backend payloads still yield safe verdicts."""

    def test_body_predicate_false_when_success_result_has_no_json_block(self):
        # _body_position: status == success but _extract_json finds no json
        # block -> {} -> position absent -> predicate False (not a crash).
        sim = _NoJsonBlockSim()
        pred = make_predicate("body_above_z", body="cube", z=0.1)
        assert pred(sim) is False

    def test_joint_predicate_false_when_observation_is_not_a_dict(self):
        # _joint_position: get_observation returns a list -> guarded to None ->
        # joint predicate False.
        sim = _NonDictObsSim()
        pred = make_predicate("joint_above", joint="elbow", value=0.0)
        assert pred(sim) is False

    def test_body_upright_false_when_backend_lacks_get_body_state(self):
        # _body_quaternion: backend exposes no get_body_state -> None -> the
        # uprightness check degrades to False instead of raising AttributeError.
        sim = _NoHelpersSim()
        pred = make_predicate("body_upright", body="bottle")
        assert pred(sim) is False

    def test_body_on_require_contact_geometric_only_when_contacts_not_a_list(self):
        # _body_contact: contacts payload is a dict, not a list -> None ->
        # body_on falls back to the geometric-only verdict (True here).
        stacked = {"cube": [0.0, 0.0, 0.10], "table": [0.0, 0.0, 0.0]}
        sim = _GeomContactSim(stacked, {"unexpected": "shape"})  # type: ignore[arg-type]
        pred = make_predicate("body_on", body_a="cube", body_b="table", require_contact=True)
        assert pred(sim) is True

    def test_body_on_require_contact_skips_non_dict_contact_entries(self):
        # _body_contact: a non-dict entry in the contacts list is skipped, and a
        # later well-formed matching entry still confirms the contact.
        stacked = {"cube": [0.0, 0.0, 0.10], "table": [0.0, 0.0, 0.0]}
        sim = _GeomContactSim(
            stacked,
            ["junk", {"geom1": "cube_g0", "geom2": "table_g1"}],  # type: ignore[list-item]
        )
        pred = make_predicate("body_on", body_a="cube", body_b="table", require_contact=True)
        assert pred(sim) is True

    def test_body_inside_false_when_body_absent(self):
        # _body_inside: target body missing -> position None -> False.
        sim = _BodyStateSim({"basket": [0.0, 0.0, 0.0]})  # 'cube' absent
        pred = make_predicate("body_inside", body="cube", container="basket")
        assert pred(sim) is False


class TestExtractJsonHelperGuard:
    """``_extract_json`` is the shared content-block reader behind every
    body/contact lookup; its non-dict guard is the innermost line of the
    "never crash" contract and is unreachable through the public predicates
    (which only ever hand it a dict), so it is pinned directly here."""

    def test_extract_json_returns_empty_for_non_dict_result(self):
        assert _extract_json(None) == {}
        assert _extract_json("not a dict") == {}  # type: ignore[arg-type]


# Name-resolution surfacing (#176 follow-up): a spec that references a body /
# joint the backend supports looking up but cannot resolve (a typo) must not be
# silent. It still degrades to a constant, but the name is logged once, and the
# LIBERO ``<name>_main`` root-body fallback now also covers the quaternion path.

_PRED_LOGGER = "strands_robots.simulation.predicates"


class TestBodyUprightLiberoMainFallback:
    """body_upright must resolve the LIBERO ``<name>_main`` root body.

    BDDL ``(upright X)`` compiles to ``body_upright(body=X)`` with the bare
    object name, but the MJCF root body of a procedurally-generated LIBERO
    object is ``X_main``. Before the fix, ``_body_quaternion`` tried only the
    bare name (unlike ``_body_position``, which has the fallback), so an upright
    mug silently scored as not-upright and the goal was never satisfiable.
    """

    def test_upright_resolves_via_main_suffix(self):
        # Only the _main root body exists (the bare BDDL name does not).
        sim = _BodyStateWithQuatSim({"mug_main": {"quaternion": [1, 0, 0, 0]}})
        pred = make_predicate("body_upright", body="mug")
        assert pred(sim) is True  # FAILS pre-fix: no _main fallback -> False

    def test_upright_tilted_via_main_suffix_is_false(self):
        # Resolves via _main but genuinely tipped over -> correctly False.
        sim = _BodyStateWithQuatSim({"mug_main": {"quaternion": [0.707, 0.707, 0.0, 0.0]}})
        pred = make_predicate("body_upright", body="mug", tol=0.1)
        assert pred(sim) is False


class TestUnresolvedNameSurfacing:
    def setup_method(self):
        _reset_resolution_warnings()

    def test_reward_term_warns_on_unresolved_body(self, caplog):
        sim = _BodyStateSim({"a": [0, 0, 0]})
        term = make_predicate("distance_neg", body_a="a", body_b="ghost_reach_1", weight=1.0)
        with caplog.at_level(logging.WARNING, logger=_PRED_LOGGER):
            value = term(sim)
        assert value == 0.0  # value contract unchanged
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1  # FAILS pre-fix: silent, zero warnings
        assert "ghost_reach_1" in warnings[0].getMessage()

    def test_bool_predicate_warns_on_unresolved_body(self, caplog):
        sim = _BodyStateSim({"a": [0, 0, 0]})
        pred = make_predicate("distance_less_than", body_a="a", body_b="ghost_reach_2", threshold=1.0)
        with caplog.at_level(logging.WARNING, logger=_PRED_LOGGER):
            assert pred(sim) is False
        assert any("ghost_reach_2" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)

    def test_joint_term_warns_on_unresolved_joint(self, caplog):
        sim = _JointObsSim({"drawer": 0.1})
        term = make_predicate("joint_progress", joint="ghost_joint_3", target=0.2, weight=1.0)
        with caplog.at_level(logging.WARNING, logger=_PRED_LOGGER):
            value = term(sim)
        assert value == 0.0
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1  # FAILS pre-fix
        assert "ghost_joint_3" in warnings[0].getMessage()

    def test_upright_warns_on_unresolved_body(self, caplog):
        sim = _BodyStateWithQuatSim({})
        pred = make_predicate("body_upright", body="ghost_up_4")
        with caplog.at_level(logging.WARNING, logger=_PRED_LOGGER):
            assert pred(sim) is False
        assert any("ghost_up_4" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)

    def test_no_warning_when_backend_lacks_introspection(self, caplog):
        # Unsupported backend (no get_body_state) is a capability gap, not a
        # spec typo -> must stay silent (pre-fix behaviour preserved).
        sim = _NoHelpersSim()
        term = make_predicate("distance_neg", body_a="a", body_b="b", weight=1.0)
        with caplog.at_level(logging.WARNING, logger=_PRED_LOGGER):
            assert term(sim) == 0.0
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_no_warning_when_body_resolves(self, caplog):
        sim = _BodyStateSim({"a": [0, 0, 0]})
        term = make_predicate("distance_neg", body_a="a", body_b="a", weight=1.0)
        with caplog.at_level(logging.WARNING, logger=_PRED_LOGGER):
            assert term(sim) == 0.0
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_warns_only_once_per_name(self, caplog):
        sim = _BodyStateSim({"a": [0, 0, 0]})
        term = make_predicate("distance_neg", body_a="a", body_b="ghost_dedup_5", weight=1.0)
        with caplog.at_level(logging.WARNING, logger=_PRED_LOGGER):
            term(sim)
            term(sim)
            term(sim)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "ghost_dedup_5" in r.getMessage()]
        assert len(warnings) == 1  # deduped across the hot loop


class TestStagedReward:
    """Behavioural tests for the ``staged_reward`` phase-machine primitive.

    Exercises the declarative phase machine end to end via the closed-registry
    ``make_predicate`` path (no ``eval``): stage-local shaping reward, one-time
    transition bonus, monotonic (forward-only) advancement, ``reset()``, and
    every factory validation error.
    """

    @staticmethod
    def _spec():
        # Stage 0: dense reward 1.0 while below; advance once cube clears z=0.5,
        # awarding a one-time bonus of 10.0. Stage 1 (final) pays a flat 2.0.
        return [
            {
                "reward": {"predicate": "constant", "value": 1.0},
                "advance_when": {"predicate": "body_above_z", "body": "cube", "z": 0.5},
                "bonus": 10.0,
            },
            {"reward": {"predicate": "constant", "value": 2.0}},
        ]

    def test_make_predicate_returns_stateful_term(self):
        term = make_predicate("staged_reward", stages=self._spec())
        # Consumers duck-type on ``reset``; the concrete type must honour the
        # StatefulRewardTerm contract so SimEnv/benchmark resets it per episode.
        assert isinstance(term, StatefulRewardTerm)
        assert hasattr(term, "reset")
        assert term.phase == 0

    def test_advances_on_gate_with_one_time_bonus(self):
        sim = _BodyStateSim({"cube": [0.0, 0.0, 0.0]})
        term = make_predicate("staged_reward", stages=self._spec())

        # Gate closed -> stage 0 dense reward only, no advance.
        assert term(sim) == pytest.approx(1.0)
        assert term.phase == 0

        # Gate fires -> emit stage-0 reward + one-time bonus, then advance.
        sim._pos["cube"] = [0.0, 0.0, 1.0]
        assert term(sim) == pytest.approx(1.0 + 10.0)
        assert term.phase == 1

        # Final stage: flat reward, bonus is NOT paid again.
        assert term(sim) == pytest.approx(2.0)
        assert term.phase == 1

    def test_phase_is_monotonic_no_regression(self):
        sim = _BodyStateSim({"cube": [0.0, 0.0, 1.0]})
        term = make_predicate("staged_reward", stages=self._spec())
        term(sim)  # advance to stage 1
        assert term.phase == 1
        # Gate condition reverses; the machine must not fall back to stage 0.
        sim._pos["cube"] = [0.0, 0.0, 0.0]
        assert term(sim) == pytest.approx(2.0)
        assert term.phase == 1

    def test_reset_returns_to_first_stage(self):
        sim = _BodyStateSim({"cube": [0.0, 0.0, 1.0]})
        term = make_predicate("staged_reward", stages=self._spec())
        term(sim)
        assert term.phase == 1
        term.reset()
        assert term.phase == 0
        # After reset it re-earns the transition bonus on the next gate hit.
        assert term(sim) == pytest.approx(1.0 + 10.0)

    def test_empty_stages_yields_zero_reward(self):
        # The factory rejects empty stages, but the term itself must be a safe
        # no-op if ever constructed empty by a consumer.
        term = _StagedReward([])
        assert term(_BodyStateSim({})) == 0.0

    def test_factory_rejects_non_list_or_empty_stages(self):
        with pytest.raises(ValueError, match="non-empty list"):
            make_predicate("staged_reward", stages=[])
        with pytest.raises(ValueError, match="non-empty list"):
            make_predicate("staged_reward", stages="not-a-list")

    def test_factory_rejects_non_dict_stage(self):
        with pytest.raises(ValueError, match=r"stage\[0\] must be a dict"):
            make_predicate("staged_reward", stages=["oops"])

    def test_factory_rejects_unknown_stage_keys(self):
        bad = [{"reward": {"predicate": "constant", "value": 1.0}, "typo": 3}]
        with pytest.raises(ValueError, match="unknown keys"):
            make_predicate("staged_reward", stages=bad)

    def test_factory_rejects_missing_or_malformed_reward(self):
        with pytest.raises(ValueError, match="reward must be a predicate-call dict"):
            make_predicate("staged_reward", stages=[{"advance_when": None}])
        with pytest.raises(ValueError, match="reward must be a predicate-call dict"):
            make_predicate("staged_reward", stages=[{"reward": {"value": 1.0}}])

    def test_factory_requires_advance_when_on_non_final_stage(self):
        two = [
            {"reward": {"predicate": "constant", "value": 1.0}},
            {"reward": {"predicate": "constant", "value": 2.0}},
        ]
        with pytest.raises(ValueError, match="must declare"):
            make_predicate("staged_reward", stages=two)

    def test_factory_rejects_malformed_advance_when(self):
        bad = [
            {
                "reward": {"predicate": "constant", "value": 1.0},
                "advance_when": {"body": "cube", "z": 0.5},
            },
            {"reward": {"predicate": "constant", "value": 2.0}},
        ]
        with pytest.raises(ValueError, match="advance_when must be a predicate-call dict"):
            make_predicate("staged_reward", stages=bad)

    def test_factory_rejects_non_numeric_bonus(self):
        base = {
            "reward": {"predicate": "constant", "value": 1.0},
            "advance_when": {"predicate": "body_above_z", "body": "cube", "z": 0.5},
        }
        for bad_bonus in ("lots", True):
            spec = [{**base, "bonus": bad_bonus}, {"reward": {"predicate": "constant", "value": 2.0}}]
            with pytest.raises(ValueError, match="bonus must be a number"):
                make_predicate("staged_reward", stages=spec)
