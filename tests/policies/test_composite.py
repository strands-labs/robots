"""Tests for ``strands_robots.policies.composite.CompositePolicy``.

CompositePolicy stacks a lower-body (locomotion) and upper-body (manipulation)
policy on one robot, querying both each tick and merging their action dicts by
joint name. These tests use stub policies so they exercise the routing / merge
contract without any model weights.
"""

import asyncio

import pytest

from strands_robots.policies import CompositePolicy, MockPolicy
from strands_robots.policies.base import Policy

LEG_JOINTS = ["hip", "knee", "ankle"]
ARM_JOINTS = ["shoulder", "elbow", "wrist"]


class StubPolicy(Policy):
    """A policy that returns a fixed chunk and records what it was queried with."""

    def __init__(self, chunk, *, requires_images=False, name="stub", horizon=1):
        self._chunk = chunk
        self._requires_images = requires_images
        self._name = name
        self._horizon = horizon
        self.state_keys: list[str] | None = None
        self.last_obs = None
        self.last_instruction = None
        self.last_kwargs = None
        self.reset_seeds: list[int | None] = []

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def requires_images(self) -> bool:
        return self._requires_images

    @property
    def execution_horizon(self) -> int:
        return self._horizon

    def set_robot_state_keys(self, robot_state_keys):
        self.state_keys = list(robot_state_keys)

    def reset(self, seed=None):
        self.reset_seeds.append(seed)

    async def get_actions(self, observation_dict, instruction, **kwargs):
        self.last_obs = observation_dict
        self.last_instruction = instruction
        self.last_kwargs = kwargs
        return [dict(d) for d in self._chunk]


def _run(policy, obs=None, instruction="", **kwargs):
    return asyncio.run(policy.get_actions(obs or {}, instruction, **kwargs))


class TestRouting:
    """Joint-name routing: each child contributes only its joint group."""

    def test_explicit_joint_groups_merge_disjoint(self):
        lower = StubPolicy([{j: 1.0 for j in LEG_JOINTS}])
        upper = StubPolicy([{j: 2.0 for j in ARM_JOINTS}])
        c = CompositePolicy(lower, upper, lower_joints=LEG_JOINTS, upper_joints=ARM_JOINTS)

        out = _run(c)
        assert len(out) == 1
        assert out[0] == {"hip": 1.0, "knee": 1.0, "ankle": 1.0, "shoulder": 2.0, "elbow": 2.0, "wrist": 2.0}

    def test_explicit_groups_drop_foreign_keys(self):
        # Each child emits the FULL joint set; routing keeps only its own group.
        full = {j: 9.0 for j in LEG_JOINTS + ARM_JOINTS}
        lower = StubPolicy([{**full, "hip": 1.0, "knee": 1.0, "ankle": 1.0}])
        upper = StubPolicy([{**full, "shoulder": 2.0, "elbow": 2.0, "wrist": 2.0}])
        c = CompositePolicy(lower, upper, lower_joints=LEG_JOINTS, upper_joints=ARM_JOINTS)

        out = _run(c)[0]
        assert out == {"hip": 1.0, "knee": 1.0, "ankle": 1.0, "shoulder": 2.0, "elbow": 2.0, "wrist": 2.0}

    def test_default_upper_fills_unclaimed_with_lower_precedence(self):
        # No joint groups: lower precedence on shared names, upper fills the rest.
        lower = StubPolicy([{"hip": 1.0, "shared": 1.0}])
        upper = StubPolicy([{"shared": 99.0, "shoulder": 2.0}])
        c = CompositePolicy(lower, upper)

        out = _run(c)[0]
        # 'shared' goes to the lower policy; upper's 99.0 is dropped.
        assert out == {"hip": 1.0, "shared": 1.0, "shoulder": 2.0}


class TestMergeChunks:
    """Chunk-length handling and re-query cadence."""

    def test_chunks_merge_to_shorter_length(self):
        lower = StubPolicy([{"hip": float(i)} for i in range(1)])  # length 1
        upper = StubPolicy([{"shoulder": float(i)} for i in range(8)], horizon=8)  # length 8
        c = CompositePolicy(lower, upper, lower_joints=["hip"], upper_joints=["shoulder"])

        out = _run(c)
        assert len(out) == 1
        assert out[0] == {"hip": 0.0, "shoulder": 0.0}

    def test_equal_length_chunks_merge_elementwise(self):
        lower = StubPolicy([{"hip": float(i)} for i in range(3)])
        upper = StubPolicy([{"shoulder": float(10 + i)} for i in range(3)])
        c = CompositePolicy(lower, upper, lower_joints=["hip"], upper_joints=["shoulder"])

        out = _run(c)
        assert out == [
            {"hip": 0.0, "shoulder": 10.0},
            {"hip": 1.0, "shoulder": 11.0},
            {"hip": 2.0, "shoulder": 12.0},
        ]

    def test_execution_horizon_is_min_of_children(self):
        lower = StubPolicy([{"hip": 0.0}], horizon=1)
        upper = StubPolicy([{"shoulder": 0.0}], horizon=8)
        c = CompositePolicy(lower, upper)
        assert c.execution_horizon == 1


class TestErrorPaths:
    """Validation and conflict surfacing - never a silent default."""

    def test_overlapping_joint_groups_rejected_at_init(self):
        lower = StubPolicy([{"hip": 0.0}])
        upper = StubPolicy([{"hip": 0.0}])
        with pytest.raises(ValueError, match="disjoint"):
            CompositePolicy(lower, upper, lower_joints=["hip", "knee"], upper_joints=["knee"])

    def test_none_children_rejected(self):
        with pytest.raises(ValueError, match="both a 'lower' and an 'upper'"):
            CompositePolicy(None, MockPolicy())  # type: ignore[arg-type]

    def test_runtime_collision_raises(self):
        # Default routing with lower precedence cannot collide; force a collision
        # by making the upper group explicitly claim a name the lower also emits.
        lower = StubPolicy([{"hip": 1.0, "shoulder": 5.0}])
        upper = StubPolicy([{"shoulder": 2.0}])
        c = CompositePolicy(lower, upper, upper_joints=["shoulder"])
        with pytest.raises(ValueError, match="both produced joint"):
            _run(c)

    def test_empty_lower_chunk_raises(self):
        c = CompositePolicy(StubPolicy([]), StubPolicy([{"shoulder": 1.0}]))
        with pytest.raises(ValueError, match="lower policy"):
            _run(c)

    def test_empty_upper_chunk_raises(self):
        c = CompositePolicy(StubPolicy([{"hip": 1.0}]), StubPolicy([]))
        with pytest.raises(ValueError, match="upper policy"):
            _run(c)

    def test_upper_fully_shadowed_by_lower_raises(self):
        # Both children drive the SAME joints, so lower precedence leaves the
        # upper policy contributing nothing: the composite commands exactly what
        # the lower policy alone would. Silently returning that reads as
        # "composed" while half the pipeline has no effect.
        lower = StubPolicy([{j: 1.0 for j in LEG_JOINTS}], name="generator")
        upper = StubPolicy([{j: 2.0 for j in LEG_JOINTS}], name="tracker")
        c = CompositePolicy(lower, upper)
        with pytest.raises(ValueError) as excinfo:
            _run(c)
        message = str(excinfo.value)
        # The refusal names the shadowed child, the count, and every joint the
        # lower policy already drives, so the caller can fix the ownership.
        assert "upper policy 'tracker'" in message
        assert "none of them reached the merged action" in message
        assert "lower policy 'generator'" in message
        for joint in LEG_JOINTS:
            assert joint in message
        assert "lower_joints" in message and "upper_joints" in message

    def test_partially_shadowed_upper_still_merges(self):
        # The refusal is scoped to a child that contributes NOTHING. Lower
        # precedence on a shared name, with the upper still contributing its own,
        # is the documented default and must keep working.
        lower = StubPolicy([{"hip": 1.0, "shared": 1.0}])
        upper = StubPolicy([{"shared": 99.0, "shoulder": 2.0}])
        assert _run(CompositePolicy(lower, upper))[0] == {"hip": 1.0, "shared": 1.0, "shoulder": 2.0}

    def test_child_commanding_nothing_is_not_a_dropped_command(self):
        # An empty action DICT (not an empty chunk) is a child commanding no
        # joint this tick. There is no command to drop, so it is not refused.
        lower = StubPolicy([{"hip": 1.0}])
        upper = StubPolicy([{}])
        assert _run(CompositePolicy(lower, upper))[0] == {"hip": 1.0}

    @pytest.mark.parametrize("role", ["lower", "upper"])
    def test_joint_group_matching_no_emitted_name_raises(self, role):
        # A group whose names do not exist in the child's output silently routed
        # that child to nothing. Name the mismatch instead of dropping it.
        emitted = {"hip": 1.0} if role == "lower" else {"shoulder": 2.0}
        groups = {f"{role}_joints": ["typo_joint"]}
        if role == "upper":
            groups["lower_joints"] = ["hip"]
        c = CompositePolicy(
            StubPolicy([{"hip": 1.0}], name="lo"),
            StubPolicy([emitted], name="up"),
            **groups,
        )
        with pytest.raises(ValueError, match=f"{role}_joints group"):
            _run(c)


class TestWholeBodyGeneratorIsNotComposable:
    """A whole-body generator and a whole-body controller cannot be composed.

    ``KimodoPolicy`` / ``MotionBricksPolicy`` emit joint targets for ALL 29 G1
    leg+waist+arm joints, and ``WBCPolicy`` drives 15 of those same joints. The
    two therefore claim overlapping joints rather than disjoint groups, so a
    composite of the pair is one child alone - the generator's reference is never
    tracked, or the controller is never consulted. Refusing that configuration is
    what keeps the mistake from looking like a working pipeline.
    """

    def test_generator_over_locomotion_controller_is_refused(self):
        from strands_robots.policies.kimodo.policy import KIMODO_G1_JOINTS
        from strands_robots.policies.wbc.policy import WBC_G1_LEG_WAIST_JOINTS

        # Every joint the controller drives is also driven by the generator.
        assert set(WBC_G1_LEG_WAIST_JOINTS) <= set(KIMODO_G1_JOINTS)

        generator = StubPolicy([{j: 0.5 for j in KIMODO_G1_JOINTS}], name="kimodo")
        controller = StubPolicy([{j: -0.9 for j in WBC_G1_LEG_WAIST_JOINTS}], name="wbc")

        with pytest.raises(ValueError) as excinfo:
            _run(CompositePolicy(generator, controller))
        message = str(excinfo.value)
        assert f"commanded {len(WBC_G1_LEG_WAIST_JOINTS)} joint(s)" in message
        assert "DISJOINT joint groups" in message

    def test_disjoint_groups_on_the_same_robot_still_compose(self):
        # The pattern CompositePolicy exists for: locomotion owns legs+waist,
        # manipulation owns the arms, each joint owned exactly once.
        from strands_robots.policies.wbc.policy import WBC_G1_ALL_JOINTS, WBC_G1_LEG_WAIST_JOINTS

        arm_joints = WBC_G1_ALL_JOINTS[len(WBC_G1_LEG_WAIST_JOINTS) :]
        lower = StubPolicy([{j: -0.9 for j in WBC_G1_LEG_WAIST_JOINTS}], name="wbc")
        upper = StubPolicy([{j: 0.3 for j in arm_joints}], name="manipulation")
        c = CompositePolicy(
            lower,
            upper,
            lower_joints=WBC_G1_LEG_WAIST_JOINTS,
            upper_joints=arm_joints,
        )

        merged = _run(c)[0]
        assert set(merged) == set(WBC_G1_ALL_JOINTS)
        assert all(merged[j] == -0.9 for j in WBC_G1_LEG_WAIST_JOINTS)
        assert all(merged[j] == 0.3 for j in arm_joints)


class TestForwarding:
    """The composite forwards lifecycle calls and obs subsets to both children."""

    def test_provider_name(self):
        c = CompositePolicy(StubPolicy([{"hip": 0.0}]), StubPolicy([{"shoulder": 0.0}]))
        assert c.provider_name == "composite"

    def test_requires_images_is_or_of_children(self):
        assert CompositePolicy(StubPolicy([{}]), StubPolicy([{}])).requires_images is False
        assert CompositePolicy(StubPolicy([{}], requires_images=True), StubPolicy([{}])).requires_images is True

    def test_set_robot_state_keys_forwarded(self):
        lower, upper = StubPolicy([{"hip": 0.0}]), StubPolicy([{"shoulder": 0.0}])
        CompositePolicy(lower, upper).set_robot_state_keys(["a", "b"])
        assert lower.state_keys == ["a", "b"]
        assert upper.state_keys == ["a", "b"]

    def test_reset_forwarded_with_seed(self):
        lower, upper = StubPolicy([{"hip": 0.0}]), StubPolicy([{"shoulder": 0.0}])
        CompositePolicy(lower, upper).reset(seed=7)
        assert lower.reset_seeds == [7]
        assert upper.reset_seeds == [7]

    def test_control_frequency_forwarded(self):
        lower, upper = StubPolicy([{"hip": 0.0}]), StubPolicy([{"shoulder": 0.0}])
        c = CompositePolicy(lower, upper)
        c.set_control_frequency(50.0)
        assert c.control_frequency == 50.0
        assert lower.control_frequency == 50.0
        assert upper.control_frequency == 50.0

    def test_instruction_and_kwargs_forwarded_to_both(self):
        lower, upper = StubPolicy([{"hip": 0.0}]), StubPolicy([{"shoulder": 0.0}])
        c = CompositePolicy(lower, upper)
        _run(c, instruction="wave", target_velocity=[0.5, 0.0, 0.0])
        assert lower.last_instruction == "wave"
        assert upper.last_instruction == "wave"
        assert lower.last_kwargs == {"target_velocity": [0.5, 0.0, 0.0]}
        assert upper.last_kwargs == {"target_velocity": [0.5, 0.0, 0.0]}

    def test_observation_subsets_routed_per_child(self):
        lower, upper = StubPolicy([{"hip": 0.0}]), StubPolicy([{"shoulder": 0.0}])
        c = CompositePolicy(lower, upper, lower_obs_keys=["base_quat"], upper_obs_keys=["observation.images.cam"])
        obs = {"base_quat": [1, 0, 0, 0], "observation.images.cam": "frame", "extra": 1}
        _run(c, obs=obs)
        assert lower.last_obs == {"base_quat": [1, 0, 0, 0]}
        assert upper.last_obs == {"observation.images.cam": "frame"}

    def test_full_observation_forwarded_by_default(self):
        lower, upper = StubPolicy([{"hip": 0.0}]), StubPolicy([{"shoulder": 0.0}])
        c = CompositePolicy(lower, upper)
        obs = {"a": 1, "b": 2}
        _run(c, obs=obs)
        assert lower.last_obs == obs
        assert upper.last_obs == obs


class TestWithRealMockPolicy:
    """End-to-end with two real MockPolicy children (no stubs)."""

    def test_two_mock_policies_compose(self):
        lower, upper = MockPolicy(), MockPolicy()
        lower.set_robot_state_keys(LEG_JOINTS)
        upper.set_robot_state_keys(ARM_JOINTS)
        c = CompositePolicy(lower, upper, lower_joints=LEG_JOINTS, upper_joints=ARM_JOINTS)
        # set_robot_state_keys via the composite forwards to BOTH; re-scope each.
        lower.set_robot_state_keys(LEG_JOINTS)
        upper.set_robot_state_keys(ARM_JOINTS)

        out = _run(c, obs={"observation.state": [0.0] * 3})
        assert len(out) == 8  # both mocks emit an 8-step chunk
        assert set(out[0]) == set(LEG_JOINTS + ARM_JOINTS)


class TestChildAccessAndRtcForwarding:
    """Public child accessors and RTC observed-delay forwarding."""

    def test_lower_and_upper_expose_the_child_policies(self):
        # The composite exposes its children so a caller can introspect or
        # re-scope them (e.g. re-running set_robot_state_keys per child).
        lower, upper = StubPolicy([{"hip": 0.0}]), StubPolicy([{"shoulder": 0.0}])
        c = CompositePolicy(lower, upper)
        assert c.lower is lower
        assert c.upper is upper

    def test_set_rtc_observed_delay_forwarded_to_composite_and_both_children(self):
        # A latency-sensitive child slices its leftover chunk by this counted
        # step delay; it must reach BOTH children and the composite itself.
        lower, upper = StubPolicy([{"hip": 0.0}]), StubPolicy([{"shoulder": 0.0}])
        c = CompositePolicy(lower, upper)
        c.set_rtc_observed_delay(3)
        assert c.rtc_observed_delay_steps == 3
        assert lower.rtc_observed_delay_steps == 3
        assert upper.rtc_observed_delay_steps == 3

    def test_set_rtc_observed_delay_none_clears_on_all(self):
        lower, upper = StubPolicy([{"hip": 0.0}]), StubPolicy([{"shoulder": 0.0}])
        c = CompositePolicy(lower, upper)
        c.set_rtc_observed_delay(5)
        c.set_rtc_observed_delay(None)
        assert c.rtc_observed_delay_steps is None
        assert lower.rtc_observed_delay_steps is None
        assert upper.rtc_observed_delay_steps is None

    def test_set_rtc_observed_delay_rejects_negative_before_forwarding(self):
        # The base contract rejects a negative count; the composite surfaces it
        # without leaving children in a half-updated state.
        lower, upper = StubPolicy([{"hip": 0.0}]), StubPolicy([{"shoulder": 0.0}])
        c = CompositePolicy(lower, upper)
        with pytest.raises(ValueError, match="rtc_observed_delay_steps must be a non-negative integer"):
            c.set_rtc_observed_delay(-1)
        assert lower.rtc_observed_delay_steps is None
        assert upper.rtc_observed_delay_steps is None
