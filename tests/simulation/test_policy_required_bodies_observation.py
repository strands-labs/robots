"""A policy's declared ``required_bodies`` reach it as body-pose observation keys.

A whole-body motion-mimic tracker consumes the world orientation of a single
*anchor* link (``torso_link`` on a Unitree G1). That link is not in the
observation schema and is not derivable from it: ``base_quat`` is the pelvis, and
the waist joints separate the two. These tests pin the declaration contract that
carries such a link to the policy - and pin that a policy declaring nothing still
sees the untouched backend observation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.composite import CompositePolicy
from strands_robots.policies.mock import MockPolicy
from strands_robots.policies.persistent import PersistentPolicy
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner

# A body that exists on the fixture's so100 arm. MuJoCo body names carry the
# robot's namespace prefix, which is what get_body_state resolves.
ARM_BODY = "alice/Lower_Arm"


class _BodyReadingPolicy(MockPolicy):
    """MockPolicy that declares bodies and records every observation it is given."""

    def __init__(self, bodies: object, **kwargs) -> None:
        super().__init__(**kwargs)
        self._declared = bodies
        self.seen: list[dict] = []

    @property
    def required_bodies(self):
        return self._declared

    async def get_actions(self, observation, instruction="", **kwargs):
        self.seen.append(dict(observation))
        return await super().get_actions(observation, instruction, **kwargs)


@pytest.fixture
def sim_with_robot():
    s = Simulation(tool_name="required_bodies_test", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    yield s
    s.cleanup()


def _run(sim, policy, **kw):
    policy.set_robot_state_keys(sim.robot_joint_names("alice"))
    kw.setdefault("duration", 0.06)
    return PolicyRunner(sim).run("alice", policy, control_frequency=50, fast_mode=True, **kw)


class TestDeclaredBodyReachesThePolicy:
    def test_declared_body_pose_is_merged_into_every_observation(self, sim_with_robot):
        """The four pose keys are present on every tick, with the right widths."""
        policy = _BodyReadingPolicy((ARM_BODY,))
        result = _run(sim_with_robot, policy)

        assert result["status"] == "success"
        assert policy.seen, "policy should have been queried at least once"
        for obs in policy.seen:
            assert [len(obs[f"body.{ARM_BODY}.{s}"]) for s in ("pos", "quat", "lin_vel", "ang_vel")] == [3, 4, 3, 3]

    def test_merged_pose_matches_the_backend_truth(self, sim_with_robot):
        """The merged values are the backend's own pose read, not a placeholder.

        Compared at a quiescent state: a rollout keeps stepping physics after
        the final ``get_actions``, so the policy's last observation is
        legitimately older than a post-rollout read.
        """
        runner = PolicyRunner(sim_with_robot)
        obs = runner._observe("alice", skip_images=True, bodies=(ARM_BODY,))
        truth = sim_with_robot.get_body_state(ARM_BODY)["content"][1]["json"]

        assert obs[f"body.{ARM_BODY}.pos"] == pytest.approx(truth["position"], abs=1e-9)
        assert obs[f"body.{ARM_BODY}.quat"] == pytest.approx(truth["quaternion"], abs=1e-9)
        assert obs[f"body.{ARM_BODY}.lin_vel"] == pytest.approx(truth["linear_velocity"], abs=1e-9)
        assert obs[f"body.{ARM_BODY}.ang_vel"] == pytest.approx(truth["angular_velocity"], abs=1e-9)

    def test_pose_is_a_live_reading_not_a_frozen_one(self, sim_with_robot):
        """A body driven by the policy reports a changing pose across ticks."""
        policy = _BodyReadingPolicy((ARM_BODY,))
        _run(sim_with_robot, policy, duration=0.4)

        poses = [tuple(o[f"body.{ARM_BODY}.pos"]) for o in policy.seen]
        assert len(set(poses)) > 1, "arm body pose never changed across the rollout"

    def test_duplicate_declarations_collapse(self, sim_with_robot):
        """Declaring a body twice is not an error and yields one key set."""
        policy = _BodyReadingPolicy((ARM_BODY, ARM_BODY))
        result = _run(sim_with_robot, policy)

        assert result["status"] == "success"
        assert f"body.{ARM_BODY}.quat" in policy.seen[0]


class TestDeclaringNothingIsFree:
    def test_no_body_keys_when_nothing_is_declared(self, sim_with_robot):
        """The default policy sees the backend observation untouched.

        Guards the cost + dataset-schema contract: body poses must not appear
        for the overwhelming majority of policies that never asked for one.
        """
        policy = _BodyReadingPolicy(())
        result = _run(sim_with_robot, policy)

        assert result["status"] == "success"
        assert policy.seen
        for obs in policy.seen:
            assert [k for k in obs if k.startswith("body.")] == []

    def test_plain_policy_without_the_attribute_still_runs(self, sim_with_robot):
        """A policy predating the contract is unaffected (getattr default)."""
        policy = MockPolicy()
        assert _run(sim_with_robot, policy)["status"] == "success"


class TestBadDeclarationsFailBeforeTheRollout:
    def test_unknown_body_raises_and_never_queries_the_policy(self, sim_with_robot):
        """An unresolvable name fails up front, not as a zero pose per tick."""
        policy = _BodyReadingPolicy(("no_such_link",))

        with pytest.raises(RuntimeError, match="no_such_link"):
            _run(sim_with_robot, policy)
        assert policy.seen == [], "rollout must not start with an unresolvable body"

    def test_bare_str_declaration_is_refused(self, sim_with_robot):
        """A str would iterate into one entry per character - refuse it."""
        policy = _BodyReadingPolicy(ARM_BODY)

        with pytest.raises(TypeError, match="not a bare str"):
            _run(sim_with_robot, policy)

    def test_non_string_entry_is_refused(self, sim_with_robot):
        policy = _BodyReadingPolicy((7,))

        with pytest.raises(TypeError, match="non-empty"):
            _run(sim_with_robot, policy)


class TestEvalPathHonoursTheContract:
    def test_eval_observations_carry_declared_body_poses(self, sim_with_robot):
        """evaluate() shares the declaration contract with run()."""
        policy = _BodyReadingPolicy((ARM_BODY,))
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        result = PolicyRunner(sim_with_robot).evaluate(
            robot_name="alice",
            policy=policy,
            n_episodes=1,
            max_steps=3,
            control_frequency=50,
        )

        assert result["status"] == "success"
        assert policy.seen
        assert f"body.{ARM_BODY}.quat" in policy.seen[0]


class TestAWrapperDoesNotHideItsChildsDeclaration:
    """A policy's declaration survives being wrapped.

    ``PersistentPolicy`` and ``CompositePolicy`` both hand their child the
    observation the runner assembles - the first verbatim, the second filtered -
    so a declaration read off the WRAPPER instead of the tree is lost for the
    policy that made it. Both halves of the contract go with it: the per-tick
    merge and the up-front scene check, leaving a rollout that reports success
    having never supplied the key. ``Policy.children`` exists so one probe walks
    to the policy that answers, and these pin that this probe does.
    """

    @staticmethod
    def _wrap(kind, sim, declared):
        """Build ``(driven, declaring_child)`` for a wrapper ``kind``."""
        joints = sim.robot_joint_names("alice")
        child = _BodyReadingPolicy(declared)
        if kind == "persistent":
            return PersistentPolicy("mock", policy_object=child), child
        composite = CompositePolicy(
            lower=MockPolicy(),
            upper=child,
            lower_joints=set(joints[:3]),
            upper_joints=set(joints[3:]),
        )
        if kind == "composite":
            return composite, child
        # Depth 2: a wrapper around a wrapper, so a fix that walks only one
        # level down still fails.
        return PersistentPolicy("mock", policy_object=composite), child

    WRAPPERS = ("persistent", "composite", "nested")

    @pytest.mark.parametrize("kind", WRAPPERS)
    def test_the_childs_declared_body_still_reaches_the_child(self, sim_with_robot, kind):
        """The four pose keys arrive in the observation the child is given."""
        driven, child = self._wrap(kind, sim_with_robot, (ARM_BODY,))
        result = _run(sim_with_robot, driven)

        assert result["status"] == "success"
        assert child.seen, "the declaring child should have been queried"
        for obs in child.seen:
            assert [len(obs[f"body.{ARM_BODY}.{s}"]) for s in ("pos", "quat", "lin_vel", "ang_vel")] == [3, 4, 3, 3]

    @pytest.mark.parametrize("kind", WRAPPERS)
    def test_a_body_the_scene_lacks_is_still_refused_up_front(self, sim_with_robot, kind):
        """The scene check is the half that fails silently: it must still fire.

        Without it the rollout runs to completion and reports success while the
        key the child declared never reaches it.
        """
        driven, child = self._wrap(kind, sim_with_robot, ("no_such_link",))

        with pytest.raises(RuntimeError, match="no_such_link"):
            _run(sim_with_robot, driven)
        assert child.seen == [], "rollout must not start with an unresolvable body"

    def test_the_refusal_names_the_declaring_policy_not_the_wrapper(self, sim_with_robot):
        """A refusal has to point at the class that has to change."""
        driven, _child = self._wrap("persistent", sim_with_robot, ("no_such_link",))

        with pytest.raises(RuntimeError) as excinfo:
            _run(sim_with_robot, driven)
        assert "_BodyReadingPolicy declares" in str(excinfo.value)

    def test_a_declaration_made_twice_in_one_tree_collapses(self, sim_with_robot):
        """Two policies naming one body yield one key set, not a duplicate merge."""
        child = _BodyReadingPolicy((ARM_BODY,))
        outer = CompositePolicy(
            lower=_BodyReadingPolicy((ARM_BODY,)),
            upper=child,
            lower_joints=set(sim_with_robot.robot_joint_names("alice")[:3]),
            upper_joints=set(sim_with_robot.robot_joint_names("alice")[3:]),
        )
        runner = PolicyRunner(sim_with_robot)

        assert runner._resolve_required_bodies(outer) == (ARM_BODY,)

    @pytest.mark.parametrize("kind", WRAPPERS)
    def test_a_wrapper_whose_child_declares_nothing_stays_free(self, sim_with_robot, kind):
        """Control: wrapping does not invent body keys for a policy that asked for none."""
        driven, child = self._wrap(kind, sim_with_robot, ())
        result = _run(sim_with_robot, driven)

        assert result["status"] == "success"
        assert child.seen
        for obs in child.seen:
            assert [k for k in obs if k.startswith("body.")] == []
