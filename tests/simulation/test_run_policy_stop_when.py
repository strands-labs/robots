"""End-to-end tests for ``run_policy(stop_when=...)`` + ``stopped_reason`` telemetry.

The semantic early-return primitive (issue #1644, the Harness-VLA ``vla_act``
pattern): a rollout ends as soon as the world reaches a predicate-DSL state,
not only when the step budget runs out, and the result json attributes WHY the
rollout ended (``stopped_reason``: ``"predicate"`` | ``"budget"`` |
``"cancelled"`` | ``"error"``) plus ``steps_used`` so an agent can decide
whether to retry.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import CooperativeStop, PolicyRunner


@pytest.fixture
def sim_with_robot_and_cube():
    """so100 + a dynamic cube spawned in the air (it free-falls under gravity),
    giving a deterministic mid-rollout state change for the stop predicates."""
    s = Simulation(tool_name="stop_when_test", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    r = s.add_object(name="cube", shape="box", position=[0.4, 0.0, 1.0], size=[0.05, 0.05, 0.05])
    assert r["status"] == "success", r
    yield s
    s.cleanup()


def _json_block(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


class TestStopWhenPredicate:
    def test_already_true_predicate_stops_after_first_step(self, sim_with_robot_and_cube):
        """body_above_z holds from step 1 (cube spawns at z=1.0) -> the rollout
        returns after exactly one applied action, not the 200-step budget."""
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            instruction="pick up the cube",
            n_steps=200,
            control_frequency=50.0,
            fast_mode=True,
            stop_when={"predicate": "body_above_z", "body": "cube", "z": 0.5},
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["stopped_reason"] == "predicate"
        assert payload["stopped_early"] is True
        assert payload["steps_used"] == 1
        assert payload["n_steps"] == 1
        assert "stop_when" in result["content"][0]["text"]

    def test_predicate_fires_mid_rollout_on_world_state_change(self, sim_with_robot_and_cube):
        """The falling cube crosses z=0.5 after ~16 control steps; the rollout
        stops there - well before the 200-step budget - proving the clause is
        checked against the LIVE sim after every applied action."""
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=200,
            control_frequency=50.0,
            fast_mode=True,
            stop_when={"predicate": "body_below_z", "body": "cube", "z": 0.5},
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["stopped_reason"] == "predicate"
        assert payload["stopped_early"] is True
        # False on step 1 (cube at z=1.0), true once fallen past 0.5 m.
        assert 1 < payload["steps_used"] < 200

    def test_group_clause_is_accepted(self, sim_with_robot_and_cube):
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=200,
            control_frequency=50.0,
            fast_mode=True,
            stop_when={
                "all": [
                    {"predicate": "body_above_z", "body": "cube", "z": 0.5},
                    {"predicate": "body_below_z", "body": "cube", "z": 2.0},
                ]
            },
        )
        assert result["status"] == "success", result
        assert _json_block(result)["stopped_reason"] == "predicate"


class TestStoppedReasonBudget:
    def test_budget_exhaustion_reports_budget(self, sim_with_robot_and_cube):
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=5,
            control_frequency=50.0,
            fast_mode=True,
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["stopped_reason"] == "budget"
        assert payload["stopped_early"] is False
        assert payload["steps_used"] == 5

    def test_never_firing_predicate_runs_to_budget(self, sim_with_robot_and_cube):
        """An armed-but-never-true clause must not change rollout semantics:
        full budget, stopped_reason='budget'."""
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=5,
            control_frequency=50.0,
            fast_mode=True,
            stop_when={"predicate": "body_above_z", "body": "cube", "z": 1000.0},
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["stopped_reason"] == "budget"
        assert payload["steps_used"] == 5


class TestStoppedReasonCancelled:
    def test_cooperative_stop_reports_cancelled(self, sim_with_robot_and_cube):
        """The existing CooperativeStop path (stop_policy's mechanism) is
        re-tagged 'cancelled' - distinguishable from a predicate hit."""
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot_and_cube.robot_joint_names("alice"))

        def cancel_at_step_3(step: int, obs: dict, action: dict) -> None:
            if step >= 2:
                raise CooperativeStop("user stop")

        result = PolicyRunner(sim_with_robot_and_cube).run(
            "alice",
            policy,
            duration=1.0,
            control_frequency=50.0,
            fast_mode=True,
            on_frame=cancel_at_step_3,
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["stopped_reason"] == "cancelled"
        assert payload["stopped_early"] is True
        assert payload["steps_used"] < 50


class TestStoppedReasonError:
    def test_raising_stop_when_is_fatal_and_reports_error(self, sim_with_robot_and_cube):
        """A raising clause means the early-return contract can no longer be
        honored - the rollout aborts (status=error, stopped_reason='error')
        instead of silently running to budget."""
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot_and_cube.robot_joint_names("alice"))

        def boom(sim) -> bool:
            raise RuntimeError("predicate exploded")

        result = PolicyRunner(sim_with_robot_and_cube).run(
            "alice",
            policy,
            duration=1.0,
            control_frequency=50.0,
            fast_mode=True,
            stop_when=boom,
        )
        assert result["status"] == "error", result
        assert "stop_when predicate raised" in result["content"][0]["text"]
        assert _json_block(result)["stopped_reason"] == "error"

    def test_rollout_failure_reports_error_reason(self, sim_with_robot_and_cube):
        """Any error result carries stopped_reason='error' (here: the fail-fast
        probe on a policy whose keys resolve to no actuator)."""

        class _WrongKeysPolicy(MockPolicy):
            async def get_actions(self, observation_dict, instruction, **kwargs):
                return [{"not_a_joint": 0.5}]

        policy = _WrongKeysPolicy()
        result = PolicyRunner(sim_with_robot_and_cube).run(
            "alice",
            policy,
            duration=1.0,
            control_frequency=50.0,
            fast_mode=True,
        )
        assert result["status"] == "error", result
        assert _json_block(result)["stopped_reason"] == "error"


class TestStopWhenValidation:
    def test_unknown_predicate_rejected_before_rollout(self, sim_with_robot_and_cube):
        """An unknown predicate name is a structured caller error naming the
        valid registry set - and nothing runs (sim time unchanged)."""
        t0 = sim_with_robot_and_cube._world.sim_time
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=50,
            fast_mode=True,
            stop_when={"predicate": "levitated", "body": "cube"},
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "Unknown predicate 'levitated'" in text
        assert "grasped" in text  # the valid list is enumerated
        assert sim_with_robot_and_cube._world.sim_time == t0

    def test_empty_clause_rejected(self, sim_with_robot_and_cube):
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=50,
            fast_mode=True,
            stop_when={},
        )
        assert result["status"] == "error", result
        assert "never fire" in result["content"][0]["text"]

    def test_float_reward_term_rejected(self, sim_with_robot_and_cube):
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=50,
            fast_mode=True,
            stop_when={"predicate": "distance_neg", "body_a": "cube", "body_b": "cube"},
        )
        assert result["status"] == "error", result
        assert "reward term" in result["content"][0]["text"]


class TestStopWhenAsyncRtc:
    def test_async_rtc_stops_within_one_step_of_condition(self, sim_with_robot_and_cube):
        """The async-RTC path checks the clause after EVERY applied action, so
        an already-true condition ends the rollout after one step even though
        MockPolicy emits 8-action chunks - pinning the documented one-control-
        step latency bound (the rest of the chunk is dropped)."""
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot_and_cube.robot_joint_names("alice"))
        result = PolicyRunner(sim_with_robot_and_cube).run(
            "alice",
            policy,
            duration=1.0,
            control_frequency=50.0,
            fast_mode=True,
            async_rtc=True,
            stop_when=lambda sim: True,
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["stopped_reason"] == "predicate"
        assert payload["steps_used"] == 1


class TestStopWhenToolDispatch:
    def test_dispatch_chain_forwards_stop_when(self, sim_with_robot_and_cube):
        """The agent-tool dispatch surface must forward stop_when all the way
        through (AGENTS.md: no silent kwarg drop) - a dispatched run_policy
        with an already-true clause returns after one step."""
        result = sim_with_robot_and_cube._dispatch_action(
            "run_policy",
            {
                "action": "run_policy",
                "robot_name": "alice",
                "policy_provider": "mock",
                "n_steps": 100,
                "control_frequency": 50.0,
                "fast_mode": True,
                "stop_when": {"predicate": "body_above_z", "body": "cube", "z": 0.5},
            },
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["stopped_reason"] == "predicate"
        assert payload["steps_used"] == 1

    def test_dispatch_rejects_unknown_predicate(self, sim_with_robot_and_cube):
        result = sim_with_robot_and_cube._dispatch_action(
            "run_policy",
            {
                "action": "run_policy",
                "robot_name": "alice",
                "policy_provider": "mock",
                "n_steps": 10,
                "fast_mode": True,
                "stop_when": {"predicate": "levitated", "body": "cube"},
            },
        )
        assert result["status"] == "error", result
        assert "Unknown predicate" in result["content"][0]["text"]


class TestStopWhenRecordingInterplay:
    def test_recorded_frame_count_equals_steps_used(self, sim_with_robot_and_cube, tmp_path):
        """stop_when composes with an active recording session: frames are
        captured up to the stop, so the reopened dataset's frame count equals
        the result's steps_used (round-trip, per AGENTS.md recording rules)."""
        from strands_robots.dataset_recorder import has_lerobot_dataset

        if not has_lerobot_dataset():
            pytest.skip("lerobot not installed")

        sim = sim_with_robot_and_cube
        root = str(tmp_path / "stopwhen_ds")
        r = sim.start_recording(repo_id="local/stopwhen", fps=50, root=root, overwrite=True)
        assert r["status"] == "success", r

        result = sim.run_policy(
            robot_name="alice",
            policy_provider="mock",
            instruction="wait for the cube to fall",
            n_steps=200,
            control_frequency=50.0,
            fast_mode=True,
            stop_when={"predicate": "body_below_z", "body": "cube", "z": 0.5},
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["stopped_reason"] == "predicate"
        steps_used = payload["steps_used"]
        assert 1 < steps_used < 200

        assert sim.stop_recording()["status"] == "success"

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id="local/stopwhen", root=root)
        assert ds.meta.total_frames == steps_used
        assert ds.meta.total_episodes == 1


class TestStopWhenMultiEpisode:
    def test_stop_when_gates_every_episode(self, sim_with_robot_and_cube):
        """n_episodes>1 forwards the compiled clause to every per-episode
        rollout (reset_between re-arms the falling cube), giving collection
        loops a per-episode success gate."""
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=200,
            control_frequency=50.0,
            fast_mode=True,
            n_episodes=2,
            stop_when={"predicate": "body_below_z", "body": "cube", "z": 0.5},
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        episodes = payload["episodes"]
        assert len(episodes) == 2
        for ep in episodes:
            assert ep["stopped_reason"] == "predicate"
            assert 1 < ep["steps_used"] < 200

    def test_multi_episode_payload_keeps_single_episode_shape(self, sim_with_robot_and_cube):
        """One tool, ONE payload shape: the n_episodes>1 aggregate carries
        stopped_reason + steps_used just like the single-episode payload
        (plus per-episode attribution in stopped_reasons), so
        payload['stopped_reason'] never becomes a KeyError when a caller
        bumps n_episodes from 1 to 2."""
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=200,
            control_frequency=50.0,
            fast_mode=True,
            n_episodes=2,
            stop_when={"predicate": "body_below_z", "body": "cube", "z": 0.5},
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["stopped_reason"] == "predicate"
        assert payload["stopped_reasons"] == ["predicate", "predicate"]
        assert payload["steps_used"] == payload["total_steps"] > 0


class TestStopWhenEntityProbe:
    """A typo'd entity name must be an up-front structured error, not a clause
    that compiles clean, silently never fires, and burns the whole budget
    reporting stopped_reason='budget' (review on #1656, item 1)."""

    def test_typo_body_rejected_before_rollout(self, sim_with_robot_and_cube):
        t0 = sim_with_robot_and_cube._world.sim_time
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=200,
            control_frequency=50.0,
            fast_mode=True,
            stop_when={"predicate": "body_above_z", "body": "cubee", "z": 0.2},
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "cubee" in text
        assert "never fire" in text
        assert _json_block(result)["stopped_reason"] == "error"
        # Nothing ran: no budget was burned on the unresolvable clause.
        assert sim_with_robot_and_cube._world.sim_time == t0

    def test_typo_body_inside_group_rejected(self, sim_with_robot_and_cube):
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=200,
            control_frequency=50.0,
            fast_mode=True,
            stop_when={
                "all": [
                    {"predicate": "body_above_z", "body": "cube", "z": 0.2},
                    {"predicate": "body_below_z", "body": "cuve", "z": 2.0},
                ]
            },
        )
        assert result["status"] == "error", result
        assert "cuve" in result["content"][0]["text"]

    def test_typo_joint_rejected_before_rollout(self, sim_with_robot_and_cube):
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=200,
            control_frequency=50.0,
            fast_mode=True,
            stop_when={"predicate": "joint_above", "joint": "alice/NoSuchJoint", "value": 0.5},
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "NoSuchJoint" in text
        assert "get_observation" in text

    def test_valid_names_pass_the_probe(self, sim_with_robot_and_cube):
        """The probe must not reject clauses whose entities DO resolve - the
        rollout proceeds and the predicate fires normally."""
        joint = sim_with_robot_and_cube.robot_joint_names("alice")[0]
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=5,
            control_frequency=50.0,
            fast_mode=True,
            stop_when={
                "any": [
                    {"predicate": "body_above_z", "body": "cube", "z": 1000.0},
                    {"predicate": "joint_above", "joint": joint, "value": 1e9},
                ]
            },
        )
        assert result["status"] == "success", result
        assert _json_block(result)["stopped_reason"] == "budget"

    def test_backend_without_body_lookup_rejected(self):
        """On a backend whose predicates cannot resolve bodies at all
        (_body_position returns None unconditionally without get_body_state),
        a body-referencing clause is rejected up front instead of silently
        never firing."""
        from tests.simulation.test_policy_kwargs_forwarding import FakeSim

        sim = FakeSim()
        result = sim.run_policy(
            robot_name="fake_robot",
            policy_provider="mock",
            n_steps=5,
            fast_mode=True,
            stop_when={"predicate": "body_above_z", "body": "cube", "z": 0.2},
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "get_body_state" in text
        assert _json_block(result)["stopped_reason"] == "error"


class TestErrorResultsCarryJson:
    """Every error exit path carries the stopped_reason='error' json block -
    'recorded on ALL exit paths' (review on #1656, item 3)."""

    def test_compile_rejection_carries_error_json(self, sim_with_robot_and_cube):
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=10,
            fast_mode=True,
            stop_when={"predicate": "levitated", "body": "cube"},
        )
        assert result["status"] == "error", result
        payload = _json_block(result)
        assert payload["stopped_reason"] == "error"
        assert payload["steps_used"] == 0

    def test_video_setup_failure_carries_error_json(self, sim_with_robot_and_cube, tmp_path):
        """The video-writer failure return was text-only; it must carry the
        same stopped_reason='error' json block as every other error exit."""
        result = sim_with_robot_and_cube.run_policy(
            robot_name="alice",
            policy_provider="mock",
            n_steps=5,
            control_frequency=50.0,
            fast_mode=True,
            video={"path": str(tmp_path / "out.mp4"), "camera": "no_such_camera"},
        )
        assert result["status"] == "error", result
        payload = _json_block(result)
        assert payload["stopped_reason"] == "error"
        assert payload["steps_used"] == 0
