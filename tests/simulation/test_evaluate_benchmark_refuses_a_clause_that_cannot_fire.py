"""``evaluate_benchmark`` refuses a spec clause that can never fire.

A benchmark spec's ``success`` / ``failure`` / ``dense_reward`` clauses are
authored in the same predicate DSL as ``run_policy(stop_when=...)`` and compile
through the same code, so they fail the same way: a name the live scene does not
answer to makes its term a constant, because predicates never raise. On a
``stop_when`` clause that wastes a step budget, which ``run_policy`` refuses up
front. On a ``success`` clause it decides the benchmark's whole output - every
episode scores a miss and the eval reports ``success_rate: 0.0`` under
``status="success"``, a number indistinguishable from an honest policy failure
and the one a caller publishes. Measured on so100 + a cube before this guard,
one character apart::

    success: {all: [{predicate: body_above_z, body: cube,   z: -1.0}]} -> 1.0
    success: {all: [{predicate: body_above_z, body: cubeee, z: -1.0}]} -> 0.0

Both under ``status="success"``. A ``dense_reward`` term went from
``avg_reward=-0.4355`` to a dead ``0.0`` the same way.

So the pins here are: every route to a permanently-decided clause is refused
before a policy is built, the two surfaces agree on the same clause, and the
two things a pre-eval probe cannot decide - a spec that loads its own scene,
and a benchmark constructed from opaque compiled callables - are evaluated
unchanged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import register_benchmark
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.mujoco.simulation import Simulation

# Trivially true for any body that resolves (the cube rests well above -1 m), so
# a control row scores 1.0 and a row that scores 0.0 did so because its term
# degraded rather than because the task was hard.
LIVE_SUCCESS = {"all": [{"predicate": "body_above_z", "body": "cube", "z": -1.0}]}


@pytest.fixture
def sim_with_robot_and_cube():
    """so100 (a FIXED-base arm: no base_pos/base_quat) plus a resting cube."""
    s = Simulation(tool_name="benchmark_probe_test", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    assert s.add_object(name="cube", shape="box", position=[0.4, 0.0, 0.5], size=[0.05] * 3)["status"] == "success"
    s.step(50)
    yield s
    s.cleanup()


def _register(name: str, **clauses: object) -> str:
    spec = {
        "name": name,
        "max_steps": 10,
        "supported_robots": ["so100"],
        "default_robot": "so100",
        **clauses,
    }
    bench = DeclarativeBenchmark.from_dict(spec)
    register_benchmark(bench.name, bench)
    return bench.name


def _evaluate(sim: Simulation, name: str) -> dict:
    return sim.evaluate_benchmark(name, robot_name="alice", policy_provider="mock", n_episodes=2)


def _json_block(result: dict) -> dict:
    return next(
        (c["json"] for c in result.get("content", []) if isinstance(c, dict) and "json" in c),
        {},
    )


# Every route a spec clause can name an entity the scene will not answer to,
# with the substring that must identify the offending name in the refusal.
UNRESOLVABLE = [
    pytest.param(
        {"success": {"all": [{"predicate": "body_above_z", "body": "cubeee", "z": -1.0}]}},
        "cubeee",
        id="success-all-body-typo",
    ),
    pytest.param(
        {"success": {"any": [{"predicate": "body_above_z", "body": "cube_", "z": -1.0}]}},
        "cube_",
        id="success-any-body-typo",
    ),
    pytest.param(
        {"success": LIVE_SUCCESS, "failure": {"any": [{"predicate": "body_below_z", "body": "nope", "z": 0.0}]}},
        "nope",
        id="failure-body-typo",
    ),
    pytest.param(
        {"success": {"all": [{"predicate": "joint_above", "joint": "alice/nosuch", "value": 0.0}]}},
        "alice/nosuch",
        id="joint-not-in-observation",
    ),
    pytest.param(
        {
            "success": LIVE_SUCCESS,
            "dense_reward": [{"predicate": "distance_neg", "body_a": "cubeee", "body_b": "cube"}],
        },
        "cubeee",
        id="dense-reward-body-typo",
    ),
]


class TestAClauseThatCannotFireIsRefusedNotScored:
    @pytest.mark.parametrize(("clauses", "offender"), UNRESOLVABLE)
    def test_refused_naming_the_entity_and_the_consequence(
        self, sim_with_robot_and_cube, clauses: dict, offender: str, request
    ):
        name = _register(f"probe-{request.node.callspec.id}", **clauses)
        result = _evaluate(sim_with_robot_and_cube, name)
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert text.startswith("evaluate_benchmark: ")
        assert offender in text, text
        # The refusal names the benchmark, so a caller running several knows which.
        assert name in text, text
        # ... and what would have happened, which is the whole reason to refuse.
        assert "0% success rate" in text, text
        # No aggregate was fabricated for a clause that could not decide anything.
        assert "success_rate" not in _json_block(result)

    def test_base_predicate_on_a_fixed_base_arm_is_refused(self, sim_with_robot_and_cube):
        """so100 reports no floating base, so every base_* term on it is a constant.

        The clause spells no robot at all - ``robot`` defaults to the sole one -
        so this is only reachable by collecting the base reference from the
        PREDICATE rather than from a kwarg.
        """
        name = _register("probe-base", success=LIVE_SUCCESS, failure={"any": [{"predicate": "base_tipped"}]})
        result = _evaluate(sim_with_robot_and_cube, name)
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "no floating base" in text, text
        assert "0% success rate" in text, text

    def test_the_refusal_lands_before_a_policy_is_built(self, sim_with_robot_and_cube, monkeypatch):
        """Refusing after ``create_policy`` would cost a checkpoint download."""
        import strands_robots.policies as policies_mod

        def _explode(*_a: object, **_k: object) -> None:
            raise AssertionError("create_policy must not run for a clause that cannot fire")

        # ``evaluate_benchmark`` imports it function-locally, so the source
        # module is where the interception has to land.
        monkeypatch.setattr(policies_mod, "create_policy", _explode)
        name = _register("probe-preflight", success={"all": [{"predicate": "body_above_z", "body": "gone", "z": 0.0}]})
        t0 = sim_with_robot_and_cube._world.sim_time
        assert _evaluate(sim_with_robot_and_cube, name)["status"] == "error"
        # Nothing stepped either: no episode ran against the dead clause.
        assert sim_with_robot_and_cube._world.sim_time == t0


class TestTheGuardDoesNotRefuseWhatItCannotDecide:
    def test_a_resolvable_clause_still_scores(self, sim_with_robot_and_cube):
        """The control: the same shape, a body the scene has, unchanged verdict."""
        name = _register("probe-live", success=LIVE_SUCCESS)
        result = _evaluate(sim_with_robot_and_cube, name)
        assert result["status"] == "success", result
        assert _json_block(result)["success_rate"] == 1.0

    def test_a_live_dense_reward_still_accumulates(self, sim_with_robot_and_cube):
        """A reward term over resolvable bodies is non-zero, so the probe is not
        refusing the whole ``dense_reward`` family."""
        name = _register(
            "probe-live-reward",
            success=LIVE_SUCCESS,
            dense_reward=[{"predicate": "distance_neg", "body_a": "cube", "body_b": "alice/Base"}],
        )
        result = _evaluate(sim_with_robot_and_cube, name)
        assert result["status"] == "success", result
        assert _json_block(result)["avg_reward"] != 0.0

    def test_a_spec_that_loads_its_own_scene_is_not_refused(self, sim_with_robot_and_cube, tmp_path):
        """``on_episode_start`` loads the declared scene BEFORE the first
        observation, so the bodies it creates cannot exist when the probe runs -
        reporting them missing would refuse a spec that is correct."""
        scene = tmp_path / "scene.xml"
        scene.write_text(
            '<mujoco><worldbody><body name="from_scene" pos="0 0 0.5">'
            '<freejoint/><geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>'
        )
        bench = DeclarativeBenchmark.from_dict(
            {
                "name": "probe-scene",
                "max_steps": 10,
                "supported_robots": ["so100"],
                "default_robot": "so100",
                "scene": str(scene),
                "success": {"all": [{"predicate": "body_above_z", "body": "from_scene", "z": -1.0}]},
            }
        )
        # The body genuinely is absent right now - that is what makes this a
        # real over-refusal risk rather than a vacuous pass.
        assert "from_scene" not in sim_with_robot_and_cube.list_bodies()["content"][1]["json"]["bodies"]
        assert bench.referenced_entities() == ([], [], [], [])
        register_benchmark(bench.name, bench)
        assert _evaluate(sim_with_robot_and_cube, bench.name)["status"] == "success"

    def test_a_benchmark_built_from_compiled_callables_is_not_refused(self, sim_with_robot_and_cube):
        """Opaque clauses cannot be probed - the contract ``run_policy`` already
        applies to a callable ``stop_when``."""
        bench = DeclarativeBenchmark(
            name="probe-opaque",
            supported_robots=["so100"],
            default_robot="so100",
            max_steps=10,
            success_fn=lambda _sim: True,
            failure_fn=lambda _sim: False,
            reward_terms=[],
        )
        assert bench.referenced_entities() == ([], [], [], [])
        register_benchmark(bench.name, bench)
        result = _evaluate(sim_with_robot_and_cube, bench.name)
        assert result["status"] == "success", result
        assert _json_block(result)["success_rate"] == 1.0


class TestBothSurfacesReadTheSameClauseTheSameWay:
    def test_run_policy_and_evaluate_benchmark_agree(self, sim_with_robot_and_cube):
        """The asymmetry this closes: one clause, one scene, one verdict.

        ``run_policy`` refused this clause before the guard existed while
        ``evaluate_benchmark`` scored it 0.0, although both compile it with
        ``compile_stop_when`` against the same sim.
        """
        clause = {"all": [{"predicate": "body_above_z", "body": "cubeee", "z": -1.0}]}
        rollout = sim_with_robot_and_cube.run_policy(
            robot_name="alice", policy_provider="mock", n_steps=5, fast_mode=True, stop_when=clause
        )
        name = _register("probe-parity", success=clause)
        evaluation = _evaluate(sim_with_robot_and_cube, name)
        assert rollout["status"] == "error", rollout
        assert evaluation["status"] == "error", evaluation
        for text in (rollout["content"][0]["text"], evaluation["content"][0]["text"]):
            assert "cubeee" in text
            assert "bodies not present in the scene" in text


class TestReferencedEntitiesIsReadOnly:
    def test_mutating_the_returned_lists_cannot_reach_the_spec(self):
        bench = DeclarativeBenchmark.from_dict(
            {
                "name": "probe-copy",
                "default_robot": "so100",
                "success": {"all": [{"predicate": "body_above_z", "body": "cube", "z": 0.1}]},
                "dense_reward": [{"predicate": "base_height", "robot": "alice", "target": 0.3}],
            }
        )
        bodies, joints, bases, _ = bench.referenced_entities()
        assert (bodies, joints, bases) == (["cube"], [], ["alice"])
        bodies.append("injected")
        bases.append("injected")
        assert bench.referenced_entities() == (["cube"], [], ["alice"], [])
