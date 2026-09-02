"""Isaac ``run_multi_policy``: the synchronized multi-robot loop WITHOUT recording.

The Isaac backend drives MULTIPLE robots, each with its own policy and
instruction, in ONE lockstep control loop (#2158, part of the #2122 parity
work): every iteration observes each robot once, re-queries a policy only when
its buffered action chunk drains, applies every robot's joint targets, then
steps physics exactly ONCE. These tests pin that behaviour - plus the loop's
Isaac-specific contracts: every Kit-touching hop is marshalled through
``run_on_main`` (#1896), a worker-thread call with no pump is refused in the
tool envelope, ``reset_between=True`` is refused citing #1895, and a rollout
whose ``control_frequency`` disagrees with an active recording's fps is
refused up front (the shared rate guard every rollout entry point applies).

The four caller knobs the pre-flight block routes through base helpers
shared with MuJoCo - ``control_frequency``, ``duration``, ``instructions``
and ``action_horizon`` - are driven through THIS entry point below, each
asserted to return the shared helper's envelope verbatim. The helpers
themselves are unit-tested on the base; the behavioural coverage that
module delegates to an entry point runs on the default backend, so these
are what make Isaac's delegation to them a checked property.

The merged-frame recording path (one ``add_frame`` per timestep with every
robot's namespaced columns, #2159) is pinned separately in
``test_run_multi_policy_recording.py``.

The engine is a skeleton ``IsaacSimulation`` built with ``__new__`` (the
established pattern from ``test_dataset_recording.py``) so the loop runs
without the Isaac Sim Kit runtime: physics is a stub ``World`` counting
``step()`` calls, articulations are stubs recording ``apply_action`` calls,
and ``isaacsim.core.utils.types`` is faked by the shared
``fake_isaacsim_types`` fixture. GPU/real-Kit coverage is a separate
``tests_integ/`` follow-up.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.base import Policy
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState
from tests.tool_result_contract import tool_json

from .test_backend_parity import fake_isaacsim_types  # noqa: F401 - fixture

_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow"]


class _StubArticulation:
    """Records every ``ArticulationAction`` applied, with the applying thread."""

    def __init__(self, n_joints: int) -> None:
        self._n = n_joints
        self.applied: list[Any] = []
        self.apply_threads: set[int] = set()

    def apply_action(self, action: Any) -> None:
        self.applied.append(action)
        self.apply_threads.add(threading.get_ident())

    def get_joint_positions(self) -> np.ndarray:
        return np.zeros(self._n, dtype=np.float32)


class _StubWorld:
    """Stub Isaac ``World`` counting physics steps and the stepping thread."""

    def __init__(self) -> None:
        self.step_calls = 0
        self.step_threads: set[int] = set()

    def step(self, render: bool = False) -> None:  # noqa: ARG002 - signature parity
        self.step_calls += 1
        self.step_threads.add(threading.get_ident())


def _make_engine(robots: dict[str, _RobotState]) -> IsaacSimulation:
    """Skeleton IsaacSimulation (no Kit runtime), per the recording-test pattern."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = IsaacConfig(render_mode="headless")
    engine._lock = threading.RLock()
    engine._world = _StubWorld()
    engine._world_created = True
    engine._robots = robots
    engine._cameras = {}
    engine._objects = {}
    engine._prim_registry = []
    engine._cams_rec_state = None
    engine._recording_state_dict = {}
    engine._action_controllers = {}
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._replicated = False
    engine._num_envs_active = 1
    engine._pump_running = False
    engine._main_tid = threading.get_ident()
    engine._main_jobs = queue.Queue()
    return engine


def _robot(name: str) -> _RobotState:
    return _RobotState(
        name=name,
        prim_path=f"/World/Robots/{name}",
        joint_names=list(_JOINTS),
        articulation=_StubArticulation(len(_JOINTS)),
    )


@pytest.fixture
def sim_two_robots(fake_isaacsim_types) -> IsaacSimulation:  # noqa: F811, ARG001 - fixture injects the fake module
    """Two stub-articulation robots in one skeleton engine."""
    return _make_engine({"alpha": _robot("alpha"), "beta": _robot("beta")})


class _ChunkCounter(Policy):
    """Counts inference calls and returns a fixed-length action chunk."""

    requires_images = False

    def __init__(self, chunk: int = 10):
        self.calls = 0
        self.chunk = chunk
        self.call_threads: set[int] = set()
        self._keys: list[str] | None = None

    def set_robot_state_keys(self, keys):
        self._keys = list(keys)

    @property
    def provider_name(self) -> str:
        return "chunk_counter"

    async def get_actions(self, obs, instruction=""):
        self.calls += 1
        self.call_threads.add(threading.get_ident())
        keys = self._keys or list(_JOINTS)
        return [{k: 0.05 * (j + 1) for k in keys} for j in range(self.chunk)]


# --------------------------------------------------------------------------- #
# The synchronized loop                                                       #
# --------------------------------------------------------------------------- #
def test_run_multi_policy_runs_and_reports_per_robot_steps(sim_two_robots):
    """The loop completes recorder-free and reports per-robot step counts."""
    sim = sim_two_robots
    r = sim.run_multi_policy(
        policies={"alpha": MockPolicy(), "beta": MockPolicy()},
        n_steps=10,
        control_frequency=500.0,
        action_horizon=4,
    )
    assert r["status"] == "success", r
    payload = tool_json(r)
    assert payload["steps"] == 10
    assert payload["per_robot_steps"] == {"alpha": 10, "beta": 10}
    assert "synchronized steps" in r["content"][0]["text"]
    assert "recorded" not in r["content"][0]["text"]
    # Both robots advanced together and were released from the running flag.
    for name in ("alpha", "beta"):
        robot = sim._robots[name]
        assert robot.policy_steps == 10
        assert robot.policy_running is False


def test_run_multi_policy_steps_physics_exactly_once_per_timestep(sim_two_robots):
    """Lockstep: N timesteps with 2 robots = N physics steps, not 2N."""
    sim = sim_two_robots
    r = sim.run_multi_policy(
        policies={"alpha": MockPolicy(), "beta": MockPolicy()},
        n_steps=6,
        control_frequency=500.0,
    )
    assert r["status"] == "success", r
    assert sim._world.step_calls == 6
    # Every robot's targets were applied on every timestep (phase-aligned).
    for name in ("alpha", "beta"):
        assert len(sim._robots[name].articulation.applied) == 6


def test_run_multi_policy_action_horizon_amortizes_inference(sim_two_robots):
    """A policy is re-queried only when its buffered chunk drains."""
    pa, pb = _ChunkCounter(chunk=10), _ChunkCounter(chunk=10)
    r = sim_two_robots.run_multi_policy(
        policies={"alpha": pa, "beta": pb},
        n_steps=20,
        control_frequency=500.0,
        action_horizon=10,
    )
    assert r["status"] == "success", r
    assert pa.calls == 2
    assert pb.calls == 2


def test_run_multi_policy_per_robot_horizon_mapping(sim_two_robots):
    """A ``{robot: horizon}`` mapping drives per-robot re-query cadence."""
    pa, pb = _ChunkCounter(chunk=10), _ChunkCounter(chunk=10)
    r = sim_two_robots.run_multi_policy(
        policies={"alpha": pa, "beta": pb},
        n_steps=20,
        control_frequency=500.0,
        action_horizon={"alpha": 1, "beta": 10},
    )
    assert r["status"] == "success", r
    # alpha re-queried every step (horizon clamped to >=1); beta batched.
    assert pa.calls == 20
    assert pb.calls == 2


def test_run_multi_policy_honors_policy_chunk_length_over_smaller_horizon(sim_two_robots):
    """A chunk-emitting policy keeps its full trained chunk in the loop.

    With ``actions_per_step=10`` and ``action_horizon=2`` over 20 steps, each
    policy runs inference exactly twice: the effective chunk is
    ``max(action_horizon, actions_per_step) == 10`` via the shared
    ``resolve_chunk_length`` rule, exactly as the single-policy runner sizes
    it. Truncating to ``action_horizon`` alone would drop the chunk tail and
    force out-of-distribution re-queries MuJoCo's loop never makes.
    """

    class _Chunked(_ChunkCounter):
        def __init__(self, actions_per_step: int = 10):
            super().__init__(chunk=actions_per_step)
            self.actions_per_step = actions_per_step
            self.supports_rtc = False

    pa, pb = _Chunked(actions_per_step=10), _Chunked(actions_per_step=10)
    r = sim_two_robots.run_multi_policy(
        policies={"alpha": pa, "beta": pb},
        n_steps=20,
        control_frequency=500.0,
        action_horizon=2,
    )
    assert r["status"] == "success", r
    assert pa.calls == 2
    assert pb.calls == 2


def test_run_multi_policy_max_steps_aliases_n_steps(sim_two_robots):
    """``max_steps`` is honoured as the legacy alias for ``n_steps``."""
    r = sim_two_robots.run_multi_policy(
        policies={"alpha": MockPolicy(), "beta": MockPolicy()},
        max_steps=8,
        control_frequency=500.0,
    )
    assert r["status"] == "success", r
    assert tool_json(r)["steps"] == 8


def test_run_multi_policy_cooperative_stop_ends_early(sim_two_robots):
    """Flipping a robot's running flag mid-loop ends the loop early but cleanly."""
    sim = sim_two_robots

    class _StopAfter(Policy):
        requires_images = False

        def __init__(self, robots, robot_name, stop_at=3):
            self._robots = robots
            self._robot_name = robot_name
            self._stop_at = stop_at
            self.calls = 0
            self._keys: list[str] | None = None

        def set_robot_state_keys(self, keys):
            self._keys = list(keys)

        @property
        def provider_name(self) -> str:
            return "stop_after"

        async def get_actions(self, obs, instruction=""):
            self.calls += 1
            if self.calls >= self._stop_at:
                # Cooperative stop: drop the running flag so the loop bails.
                self._robots[self._robot_name].policy_running = False
            keys = self._keys or list(_JOINTS)
            return [{k: 0.0 for k in keys}]

    pa = _StopAfter(sim._robots, "alpha", stop_at=3)
    r = sim.run_multi_policy(
        policies={"alpha": pa, "beta": MockPolicy()},
        n_steps=50,
        control_frequency=500.0,
        action_horizon=1,
    )
    assert r["status"] == "success", r
    assert "stopped early" in r["content"][0]["text"]
    assert tool_json(r)["steps"] < 50
    # Running flags are cleared on the way out regardless of early stop.
    for name in ("alpha", "beta"):
        assert sim._robots[name].policy_running is False


def test_run_multi_policy_raises_on_empty_action_chunk(sim_two_robots):
    """An empty action chunk fails loudly instead of a zero-valued substitute."""

    class _Empty(Policy):
        requires_images = False

        def set_robot_state_keys(self, keys):
            pass

        @property
        def provider_name(self) -> str:
            return "empty"

        async def get_actions(self, obs, instruction=""):
            return []

    with pytest.raises(RuntimeError, match="empty action chunk"):
        sim_two_robots.run_multi_policy(
            policies={"alpha": _Empty(), "beta": _Empty()},
            n_steps=5,
            control_frequency=500.0,
        )
    # The failure path still released the running flags.
    for name in ("alpha", "beta"):
        assert sim_two_robots._robots[name].policy_running is False


def test_run_multi_policy_warns_on_distinct_instructions(sim_two_robots, caplog):
    """Distinct per-robot instructions warn, attributed to the Isaac module logger."""
    import logging

    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.isaac.simulation"):
        r = sim_two_robots.run_multi_policy(
            policies={"alpha": MockPolicy(), "beta": MockPolicy()},
            instructions={"alpha": "pour", "beta": "catch"},
            n_steps=4,
            control_frequency=500.0,
        )
    assert r["status"] == "success", r
    assert any("distinct per-robot instructions" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# Caller-error envelope                                                       #
# --------------------------------------------------------------------------- #
def test_run_multi_policy_rejects_empty_policies(sim_two_robots):
    assert sim_two_robots.run_multi_policy(policies={})["status"] == "error"


def test_run_multi_policy_rejects_unknown_robot(sim_two_robots):
    r = sim_two_robots.run_multi_policy(policies={"ghost": MockPolicy()}, n_steps=2)
    assert r["status"] == "error"
    assert "ghost" in r["content"][0]["text"]


def test_run_multi_policy_rejects_nonpositive_n_steps(sim_two_robots):
    r = sim_two_robots.run_multi_policy(
        policies={"alpha": MockPolicy()},
        n_steps=0,
        control_frequency=500.0,
    )
    assert r["status"] == "error"
    assert "n_steps must be a positive integer" in r["content"][0]["text"]


# --------------------------------------------------------------------------- #
# The four shared knob domains, driven through THIS entry point               #
# --------------------------------------------------------------------------- #
# The pre-flight block above the loop routes four caller knobs -
# ``control_frequency``, ``duration``, ``instructions`` and ``action_horizon`` -
# through base helpers shared with MuJoCo, and its own comments state the
# intent four times ("one refusal text for every backend", "guards the same
# domain as run_policy", "MuJoCo parity", "the shared positive-int domain").
# Those helpers are unit-tested on the base in
# ``tests/simulation/test_run_multi_policy_base_contract.py``, and the
# behavioural coverage that module delegates to an entry point runs on
# ``create_simulation()`` - the default backend - so no test drove these four
# refusals through the Isaac loop. Each one is asserted here to return the
# shared helper's own envelope verbatim, which is what makes "cannot drift
# from MuJoCo's refusal texts" a checked property rather than an intention,
# and needs neither MuJoCo nor a Kit runtime to state.


def _both() -> dict[str, Any]:
    """The two-robot policy mapping the fixture's engine drives."""
    return {"alpha": MockPolicy(), "beta": MockPolicy()}


def _cost_nothing(sim: IsaacSimulation) -> None:
    """Assert a refusal advanced no physics and applied no joint targets."""
    assert sim._world.step_calls == 0
    for name, robot in sim._robots.items():
        assert robot.articulation.applied == [], name
        assert getattr(robot, "policy_running", False) is False, name


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan")], ids=["zero", "negative", "nan"])
def test_run_multi_policy_rejects_a_frequency_it_cannot_divide_by(sim_two_robots, bad: float) -> None:
    """``control_frequency`` is refused with the shared helper's own envelope.

    It is validated before ``_resolve_horizon`` because that helper divides by
    it. The assertion is envelope EQUALITY rather than a substring: it pins
    that the loop returns the shared verdict, not a locally re-worded copy
    that could drift from the sibling backends.
    """
    sim = sim_two_robots
    result = sim.run_multi_policy(policies=_both(), n_steps=4, control_frequency=bad)

    assert result == IsaacSimulation._validate_positive_frequency(bad, "run_multi_policy")
    assert result["status"] == "error"
    _cost_nothing(sim)


@pytest.mark.parametrize("bad", [0.0, -1.0, True], ids=["zero", "negative", "bool"])
def test_run_multi_policy_rejects_a_duration_it_cannot_honor(sim_two_robots, bad: Any) -> None:
    """``duration`` - the horizon when no step count is given - is refused too."""
    sim = sim_two_robots
    result = sim.run_multi_policy(policies=_both(), duration=bad, control_frequency=500.0)

    assert result == IsaacSimulation._validate_duration(bad, "run_multi_policy", 500.0)
    assert result["status"] == "error"
    _cost_nothing(sim)


def test_run_multi_policy_checks_duration_only_when_it_is_the_effective_horizon(sim_two_robots) -> None:
    """A step count supersedes ``duration``, so an unusable one stays inert.

    The mirror of the test above. ``_resolve_horizon`` REBINDS ``duration`` to
    ``n_steps / control_frequency`` (measured: 0.0 in, 0.008 out), so the
    caller's unusable value never reaches ``_validate_duration`` and the
    ``if n_steps is None`` gate above it is belt-and-braces rather than the
    thing that makes this pass - removing that gate is behaviour-preserving.
    What this pins is the caller-visible contract, that an ignored knob is not
    a refusal, which is what breaks if a later change validates the ARGUMENT
    instead of the resolved horizon.
    """
    sim = sim_two_robots
    result = sim.run_multi_policy(policies=_both(), n_steps=4, duration=0.0, control_frequency=500.0)

    assert result["status"] == "success", result
    assert sim._world.step_calls == 4


@pytest.mark.parametrize(
    "bad",
    [{"nosuch": "pick the cube"}, ["pick", "hold"]],
    ids=["undriven-robot", "non-mapping"],
)
def test_run_multi_policy_rejects_instructions_the_shared_helper_refuses(sim_two_robots, bad: Any) -> None:
    """``instructions`` normalization refuses with the shared helper's envelope."""
    sim = sim_two_robots
    policies = _both()
    _, expected = IsaacSimulation._normalize_multi_policy_instructions(policies, bad, "run_multi_policy")
    assert expected is not None, "probe value must be outside the shared domain"

    result = sim.run_multi_policy(policies=policies, instructions=bad, n_steps=4)

    assert result == expected
    _cost_nothing(sim)


@pytest.mark.parametrize("bad", [0, {"nosuch": 4}], ids=["nonpositive-scalar", "undriven-robot"])
def test_run_multi_policy_rejects_an_action_horizon_the_shared_helper_refuses(sim_two_robots, bad: Any) -> None:
    """``action_horizon`` normalization refuses with the shared helper's envelope.

    ``default_horizon=8`` mirrors the value the loop passes, so the expected
    envelope is the one the production call really produces.
    """
    sim = sim_two_robots
    policies = _both()
    _, expected = IsaacSimulation._normalize_multi_policy_horizons(policies, bad, "run_multi_policy", default_horizon=8)
    assert expected is not None, "probe value must be outside the shared domain"

    result = sim.run_multi_policy(policies=policies, action_horizon=bad, n_steps=4)

    assert result == expected
    _cost_nothing(sim)


def test_run_multi_policy_requires_world():
    """Without a created world the loop returns a graceful error, not a crash."""
    engine = _make_engine({})
    engine._world_created = False
    engine._world = None
    r = engine.run_multi_policy(policies={"alpha": MockPolicy()}, n_steps=2)
    assert r["status"] == "error"
    assert "world" in r["content"][0]["text"].lower()


def test_run_multi_policy_rejects_robot_with_running_policy(sim_two_robots):
    """A robot already driven by another rollout is refused up front.

    ``run_multi_policy`` advances physics; letting it also drive a robot some
    other loop is already stepping would double-step that robot. Isaac has no
    background ``start_policy`` futures to prune (the base ``start_policy`` is
    a synchronous passthrough), so the guard reads the per-robot
    ``policy_running`` flag every Isaac policy-driving loop sets - the same
    flag the recording hook and this loop itself maintain.
    """
    sim = sim_two_robots
    sim._robots["alpha"].policy_running = True
    r = sim.run_multi_policy(policies={"alpha": MockPolicy(), "beta": MockPolicy()}, n_steps=2)
    assert r["status"] == "error", r
    msg = r["content"][0]["text"]
    assert "already running" in msg
    assert "alpha" in msg
    # The refusal touched nothing: no physics step, no targets applied.
    assert sim._world.step_calls == 0
    assert sim._robots["beta"].articulation.applied == []


def test_run_multi_policy_reset_between_returns_the_1895_error(sim_two_robots):
    """``reset_between=True`` is a structured refusal citing #1895, not a silent skip.

    Isaac's ``reset()`` tears down the articulation physics-tensor views on
    the pip wheels (#1895), so a mid-run reset would leave every robot
    unobservable. The keyword exists for forward-compat with ``run_policy``'s
    multi-episode semantics; until #1895 is fixed it must refuse loudly (Key
    Conventions #5/#6).
    """
    sim = sim_two_robots
    r = sim.run_multi_policy(
        policies={"alpha": MockPolicy(), "beta": MockPolicy()},
        n_steps=4,
        control_frequency=500.0,
        reset_between=True,
    )
    assert r["status"] == "error", r
    msg = r["content"][0]["text"]
    assert "#1895" in msg
    assert "reset_between" in msg
    # Refused before any physics advanced.
    assert sim._world.step_calls == 0


def test_run_multi_policy_refuses_rate_the_recording_cannot_describe(sim_two_robots):
    """A control_frequency the active recording's fps cannot describe is refused.

    Replaces the pre-#2159 premise test (a blanket refusal while any recording
    was active): the loop now records merged frames, so what remains refusable
    is the shared rate disagreement every rollout entry point guards -
    LeRobot timestamps frames positionally at the dataset fps, so a differing
    capture rate cannot be honored, only mislabelled. Refused before any
    physics advances (the same ``_validate_recording_rate`` contract as
    ``run_policy`` / MuJoCo's ``run_multi_policy``).
    """
    sim = sim_two_robots

    class _RecorderAt30:
        class dataset:  # noqa: N801 - attribute surface of DatasetRecorder
            fps = 30

    sim._recording_state_dict = {"recording": True, "dataset_recorder": _RecorderAt30()}
    r = sim.run_multi_policy(
        policies={"alpha": MockPolicy(), "beta": MockPolicy()},
        n_steps=2,
        control_frequency=50.0,
    )
    assert r["status"] == "error", r
    msg = r["content"][0]["text"]
    assert "30" in msg and "50" in msg
    assert sim._world.step_calls == 0


# --------------------------------------------------------------------------- #
# Thread affinity (#1896)                                                     #
# --------------------------------------------------------------------------- #
def test_run_multi_policy_marshals_hops_through_run_on_main(sim_two_robots):
    """Every Kit-touching hop goes through ``run_on_main``: two per timestep."""
    sim = sim_two_robots
    calls: list[int] = []
    real_run_on_main = sim.run_on_main

    def _spy(fn, timeout=None):
        calls.append(threading.get_ident())
        return real_run_on_main(fn, timeout)

    sim.run_on_main = _spy
    r = sim.run_multi_policy(
        policies={"alpha": MockPolicy(), "beta": MockPolicy()},
        n_steps=3,
        control_frequency=500.0,
    )
    assert r["status"] == "success", r
    # One observe-all hop + one apply-all-and-step hop per timestep.
    assert len(calls) == 2 * 3


def test_run_multi_policy_from_worker_thread_runs_kit_work_on_the_pump_thread(sim_two_robots):
    """Called off the owning thread, physics/apply run on the pump owner while
    policy inference stays on the worker (the #1896 split)."""
    sim = sim_two_robots
    sim._pump_running = True
    main_tid = threading.get_ident()
    policy_a, policy_b = _ChunkCounter(chunk=4), _ChunkCounter(chunk=4)
    box: dict[str, Any] = {}

    def _worker() -> None:
        box["tid"] = threading.get_ident()
        box["result"] = sim.run_multi_policy(
            policies={"alpha": policy_a, "beta": policy_b},
            n_steps=4,
            control_frequency=500.0,
            action_horizon=4,
        )

    t = threading.Thread(target=_worker)
    t.start()
    # Emulate run_pump_forever: drain the main-jobs queue on this ("main") thread.
    while t.is_alive():
        try:
            job = sim._main_jobs.get(timeout=0.05)
        except queue.Empty:
            continue
        job()
    t.join(timeout=10.0)

    assert box["result"]["status"] == "success", box["result"]
    # Kit work (physics step, articulation apply) ran on the pump owner...
    assert sim._world.step_threads == {main_tid}
    for name in ("alpha", "beta"):
        assert sim._robots[name].articulation.apply_threads == {main_tid}
    # ...while policy inference stayed off the main thread.
    assert main_tid not in policy_a.call_threads
    assert main_tid not in policy_b.call_threads
    assert box["tid"] in policy_a.call_threads


def test_run_multi_policy_from_worker_thread_without_a_pump_is_refused(sim_two_robots):
    """No pump on the owning thread: the call is refused in the tool envelope
    (the hops would block forever), rather than raising like step()/reset()."""
    sim = sim_two_robots
    box: dict[str, Any] = {}

    def _worker() -> None:
        box["result"] = sim.run_multi_policy(
            policies={"alpha": MockPolicy(), "beta": MockPolicy()},
            n_steps=2,
            control_frequency=500.0,
        )

    t = threading.Thread(target=_worker)
    t.start()
    t.join(timeout=10.0)
    assert not t.is_alive(), "the worker-thread call must not hang"
    assert box["result"]["status"] == "error", box["result"]
    assert "run_pump_forever" in box["result"]["content"][0]["text"]
    assert sim._world.step_calls == 0
