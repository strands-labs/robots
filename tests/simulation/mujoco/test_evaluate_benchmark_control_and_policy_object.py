"""``evaluate_benchmark`` parity with ``run_policy`` / ``eval_policy``.

``evaluate_benchmark`` was the sole policy-evaluation entry point that could
neither run a pre-built policy (``policy_object=``) nor set the control-loop
rate (``control_frequency`` / ``control_substeps``); it always built a fresh
policy via ``create_policy`` and stepped physics at a hardcoded 50 Hz. Both
capabilities already exist on the shared ``PolicyRunner.evaluate`` /
``_evaluate_with_spec`` plumbing - the facade just did not expose them.

These tests drive a real MuJoCo physics benchmark (inline MJCF, dt=0.002,
no rendering - the policy declares ``requires_images=False``) and assert:

* ``policy_object=`` is accepted and the benchmark runs against it WITHOUT a
  ``create_policy`` round-trip (a heavy checkpoint is not reloaded);
* ``control_frequency`` flows through to the physics substeps stepped per
  action (``round(1 / control_frequency / physics_timestep)``), so a benchmark
  can run at the rate the policy was trained at;
* a non-positive ``control_frequency`` is rejected with a structured error.

Each assertion fails on the pre-fix facade (``TypeError: evaluate_benchmark()
got an unexpected keyword argument ...``).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy  # noqa: E402
from strands_robots.simulation.benchmark import (  # noqa: E402
    _BENCHMARK_REGISTRY,
    BenchmarkProtocol,
    StepInfo,
    register_benchmark,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


class _ProbeBenchmark(BenchmarkProtocol):
    """Minimal spec: runs the full horizon, never succeeds/fails early."""

    max_steps = 4

    @property
    def supported_robots(self) -> list[str]:
        return []  # any loaded robot

    @property
    def default_robot(self) -> str:
        return "arm1"

    def on_step(self, sim, obs, action) -> StepInfo:
        return StepInfo(reward=0.0)

    def is_success(self, sim) -> bool:
        return False

    def is_failure(self, sim) -> bool:
        return False


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


@pytest.fixture
def sim_with_robot():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "arm.xml")
    with open(path, "w") as f:
        f.write(ROBOT_XML)
    s = Simulation(tool_name="bench_parity_sim", mesh=False)
    s.create_world()
    s.add_robot("arm1", urdf_path=path)
    yield s
    s.cleanup()
    shutil.rmtree(tmpdir, ignore_errors=True)


def _capture_substeps(sim) -> list[int]:
    """Wrap sim.send_action to record the n_substeps it is driven with."""
    captured: list[int] = []
    orig = sim.send_action

    def _spy(action, robot_name=None, n_substeps: int = 1):
        captured.append(int(n_substeps))
        return orig(action, robot_name=robot_name, n_substeps=n_substeps)

    sim.send_action = _spy  # type: ignore[method-assign]
    return captured


def test_policy_object_runs_benchmark_without_create_policy(sim_with_robot, monkeypatch):
    """A pre-built policy is evaluated as-is; create_policy is never called."""
    register_benchmark("probe", _ProbeBenchmark())

    def _boom(*_a: Any, **_k: Any):
        raise AssertionError("create_policy must not be called on the policy_object path")

    # create_policy is imported lazily inside evaluate_benchmark from this module.
    monkeypatch.setattr("strands_robots.policies.create_policy", _boom)

    policy = MockPolicy()
    result = sim_with_robot.evaluate_benchmark(
        benchmark_name="probe",
        robot_name="arm1",
        n_episodes=1,
        policy_object=policy,
    )
    assert result["status"] == "success", result


def test_control_frequency_sets_physics_substeps(sim_with_robot):
    """control_frequency derives substeps/action = round(1/cf / dt), dt=0.002."""
    register_benchmark("probe", _ProbeBenchmark())

    for control_frequency, expected in ((25.0, 20), (50.0, 10)):
        captured = _capture_substeps(sim_with_robot)
        result = sim_with_robot.evaluate_benchmark(
            benchmark_name="probe",
            robot_name="arm1",
            n_episodes=1,
            policy_object=MockPolicy(),
            control_frequency=control_frequency,
        )
        assert result["status"] == "success", result
        assert captured, "send_action was never called"
        assert set(captured) == {expected}, (
            f"control_frequency={control_frequency} -> expected n_substeps={expected}, got {sorted(set(captured))}"
        )


def test_control_substeps_override(sim_with_robot):
    """control_substeps overrides the control_frequency-derived value."""
    register_benchmark("probe", _ProbeBenchmark())
    captured = _capture_substeps(sim_with_robot)
    result = sim_with_robot.evaluate_benchmark(
        benchmark_name="probe",
        robot_name="arm1",
        n_episodes=1,
        policy_object=MockPolicy(),
        control_frequency=50.0,
        control_substeps=7,
    )
    assert result["status"] == "success", result
    assert set(captured) == {7}, sorted(set(captured))


def test_non_positive_control_frequency_rejected(sim_with_robot):
    """A non-positive control_frequency returns a structured error, not a raise."""
    register_benchmark("probe", _ProbeBenchmark())
    result = sim_with_robot.evaluate_benchmark(
        benchmark_name="probe",
        robot_name="arm1",
        n_episodes=1,
        policy_object=MockPolicy(),
        control_frequency=0,
    )
    assert result["status"] == "error"
    assert "control_frequency" in result["content"][0]["text"]
