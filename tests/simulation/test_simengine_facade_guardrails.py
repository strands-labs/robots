"""Behavioural tests for the backend-agnostic ``SimEngine`` facade guardrails.

These pin the agent-facing validation and error-reporting contract that the
concrete facades on :class:`strands_robots.simulation.base.SimEngine` promise:

* ``eval_policy`` accepts a pre-built ``policy_object`` and runs it.
* ``evaluate_benchmark`` returns structured error dicts (never raises) when the
  sim has no robots or when the robot is ambiguous in a multi-robot scene.
* ``register_benchmark_from_file`` validates its arguments and converts loader
  exceptions into structured error dicts rather than propagating them.
* ``start_policy`` transparently passes through to ``run_policy``.
* ``__del__`` swallows (and logs) cleanup failures during GC.

All run against a pure-Python fake engine - no MuJoCo, no GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.base import SimEngine


class FakeSim(SimEngine):
    """Minimal in-memory ``SimEngine`` with a configurable robot set."""

    def __init__(self, robots: tuple[str, ...] = ("fake_robot",)) -> None:
        self._joint_names = ["j0", "j1", "j2"]
        self._robots = {name: self._joint_names for name in robots}

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        return {"status": "success"}

    def get_state(self):
        return {"sim_time": 0.0, "step_count": 0}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._robots.get(robot_name, []))

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {n: 0.0 for n in self._joint_names}

    def send_action(self, action, robot_name=None, n_substeps=1):
        return {"status": "success"}

    def render(self, camera_name="default", width=None, height=None):
        return {"image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8)}


# eval_policy


def test_eval_policy_resolves_sole_robot_when_name_omitted():
    """None robot_name auto-selects the only robot, mirroring run_policy.

    eval_policy and run_policy are siblings; a single-robot scene must
    resolve identically for both so a policy run with run_policy() can be
    evaluated the same way with eval_policy() (no spurious hard error).
    """
    result = FakeSim().eval_policy(policy_object=MockPolicy(), n_episodes=1, max_steps=2, control_frequency=10.0)
    assert result["status"] == "success", result


def test_eval_policy_ambiguous_multi_robot_lists_candidates():
    """None robot_name in a multi-robot scene errors with the candidate list."""
    result = FakeSim(robots=("arm_a", "arm_b")).eval_policy(policy_object=MockPolicy())
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "arm_a" in text and "arm_b" in text


def test_eval_policy_empty_world_reports_no_robots():
    result = FakeSim(robots=()).eval_policy(policy_object=MockPolicy())
    assert result["status"] == "error"
    assert "No robots" in result["content"][0]["text"]


def test_eval_policy_unknown_robot_reports_not_found():
    result = FakeSim().eval_policy(robot_name="ghost")
    assert result["status"] == "error"
    assert "ghost" in result["content"][0]["text"]


def test_eval_policy_runs_prebuilt_policy_object():
    """A caller-supplied policy_object is used directly (skips create_policy)."""
    sim = FakeSim()
    policy = MockPolicy()
    result = sim.eval_policy(
        robot_name="fake_robot",
        policy_object=policy,
        n_episodes=1,
        max_steps=2,
        control_frequency=10.0,
    )
    assert result["status"] == "success"
    json_blocks = [c["json"] for c in result["content"] if "json" in c]
    assert json_blocks and "success_rate" in json_blocks[0]


# evaluate_benchmark


def test_evaluate_benchmark_no_robots_is_structured_error(monkeypatch):
    """A valid benchmark with an empty sim returns an error, not a traceback."""
    import strands_robots.simulation.benchmark as bench

    monkeypatch.setattr(bench, "get_benchmark", lambda name: object())
    result = FakeSim(robots=()).evaluate_benchmark("any_bench")
    assert result["status"] == "error"
    assert "No robots" in result["content"][0]["text"]


def test_evaluate_benchmark_ambiguous_multi_robot_requires_name(monkeypatch):
    """Multi-robot scene + no robot_name -> error listing the candidates."""
    import strands_robots.simulation.benchmark as bench

    monkeypatch.setattr(bench, "get_benchmark", lambda name: object())
    result = FakeSim(robots=("arm_a", "arm_b")).evaluate_benchmark("any_bench")
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "robot_name" in text
    assert "arm_a" in text and "arm_b" in text


def test_evaluate_benchmark_explicit_unknown_robot_reports_not_found(monkeypatch):
    """An explicit robot_name absent from the sim is a structured not-found error."""
    import strands_robots.simulation.benchmark as bench

    monkeypatch.setattr(bench, "get_benchmark", lambda name: object())
    result = FakeSim(robots=("arm_a", "arm_b")).evaluate_benchmark("any_bench", robot_name="ghost")
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "ghost" in text and "not found" in text


def test_evaluate_benchmark_unknown_name_lists_registered():
    """An unregistered benchmark name surfaces the available set."""
    result = FakeSim().evaluate_benchmark("does_not_exist")
    assert result["status"] == "error"
    assert "does_not_exist" in result["content"][0]["text"]


# register_benchmark_from_file


def test_register_benchmark_from_file_rejects_empty_spec_path():
    result = FakeSim().register_benchmark_from_file("bench", "")
    assert result["status"] == "error"
    assert "spec_path" in result["content"][0]["text"]


def test_register_benchmark_from_file_rejects_empty_name():
    result = FakeSim().register_benchmark_from_file("", "/tmp/x.yaml")
    assert result["status"] == "error"
    assert "benchmark_name" in result["content"][0]["text"]


def test_register_benchmark_from_file_import_error_surfaces_hint(monkeypatch):
    """A missing optional dep (ImportError) is reported verbatim, not raised."""
    import strands_robots.simulation.benchmark_spec as spec_mod

    def _boom(name, path):
        raise ImportError("pyyaml is required for YAML benchmark specs")

    monkeypatch.setattr(spec_mod, "register_benchmark_from_file", _boom)
    result = FakeSim().register_benchmark_from_file("bench", "/tmp/spec.yaml")
    assert result["status"] == "error"
    assert "pyyaml" in result["content"][0]["text"]


def test_register_benchmark_from_file_unexpected_error_is_wrapped(monkeypatch):
    """An unforeseen loader failure is wrapped, not propagated past dispatch."""
    import strands_robots.simulation.benchmark_spec as spec_mod

    def _boom(name, path):
        raise RuntimeError("disk gremlin")

    monkeypatch.setattr(spec_mod, "register_benchmark_from_file", _boom)
    result = FakeSim().register_benchmark_from_file("bench", "/tmp/spec.yaml")
    assert result["status"] == "error"
    assert "unexpected error" in result["content"][0]["text"]
    assert "disk gremlin" in result["content"][0]["text"]


# start_policy passthrough


def test_start_policy_passes_through_to_run_policy():
    """The default start_policy is a synchronous run_policy passthrough."""
    sim = FakeSim()
    captured: dict[str, Any] = {}

    def fake_run_policy(robot_name=None, **kwargs):
        captured["robot_name"] = robot_name
        captured["kwargs"] = kwargs
        return {"status": "success", "content": [{"text": "ran"}]}

    sim.run_policy = fake_run_policy  # type: ignore[method-assign]
    result = sim.start_policy(robot_name="fake_robot", instruction="pick")
    assert result["status"] == "success"
    assert captured["robot_name"] == "fake_robot"
    assert captured["kwargs"]["instruction"] == "pick"


def test_start_policy_resolves_single_robot_when_unspecified():
    """start_policy(None) resolves to the lone robot before delegating."""
    sim = FakeSim()
    captured: dict[str, Any] = {}

    def fake_run_policy(robot_name=None, **kwargs):
        captured["robot_name"] = robot_name
        return {"status": "success", "content": [{"text": "ran"}]}

    sim.run_policy = fake_run_policy  # type: ignore[method-assign]
    sim.start_policy()
    assert captured["robot_name"] == "fake_robot"


# __del__ cleanup robustness


def test_del_swallows_cleanup_errors(caplog):
    """A failing cleanup during GC is logged, not raised (CPython __del__)."""
    import gc
    import logging

    class Exploding(FakeSim):
        def cleanup(self) -> None:
            raise RuntimeError("cleanup blew up")

    sim = Exploding()
    with caplog.at_level(logging.WARNING):
        # Drive finalization through real garbage collection rather than an
        # explicit dunder call: dropping the last reference triggers prompt
        # refcount-based finalization under CPython, with gc.collect() covering
        # any reference-cycle case. This exercises the genuine destructor path.
        del sim
        gc.collect()
    assert any("Cleanup error during __del__" in rec.message for rec in caplog.records)


# verify_dataset_episodes - on-disk episode-count verification contract


def test_verify_dataset_episodes_rejects_negative_expected():
    """A negative ``expected`` is a caller error reported as a structured dict."""
    sim = FakeSim()
    result = sim.verify_dataset_episodes(-1)
    assert result["status"] == "error"
    assert "non-negative int" in result["content"][0]["text"]


def test_verify_dataset_episodes_rejects_non_int_expected():
    """A non-int ``expected`` (e.g. a float) is rejected before touching disk."""
    sim = FakeSim()
    result = sim.verify_dataset_episodes(1.5)  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "non-negative int" in result["content"][0]["text"]


def test_verify_dataset_episodes_no_active_dataset_is_structured_error():
    """The base engine records nothing, so there is no dataset root to verify.

    Exercises the base ``_active_dataset_root`` returning ``None`` and the
    "record one first" guidance the caller gets instead of an exception.
    """
    sim = FakeSim()
    assert sim._active_dataset_root() is None
    result = sim.verify_dataset_episodes(1)
    assert result["status"] == "error"
    assert "no active or recently-recorded dataset" in result["content"][0]["text"]


def test_verify_dataset_episodes_missing_parquet_reports_json_diagnostics(monkeypatch):
    """A missing dataset parquet surfaces as a structured error with diagnostics.

    When ``_active_dataset_root`` points at a location whose parquet cannot be
    read, the reader raises ``FileNotFoundError``; the facade must convert that
    into a ``status=error`` dict carrying the machine-readable ``json`` block
    (expected/actual/root) rather than letting the exception escape.
    """
    import strands_robots.dataset_recorder as dr

    class RootedSim(FakeSim):
        def _active_dataset_root(self) -> str:
            return "/tmp/does-not-exist-dataset-root"

    def _raise(root):
        raise FileNotFoundError("meta/info.json not found")

    monkeypatch.setattr(dr, "read_dataset_episode_indices", _raise)

    result = RootedSim().verify_dataset_episodes(3)
    assert result["status"] == "error"
    diagnostics = result["content"][1]["json"]
    assert diagnostics["expected"] == 3
    assert diagnostics["actual"] == 0
    assert diagnostics["sources_agree"] is False
    assert diagnostics["root"] == "/tmp/does-not-exist-dataset-root"


def test_verify_dataset_episodes_import_error_is_structured_error(monkeypatch):
    """A missing optional dep behind the reader degrades to a structured error."""
    import strands_robots.dataset_recorder as dr

    class RootedSim(FakeSim):
        def _active_dataset_root(self) -> str:
            return "/tmp/some-dataset-root"

    def _raise(root):
        raise ImportError("pandas is required to read dataset episodes")

    monkeypatch.setattr(dr, "read_dataset_episode_indices", _raise)

    result = RootedSim().verify_dataset_episodes(1)
    assert result["status"] == "error"
    assert "pandas is required" in result["content"][0]["text"]


# Backend hooks the base facade leaves unimplemented


def test_set_obs_noise_not_implemented_on_base_facade():
    """The base engine has no physics, so ``set_obs_noise`` is not implemented."""
    with pytest.raises(NotImplementedError, match="set_obs_noise"):
        FakeSim().set_obs_noise(joint_std=0.01)


def test_get_contacts_not_implemented_on_base_facade():
    """Contact queries require a concrete physics backend to override the hook."""
    with pytest.raises(NotImplementedError, match="get_contacts"):
        FakeSim().get_contacts()


# Actionable "robot not found" on the policy-execution sites (#1306 follow-up)
#
# run_policy / eval_policy / evaluate_benchmark used to emit a bare
# "Robot 'X' not found." (or ".. Loaded: [...]") on a mistyped robot_name,
# unlike the ~9 lookup sites #1306 routed through the backend helper. These pin
# that all three now surface the close-match + available-list + discovery action
# via the backend-agnostic SimEngine._unknown_robot_msg helper (list_robots-backed,
# so every backend inherits it). They fail on the pre-fix bare strings.


def test_unknown_robot_msg_offers_close_match_and_discovery_action():
    """The base helper names the robot, suggests a close match, lists the world,
    and points at the discovery action - not a dead-end string."""
    msg = FakeSim(robots=("so100", "so101"))._unknown_robot_msg("so10")
    assert "Robot 'so10' not found." in msg
    assert "Did you mean:" in msg and "so100" in msg
    assert "Available robots:" in msg
    assert "list_robots" in msg


def test_unknown_robot_msg_empty_world_points_at_add_robot():
    """With no robots, the helper omits the close-match and points at add_robot."""
    msg = FakeSim(robots=())._unknown_robot_msg("ghost")
    assert "Robot 'ghost' not found." in msg
    assert "add_robot" in msg
    assert "Did you mean" not in msg


def test_run_policy_unknown_robot_is_actionable():
    """run_policy's not-found error carries the close-match + discovery action."""
    result = FakeSim(robots=("so100",)).run_policy(robot_name="so10", policy_object=MockPolicy())
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "Robot 'so10' not found." in text
    assert "Did you mean:" in text and "so100" in text
    assert "list_robots" in text


def test_eval_policy_unknown_robot_is_actionable():
    """eval_policy's not-found error carries the close-match + discovery action."""
    result = FakeSim(robots=("so100",)).eval_policy(robot_name="so10", policy_object=MockPolicy())
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "Robot 'so10' not found." in text
    assert "Did you mean:" in text and "so100" in text
    assert "list_robots" in text


def test_evaluate_benchmark_unknown_robot_is_actionable(monkeypatch):
    """evaluate_benchmark's not-found error carries the close-match + discovery
    action (replacing the older bare 'Loaded: [...]' tail)."""
    import strands_robots.simulation.benchmark as bench

    monkeypatch.setattr(bench, "get_benchmark", lambda name: object())
    result = FakeSim(robots=("so100", "so101")).evaluate_benchmark("any_bench", robot_name="so10")
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "Robot 'so10' not found." in text
    assert "Did you mean:" in text and "so100" in text
    assert "list_robots" in text
