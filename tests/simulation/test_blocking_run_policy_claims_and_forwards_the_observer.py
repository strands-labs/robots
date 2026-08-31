"""The MuJoCo blocking entry claims the robot AND forwards the observer.

``MuJoCoSimEngine.run_policy`` is the only surface between a caller and
:meth:`PolicyRunner.run` on the default backend, and it carries two independent
obligations that arrived from opposite directions:

* it claims the robot on the launching thread (``_announce_rollout``) and drives
  the loop through ``_drive_rollout``, so a stop issued before the rollout's
  first frame lands on a raised flag rather than being answered "was not
  running" (#2833/#3060);
* it forwards ``observer`` to the runner, so a read-only consumer of the rollout
  event lane actually receives events (#2907).

Both are satisfied by *forwarding*, and a forward is exactly what a refactor
drops silently: the parameter stays in the signature, the call still returns
``success``, and the rollout applies the same actions. Measured on a real
12-step SO-101 rollout with the forward removed:

    observer= supplied     events delivered   observer_failures in payload
    forward present        14 (1+12+1)        0
    forward removed        0                  ABSENT

Neither obligation was graded here before this file. The observer lane's own
suite drives :meth:`PolicyRunner.run` through a stand-in engine and never calls
``MuJoCoSimEngine.run_policy``; the launch-window suite drives ``start_policy``
exclusively. So each of the two one-sided readings of this method passed every
test in the tree.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import threading
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation import create_simulation
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.model_registry import resolve_model_path
from strands_robots.simulation.observers import (
    SCHEMA_VERSION,
    RunPolicyEnded,
    RunPolicyStarted,
    RunPolicyStep,
)

N_STEPS = 12
CONTROL_HZ = 30.0

#: The parameter under test. Named once so the derived scan below and the
#: behavioural cells above cannot drift apart.
_PARAM = "observer"


@pytest.fixture
def sim() -> Any:
    """A real MuJoCo sim holding one arm, torn down after the test."""
    engine = create_simulation("mujoco")
    engine.create_world()
    engine.add_robot(name="arm", urdf_path=str(resolve_model_path("so101")))
    try:
        yield engine
    finally:
        engine.cleanup()


def _rollout(sim: Any, events: list[Any]) -> dict[str, Any]:
    result = sim.run_policy(
        robot_name="arm",
        policy_provider="mock",
        instruction="wave",
        n_steps=N_STEPS,
        control_frequency=CONTROL_HZ,
        fast_mode=True,
        observer=events.append,
    )
    assert result["status"] == "success", result
    return result


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    return next(block["json"] for block in result["content"] if "json" in block)


class TestTheObserverReachesTheRunnerThroughTheBackend:
    """A caller of the backend's own ``run_policy`` receives the event lane."""

    def test_the_lifecycle_is_delivered(self, sim: Any) -> None:
        events: list[Any] = []
        _rollout(sim, events)

        assert [type(e) for e in events] == [RunPolicyStarted] + [RunPolicyStep] * N_STEPS + [RunPolicyEnded], (
            f"the backend delivered {[type(e).__name__ for e in events]}"
        )

    def test_the_stream_is_dense_and_belongs_to_one_rollout(self, sim: Any) -> None:
        events: list[Any] = []
        _rollout(sim, events)

        assert [e.event_seq for e in events] == list(range(len(events)))
        assert len({e.run_id for e in events}) == 1
        assert {e.schema_version for e in events} == {SCHEMA_VERSION}

    def test_the_result_payload_reports_the_observers_own_health(self, sim: Any) -> None:
        """``observer_failures`` is present because the lane ran, and is zero."""
        events: list[Any] = []
        result = _rollout(sim, events)

        assert _payload(result)["observer_failures"] == 0

    def test_the_backends_own_hook_still_ran_beside_the_observer(self, sim: Any) -> None:
        """The whole point of the lane: watching does not take the cancel hook.

        MuJoCo installs its own ``on_frame`` (cooperative stop, trajectory
        mirror, mesh telemetry, dataset recording). Every step event reporting
        ``"ok"`` rather than ``"absent"`` is that hook confirming it ran, on the
        real backend, while an observer was attached.
        """
        events: list[Any] = []
        _rollout(sim, events)

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        assert [e.legacy_hook_outcome for e in steps] == ["ok"] * N_STEPS

    def test_the_lane_reports_what_the_world_did(self, sim: Any) -> None:
        """Every applied action resolved against the arm's own actuators."""
        events: list[Any] = []
        result = _rollout(sim, events)

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        ended = next(e for e in events if isinstance(e, RunPolicyEnded))
        joints = set(sim.robot_joint_names("arm"))

        assert [e.applied_action_index for e in steps] == list(range(N_STEPS))
        assert {e.action_resolution for e in steps} == {"full"}
        assert all(set(e.applied_action_keys) <= joints for e in steps)
        assert all(e.unresolved_action_keys == () for e in steps)
        assert ended.applied_actions == N_STEPS == _payload(result)["steps_used"]
        assert ended.stopped_reason == _payload(result)["stopped_reason"]


class TestTheBlockingEntryClaimsTheRobotBeforeItsFirstFrame:
    """A stop issued before the first frame is answered as a stop."""

    def test_a_stop_in_the_launch_window_is_honoured(self, sim: Any) -> None:
        """Park the hook factory, then stop the rollout from another thread.

        ``stop_policy`` reports "Stopped on" or "Was not running on" straight
        from ``policy_running``, so claiming the robot on the first frame rather
        than on the calling thread leaves this window answering "was not
        running" about a rollout that then runs to its full budget (#2833). The
        launching thread IS the caller for the blocking entry, so only another
        thread can observe the window - which is exactly where a stop comes
        from.
        """
        at_the_hook = threading.Event()
        release = threading.Event()
        real_factory = sim._make_run_policy_hook

        def parked_factory(robot_name: str, instruction: str) -> Any:
            at_the_hook.set()
            assert release.wait(30), "test never released the parked rollout"
            return real_factory(robot_name, instruction)

        sim._make_run_policy_hook = parked_factory
        done: list[dict[str, Any]] = []
        worker = threading.Thread(
            target=lambda: done.append(
                sim.run_policy(
                    robot_name="arm",
                    policy_provider="mock",
                    n_steps=N_STEPS,
                    control_frequency=CONTROL_HZ,
                    fast_mode=True,
                )
            ),
            daemon=True,
        )
        worker.start()
        try:
            assert at_the_hook.wait(30), "the rollout never reached the hook factory"
            robot = sim._world.robots["arm"]
            assert robot.policy_steps == 0, "the window is not open - a frame was already taken"
            assert robot.policy_running is True, "the rollout was not claimed by the thread that launched it"

            stopped = sim.stop_policy("arm")
            assert stopped["status"] == "success"
            assert stopped["content"][0]["text"] == "Stopped on 'arm'"
        finally:
            release.set()
            worker.join(timeout=60)

        assert done, "the rollout never returned"
        assert sim._world.robots["arm"].policy_steps == 0, "the stop did not land before the first frame"
        assert sim._world.robots["arm"].policy_running is False, "the rollout ended without releasing the robot"


def _scan_root() -> pathlib.Path:
    # Derived from a symbol rather than a path literal, so a scan rooted
    # elsewhere fails the non-vacuity assertion below instead of passing.
    return pathlib.Path(inspect.getfile(SimEngine)).parent


def _consumes_the_parameter(fn: ast.AST) -> bool:
    """The owner reads the lane: it tests it against ``None`` and calls it."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == _PARAM:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _PARAM:
            return True
    return False


def _forwards_the_parameter(fn: ast.AST) -> bool:
    """A surface that hands the lane on by keyword delegates to the owner."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == _PARAM and isinstance(kw.value, ast.Name) and kw.value.id == _PARAM:
                    return True
    return False


def _planted_method(source: str) -> ast.FunctionDef:
    """The single method of a one-class snippet, narrowed rather than suppressed.

    ``ast.parse(...).body[0].body[0]`` is typed ``stmt``, which carries no
    ``body``, so reaching through it needs either a suppression or these two
    assertions. The assertions also make a malformed exemplar fail here, naming
    the snippet, instead of at the predicate under test.
    """
    cls = ast.parse(source).body[0]
    assert isinstance(cls, ast.ClassDef), source
    fn = cls.body[0]
    assert isinstance(fn, ast.FunctionDef), source
    return fn


def _observer_surfaces() -> list[tuple[str, ast.AST]]:
    found: list[tuple[str, ast.AST]] = []
    for path in sorted(_scan_root().rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - the package parses
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            if cls.name.startswith("_"):
                continue
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]:
                if fn.name.startswith("_"):
                    continue
                if _PARAM in [a.arg for a in fn.args.args + fn.args.kwonlyargs]:
                    found.append((f"{path.name}::{cls.name}.{fn.name}", fn))
    return found


class TestNoObserverSurfaceAcceptsTheLaneAndDropsIt:
    """Every public method taking ``observer`` either consumes it or forwards it.

    Derived from the package rather than listed, so a fourth surface - another
    backend override, or a second entry point growing the lane - is held to the
    same rule the moment it lands, which is when the forward is cheapest to add.
    """

    def test_the_known_surfaces_are_the_ones_found(self) -> None:
        # Non-vacuity: a scan that resolves nothing would satisfy the negative
        # assertion below by matching nothing.
        assert {name for name, _fn in _observer_surfaces()} == {
            "base.py::SimEngine.run_policy",
            "policy_runner.py::PolicyRunner.run",
            "simulation.py::MuJoCoSimEngine.run_policy",
        }

    def test_every_surface_consumes_or_forwards_the_lane(self) -> None:
        adrift = [
            name
            for name, fn in _observer_surfaces()
            if not (_consumes_the_parameter(fn) or _forwards_the_parameter(fn))
        ]
        assert adrift == [], f"{adrift} accept {_PARAM} and neither consume nor forward it"

    def test_the_scanner_detects_a_planted_surface_that_drops_the_lane(self) -> None:
        fn = _planted_method(
            "class Engine:\n    def run_policy(self, observer=None):\n        return super().run_policy()\n"
        )
        assert not _consumes_the_parameter(fn)
        assert not _forwards_the_parameter(fn)

    def test_the_scanner_accepts_a_planted_forwarder_and_a_planted_owner(self) -> None:
        forwarder = _planted_method(
            "class Engine:\n"
            "    def run_policy(self, observer=None):\n"
            "        return self._drive_rollout('arm', observer=observer)\n"
        )
        owner = _planted_method(
            "class Runner:\n"
            "    def run(self, observer=None):\n"
            "        if observer is None:\n            return\n        observer(1)\n"
        )
        assert _forwards_the_parameter(forwarder)
        assert _consumes_the_parameter(owner)
