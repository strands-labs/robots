"""A queued joint write with no pump to drain it is refused, not reported applied.

``set_joint_positions`` applies the pose inline on the ``SimulationApp``-owning
thread and otherwise queues it for :meth:`IsaacSimulation.pump`. ``pump`` is the
only consumer of that queue and ``run_pump_forever`` is the only thing that runs
it, so off the main thread with no pump engaged the put stranded the action in a
queue nobody reads - and answered:

    {"status": "success", "content": [{"text": "Set joint positions (queued)."}]}

while the articulation kept its previous pose. Measured from a worker thread with
no pump: success reported, joints unchanged, one action left sitting in the queue.

This is the shape the method's own docstring says it avoids for bad *values*:

    Validation runs synchronously, before the write is queued for the main
    thread, so a rejected value is reported to the caller rather than raised on
    the pump thread - where the queued-action handler swallows it after this
    method has already answered ``status="success"``.

The same reasoning covers a queue with no consumer, where there is not even a
swallowed exception to find afterwards. ``set_joint_positions`` was the only one
of this backend's ``_action_q`` producers with no pump guard; every other
main-thread-affine surface either routes through
``_marshal_main_thread_affine`` or checks ``_pump_running`` directly, as
``get_body_state`` does.

The refusal is an error dict rather than the ``RuntimeError``
``_marshal_main_thread_affine`` raises for ``reset``/``step``: this surface's
contract is the envelope throughout, and unlike those it cannot deadlock - it
returned promptly having done nothing, which is exactly why the silence needed
closing rather than the block.

Nothing here needs Isaac Sim; the articulation is a stand-in that records what it
was written.
"""

from __future__ import annotations

import queue
import threading
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402


class _Articulation:
    """Records the pose it was written, so an unapplied write is visible."""

    def __init__(self) -> None:
        self.q: list[float] = [0.0, 0.0]
        self.writes = 0

    def get_joint_positions(self) -> list[float]:
        return list(self.q)

    def set_joint_positions(self, arr: Any) -> None:
        self.q = [float(v) for v in np.asarray(arr, dtype=float)]
        self.writes += 1


def _engine(*, pump_running: bool) -> tuple[Any, _Articulation]:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._world_created = True
    engine._world = types.SimpleNamespace()
    engine._action_q = queue.Queue()
    engine._pump_running = pump_running
    # Recorded on THIS thread, so a call from a spawned thread is off-main.
    engine._main_tid = threading.get_ident()
    articulation = _Articulation()
    engine._robots = {
        "arm": types.SimpleNamespace(  # type: ignore[dict-item]
            articulation=articulation, joint_names=["j0", "j1"]
        )
    }
    return engine, articulation


def _from_worker(engine: Any, **kwargs: Any) -> dict[str, Any]:
    """Call ``set_joint_positions`` from a thread that does not own the app."""
    out: dict[str, Any] = {}

    def _run() -> None:
        out["result"] = engine.set_joint_positions(**kwargs)

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join()
    return out["result"]


class TestAWorkerWriteWithNoPumpIsRefused:
    def test_it_reports_an_error(self) -> None:
        engine, _ = _engine(pump_running=False)

        result = _from_worker(engine, positions={"j0": 1.5}, robot_name="arm")

        assert result["status"] == "error", result

    def test_the_pose_is_not_applied(self) -> None:
        engine, articulation = _engine(pump_running=False)

        _from_worker(engine, positions={"j0": 1.5}, robot_name="arm")

        assert articulation.q == [0.0, 0.0]
        assert articulation.writes == 0

    def test_nothing_is_left_stranded_in_the_queue(self) -> None:
        """The refusal happens instead of the put, so a pump started later does
        not silently apply a pose the caller was already told had failed."""
        engine, _ = _engine(pump_running=False)

        _from_worker(engine, positions={"j0": 1.5}, robot_name="arm")

        assert engine._action_q.qsize() == 0

    def test_the_message_says_the_pose_was_not_applied(self) -> None:
        engine, _ = _engine(pump_running=False)

        text = _from_worker(engine, positions={"j0": 1.5}, robot_name="arm")["content"][0]["text"]

        assert "NOT applied" in text

    def test_the_message_names_both_remedies(self) -> None:
        engine, _ = _engine(pump_running=False)

        text = _from_worker(engine, positions={"j0": 1.5}, robot_name="arm")["content"][0]["text"]

        assert "owning thread" in text
        assert "run_pump_forever" in text

    def test_the_list_form_is_refused_the_same_way(self) -> None:
        engine, articulation = _engine(pump_running=False)

        result = _from_worker(engine, positions=[1.5, 2.0], robot_name="arm")

        assert result["status"] == "error", result
        assert articulation.q == [0.0, 0.0]


class TestTheAcceptedPathsAreUnchanged:
    """Controls: refusing everything off-main would pass every test above."""

    def test_the_owning_thread_still_applies_inline(self) -> None:
        engine, articulation = _engine(pump_running=False)

        result = engine.set_joint_positions({"j0": 1.5}, robot_name="arm")

        assert result["status"] == "success", result
        assert articulation.q == [1.5, 0.0]

    def test_a_worker_still_queues_when_a_pump_is_running(self) -> None:
        engine, articulation = _engine(pump_running=True)

        result = _from_worker(engine, positions={"j0": 1.5}, robot_name="arm")

        assert result["status"] == "success", result
        assert engine._action_q.qsize() == 1
        assert articulation.q == [0.0, 0.0], "queued, so not applied until the pump runs"

    def test_the_queued_write_applies_when_the_pump_drains_it(self) -> None:
        """The end-to-end shape the accepted path promises."""
        engine, articulation = _engine(pump_running=True)
        _from_worker(engine, positions={"j0": 1.5}, robot_name="arm")

        engine._action_q.get_nowait()()

        assert articulation.q == [1.5, 0.0]


class TestValidationStillPrecedesTheThreadingCheck:
    """A bad value must keep its own diagnosis rather than inherit the pump one.

    Ordering matters: the pump refusal is about *where* the call came from, and a
    caller who passed a bad joint name has a different thing to fix. Reporting
    the threading problem first would send them to start a pump and then hit the
    real error, one round later.
    """

    def test_an_unresolved_joint_name_is_reported_as_such(self) -> None:
        engine, _ = _engine(pump_running=False)

        text = _from_worker(engine, positions={"nope": 1.5}, robot_name="arm")["content"][0]["text"]

        assert "unresolved" in text
        assert "pump" not in text

    def test_a_non_finite_value_is_reported_as_such(self) -> None:
        engine, _ = _engine(pump_running=False)

        text = _from_worker(engine, positions={"j0": float("nan")}, robot_name="arm")["content"][0]["text"]

        assert "pump" not in text

    def test_an_empty_mapping_is_reported_as_such(self) -> None:
        engine, _ = _engine(pump_running=False)

        text = _from_worker(engine, positions={}, robot_name="arm")["content"][0]["text"]

        assert "empty" in text
        assert "pump" not in text

    def test_a_length_mismatch_is_reported_as_such(self) -> None:
        engine, _ = _engine(pump_running=False)

        text = _from_worker(engine, positions=[1.0], robot_name="arm")["content"][0]["text"]

        assert "does not match" in text
        assert "pump" not in text
