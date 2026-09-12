"""The kinematic state writers apply one joint-state domain on every backend.

``set_joint_positions`` writes joint state directly, bypassing the actuators. The
MuJoCo backend refuses a value the engine cannot honor - a boolean (``float(True)``
is a silent 1-radian target), a ``nan`` / ``inf`` (``mj_forward`` propagates it
across the whole kinematic state), a non-numeric value (it would raise past the
structured-error contract), an unresolvable joint name (the write used to be
skipped and reported as a complete pose) and a vector of the wrong width. The
Isaac backend accepted all of them, so the accepted domain depended on which
engine the caller happened to be driving.

The value half of that contract now lives on the ABC as
``SimEngine._coerce_joint_state_map``, so the two cannot drift apart again; the
structural checks are per-backend because each engine resolves joint names
against a different authority (MuJoCo against the compiled model, Isaac against
the articulation's DOF order).

Isaac amplifies every case: the write is queued for the main thread when the call
arrives from a worker, and the pump's queued-action handler swallows
``ValueError`` / ``TypeError`` as best-effort. So a value that raised on the main
thread returned ``status="success"`` from a worker and failed with no channel at
all. Validation therefore has to run synchronously, before the write is queued -
which is what ``TestARefusalReachesTheCaller`` pins.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
import queue
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation import base as sim_base
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation, _RobotState

JOINTS = ["shoulder", "elbow", "wrist"]
HOME = [0.10, 0.20, 0.30]


class FakeArticulation:
    """Records what reaches the articulation, so a write is measurable.

    Deliberately permissive: it accepts any width and any value, exactly as
    Isaac's articulation handle does. A double that validated would hide the
    defect this module is about.
    """

    def __init__(self) -> None:
        self.q: list[float] = list(HOME)
        self.writes: list[list[float]] = []

    def get_joint_positions(self) -> list[float]:
        return list(self.q)

    def set_joint_positions(self, arr: Any) -> None:
        self.writes.append(list(np.asarray(arr, dtype=float).tolist()))
        self.q = list(np.asarray(arr, dtype=float).tolist())


def _engine(*, queued: bool = False) -> tuple[IsaacSimulation, FakeArticulation]:
    """A skeleton engine whose ``set_joint_positions`` reaches the real code.

    ``__new__`` leaves ``_init_complete`` at its class default of ``False`` so the
    finalizer skips a teardown this instance never set up.
    """
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._world_created = True
    engine._config = IsaacConfig()
    # A worker-thread call takes the queued branch; -1 is never a real thread id.
    engine._main_tid = -1 if queued else threading.get_ident()
    # Queuing is only a write if something drains the queue, and ``pump`` - run by
    # ``run_pump_forever`` - is the sole consumer. So the queued cases model the
    # real deployment shape, worker thread PLUS a running pump; without the pump
    # the write is refused rather than applied, because it would sit in a queue
    # nobody reads while the caller was told it succeeded.
    engine._pump_running = queued
    engine._action_q = queue.Queue()
    articulation = FakeArticulation()
    engine._robots = {
        "arm": _RobotState(
            name="arm",
            prim_path="/World/Robots/arm",
            joint_names=list(JOINTS),
            articulation=articulation,
        )
    }
    return engine, articulation


def _drain(engine: IsaacSimulation) -> str | None:
    """Apply queued work the way the production pump does, reporting what it ate.

    ``_pump_once`` catches ``ValueError`` / ``TypeError`` from a queued action as
    best-effort, so a failure there has no channel back to the caller.
    """
    swallowed: str | None = None
    while not engine._action_q.empty():
        fn = engine._action_q.get_nowait()
        try:
            fn()
        except (RuntimeError, ValueError, AttributeError, TypeError, KeyError, IndexError) as exc:
            swallowed = f"{type(exc).__name__}: {exc}"
    return swallowed


UNUSABLE_VALUES = [
    pytest.param(True, "bool", id="python-true"),
    pytest.param(False, "bool", id="python-false"),
    pytest.param(np.bool_(True), "bool", id="numpy-true"),
    pytest.param(float("nan"), "finite", id="nan"),
    pytest.param(float("inf"), "finite", id="inf"),
    pytest.param(float("-inf"), "finite", id="negative-inf"),
    pytest.param("abc", "must be a number", id="non-numeric-string"),
    pytest.param(None, "must be a number", id="none"),
    pytest.param([0.5], "must be a number", id="list-value"),
]


class TestIsaacAppliesTheSharedValueDomain:
    """A value the engine cannot honor is refused, and nothing is written."""

    @pytest.mark.parametrize(("value", "reason"), UNUSABLE_VALUES)
    def test_an_unusable_value_is_refused(self, value: Any, reason: str) -> None:
        engine, articulation = _engine()
        result = engine.set_joint_positions(positions={"shoulder": value}, robot_name="arm")
        assert result["status"] == "error", f"{value!r} was accepted as a joint position"
        text = " ".join(c.get("text", "") for c in result["content"] if "text" in c)
        assert reason in text, text
        assert "shoulder" in text, "the refusal must name the joint it is about"
        assert articulation.writes == [], "a refused value must not reach the articulation"
        assert articulation.q == HOME

    @pytest.mark.parametrize(("value", "reason"), UNUSABLE_VALUES)
    def test_the_list_form_is_held_to_the_same_domain(self, value: Any, reason: str) -> None:
        """The two accepted shapes must not have different value domains."""
        engine, articulation = _engine()
        result = engine.set_joint_positions(positions=[value, 0.2, 0.3], robot_name="arm")
        assert result["status"] == "error", f"{value!r} was accepted as a vector entry"
        assert articulation.q == HOME

    def test_a_non_finite_value_never_reaches_the_articulation(self) -> None:
        """PhysX surfaces a non-finite articulation from a later step, not this call.

        So a ``nan`` written here is reported as success and diagnosed elsewhere.
        """
        engine, articulation = _engine()
        engine.set_joint_positions(positions={"elbow": float("nan")}, robot_name="arm")
        assert all(math.isfinite(v) for v in articulation.q)

    def test_a_boolean_is_not_read_as_one_radian(self) -> None:
        engine, articulation = _engine()
        engine.set_joint_positions(positions={"shoulder": True}, robot_name="arm")
        assert articulation.q[0] != 1.0, "float(True) was written as a 1-radian target"
        assert articulation.q == HOME


class TestIsaacResolvesEveryJointNameBeforeWriting:
    """The write is all-or-nothing: a name that does not resolve refuses the call."""

    def test_a_typo_is_refused_rather_than_skipped(self) -> None:
        engine, articulation = _engine()
        result = engine.set_joint_positions(positions={"shoudler": 0.9}, robot_name="arm")
        assert result["status"] == "error"
        text = " ".join(c.get("text", "") for c in result["content"] if "text" in c)
        assert "shoudler" in text, "the refusal must name the key that did not resolve"
        assert "shoulder" in text, "and list the joints the robot does have"
        assert articulation.writes == []

    def test_a_partly_resolvable_mapping_writes_nothing(self) -> None:
        """The worst case: half the requested pose applied, reported as all of it."""
        engine, articulation = _engine()
        result = engine.set_joint_positions(positions={"shoulder": 0.9, "nope": 0.5}, robot_name="arm")
        assert result["status"] == "error"
        assert articulation.q == HOME, "a partial pose was applied"

    def test_an_empty_mapping_is_refused(self) -> None:
        """There is no write whose success could be reported."""
        engine, articulation = _engine()
        result = engine.set_joint_positions(positions={}, robot_name="arm")
        assert result["status"] == "error"
        text = " ".join(c.get("text", "") for c in result["content"] if "text" in c)
        assert "empty" in text
        assert articulation.writes == []


class TestIsaacRefusesAVectorOfTheWrongWidth:
    """The list form binds positionally, so its length is part of the contract."""

    @pytest.mark.parametrize("ordered", [[0.5, 0.6], [0.5, 0.6, 0.7, 0.8], []], ids=["short", "long", "empty"])
    def test_a_mismatched_vector_is_refused(self, ordered: list[float]) -> None:
        engine, articulation = _engine()
        result = engine.set_joint_positions(positions=ordered, robot_name="arm")
        assert result["status"] == "error"
        assert articulation.writes == []
        assert len(articulation.q) == len(JOINTS), "the articulation's joint array was resized"

    def test_the_refusal_names_both_counts(self) -> None:
        engine, _ = _engine()
        result = engine.set_joint_positions(positions=[0.5, 0.6], robot_name="arm")
        text = " ".join(c.get("text", "") for c in result["content"] if "text" in c)
        assert "2" in text and "3" in text, text
        assert "dict" in text, "and point at the shape that does support a partial update"

    @pytest.mark.parametrize("value", ["abc", 0.5, {0.5}], ids=["string", "scalar", "set"])
    def test_a_value_that_is_neither_mapping_nor_vector_is_refused(self, value: Any) -> None:
        """A str is iterable, so it would otherwise be read as one entry per character."""
        engine, articulation = _engine()
        result = engine.set_joint_positions(positions=value, robot_name="arm")
        assert result["status"] == "error"
        assert articulation.writes == []


class TestARefusalReachesTheCaller:
    """Validation is synchronous, so a worker-thread call is answered, not queued.

    Placed after the world/robot checks and before ``_action_q.put``: the pump
    swallows a queued failure, so a check that ran inside the queued work would
    leave the caller holding ``status="success"`` for a write that never happened.
    """

    @pytest.mark.parametrize(
        "positions",
        [{"shoulder": True}, {"shoulder": float("nan")}, {"shoulder": "abc"}, {"shoudler": 0.9}, [0.5, 0.6]],
        ids=["bool", "nan", "non-numeric", "typo", "wrong-width"],
    )
    def test_an_unusable_value_is_never_queued(self, positions: Any) -> None:
        engine, articulation = _engine(queued=True)
        result = engine.set_joint_positions(positions=positions, robot_name="arm")
        assert result["status"] == "error", "a worker-thread call must be refused too"
        assert engine._action_q.empty(), "the refused write was queued instead of reported"
        assert _drain(engine) is None
        assert articulation.q == HOME

    def test_the_two_call_paths_agree(self) -> None:
        """The same value must not be refused on one thread and accepted on another."""
        for positions in ({"shoulder": "abc"}, {"shoulder": True}, [0.5, 0.6], {}):
            main_engine, _ = _engine()
            worker_engine, _ = _engine(queued=True)
            assert (
                main_engine.set_joint_positions(positions=positions, robot_name="arm")["status"]
                == worker_engine.set_joint_positions(positions=positions, robot_name="arm")["status"]
            ), f"verdicts differ by calling thread for {positions!r}"


class TestAcceptedValuesStillWrite:
    """The domain is additive: every usable call keeps working, on both paths."""

    @pytest.mark.parametrize("queued", [False, True], ids=["main", "queued"])
    def test_a_partial_mapping_writes_only_the_named_joints(self, queued: bool) -> None:
        engine, articulation = _engine(queued=queued)
        result = engine.set_joint_positions(positions={"shoulder": 0.9}, robot_name="arm")
        assert result["status"] == "success"
        _drain(engine)
        assert articulation.q == pytest.approx([0.9, HOME[1], HOME[2]])

    @pytest.mark.parametrize("queued", [False, True], ids=["main", "queued"])
    def test_a_full_vector_writes_every_joint(self, queued: bool) -> None:
        engine, articulation = _engine(queued=queued)
        result = engine.set_joint_positions(positions=[0.5, 0.6, 0.7], robot_name="arm")
        assert result["status"] == "success"
        _drain(engine)
        assert articulation.q == pytest.approx([0.5, 0.6, 0.7])

    @pytest.mark.parametrize(
        "value",
        [0, 0.0, -1.25, np.float64(0.75), np.int64(1), "0.5"],
        ids=["int0", "float0", "negative", "np-float", "np-int", "numeric-string"],
    )
    def test_a_usable_spelling_of_a_number_is_accepted(self, value: Any) -> None:
        """The domain is finiteness, not a narrow type - a numeric string is a number."""
        engine, articulation = _engine()
        result = engine.set_joint_positions(positions={"elbow": value}, robot_name="arm")
        assert result["status"] == "success", f"{value!r} was refused"
        assert articulation.q[1] == pytest.approx(float(value))


class TestTheSharedValueDomainIsOnTheABC:
    """One owner for the value half, so the backends cannot diverge again."""

    def test_the_domain_lives_on_the_engine_base(self) -> None:
        assert hasattr(SimEngine, "_coerce_joint_state_map")

    @pytest.mark.parametrize(("value", "reason"), UNUSABLE_VALUES)
    def test_the_shared_domain_refuses_what_the_backends_refuse(self, value: Any, reason: str) -> None:
        coerced, error = SimEngine._coerce_joint_state_map({"j0": value}, "positions", "set_joint_positions")
        assert error is not None
        assert coerced == {}, "a rejected map must not be returned half-coerced"
        assert reason in error["content"][0]["text"]

    def test_it_reports_rather_than_raises(self) -> None:
        """The structured-error contract: a caller never has to catch."""
        coerced, error = SimEngine._coerce_joint_state_map({"j0": "abc"}, "positions", "set_joint_positions")
        assert error is not None and error["status"] == "error"
        assert coerced == {}

    def test_it_rejects_atomically(self) -> None:
        """One bad entry rejects the whole map, so a write cannot be partial."""
        coerced, error = SimEngine._coerce_joint_state_map(
            {"j0": 0.5, "j1": float("nan"), "j2": -0.25}, "positions", "set_joint_positions"
        )
        assert error is not None
        assert coerced == {}


def _joint_state_writers() -> dict[tuple[str, str], str]:
    """Every public joint-state writer defined by a simulation backend.

    Keyed by ``(backend, method)`` so the failure message names the surface that
    drifted rather than a line number.
    """
    package = pathlib.Path(inspect.getfile(sim_base)).parent
    writers: dict[tuple[str, str], str] = {}
    for path in sorted(package.rglob("*.py")):
        backend = path.parent.name
        if backend == package.name:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in ast.iter_child_nodes(cls):
                if not isinstance(fn, ast.FunctionDef):
                    continue
                if fn.name not in ("set_joint_positions", "set_joint_velocities"):
                    continue
                writers[(backend, fn.name)] = ast.unparse(fn)
    return writers


class TestEveryBackendJointStateWriterAppliesTheSharedDomain:
    """A fourth backend cannot ship a joint-state writer with its own domain."""

    def test_the_scan_finds_the_known_writers(self) -> None:
        """A scan that found nothing would pass every assertion below."""
        assert set(_joint_state_writers()) == {
            ("mujoco", "set_joint_positions"),
            ("mujoco", "set_joint_velocities"),
            ("isaac", "set_joint_positions"),
        }

    def test_every_writer_calls_the_shared_domain(self) -> None:
        adrift = sorted(k for k, src in _joint_state_writers().items() if "_coerce_joint_state_map" not in src)
        assert not adrift, (
            f"these joint-state writers do not apply the shared value domain: {adrift}. "
            "A boolean reaches the joint state as a 1-radian target and a nan/inf is "
            "written unexamined, under status='success'."
        )

    def test_the_scanner_detects_a_planted_writer(self) -> None:
        """Without this, a scanner that silently matched nothing would look clean."""
        planted = ast.unparse(
            ast.parse("class E:\n    def set_joint_positions(self, positions=None):\n        return float(positions)\n")
        )
        assert "_coerce_joint_state_map" not in planted
