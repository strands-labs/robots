"""Regression tests: no backend's ``step`` holds its lock for the whole count.

Follow-up to #1868 / #1869, which settled the ``step(n_steps)`` input *domain*
across all three backends, and to #1871, which reported what the domain change
deliberately left: the per-call ceiling and the batched lock release were
MuJoCo-only. This module settles the **lock hold** half. The ceiling stays open
as #1871 and is pinned as out of scope by
``test_step_count_domain_across_backends.py``.

Two defects, and the second is the reason they are one change rather than two.

## 1. Isaac and Newton held the lock for an unbounded step count

Measured solver-free on ``d2a6ddc``, counting lock acquisitions for one
``step(100_001)``:

| | lock acquisitions | ticks |
| --- | --- | --- |
| MuJoCo | refuses the count (its own ceiling) | 0 |
| Isaac | **1** | 100_001 |
| Newton | **1** | 100_001 control = 400_004 solver steps at ``substeps=4`` |

At Isaac's ~2 ms ``world.step`` that single hold is over three minutes in which
every other locked method on the engine blocks. That is not hypothetical on
Isaac: ``pump`` / ``run_pump_forever`` drive the sim from the owning main thread
precisely because a web UI serves ``get_observation`` / ``send_action`` on worker
threads, so a worker's locked read is exactly what waits. Post-fix the same call
takes 102 and 101 acquisitions.

MuJoCo already batched, and its constant is now
``SimEngine._STEPS_PER_BATCH`` - shared because the *reason* is shared, unlike
the ceiling, whose value cannot be (see the note on the constant).

## 2. The batching MuJoCo already had was unsafe at its own boundaries

This is the finding #1871 did not have, and it is why batching could not simply
be copied to two more backends. ``cleanup`` nulls ``self._world`` under a bounded
``self._lock`` acquire specifically so a worker is never inside
``mj_step(world._model, world._data)`` on freed arrays (GH #116). A batched
``step`` releases that same lock every 1000 steps - so the handoff can land
*between two batches*, which no check covered:

```
MuJoCo step(3000), world nulled on the release ending batch 1
  pre-fix : AttributeError: 'NoneType' object has no attribute '_model'
            raised past the structured envelope, after 1000 of 3000 ticks
  post-fix: status=error, "step: world was destroyed mid-run after 1000 of
            3000 steps; aborting. ..."
```

So adding the release to Isaac and Newton without a boundary re-check would
have propagated a known-hazard shape to two more backends. The reference
backend had already settled the pattern one class over:
``_primitive_abort_reason`` re-checks the world on every tick boundary *because*
"the primitive loops release the lock between control ticks". ``step`` releases
the lock on the same schedule and did not. Hence one contract, stated on
``SimEngine.step``: release the lock at least every ``_STEPS_PER_BATCH`` steps,
and re-check the world on each boundary before advancing it.

The refusal names the steps completed because some of them were - a bare "no
world" would read as the call having done nothing, the same
report-disagrees-with-the-world defect the step-count domain removed.

What is deliberately NOT decided here is pinned by
``TestModelIdentityStaysOutOfScope``: a mid-run *recompile* (as opposed to a
teardown) is still invisible to ``step``, though ``_primitive_abort_reason``
catches it by comparing model identity.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap
import threading
import types
from typing import Any

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.models import SimWorld
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
from strands_robots.simulation.newton.simulation import NewtonSimEngine

#: Batch size used by the timing tests. Deliberately not
#: ``SimEngine._STEPS_PER_BATCH``: at the real 1000 a tick slow enough to be
#: measurable makes one batch take a second, so the tests would trade their own
#: runtime for nothing. The mechanism under test is the release, not the number;
#: the number itself is asserted by ``test_the_batch_size_is_shared``.
PROBE_BATCH = 5


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


class CountingLock:
    """An RLock that counts entries and can run a hook once it has released.

    ``_teardown_after`` stands in for a concurrent ``cleanup`` / ``destroy``
    winning the lock in the gap between two batches: the hook
    runs on that release, which is the only moment the engine does not hold the
    lock. ``_on_release`` is the same seam used by the interleaving tests, which
    need to observe the gap rather than mutate through it.
    """

    def __init__(self, teardown_after: int | None = None, hook: Any = None) -> None:
        self._lock = threading.RLock()
        self.acquires = 0
        self.releases = 0
        self._teardown_after = teardown_after
        self._hook = hook
        self._on_release: Any = None

    def __enter__(self) -> Any:
        self.acquires += 1
        return self._lock.__enter__()

    def __exit__(self, *exc: Any) -> None:
        # Returns ``None``, i.e. never suppresses. Forwarding
        # ``RLock.__exit__``'s result would have read as passing a suppression
        # decision through, but that method returns ``None`` unconditionally, so
        # the only thing forwarding could do here is mislead - and a lock that
        # swallowed the exception a batch raised is the last thing these tests
        # want.
        self._lock.__exit__(*exc)
        self.releases += 1
        if self._teardown_after is not None and self.releases == self._teardown_after:
            self._hook()
        if self._on_release is not None:
            self._on_release()

    # ``cleanup`` uses a bounded ``acquire(timeout=...)`` rather than ``with``.
    def acquire(self, timeout: float = -1) -> bool:
        return bool(self._lock.acquire(timeout=timeout))

    def release(self) -> None:
        self._lock.release()


# --------------------------------------------------------------------------- #
# Backend stand-ins (every guard precedes the solver and the stage)           #
# --------------------------------------------------------------------------- #
def _mujoco_stub(lock: CountingLock, tick: Any = None, batch: int | None = None) -> tuple[Any, dict[str, int]]:
    calls = {"n": 0}

    def mj_step(model: Any, data: Any) -> None:
        calls["n"] += 1
        data.time += 0.002
        if tick is not None:
            tick()

    stub = types.SimpleNamespace(
        _world=types.SimpleNamespace(
            _model=object(),
            _data=types.SimpleNamespace(time=0.0),
            sim_time=0.0,
            step_count=0,
            _backend_state={},
        ),
        _mj=types.SimpleNamespace(mj_step=mj_step, mj_forward=lambda model, data: None),
        _lock=lock,
        _MAX_STEPS_PER_CALL=MuJoCoSimEngine._MAX_STEPS_PER_CALL,
        _STEPS_PER_BATCH=MuJoCoSimEngine._STEPS_PER_BATCH if batch is None else batch,
        _apply_kinematic_attachments=lambda: None,
        _publish_ros_telemetry=lambda: None,
    )
    return stub, calls


def _isaac_stub(lock: CountingLock, tick: Any = None, batch: int | None = None) -> tuple[Any, dict[str, int]]:
    calls = {"n": 0}

    def world_step(render: bool = False) -> None:
        calls["n"] += 1
        if tick is not None:
            tick()

    stub: Any = types.SimpleNamespace(
        _lock=lock,
        _world_created=True,
        _STEPS_PER_BATCH=IsaacSimulation._STEPS_PER_BATCH if batch is None else batch,
        _config=IsaacConfig(),
        _sim_time=0.0,
        _step_count=0,
        _world=types.SimpleNamespace(step=world_step),
        # Main-thread-affinity opt-out (#1896): these tests measure lock
        # scheduling, not kit-thread affinity, and the interleave case
        # deliberately drives ``step`` from a worker thread while a contender
        # takes the lock. Declaring whichever thread runs the call as the
        # owning thread keeps the genuinely-bound marshal helper on its
        # inline path so the lock schedule stays the measurement.
        _on_main_thread=lambda: True,
        _pump_running=False,
    )
    stub._marshal_main_thread_affine = lambda name, fn: IsaacSimulation._marshal_main_thread_affine(stub, name, fn)
    return stub, calls


def _newton_stub(lock: CountingLock, tick: Any = None, batch: int | None = None) -> tuple[Any, SimWorld]:
    """A Newton stand-in on the real inherited ``_advance``, solver-free."""
    world = SimWorld()
    stub: Any = types.SimpleNamespace(
        _world=world,
        _model=types.SimpleNamespace(body_label=["ground"]),
        _lock=lock,
        _solver=None,
        _sync_viewer=lambda: tick() if tick is not None else None,
        _STEPS_PER_BATCH=NewtonSimEngine._STEPS_PER_BATCH if batch is None else batch,
        substeps=1,
    )
    stub._advance = lambda n_steps: NewtonSimEngine._advance(stub, n_steps)
    return stub, world


# --------------------------------------------------------------------------- #
# The lock is released between batches                                        #
# --------------------------------------------------------------------------- #
class TestTheLockIsReleasedBetweenBatches:
    """A long count costs many short holds, not one unbounded one."""

    def test_the_batch_size_is_shared(self) -> None:
        """One constant, on the base class, so a new backend inherits it.

        The ceiling is deliberately not shared this way - see
        ``test_step_count_domain_across_backends.py``. This asserts the split:
        the granularity is inherited, ``_MAX_STEPS_PER_CALL`` is MuJoCo's own.
        """
        assert SimEngine._STEPS_PER_BATCH == 1000
        for engine in (MuJoCoSimEngine, IsaacSimulation, NewtonSimEngine):
            assert engine._STEPS_PER_BATCH == SimEngine._STEPS_PER_BATCH
            assert "_STEPS_PER_BATCH" not in vars(engine), (
                f"{engine.__name__} shadows the shared batch size; the reason for it is shared"
            )
        assert not hasattr(IsaacSimulation, "_MAX_STEPS_PER_CALL")
        assert not hasattr(NewtonSimEngine, "_MAX_STEPS_PER_CALL")

    def test_mujoco_releases_the_lock_every_batch(self) -> None:
        lock = CountingLock()
        stub, calls = _mujoco_stub(lock, batch=PROBE_BATCH)
        assert MuJoCoSimEngine.step(stub, PROBE_BATCH * 4)["status"] == "success"
        assert calls["n"] == PROBE_BATCH * 4
        assert lock.acquires == 4

    def test_isaac_releases_the_lock_every_batch(self) -> None:
        lock = CountingLock()
        stub, calls = _isaac_stub(lock, batch=PROBE_BATCH)
        assert IsaacSimulation.step(stub, PROBE_BATCH * 4)["status"] == "success"
        assert calls["n"] == PROBE_BATCH * 4
        # One extra for the precondition read, which precedes the loop.
        assert lock.acquires == 5

    def test_newton_releases_the_lock_every_batch(self) -> None:
        lock = CountingLock()
        stub, world = _newton_stub(lock, batch=PROBE_BATCH)
        assert NewtonSimEngine.step(stub, PROBE_BATCH * 4)["status"] == "success"
        assert world.step_count == PROBE_BATCH * 4
        assert lock.acquires == 4

    @pytest.mark.parametrize("n_steps", [1, 7, 1000, 4001])
    def test_no_backend_holds_the_lock_for_more_than_one_batch(self, n_steps: int) -> None:
        """The invariant, stated as steps-per-acquisition rather than a count.

        Parametrized across a batch boundary because an off-by-one in the
        ``min(remaining, batch)`` arithmetic is the way this regresses: it would
        still release *a* lock, just not on the schedule the contract promises.
        """
        for name, run in (
            ("mujoco", lambda lk: MuJoCoSimEngine.step(_mujoco_stub(lk)[0], n_steps)),
            ("isaac", lambda lk: IsaacSimulation.step(_isaac_stub(lk)[0], n_steps)),
            ("newton", lambda lk: NewtonSimEngine.step(_newton_stub(lk)[0], n_steps)),
        ):
            lock = CountingLock()
            assert run(lock)["status"] == "success", name
            batches = -(-n_steps // SimEngine._STEPS_PER_BATCH)  # ceil
            assert lock.acquires >= batches, (name, lock.acquires, batches)
            steps_per_acquire = n_steps / lock.acquires
            assert steps_per_acquire <= SimEngine._STEPS_PER_BATCH, (name, steps_per_acquire)


# --------------------------------------------------------------------------- #
# A concurrent locked caller interleaves                                      #
# --------------------------------------------------------------------------- #
class TestAConcurrentCallerInterleaves:
    """The point of the release, asserted through real threads.

    The assertion is that the lock is genuinely free between batches while the
    call has not returned - which is what a worker thread's locked read needs.
    It is deliberately NOT "a waiter is served promptly", because that is false
    on CPython and measuring it would make this suite flaky rather than strict.

    Measured on this stub shape at ``batch=5`` with a 2 ms tick and a 400 ms
    total call: a thread already blocked on the lock waited **333 ms** to be
    handed it. Python locks are unfair, and the stepper's release-to-re-acquire
    gap is a handful of bytecodes with no yield point, so the stepper usually
    barges past a blocked waiter. A contender retrying with
    ``acquire(timeout=0.02)`` was never served at all, because each timeout
    re-queues it. So the release bounds the *hold*, and handover latency remains
    at the scheduler's discretion - worth knowing before anyone reads
    ``_STEPS_PER_BATCH`` as a latency guarantee.

    The ``_on_release`` seam removes that race from the test rather than from
    the behaviour: the stepper pauses after releasing until the contender has
    had the lock, so a pass means the lock was actually free, not that the test
    won a coin flip.

    Two things the seam does *not* license, both learned the hard way:

    The assertion samples ``step_done`` on the **contender** thread, inside the
    ``with lock:`` block, *before* calling ``acquired.set()``. At that instant
    the stepper is provably parked in ``_on_release`` (it cannot proceed until
    ``acquired`` is set or the 5 s timeout lapses), so ``step_done`` cannot
    race: if the lock was won during a batch gap, the stepper has not returned
    yet and ``not step_done.is_set()`` is True; if the lock was only free after
    ``step`` returned, ``step_done`` is already set and the test correctly
    fails. This keeps the suite strict about the property that is true (the
    lock is genuinely free mid-call) without re-importing the barging race the
    ``_on_release`` seam was built to remove.

    **Do not park on a release the contender cannot answer.** Isaac reads its
    precondition under a separate acquire before the loop, so that release
    arrives before any tick, while the contender is still waiting on
    ``underway``. Parking there cannot be answered and merely burns the hook's
    timeout - 5 s of the suite's runtime, every run. The hook therefore stays
    out of the way until the first tick has run.
    """

    @pytest.mark.parametrize(
        "backend,build,run",
        [
            ("mujoco", _mujoco_stub, MuJoCoSimEngine.step),
            ("isaac", _isaac_stub, IsaacSimulation.step),
            ("newton", _newton_stub, NewtonSimEngine.step),
        ],
    )
    def test_the_lock_is_free_between_batches_while_step_runs(self, backend: str, build: Any, run: Any) -> None:
        total_steps = PROBE_BATCH * 3

        # What the whole call costs in releases, uncontended and single-threaded.
        # Measured rather than assumed because it is backend-specific - Isaac
        # reads its precondition under a separate acquire - and this doubles as
        # the check that the uncontended path still works.
        reference = CountingLock()
        assert run(build(reference, batch=PROBE_BATCH)[0], total_steps)["status"] == "success", (
            f"{backend}: the uncontended path still works"
        )
        total_releases = reference.releases
        assert total_releases > 1, (
            f"{backend}: {total_steps} steps at batch {PROBE_BATCH} cost {total_releases} hold(s); "
            f"the lock was never released mid-call"
        )

        underway = threading.Event()
        acquired = threading.Event()
        step_done = threading.Event()
        #: Whether the contender held the lock while step was still running.
        #: Written by the contender under the lock (deterministic: the stepper
        #: is parked in ``_on_release`` and cannot proceed until ``acquired`` is
        #: set), read by the main thread only after both threads have joined.
        served_mid_call = [False]

        lock = CountingLock()
        # Hold the gap open until the contender has been served, so the
        # unfair-lock barging described above cannot decide the outcome - but
        # only from the first tick on, since a release preceding every tick
        # leaves the contender, still waiting on ``underway``, no way to answer.
        lock._on_release = lambda: None if acquired.is_set() or not underway.is_set() else acquired.wait(5.0)
        stub = build(lock, tick=underway.set, batch=PROBE_BATCH)[0]

        def _step() -> None:
            try:
                run(stub, total_steps)
            finally:
                step_done.set()

        def _contend() -> None:
            underway.wait(10.0)
            with lock:
                served_mid_call[0] = not step_done.is_set()
                acquired.set()

        stepper = threading.Thread(target=_step)
        contender = threading.Thread(target=_contend)
        stepper.start()
        contender.start()
        try:
            assert acquired.wait(20.0), f"{backend}: the lock never became free between batches"
        finally:
            acquired.set()  # never leave the stepper parked if we failed above
            step_done.wait(30.0)
            stepper.join(timeout=30.0)
            contender.join(timeout=30.0)

        assert served_mid_call[0], f"{backend}: the lock was only free after step returned"


# --------------------------------------------------------------------------- #
# A world torn down on a boundary aborts rather than raises                   #
# --------------------------------------------------------------------------- #
class TestATornDownWorldAborts:
    """Releasing the lock is what makes a mid-call teardown reachable.

    Each case nulls the world on the release that ends the first batch - the
    ``cleanup`` world handoff of GH #116 winning its bounded acquire in the only
    gap there is - and asserts the call reports through its documented channel
    instead of raising, naming the steps that did happen.
    """

    def test_mujoco_aborts_with_the_count_completed(self) -> None:
        lock = CountingLock()
        stub, calls = _mujoco_stub(lock, batch=PROBE_BATCH)
        lock._teardown_after, lock._hook = 1, lambda: setattr(stub, "_world", None)

        result = MuJoCoSimEngine.step(stub, PROBE_BATCH * 3)
        assert result["status"] == "error"
        assert f"after {PROBE_BATCH} of {PROBE_BATCH * 3} steps" in _text(result)
        assert calls["n"] == PROBE_BATCH, "only the completed batch ran"

    def test_isaac_aborts_with_the_count_completed(self) -> None:
        lock = CountingLock()
        stub, calls = _isaac_stub(lock, batch=PROBE_BATCH)
        # Release 1 is the precondition read; the first batch ends on release 2.
        lock._teardown_after, lock._hook = 2, lambda: setattr(stub, "_world", None)

        result = IsaacSimulation.step(stub, PROBE_BATCH * 3)
        assert result["status"] == "error"
        assert f"after {PROBE_BATCH} of {PROBE_BATCH * 3} steps" in _text(result)
        assert calls["n"] == PROBE_BATCH

    def test_newton_aborts_with_the_count_completed(self) -> None:
        lock = CountingLock()
        stub, world = _newton_stub(lock, batch=PROBE_BATCH)
        lock._teardown_after, lock._hook = 1, lambda: setattr(stub, "_model", None)

        result = NewtonSimEngine.step(stub, PROBE_BATCH * 3)
        assert result["status"] == "error"
        assert f"after {PROBE_BATCH} of {PROBE_BATCH * 3} steps" in _text(result)
        assert world.step_count == PROBE_BATCH

    @pytest.mark.parametrize(
        "backend,build,run,teardown_after,kill",
        [
            ("mujoco", _mujoco_stub, MuJoCoSimEngine.step, 1, "_world"),
            ("isaac", _isaac_stub, IsaacSimulation.step, 2, "_world"),
            ("newton", _newton_stub, NewtonSimEngine.step, 1, "_model"),
        ],
    )
    def test_nothing_raises_past_the_structured_envelope(
        self, backend: str, build: Any, run: Any, teardown_after: int, kill: str
    ) -> None:
        """The contract these methods document is their only failure channel.

        Pre-fix every row here raised ``AttributeError`` - the defect, not a
        detail: one caller takes its step count from a remote process, and a
        bare exception is not a result it can read.
        """
        lock = CountingLock()
        stub = build(lock, batch=PROBE_BATCH)[0]
        lock._teardown_after, lock._hook = teardown_after, lambda: setattr(stub, kill, None)

        result = run(stub, PROBE_BATCH * 3)  # must not raise
        assert result["status"] == "error", backend
        assert "aborting" in _text(result), backend

    def test_the_refusal_is_worded_identically_across_backends(self) -> None:
        """Identical in wording, not merely in verdict - as the other shared
        domains are, because an agent matches on the text."""
        texts = set()
        for build, run, teardown_after, kill in (
            (_mujoco_stub, MuJoCoSimEngine.step, 1, "_world"),
            (_isaac_stub, IsaacSimulation.step, 2, "_world"),
            (_newton_stub, NewtonSimEngine.step, 1, "_model"),
        ):
            lock = CountingLock()
            stub = build(lock, batch=PROBE_BATCH)[0]
            lock._teardown_after, lock._hook = teardown_after, lambda s=stub, k=kill: setattr(s, k, None)
            texts.add(_text(run(stub, PROBE_BATCH * 3)))
        assert len(texts) == 1, texts

    def test_a_world_absent_before_the_call_still_reports_that(self) -> None:
        """The abort message must not displace the no-world message.

        ``step`` on an engine that never had a world says so; only a world lost
        *during* the call reports a partial count. Both directions matter: the
        abort text asserting "0 of N steps" would be a false statement about a
        teardown that never happened.
        """
        lock = CountingLock()
        stub, _ = _isaac_stub(lock)
        stub._world_created = False
        assert "No world created." in _text(IsaacSimulation.step(stub, 3))

        stub, _ = _mujoco_stub(lock)
        stub._world = None
        assert "aborting" not in _text(MuJoCoSimEngine.step(stub, 3))

        stub, _ = _newton_stub(lock)
        stub._model = None
        assert "Call create_world first" in _text(NewtonSimEngine.step(stub, 3))


# --------------------------------------------------------------------------- #
# Structural: a new backend cannot skip the batching                          #
# --------------------------------------------------------------------------- #
#: Guard names every concrete ``step`` must reference: the batch size it slices
#: on, and the shared refusal it aborts a lost world with.
_REQUIRED_STEP_GUARDS = ("_STEPS_PER_BATCH", "step_aborted_msg")

_KNOWN_STEP_SURFACES = ("isaac", "mujoco", "newton")


def _scan_step_surfaces(root: pathlib.Path) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    """Map backend dir -> guard names its ``step`` references, plus those adrift."""
    found: dict[str, tuple[str, ...]] = {}
    adrift: list[str] = []
    for path in sorted(root.glob("*/simulation.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step"):
                names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
                names |= {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
                present = tuple(g for g in _REQUIRED_STEP_GUARDS if g in names)
                found[path.parent.name] = present
                if len(present) != len(_REQUIRED_STEP_GUARDS):
                    missing = sorted(set(_REQUIRED_STEP_GUARDS) - set(present))
                    adrift.append(f"{path.parent.name}:{cls.name}.step missing {missing}")
    return found, adrift


class TestNoStepLockHoldDrifts:
    """A backend whose ``step`` advances a count must batch and re-check."""

    def test_every_backend_step_batches_and_rechecks(self) -> None:
        root = pathlib.Path(inspect.getfile(NewtonSimEngine)).parent.parent
        found, adrift = _scan_step_surfaces(root)
        assert adrift == [], "these hold the lock for the whole count: " + ", ".join(adrift)
        assert tuple(sorted(found)) == _KNOWN_STEP_SURFACES, f"the set of step surfaces changed: {sorted(found)}"

    def test_the_scanner_reports_a_planted_omission(self, tmp_path: pathlib.Path) -> None:
        """Without this, an empty ``adrift`` could mean a scanner matching nothing."""
        backend = tmp_path / "newton"
        backend.mkdir()
        (backend / "simulation.py").write_text(
            textwrap.dedent(
                """
                class Engine:
                    def step(self, n_steps=1):
                        with self._lock:
                            for _ in range(n_steps):
                                self._world.step()
                        return {"status": "success"}
                """
            ),
            encoding="utf-8",
        )
        found, adrift = _scan_step_surfaces(tmp_path)
        assert found == {"newton": ()}
        assert len(adrift) == 1
        assert "Engine.step" in adrift[0]

    def test_the_scanner_sees_a_planted_batched_step(self, tmp_path: pathlib.Path) -> None:
        """The other direction: a batched surface must not read as adrift."""
        backend = tmp_path / "newton"
        backend.mkdir()
        (backend / "simulation.py").write_text(
            textwrap.dedent(
                """
                class Engine:
                    def step(self, n_steps=1):
                        remaining = n_steps
                        while remaining > 0:
                            batch = min(remaining, self._STEPS_PER_BATCH)
                            with self._lock:
                                if self._world is None:
                                    return {"text": step_aborted_msg(0, n_steps)}
                                self._advance(batch)
                            remaining -= batch
                        return {"status": "success"}
                """
            ),
            encoding="utf-8",
        )
        found, adrift = _scan_step_surfaces(tmp_path)
        assert found == {"newton": _REQUIRED_STEP_GUARDS}
        assert adrift == []


# --------------------------------------------------------------------------- #
# The boundary: what this change deliberately does not decide                  #
# --------------------------------------------------------------------------- #
class TestModelIdentityStaysOutOfScope:
    """A mid-run recompile is still invisible to ``step``.

    Asserted rather than omitted so the gap cannot be mistaken for settled.
    ``_primitive_abort_reason`` catches this case by comparing model *identity*
    (``world._model is not model``), because a recompile swaps the model while
    leaving the world present. ``step``'s boundary check tests existence only,
    so a recompile between two batches continues the count on the new model
    under a success result.

    Not fixed here because it is a different failure - a swap rather than a
    teardown - with its own question: MuJoCo is the only backend with a
    recompile path, so unlike the lock hold there is no cross-backend contract
    to state. Replace this pin rather than delete it when that is settled.
    """

    def test_a_model_swapped_on_a_boundary_is_not_detected(self) -> None:
        lock = CountingLock()
        stub, calls = _mujoco_stub(lock, batch=PROBE_BATCH)
        original = stub._world._model
        lock._teardown_after = 1
        lock._hook = lambda: setattr(stub._world, "_model", object())

        result = MuJoCoSimEngine.step(stub, PROBE_BATCH * 3)

        assert result["status"] == "success", "current behaviour, not desired behaviour"
        assert calls["n"] == PROBE_BATCH * 3, "the count continued on the new model"
        assert stub._world._model is not original
