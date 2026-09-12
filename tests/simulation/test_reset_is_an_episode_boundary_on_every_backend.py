# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: ``reset`` closes the open dataset episode on every backend.

:meth:`~strands_robots.simulation.recording.DatasetRecordingMixin.save_episode`
states it as a fact about the class every engine inherits -- "``reset()`` is
itself an episode boundary while recording: it flushes buffered frames before
teleporting" -- and ``docs/recording.md`` repeats it without naming a backend.
Until this fix only MuJoCo did it. Measured on MuJoCo 3.10.0 with the flush
removed, one ``start_recording`` then two ten-step ``run_policy`` rollouts with
a ``reset()`` between them:

* ``reset()`` answered ``Reset to initial state.`` -- no episode written;
* ``stop_recording`` answered ``status="success"``, ``frame_count=20``,
  ``episode_count=1``, ``parquet_episode_count=1``,
  ``episode_count_mismatch=False``;
* ``meta/info.json`` held ``total_episodes=1``, ``total_frames=20``.

So both rollouts landed in ``episode_index=0`` as one 20-frame trajectory with
the reset teleport in the middle of it, and every reporting surface agreed: the
recorder and the parquet both counted one episode, so ``stop_recording``'s
author-versus-parquet gate had nothing to compare, and
``verify_dataset_episodes`` counts episodes rather than comparing them to the
rollouts that were run. The same script with the flush in place wrote
``total_episodes=2``, ``total_frames=20``, ten frames per episode.

The fix states the rule once, in
``DatasetRecordingMixin._flush_open_episode_before_reset``, and all three
``reset`` implementations ask it -- a copy per backend is what left two of them
without the boundary while the shared docstring promised it.

These tests need no GPU and neither ``newton``/``warp`` nor ``isaacsim``: the
flush runs before either backend touches a solver or the kit runtime, so the
unbound-instance stand-in pattern
(``tests/simulation/test_reserved_camera_name_at_creation.py``) reaches it. The
MuJoCo half drives a real engine and a real ``create_world``.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.simulation.recording import DatasetRecordingMixin

#: The concrete engines that inherit the rule. Literal rather than derived from
#: the package, so a backend dropped from the list fails
#: ``TestTheRuleHasOneOwner`` instead of quietly leaving the population.
BACKENDS = ("mujoco", "newton", "isaac")


class _FakeRecorder:
    """Stand-in for the LeRobot writer, shaped like the real one at the seam.

    ``save_episode`` is the only call the flush makes, and the real recorder
    answers it with the episode/frame counts the reset text quotes -- or an
    error, after which it has closed itself.
    """

    def __init__(self, pending: int = 0, *, fail: bool = False) -> None:
        self.episode_frame_count = pending
        self.frame_count = pending
        self.episode_count = 0
        self.save_calls = 0
        self._fail = fail

    def save_episode(self) -> dict[str, Any]:
        self.save_calls += 1
        if self._fail:
            return {"status": "error", "message": "parquet write refused"}
        self.episode_count += 1
        written = self.episode_frame_count
        self.episode_frame_count = 0
        return {
            "status": "success",
            "episode": self.episode_count,
            "episode_frames": written,
            "total_frames": self.frame_count,
        }


def _newton_engine() -> tuple[Any, dict[str, Any], list[bool]]:
    """A Newton engine whose reset reaches the flush without a solver."""
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    # Typed ``Any``: the engine is built without ``__init__``, so every attribute
    # below is injected rather than declared, and the reset under test reads them
    # through the same names the real constructor sets.
    engine: Any = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._lock = threading.RLock()
    engine._targets = {("robot", "joint"): 1.0}
    state: dict[str, Any] = {}
    engine._world = SimpleNamespace(sim_time=5.0, step_count=99, _backend_state=state)
    rebuilt: list[bool] = []

    def _rebuild() -> None:
        rebuilt.append(True)

    engine._rebuild = _rebuild
    return engine, state, rebuilt


def _isaac_engine() -> tuple[Any, dict[str, Any], list[bool]]:
    """An Isaac engine whose reset reaches the flush without the kit runtime."""
    from strands_robots.simulation.isaac.simulation import IsaacSimulation

    engine: Any = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._world_created = True
    engine._world = None
    state: dict[str, Any] = {}
    engine._recording_state_dict = state
    engine._sim_time = 5.0
    engine._step_count = 99
    engine._main_tid = threading.get_ident()
    rebuilt: list[bool] = []

    def _marshal(_method_name: str, fn: Any) -> Any:
        """Run the reset body here: the marshal targets the kit runtime, not it."""
        rebuilt.append(True)
        return fn()

    engine._marshal_main_thread_affine = _marshal
    return engine, state, rebuilt


def _mujoco_engine() -> tuple[Any, dict[str, Any], list[bool]]:
    """A real MuJoCo engine with a compiled world."""
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    engine: Any = MuJoCoSimEngine()
    assert engine.create_world()["status"] == "success"
    engine.step(2)
    return engine, engine._world._backend_state, []


ENGINES = {"mujoco": _mujoco_engine, "newton": _newton_engine, "isaac": _isaac_engine}


def _recording(state: dict[str, Any], recorder: _FakeRecorder) -> None:
    """Open a recording session in the engine's own state seam."""
    state["recording"] = True
    state["dataset_recorder"] = recorder
    state["trajectory"] = [object()]


@pytest.mark.parametrize("backend", BACKENDS)
class TestEveryBackendCutsTheBoundary:
    """The rule the shared docstring states, graded on each engine that inherits it."""

    def test_buffered_frames_are_flushed_as_their_own_episode(self, backend: str) -> None:
        """The frames of the rollout that just ended become one episode."""
        engine, state, _ = ENGINES[backend]()
        recorder = _FakeRecorder(pending=10)
        _recording(state, recorder)

        result = engine.reset()

        assert result["status"] == "success", result
        assert recorder.save_calls == 1, (
            f"{backend}: reset did not flush the open episode, so the next rollout appends "
            "to the same buffer and both land in episode_index=0"
        )
        assert recorder.episode_count == 1
        assert recorder.episode_frame_count == 0

    def test_the_reset_names_the_episode_it_wrote(self, backend: str) -> None:
        """An operator reading the reset sees the boundary it cut."""
        engine, state, _ = ENGINES[backend]()
        _recording(state, _FakeRecorder(pending=10))

        text = engine.reset()["content"][0]["text"]

        assert "Episode 1 saved -- 10 frames" in text, (backend, text)

    def test_a_failed_flush_is_reported_and_the_world_is_not_reset(self, backend: str) -> None:
        """A poisoned recorder stops the reset instead of being reset over."""
        engine, state, rebuilt = ENGINES[backend]()
        _recording(state, _FakeRecorder(pending=10, fail=True))
        # MuJoCo drives a real world, so the clock itself says whether the reset
        # happened; the two stand-ins record the rebuild they were asked for.
        before = engine._world._data.time if backend == "mujoco" else None

        result = engine.reset()

        assert result["status"] == "error", result
        assert "save_episode failed" in result["content"][0]["text"]
        assert rebuilt == [], f"{backend}: the world was re-initialized over a poisoned recorder"
        if before is not None:
            assert engine._world._data.time == before > 0.0

    def test_nothing_buffered_leaves_the_reset_as_it_was(self, backend: str) -> None:
        """Control: a reset not preceded by recorded frames is untouched."""
        engine, state, _ = ENGINES[backend]()
        recorder = _FakeRecorder(pending=0)
        _recording(state, recorder)

        result = engine.reset()

        assert result["status"] == "success", result
        assert recorder.save_calls == 0
        assert "Episode" not in result["content"][0]["text"]

    def test_a_reset_outside_a_recording_never_asks_the_recorder(self, backend: str) -> None:
        """Control: the boundary belongs to a recording, not to every reset."""
        engine, state, _ = ENGINES[backend]()
        recorder = _FakeRecorder(pending=10)
        state["recording"] = False
        state["dataset_recorder"] = recorder

        result = engine.reset()

        assert result["status"] == "success", result
        assert recorder.save_calls == 0


class TestTheRuleHasOneOwner:
    """One definition, asked by every reset - a copy per backend is the defect."""

    OWNER = "_flush_open_episode_before_reset"

    @staticmethod
    def _reset_source(backend: str) -> str:
        module = __import__(f"strands_robots.simulation.{backend}.simulation", fromlist=["x"])
        engine = next(
            cls
            for name, cls in vars(module).items()
            if isinstance(cls, type) and cls.__module__ == module.__name__ and "reset" in vars(cls)
        )
        return inspect.getsource(vars(engine)["reset"])

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_the_reset_asks_the_owner(self, backend: str) -> None:
        """Grade the call, not the text: a mention in prose is not a call."""
        tree = ast.parse(textwrap.dedent(self._reset_source(backend)))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert self.OWNER in called, (
            f"{backend}.reset does not ask {self.OWNER}, so its recording keeps buffering across the teleport"
        )

    def test_the_owner_exists_and_is_the_only_definition(self) -> None:
        """Non-vacuity: the cells above pass trivially if the name is gone."""
        assert hasattr(DatasetRecordingMixin, self.OWNER)
        root = Path(inspect.getfile(DatasetRecordingMixin)).parent
        defining = [path for path in root.rglob("*.py") if f"def {self.OWNER}" in path.read_text()]
        assert len(defining) == 1, f"the rule has {len(defining)} definitions: {defining}"

    def test_the_documented_claim_is_what_this_pins(self) -> None:
        """The premise: the shared docstring promises the boundary for every engine."""
        doc = inspect.getdoc(DatasetRecordingMixin.save_episode) or ""
        assert "is itself an episode boundary while recording" in doc


class TestIsaacRefusesAResetItCannotPerform:
    """Isaac's ``env_ids`` is refused, so there is no partial reset to except.

    This class replaces one that pinned the opposite - a partial reset leaving
    the recording buffer open, on the stated grounds that whether the recorded
    robot's rollout ended is not knowable from ``env_ids``. That reasoning was
    sound about a partial reset and the backend never performed one:
    ``world.reset()`` was called unconditionally and ``env_ids`` selected only
    the message. So the premise was false, and the behaviour it justified was
    the harmful half - the frames either side of a whole-world teleport were
    concatenated into one open episode, putting a physically impossible
    transition in a dataset with nothing raised. The conclusion does not survive
    for another reason, which is why this is a replacement and not a rename.
    """

    def test_env_ids_is_refused(self) -> None:
        engine, state, _ = _isaac_engine()

        result = engine.reset(env_ids=[0])

        assert result["status"] == "error", result
        assert "env_ids is not supported" in result["content"][0]["text"]

    def test_a_refused_reset_leaves_the_recording_untouched(self) -> None:
        """Refused ahead of every side effect: no flush, and no discard either."""
        engine, state, _ = _isaac_engine()
        recorder = _FakeRecorder(pending=10)
        _recording(state, recorder)

        result = engine.reset(env_ids=[0])

        assert result["status"] == "error", result
        assert recorder.save_calls == 0
        assert recorder.episode_frame_count == 10

    def test_an_empty_selection_is_refused_too(self) -> None:
        """``[]`` is a selection that names nothing, not an absent one - the
        subset-selector rule. Read by truthiness it would reach the whole-world
        reset and be reported as the caller's own request."""
        engine, state, _ = _isaac_engine()

        assert engine.reset(env_ids=[])["status"] == "error"

    def test_the_whole_world_reset_is_still_a_boundary(self) -> None:
        """The control: removing the branch must not cost the flush."""
        engine, state, _ = _isaac_engine()
        recorder = _FakeRecorder(pending=10)
        _recording(state, recorder)

        result = engine.reset()

        assert result["status"] == "success", result
        assert recorder.save_calls == 1
