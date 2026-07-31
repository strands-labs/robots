"""The finalizer's report must mean what it says, on every engine.

Two halves of one contract were already agreed in the tree, each pinned in a
place that could not see the other:

* a partially-built engine must **not** log a cleanup error during ``__del__``
  (pinned for Newton in ``tests/simulation/newton``), and
* a genuine cleanup failure **must** be logged (pinned for the base facade in
  ``tests/simulation``).

Neither held everywhere. ``SimEngine.__del__`` called ``cleanup()`` on any
instance, so on one that never finished ``__init__`` it reported whichever
attribute construction had not reached yet as a cleanup failure - noise
indistinguishable from a real leak, and a red herring for whoever read it.
``MuJoCoSimEngine`` avoided that noise by overriding ``__del__`` to swallow
every exception silently, which also discarded real failures, so the base
class's own pinned guarantee did not hold for the default backend. And
``IsaacSimulation.__repr__`` raised on such an instance, so a traceback or a
failing assertion rendered ``[AttributeError ... raised in repr()]`` naming an
attribute the failure did not turn on.

These tests pin the contract in both directions: nothing is reported for an
instance that never acquired anything, everything is reported for one that
failed to release something, and ``repr`` never raises. The parity check at the
bottom keeps a future backend from shipping without declaring construction
complete.
"""

from __future__ import annotations

import ast
import gc
import inspect
import logging
import pathlib
from typing import Any

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

_BASE_LOGGER = "strands_robots.simulation.base"
_CLEANUP_WARNING = "Cleanup error during __del__"


def _cleanup_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every finalizer cleanup-failure warning captured so far."""
    return [rec.getMessage() for rec in caplog.records if _CLEANUP_WARNING in rec.getMessage()]


class _Engine(SimEngine):
    """Minimal concrete engine that records whether ``cleanup`` was reached.

    Mirrors a real backend's lifecycle: ``_init_complete`` is assigned as the
    final statement of ``__init__``, so the finalizer treats a fully
    constructed instance as owning resources.

    Args:
        cleanup_raises: Make ``cleanup`` fail, standing in for an engine that
            acquired something and could not release it.
        declare_complete: Assign the sentinel. ``False`` leaves it unassigned,
            which is what a real backend looks like when ``__init__`` raises
            before reaching its final statement: every earlier field is set and
            the sentinel never is, so the class-level floor is what the
            finalizer reads.
    """

    def __init__(self, *, cleanup_raises: bool = False, declare_complete: bool = True) -> None:
        self.cleanup_calls = 0
        self._cleanup_raises = cleanup_raises
        if declare_complete:
            self._init_complete = True

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self._cleanup_raises:
            raise RuntimeError("release failed")

    # Abstract surface - none of it is exercised here. The four methods with a
    # wide parameter list take ``*args``/``**kwargs`` rather than restating the
    # ABC's signature: they exist only to make the class concrete.
    def create_world(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def destroy(self) -> dict[str, Any]:
        return {"status": "success"}

    def reset(self) -> dict[str, Any]:
        return {"status": "success"}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        return {"status": "success"}

    def get_state(self) -> dict[str, Any]:
        return {"sim_time": 0.0, "step_count": 0}

    def add_robot(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def remove_robot(self, name: str) -> dict[str, Any]:
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return []

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return []

    def add_object(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def remove_object(self, name: str) -> dict[str, Any]:
        return {"status": "success"}

    def get_observation(self, robot_name=None, skip_images: bool = False) -> dict[str, Any]:
        return {}

    def send_action(self, action, robot_name=None, n_substeps: int = 1) -> dict[str, Any]:
        return {"status": "success"}

    def render(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}


class TestNothingIsReportedForAnEngineThatNeverAcquiredAnything:
    def test_the_class_attribute_is_the_floor_so_the_read_never_raises(self) -> None:
        """``__del__`` reads ``_init_complete`` before anything else.

        Declaring it on the class means the read resolves even on an instance
        whose ``__init__`` never ran, so the finalizer cannot itself raise the
        AttributeError it exists to stop reporting.
        """
        assert SimEngine._init_complete is False
        assert IsaacSimulation.__new__(IsaacSimulation)._init_complete is False

    def test_a_skeleton_instance_logs_no_cleanup_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = _Engine.__new__(_Engine)
        with caplog.at_level(logging.WARNING, logger=_BASE_LOGGER):
            del engine
            gc.collect()
        assert _cleanup_warnings(caplog) == []

    def test_a_skeleton_instance_is_not_offered_to_cleanup_at_all(self) -> None:
        """The finalizer skips, rather than calling ``cleanup`` and silencing it.

        Silencing would also discard a real failure; skipping keeps the channel
        meaningful. Observing the call count is what tells the two apart.
        """
        calls: list[int] = []

        class _Recording(_Engine):
            def cleanup(self) -> None:
                calls.append(1)

        engine = _Recording.__new__(_Recording)
        del engine
        gc.collect()
        assert calls == []

    def test_an_engine_that_never_declares_construction_complete_is_not_finalized(self) -> None:
        """The flag is the contract, and it is opt-in.

        An engine that never sets it is treated as never having acquired
        anything, however far its ``__init__`` otherwise got. The parity check
        below is what holds this package's own engines to declaring it.
        """
        calls: list[int] = []

        class _Recording(_Engine):
            def cleanup(self) -> None:
                calls.append(1)

        engine = _Recording(declare_complete=False)
        # Never assigned, rather than assigned False: the read resolves to the
        # class-level floor, which is the state a backend is left in when
        # ``__init__`` raises before its final statement.
        assert "_init_complete" not in vars(engine)
        del engine
        gc.collect()
        assert calls == []


class TestEverythingIsReportedForAnEngineThatFailedToRelease:
    def test_a_constructed_engine_is_still_cleaned_up_on_gc(self) -> None:
        """The safety net must survive the fix - skipping is for skeletons only."""
        calls: list[int] = []

        class _Recording(_Engine):
            def cleanup(self) -> None:
                calls.append(1)

        engine = _Recording()
        del engine
        gc.collect()
        assert calls == [1]

    def test_a_failing_cleanup_on_a_constructed_engine_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = _Engine(cleanup_raises=True)
        with caplog.at_level(logging.WARNING, logger=_BASE_LOGGER):
            del engine
            gc.collect()
        assert any("release failed" in message for message in _cleanup_warnings(caplog))


class TestMuJoCoUsesTheSharedFinalizer:
    """The default backend must obey the base class's own pinned guarantees.

    It previously overrode ``__del__`` to swallow every exception, so a real
    cleanup failure had no channel at all - the base facade's pin could not see
    that because it exercises a direct ``SimEngine`` subclass.
    """

    def test_a_real_cleanup_failure_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        class _Exploding(MuJoCoSimEngine):
            def cleanup(self, policy_stop_timeout: float | None = None) -> None:
                raise RuntimeError("mujoco release failed")

        engine = _Exploding(tool_name="finalizer_probe")
        with caplog.at_level(logging.WARNING, logger=_BASE_LOGGER):
            del engine
            gc.collect()
        assert any("mujoco release failed" in message for message in _cleanup_warnings(caplog))

    def test_a_skeleton_logs_nothing_and_is_not_offered_to_cleanup(self, caplog: pytest.LogCaptureFixture) -> None:
        """Its own ``cleanup()`` does raise on a skeleton, so this is the skip.

        Asserting both halves separates "skipped" from "attempted and silenced".
        """
        skeleton = MuJoCoSimEngine.__new__(MuJoCoSimEngine)
        with pytest.raises(AttributeError):
            skeleton.cleanup()

        calls: list[int] = []

        class _Recording(MuJoCoSimEngine):
            def cleanup(self, policy_stop_timeout: float | None = None) -> None:
                calls.append(1)

        engine = _Recording.__new__(_Recording)
        with caplog.at_level(logging.WARNING, logger=_BASE_LOGGER):
            del engine
            gc.collect()
        assert _cleanup_warnings(caplog) == []
        assert calls == []


class TestIsaacReportsNoBogusCleanupFailure:
    def test_a_rejected_kwarg_does_not_also_produce_a_cleanup_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A plain caller typo reaches this, with no test skeleton involved.

        ``IsaacSimulation.__init__`` rejects an unknown kwarg deliberately, and
        that rejection happens before the attributes ``cleanup()`` reads are
        assigned. The TypeError is the whole story; a second warning naming an
        unrelated attribute is not part of it.
        """
        with caplog.at_level(logging.WARNING, logger=_BASE_LOGGER):
            try:
                IsaacSimulation(headles=False)
            except TypeError as exc:
                assert "headles" in str(exc)
            else:
                pytest.fail("expected IsaacSimulation to reject the unknown kwarg")
            # The bound exception is dropped when the except clause ends, which
            # releases the traceback holding the half-built instance.
            gc.collect()
        assert _cleanup_warnings(caplog) == []

    def test_a_skeleton_instance_logs_no_cleanup_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = IsaacSimulation.__new__(IsaacSimulation)
        with caplog.at_level(logging.WARNING, logger=_BASE_LOGGER):
            del engine
            gc.collect()
        assert _cleanup_warnings(caplog) == []


class TestIsaacReprNeverHidesAFailure:
    def test_repr_of_a_half_built_instance_does_not_raise(self) -> None:
        text = repr(IsaacSimulation.__new__(IsaacSimulation))
        assert "partially constructed" in text

    def test_repr_of_a_half_built_instance_names_no_attribute(self) -> None:
        """Naming one sends the reader after a red herring.

        The lifecycle fact is what is relevant; the attribute construction
        happened to stop before is not.
        """
        text = repr(IsaacSimulation.__new__(IsaacSimulation))
        assert "_config" not in text
        assert "_world_created" not in text
        assert "AttributeError" not in text

    def test_repr_still_describes_an_instance_that_has_its_state(self) -> None:
        """The tolerance must not cost the informative form."""
        engine = IsaacSimulation.__new__(IsaacSimulation)
        engine._config = type("_Cfg", (), {"num_envs": 4, "device": "cpu", "headless": True})()
        engine._world_created = False
        text = repr(engine)
        assert "num_envs=4" in text
        assert "device='cpu'" in text
        assert "world=none" in text


# Structural parity: no backend ships without declaring construction complete.

_SENTINEL = "_init_complete"


def _engines_missing_the_sentinel(source: str) -> list[str]:
    """Names of ``SimEngine`` subclasses in ``source`` that do not declare it.

    A class qualifies when its ``__init__`` does not end with
    ``self._init_complete = True``. Requiring it *last* is the point: the flag
    means "every statement of ``__init__`` ran", so an earlier position would
    let a later fallible step leave it claiming more than happened.
    """
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {ast.unparse(base).split(".")[-1] for base in node.bases}
        if "SimEngine" not in bases:
            continue
        init = next(
            (m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"),
            None,
        )
        if init is None:
            offenders.append(f"{node.name} (no __init__)")
            continue
        last = init.body[-1]
        ok = (
            isinstance(last, ast.Assign)
            and len(last.targets) == 1
            and isinstance(last.targets[0], ast.Attribute)
            and last.targets[0].attr == _SENTINEL
            and isinstance(last.value, ast.Constant)
            and last.value.value is True
        )
        if not ok:
            offenders.append(f"{node.name} (last statement is {ast.unparse(last)[:60]!r})")
    return offenders


def _engine_sources() -> dict[str, str]:
    """Every ``<backend>/simulation.py`` beside the ABC, keyed by backend."""
    root = pathlib.Path(inspect.getfile(SimEngine)).parent
    return {path.parent.name: path.read_text(encoding="utf-8") for path in sorted(root.glob("*/simulation.py"))}


class TestEveryBackendDeclaresConstructionComplete:
    def test_the_known_backends_are_all_scanned(self) -> None:
        """An empty scan would pass the check below vacuously."""
        assert {"mujoco", "isaac", "newton"} <= set(_engine_sources())

    @pytest.mark.parametrize("backend", sorted(_engine_sources()))
    def test_backend_sets_the_sentinel_last_in_init(self, backend: str) -> None:
        offenders = _engines_missing_the_sentinel(_engine_sources()[backend])
        assert offenders == [], (
            f"{backend}: {offenders} - set `self.{_SENTINEL} = True` as the final "
            f"statement of __init__, or SimEngine.__del__ will treat the engine "
            f"as holding no resources and skip its cleanup"
        )

    def test_the_scan_detects_a_planted_omission(self) -> None:
        """A scanner that silently matches nothing would look like a clean tree."""
        missing = "class Broken(SimEngine):\n    def __init__(self) -> None:\n        self._world = None\n"
        not_last = (
            "class Late(SimEngine):\n"
            "    def __init__(self) -> None:\n"
            "        self._init_complete = True\n"
            "        self._world = None\n"
        )
        assert [name.split()[0] for name in _engines_missing_the_sentinel(missing)] == ["Broken"]
        assert [name.split()[0] for name in _engines_missing_the_sentinel(not_last)] == ["Late"]

    def test_a_compliant_engine_passes_the_scan(self) -> None:
        compliant = (
            "class Fine(SimEngine):\n"
            "    def __init__(self) -> None:\n"
            "        self._world = None\n"
            "        self._init_complete = True\n"
        )
        assert _engines_missing_the_sentinel(compliant) == []
