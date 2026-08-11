"""Every backend must *report* each way the dataset stack can be unavailable.

``start_recording`` writes a LeRobotDataset, so before it builds one it resolves
the recorder out of :mod:`strands_robots.dataset_recorder` and reports failure
rather than raising. Three different things can go wrong there, and each one has
a *different remedy*:

============================================  ==================================
cause                                         what the caller has to do
============================================  ==================================
the lerobot extra is absent                    install the extra the message names
:mod:`strands_robots.dataset_recorder` did     repair a partial / drifted
not import                                     **strands-robots** install
the module imported but supplied no            same, on a module that loaded but
``DatasetRecorder``                            carries no recorder symbol
============================================  ==================================

All three backends carry the same block, so there are nine cells. Three had ever
run: MuJoCo drove the absent extra and the failed import, Newton drove the absent
extra, and Isaac drove none of them - its unavailability report had never
executed at all. A collapse of the three into one message would leave every
caller with the same instruction for three different faults, and nothing failed.

This module drives all nine, so each cause keeps its own diagnosis on each
backend, and adds the two properties that make those diagnoses worth having:
they stay pairwise distinct, and the fallback each one recommends really exists
on the backend that recommended it (Newton has no ``start_cameras_recording``
and correctly points at ``run_policy(video=...)`` instead).

Complete accounting for the block: the ``fps`` / posture / ``cameras`` refusals
above it are owned by
``tests/simulation/test_recording_preflight_refusals_across_backends.py``, and
the rate refusal between them is provably unreachable on Newton and Isaac for
the reason that module records.

The Isaac and Newton skeletons are the ones that module already builds with
``__new__``; this block runs before either backend's lock, so neither the Isaac
Sim Kit runtime nor the Newton/Warp stack is needed. Only the MuJoCo cells need
a compiled model, so that factory alone is gated.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from strands_robots import dataset_recorder
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.newton.simulation import NewtonSimEngine

from .test_recording_preflight_refusals_across_backends import _isaac_engine, _newton_engine

# Verbatim copy of the diagnosis ``lerobot_dataset_import_error`` produces when
# the extra is missing, so the "surfaced verbatim" assertions below compare
# against what a caller really sees.
_EXTRA_ABSENT_REASON = (
    "lerobot is not installed (ModuleNotFoundError: No module named 'lerobot'). "
    "Install lerobot >= 0.6.0 with: pip install 'strands-robots[lerobot]'"
)
_MODULE_DID_NOT_IMPORT = "strands_robots.dataset_recorder is unavailable ("
_NO_RECORDER_SYMBOL = "strands_robots.dataset_recorder did not provide DatasetRecorder."


# --- the three causes, as monkeypatches -------------------------------------


def _absent_lerobot_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extra is cleanly missing: the probe returns a reason."""
    monkeypatch.setattr(dataset_recorder, "lerobot_dataset_import_error", lambda: _EXTRA_ABSENT_REASON)


def _module_did_not_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial install: the in-function ``from ... import`` raises ImportError."""
    monkeypatch.delattr(dataset_recorder, "DatasetRecorder", raising=False)


def _module_supplied_no_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    """A drifted install: the module imports but its recorder symbol is ``None``.

    The probe is pinned to "no reason" as well, because it wins the precedence
    above this branch - without that, the case would only reach the branch on an
    install that happens to have lerobot, and the contract has to hold on both.
    """
    monkeypatch.setattr(dataset_recorder, "lerobot_dataset_import_error", lambda: None)
    monkeypatch.setattr(dataset_recorder, "DatasetRecorder", None)


# The table is typed rather than read back off the ``pytest.param`` objects,
# whose recorded values are plain ``object`` - the distinctness test calls these.
_CAUSE_TABLE: list[tuple[Callable[[pytest.MonkeyPatch], None], str]] = [
    (_absent_lerobot_extra, _EXTRA_ABSENT_REASON),
    (_module_did_not_import, _MODULE_DID_NOT_IMPORT),
    (_module_supplied_no_recorder, _NO_RECORDER_SYMBOL),
]
CAUSES = [
    pytest.param(apply_cause, marker, id=apply_cause.__name__.strip("_").replace("_", "-"))
    for apply_cause, marker in _CAUSE_TABLE
]


# --- one engine per backend -------------------------------------------------


@contextlib.contextmanager
def _mujoco_engine() -> Iterator[Any]:
    """A real MuJoCo engine over an empty world - the only cell needing a model."""
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    sim = Simulation(tool_name="dataset_stack_probe", mesh=False)
    sim.create_world()
    try:
        yield sim
    finally:
        sim.cleanup()


@contextlib.contextmanager
def _newton_skeleton() -> Iterator[Any]:
    """No Newton/Warp stack was built, so there is nothing to release."""
    yield _newton_engine()


@contextlib.contextmanager
def _isaac_skeleton() -> Iterator[Any]:
    """No Isaac Sim Kit runtime was started, so there is nothing to release."""
    yield _isaac_engine()


BACKENDS = [
    pytest.param(_mujoco_engine, id="mujoco"),
    pytest.param(_newton_skeleton, id="newton"),
    pytest.param(_isaac_skeleton, id="isaac"),
]


def _start(engine: Any, root: Path) -> dict[str, Any]:
    return dict(engine.start_recording(repo_id="local/dataset_stack_probe", root=str(root)))


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


# --- the mechanisms really produce the states they model ---------------------


class TestTheThreeCausesAreRealStates:
    """Non-vacuity: each monkeypatch produces the interpreter state it claims."""

    def test_deleting_the_symbol_makes_the_in_function_import_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _module_did_not_import(monkeypatch)

        with pytest.raises(ImportError):
            from strands_robots.dataset_recorder import DatasetRecorder  # noqa: F401

    def test_a_none_symbol_makes_the_in_function_import_bind_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``from ... import`` is a ``getattr``, so a ``None`` attribute binds ``None``."""
        _module_supplied_no_recorder(monkeypatch)

        from strands_robots.dataset_recorder import DatasetRecorder as bound

        assert bound is None

    def test_the_absent_extra_reason_is_what_the_probe_reports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _absent_lerobot_extra(monkeypatch)

        assert dataset_recorder.lerobot_dataset_import_error() == _EXTRA_ABSENT_REASON


# --- every backend reports every cause --------------------------------------


class TestEveryBackendReportsEveryCause:
    """Nine cells: three causes on three backends, each reported not raised."""

    @pytest.mark.parametrize("factory", BACKENDS)
    @pytest.mark.parametrize(("apply_cause", "marker"), CAUSES)
    def test_the_call_is_refused_and_names_the_cause(
        self,
        factory: Any,
        apply_cause: Any,
        marker: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        apply_cause(monkeypatch)
        with factory() as engine:
            result = _start(engine, tmp_path / "dataset")

        assert result["status"] == "error"
        assert marker in _text(result)

    @pytest.mark.parametrize("factory", BACKENDS)
    @pytest.mark.parametrize(("apply_cause", "marker"), CAUSES)
    def test_the_report_names_the_dataset_stack_it_needs(
        self,
        factory: Any,
        apply_cause: Any,
        marker: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Whatever the cause, the report says which capability is unavailable."""
        apply_cause(monkeypatch)
        with factory() as engine:
            text = _text(_start(engine, tmp_path / "dataset"))

        assert "lerobot's dataset stack" in text


class TestTheThreeCausesGetThreeDifferentDiagnoses:
    """Three faults, three remedies - so one message for all three is wrong."""

    @pytest.mark.parametrize("factory", BACKENDS)
    def test_the_three_reports_are_pairwise_distinct(
        self, factory: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        texts: list[str] = []
        for index, (apply_cause, _marker) in enumerate(_CAUSE_TABLE):
            with monkeypatch.context() as patch:
                apply_cause(patch)
                with factory() as engine:
                    texts.append(_text(_start(engine, tmp_path / f"dataset{index}")))

        assert len(set(texts)) == len(CAUSES), texts


class TestTheRecommendedFallbackExistsOnThatBackend:
    """The plain-MP4 route each report names must be reachable on that backend.

    Newton has no ``start_cameras_recording`` and points at ``run_policy``'s
    ``video=`` instead; a message copied from a sibling would name a method the
    caller cannot call.
    """

    @pytest.mark.parametrize("factory", BACKENDS)
    def test_the_named_fallback_is_a_real_surface(
        self, factory: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _absent_lerobot_extra(monkeypatch)
        with factory() as engine:
            text = _text(_start(engine, tmp_path / "dataset"))
            if "start_cameras_recording" in text:
                assert callable(getattr(engine, "start_cameras_recording", None))
            else:
                assert "video=" in text
                assert "video" in inspect.signature(engine.run_policy).parameters

    def test_newton_has_no_cameras_recording_so_its_advice_has_to_differ(self) -> None:
        """The premise behind the branch above, pinned so it cannot drift silently."""
        assert getattr(NewtonSimEngine, "start_cameras_recording", None) is None
        assert callable(getattr(IsaacSimulation, "start_cameras_recording", None))


class TestARefusedStartRecordsNothing:
    """An unavailable dataset stack leaves no session and no directory."""

    @pytest.mark.parametrize("factory", BACKENDS)
    @pytest.mark.parametrize(("apply_cause", "marker"), CAUSES)
    def test_no_session_is_opened_and_no_root_is_created(
        self,
        factory: Any,
        apply_cause: Any,
        marker: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        apply_cause(monkeypatch)
        root = tmp_path / "dataset"
        with factory() as engine:
            assert _start(engine, root)["status"] == "error"
            assert engine._is_recording() is False
            assert engine._active_recorder() is None

        assert not root.exists()
