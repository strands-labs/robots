"""A resumed recording writes the base it observed, not a base at the origin.

``DatasetRecorder.create`` takes ``extra_state_specs`` so a floating-base robot's
dataset carries its base columns: ``base_pos`` / ``base_quat`` /
``base_lin_vel`` / ``base_ang_vel`` arrive in the observation as VECTORS, and the
schema declares them expanded per component (``base_pos.x`` ...). ``add_frame``
bridges the two by reading the SOURCE keys and flattening each in schema order,
which it can only do because ``create`` records those sources on the recorder as
``_state_source_keys``.

``resume`` did not, and could not derive them: it inherits the expanded column
names from the dataset on disk, and nothing there says which source key a run of
components was flattened from. So the fallback read the expanded names,
``observation.get("base_pos.x")`` answered ``None`` for every one of them, and
the zero-fill beside it supplied ``0.0`` per component, per frame.

Nothing raised and nothing logged. The flattened vector still had the schema's
length, so no width check could see it; and the simulation backends' recording
hooks all pass ``required_action_keys``, which is what disables the
missing-state-column refusal that would otherwise have caught a ``None``. Every
appended episode therefore recorded the base at the origin, at rest, under
``status: success`` - the "zero pose the robot is not in" corruption
``undriven_robot_state`` names, and precisely the base-blind dataset
``extra_state_specs`` exists to prevent, reintroduced on the append path.

Measured before the fix, driving ``DatasetRecorder`` directly with no simulation
backend involved: an observation carrying ``base_pos=[1.0, 2.0, 9.0]`` recorded
``[0.0, 0.0, 0.0]``, while the joint columns beside it recorded correctly. That
asymmetry is what kept it invisible - a spot check of a resumed dataset shows
live joints.

All three simulation backends pass ``extra_state_specs`` to ``create`` and call
``resume``, so this is graded on the shared recorder rather than per backend, and
the parameter is passed by all three. The per-backend half is
:class:`TestEveryBackendPassesTheSourcesOnResume`, derived from the source so a
fourth backend is held to it on arrival.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

pytest.importorskip("lerobot")

from strands_robots.dataset_recorder import DatasetRecorder  # noqa: E402

#: A two-joint arm on a floating base: the scalar sources, then the vector ones.
JOINTS = ["j0", "j1"]
SPECS: list[tuple[str, list[str]]] = [
    ("base_pos", ["x", "y", "z"]),
    ("base_quat", ["w", "x", "y", "z"]),
]

#: The z the appended frame observes. Deliberately not 0.0, and not the z the
#: created episode used, so a zero-fill and a stale first-episode value are both
#: distinguishable from the real reading.
APPENDED_Z = 9.0


def _observation(z: float) -> dict[str, Any]:
    return {
        "j0": 0.1,
        "j1": 0.2,
        "base_pos": [1.0, 2.0, z],
        "base_quat": [1.0, 0.0, 0.0, 0.0],
    }


def _state_matrix(dataset_dir: pathlib.Path) -> tuple[np.ndarray, list[str]]:
    """Every recorded ``observation.state`` row, and the column names."""
    pandas = pytest.importorskip("pandas")
    frames = [pandas.read_parquet(parquet) for parquet in sorted((dataset_dir / "data").rglob("*.parquet"))]
    table = pandas.concat(frames, ignore_index=True)
    return np.stack(table["observation.state"].to_numpy()), _column_names(dataset_dir)


def _column_names(dataset_dir: pathlib.Path) -> list[str]:
    import json

    info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    return list(info["features"]["observation.state"]["names"])


def _record_then_append(root: pathlib.Path, *, pass_sources: bool) -> tuple[np.ndarray, list[str]]:
    """One created episode, then one appended through ``resume``.

    ``pass_sources`` is the fix under test: with it ``False`` this reproduces the
    defect, which is what makes the assertions below non-vacuous.
    """
    dataset_dir = root / "dataset"
    recorder = DatasetRecorder.create(
        repo_id="local/base_resume_probe",
        fps=30,
        robot_type="probe",
        joint_names=JOINTS,
        action_names=JOINTS,
        extra_state_specs=SPECS,
        camera_keys=[],
        camera_dims={},
        task="probe",
        root=str(dataset_dir),
        use_videos=False,
    )
    for _ in range(3):
        recorder.add_frame(_observation(0.5), {"j0": 0.0, "j1": 0.0}, required_action_keys=JOINTS)
    recorder.save_episode()
    recorder.finalize()

    resume_kwargs: dict[str, Any] = {"repo_id": "local/base_resume_probe", "root": str(dataset_dir), "task": "probe"}
    if pass_sources:
        resume_kwargs["joint_names"] = JOINTS
        resume_kwargs["extra_state_specs"] = SPECS
    resumed = DatasetRecorder.resume(**resume_kwargs)
    for _ in range(3):
        resumed.add_frame(_observation(APPENDED_Z), {"j0": 0.0, "j1": 0.0}, required_action_keys=JOINTS)
    resumed.save_episode()
    resumed.finalize()
    return _state_matrix(dataset_dir)


def _base_columns(names: list[str]) -> list[int]:
    return [i for i, name in enumerate(names) if name.startswith("base_")]


class TestTheAppendedEpisodeCarriesTheObservedBase:
    def test_the_base_columns_are_not_all_zero(self, tmp_path: pathlib.Path) -> None:
        """The headline. Every base component zero is the corruption itself."""
        state, names = _record_then_append(tmp_path, pass_sources=True)

        appended = state[3]
        base = appended[_base_columns(names)]

        assert not np.all(base == 0.0), f"the appended episode recorded a base at the origin: {base}"

    def test_it_records_the_z_it_observed(self, tmp_path: pathlib.Path) -> None:
        """Not merely non-zero: the value the observation actually carried."""
        state, names = _record_then_append(tmp_path, pass_sources=True)

        z_index = names.index("base_pos.z")

        assert state[3][z_index] == pytest.approx(APPENDED_Z)

    def test_the_whole_vector_round_trips(self, tmp_path: pathlib.Path) -> None:
        state, names = _record_then_append(tmp_path, pass_sources=True)

        appended = state[3]
        by_name = {name: appended[i] for i, name in enumerate(names)}

        assert by_name["base_pos.x"] == pytest.approx(1.0)
        assert by_name["base_pos.y"] == pytest.approx(2.0)
        assert by_name["base_pos.z"] == pytest.approx(APPENDED_Z)
        assert by_name["base_quat.w"] == pytest.approx(1.0)

    def test_the_joint_columns_were_never_the_problem(self, tmp_path: pathlib.Path) -> None:
        """Stated so the asymmetry that hid this is on the record: the scalar
        columns recorded correctly throughout, which is why a spot check of a
        resumed dataset looked healthy."""
        state, names = _record_then_append(tmp_path, pass_sources=True)

        assert state[3][names.index("j0")] == pytest.approx(0.1)
        assert state[3][names.index("j1")] == pytest.approx(0.2)

    def test_the_created_episode_is_unchanged(self, tmp_path: pathlib.Path) -> None:
        """The control: ``create`` always worked, and must keep working."""
        state, names = _record_then_append(tmp_path, pass_sources=True)

        assert state[0][names.index("base_pos.z")] == pytest.approx(0.5)


class TestTheDefectReproducesWithoutTheSources:
    """The mutation, expressed as a test rather than performed by hand.

    Without the source keys the appended base columns are ALL zero while the
    joints beside them are correct - so if a later change stops passing them,
    these two cells are what says the corruption is back rather than a schema
    error somewhere.
    """

    def test_omitting_the_sources_records_zeros(self, tmp_path: pathlib.Path) -> None:
        state, names = _record_then_append(tmp_path, pass_sources=False)

        base = state[3][_base_columns(names)]

        assert np.all(base == 0.0), f"expected the pre-fix zero-fill, got {base}"

    def test_and_the_joints_are_still_right(self, tmp_path: pathlib.Path) -> None:
        state, names = _record_then_append(tmp_path, pass_sources=False)

        assert state[3][names.index("j0")] == pytest.approx(0.1)


class TestResumeCarriesTheSourceKeys:
    """The mechanism, so a rename cannot quietly detach the fix from its reason."""

    def test_resume_accepts_the_two_parameters_create_takes(self) -> None:
        import inspect

        params = inspect.signature(DatasetRecorder.resume).parameters

        assert "joint_names" in params
        assert "extra_state_specs" in params

    def test_the_resumed_recorder_knows_its_sources(self, tmp_path: pathlib.Path) -> None:
        dataset_dir = tmp_path / "dataset"
        recorder = DatasetRecorder.create(
            repo_id="local/sources_probe",
            fps=30,
            robot_type="probe",
            joint_names=JOINTS,
            action_names=JOINTS,
            extra_state_specs=SPECS,
            camera_keys=[],
            camera_dims={},
            task="probe",
            root=str(dataset_dir),
            use_videos=False,
        )
        recorder.add_frame(_observation(0.5), {"j0": 0.0, "j1": 0.0}, required_action_keys=JOINTS)
        recorder.save_episode()
        recorder.finalize()

        resumed = DatasetRecorder.resume(
            repo_id="local/sources_probe",
            root=str(dataset_dir),
            task="probe",
            joint_names=JOINTS,
            extra_state_specs=SPECS,
        )

        assert resumed._state_source_keys == ["j0", "j1", "base_pos", "base_quat"]

    def test_a_dataset_with_no_vector_columns_is_unaffected(self, tmp_path: pathlib.Path) -> None:
        """``extra_state_specs`` absent leaves the sources unset, which is the
        behaviour every scalar-only recording already relied on."""
        dataset_dir = tmp_path / "dataset"
        recorder = DatasetRecorder.create(
            repo_id="local/scalar_probe",
            fps=30,
            robot_type="probe",
            joint_names=JOINTS,
            action_names=JOINTS,
            camera_keys=[],
            camera_dims={},
            task="probe",
            root=str(dataset_dir),
            use_videos=False,
        )
        recorder.add_frame({"j0": 0.1, "j1": 0.2}, {"j0": 0.0, "j1": 0.0}, required_action_keys=JOINTS)
        recorder.save_episode()
        recorder.finalize()

        resumed = DatasetRecorder.resume(
            repo_id="local/scalar_probe", root=str(dataset_dir), task="probe", joint_names=JOINTS
        )

        assert resumed._state_source_keys is None


class TestEveryBackendPassesTheSourcesOnResume:
    """Derived from the source, because the defect is a call site that omits an
    argument - which no behavioural test of one backend can see on the others.

    All three pass ``extra_state_specs`` to ``create`` and all three call
    ``resume``, so all three owed this.
    """

    @pytest.mark.parametrize("backend", ["isaac", "mujoco", "newton"])
    def test_the_resume_call_carries_the_specs(self, backend: str) -> None:
        source = (
            pathlib.Path(__file__).resolve().parents[1] / "strands_robots" / "simulation" / backend / "recording.py"
        ).read_text(encoding="utf-8")

        assert "extra_state_specs" in source, f"{backend} declares no base columns at all"
        resume_at = source.index(".resume(")
        window = source[resume_at : resume_at + 400]
        assert "extra_state_specs" in window, (
            f"{backend}'s resume() call omits extra_state_specs, so its appended floating-base episodes record zeros"
        )
        assert "joint_names" in window, f"{backend}'s resume() call omits joint_names"
