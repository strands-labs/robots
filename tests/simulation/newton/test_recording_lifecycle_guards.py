"""Newton recording lifecycle guards: the error, guard, and no-op paths.

The happy path (start -> capture -> save_episode -> parquet) is covered by
``test_dataset_recording``. This module pins the surrounding contracts that a
happy-path recording never exercises:

* ``start_recording`` fails loudly and actionably when the ``lerobot`` extra is
  missing, instead of dead-ending later in dataset creation.
* A recorder-creation failure resets ``recording`` to False (so a subsequent
  attempt is not wedged "recording" with no recorder) and returns an error
  rather than raising past the tool boundary.
* The default ``root=None`` resolves the on-disk dataset dir from ``repo_id``.
* An existing on-disk dataset resumes (appends) instead of recreating.
* The per-frame capture hook is a safe no-op when the robot is unknown or no
  recorder is attached, so it never raises inside the run-policy loop.

The engine is built through ``__new__`` (as in ``test_dataset_recording``) so
the recording lifecycle runs without the optional Newton/Warp physics stack.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")

import strands_robots.dataset_recorder as dataset_recorder
from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_SO100_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def _make_engine(world: SimWorld) -> NewtonSimEngine:
    """Build a NewtonSimEngine bound to ``world`` without the Warp stack."""
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = object()  # non-None sentinel: "world created"
    engine.default_width = 64
    engine.default_height = 48
    return engine


def _world_with_robot(name: str = "so100") -> SimWorld:
    world = SimWorld()
    world.robots[name] = SimRobot(
        name=name, urdf_path="so100.xml", data_config="so100", joint_names=list(_SO100_JOINTS)
    )
    return world


class TestStartRecordingGuards:
    def test_missing_lerobot_extra_returns_actionable_error(self, monkeypatch):
        # When the lerobot extra is absent, start_recording must not dead-end in
        # dataset creation - it returns an error that names the install extra.
        monkeypatch.setattr(dataset_recorder, "has_lerobot_dataset", lambda: False)
        engine = _make_engine(_world_with_robot())

        result = engine.start_recording(repo_id="local/sim_recording")

        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "lerobot" in text
        assert "strands-robots[lerobot]" in text

    def test_no_world_returns_error(self):
        engine = NewtonSimEngine.__new__(NewtonSimEngine)
        engine._world = None
        engine._model = None

        result = engine.start_recording(repo_id="local/sim_recording")

        assert result["status"] == "error"
        assert "create_world" in result["content"][0]["text"]

    def test_recorder_creation_failure_resets_recording_flag(self, monkeypatch):
        # root=None exercises the repo_id -> HF-cache dir resolution. A failing
        # create() must reset the recording flag (so the world is not wedged in
        # a "recording" state with no recorder) and surface the cause.
        def _boom(**_kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(dataset_recorder.DatasetRecorder, "create", staticmethod(_boom))
        engine = _make_engine(_world_with_robot())

        result = engine.start_recording(repo_id="owner/name", root=None)

        assert result["status"] == "error"
        assert "disk full" in result["content"][0]["text"]
        assert engine._world._backend_state["recording"] is False
        # The dir was resolved into the HF cache tree from the owner/name id.
        assert "owner/name" in engine._world._backend_state["last_dataset_root"]

    def test_existing_dataset_resumes_instead_of_recreating(self, monkeypatch, tmp_path):
        # A dataset dir with a meta/ dir on disk must take the resume (append)
        # branch: DatasetRecorder.resume is called, create() is not.
        dataset_dir = tmp_path / "existing_ds"
        (dataset_dir / "meta").mkdir(parents=True)

        resumed_sentinel = object()
        created = []

        monkeypatch.setattr(
            dataset_recorder.DatasetRecorder,
            "resume",
            staticmethod(lambda **_kwargs: resumed_sentinel),
        )
        monkeypatch.setattr(
            dataset_recorder.DatasetRecorder,
            "create",
            staticmethod(lambda **_kwargs: created.append(True)),
        )
        engine = _make_engine(_world_with_robot())
        monkeypatch.setattr(engine, "_verify_resume_schema", lambda *a, **k: None)

        result = engine.start_recording(repo_id="owner/name", root=str(dataset_dir))

        assert result["status"] == "success"
        assert engine._world._backend_state["dataset_recorder"] is resumed_sentinel
        assert created == []  # create() must not run on the resume branch


class TestRunPolicyHookGuards:
    def test_hook_is_none_for_unknown_robot(self):
        engine = _make_engine(_world_with_robot())
        assert engine._make_run_policy_hook("ghost", "pick") is None

    def test_hook_is_noop_without_recorder(self):
        # recording flagged True but no recorder attached: the hook must return
        # early rather than raise, so a run-policy loop never crashes mid-rollout.
        engine = _make_engine(_world_with_robot())
        hook = engine._make_run_policy_hook("so100", "pick")
        assert hook is not None
        engine._world._backend_state["recording"] = True
        engine._world._backend_state["dataset_recorder"] = None

        obs = {j: 0.0 for j in _SO100_JOINTS}
        action = {j: 0.0 for j in _SO100_JOINTS}
        hook(0, obs, action)  # must not raise

        assert engine._world.robots["so100"].policy_steps == 1
