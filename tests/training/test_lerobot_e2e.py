"""End-to-end: LerobotTrainer trains a real ACT checkpoint from a recorded
LeRobotDataset, and the result is loadable via create_policy.

Slow + requires lerobot + mujoco. Skipped automatically if either is absent.
Runs CPU-only, 2 steps - just enough to prove the record->train->load loop.
"""

import os
import sys

import pytest

lerobot = pytest.importorskip("lerobot")
pytest.importorskip("mujoco")

from strands_robots.training import TrainSpec, create_trainer  # noqa: E402


@pytest.fixture(scope="module")
def recorded_dataset(tmp_path_factory):
    """Record a tiny dataset in MuJoCo sim (one short episode)."""
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    from strands_robots import MockPolicy, Robot

    root = str(tmp_path_factory.mktemp("e2e_ds"))
    sim = Robot("so100", mesh=False)
    sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])
    start = sim.start_recording(
        repo_id="local/e2e",
        root=root,
        fps=30,
        task="pick up the red cube",
        overwrite=True,
    )
    assert start["status"] == "success", start
    sim.run_policy(
        robot_name="so100",
        policy_object=MockPolicy(),
        instruction="pick up the red cube",
        n_steps=20,
    )
    stop = sim.stop_recording()
    assert stop["status"] == "success", stop
    assert os.path.isfile(os.path.join(root, "meta", "info.json"))
    return root


@pytest.mark.slow
def test_record_train_load_loop(recorded_dataset, tmp_path):
    out = str(tmp_path / "e2e_out")
    trainer = create_trainer("lerobot_local", device="cpu")

    spec = TrainSpec(
        dataset_root=recorded_dataset,
        base_model="",  # ACT from scratch - smallest CPU path
        output_dir=out,
        steps=2,
        save_freq=2,
        global_batch_size=2,
        extra={"policy_type": "act", "num_workers": 0},
    )

    assert trainer.validate(spec) == []

    result = trainer.train(spec)
    assert result.status == "success", result.message
    assert result.checkpoint_dir and os.path.isdir(result.checkpoint_dir)
    assert os.path.isfile(os.path.join(result.checkpoint_dir, "model.safetensors"))

    # export() returns a create_policy-loadable path (default passthrough).
    exported = trainer.export(spec, result.checkpoint_dir)
    assert os.path.isdir(exported)

    # The trained checkpoint loads back as a Policy - loop closed.
    os.environ.setdefault("STRANDS_TRUST_REMOTE_CODE", "1")
    from strands_robots import create_policy

    policy = create_policy(exported, device="cpu")
    assert policy.provider_name == "lerobot_local"
