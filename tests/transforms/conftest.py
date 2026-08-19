"""Shared fixtures for the dataset-transform tests that touch real datasets.

Every recording call here names its own dataset root under ``tmp_path``
(AGENTS.md rule 15): a rootless ``repo_id`` would resolve into the developer's
shared ``$HF_LEROBOT_HOME`` cache and be inspected there.
"""

import numpy as np
import pytest


@pytest.fixture
def record_source_dataset(tmp_path):
    """Record a small real LeRobotDataset and return its root.

    Returns a callable ``(episode_pixel_values, frames_per_episode=4) -> str``
    where ``episode_pixel_values`` gives each episode's constant pixel value -
    so a pixel-statistic verdict function can be aimed at a known number. The
    action and state columns carry distinct non-zero values per frame so a
    byte-equality assertion cannot pass vacuously on zeros.
    """
    pytest.importorskip("lerobot")
    from strands_robots.dataset_recorder import DatasetRecorder

    def _record(episode_pixel_values, frames_per_episode: int = 4) -> str:
        root = tmp_path / "source"
        recorder = DatasetRecorder.create(
            "local/source",
            fps=10,
            camera_keys=["cam"],
            camera_dims={"cam": (64, 64)},
            joint_names=["j1", "j2"],
            root=str(root),
        )
        for ep, pixel in enumerate(episode_pixel_values):
            for t in range(frames_per_episode):
                img = np.full((64, 64, 3), pixel, dtype=np.uint8)
                observation = {"j1": 0.125 + ep + t, "j2": -0.25 - t, "cam": img}
                action = {"j1": 0.5 + t, "j2": 1.5 + ep + t}
                recorder.add_frame(observation, action, task=f"task-{ep}")
            recorder.save_episode()
        recorder.finalize()
        return str(root)

    return _record


def episode_column(ds, episode: int, key: str) -> np.ndarray:
    """Stack one column of one episode from an opened LeRobotDataset."""
    info = ds.meta.episodes[episode]
    start, stop = int(info["dataset_from_index"]), int(info["dataset_to_index"])
    return np.stack([ds[i][key].numpy() for i in range(start, stop)])
