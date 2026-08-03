"""GPU integration tests for LeRobotDataset recording on the Isaac backend.

Closes the acceptance criteria of the IsaacRecordingMixin issue: on a real
Isaac Sim runtime, ``start_recording -> run_policy(policy_object=MockPolicy())
-> stop_recording`` must produce a valid LeRobotDataset (parquet + per-camera
MP4 + ``meta/``), round-trip verified by REOPENING the on-disk dataset
(AGENTS.md: "round-trip tests for recording") - through the exact same facade
that works on the MuJoCo and Newton backends, with only the backend swapped.

Requirements match ``test_isaac_gpu.py``: NVIDIA GPU + CUDA, Isaac Sim 6.0+
installed out-of-band, ``pip install 'strands-robots[sim-isaac,lerobot]'``,
and ``STRANDS_GPU_TEST=1``. Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ \\
        tests_integ/simulation/test_isaac_recording_gpu.py -m gpu -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("strands_robots.simulation.isaac")
pytest.importorskip("lerobot")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]


def _skip_if_isaac_unavailable() -> None:
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


def _assets_root_path() -> str:
    """Resolve the Isaac Sim bundled-assets root (modern then legacy path)."""
    try:
        from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]
    except ImportError:
        from omni.isaac.nucleus import get_assets_root_path  # type: ignore[import-not-found]

    assets_root = get_assets_root_path()
    assert assets_root, "get_assets_root_path() returned empty"
    return assets_root


class TestIsaacDatasetRecording:
    """The MuJoCo/Newton record-and-stream workflow with backend='isaac'."""

    def test_record_run_policy_stop_round_trip(self, tmp_path):
        """start_recording -> run_policy(MockPolicy) -> stop_recording -> reopen.

        Records two RTX cameras alongside the Franka joint state, drives the
        rollout with the stock MockPolicy (requires_images=False - pinning the
        recording-forces-images override on a real RTX stream), then reopens
        the dataset from disk and asserts parquet truth + per-camera MP4s.
        """
        from strands_robots.dataset_recorder import read_dataset_episode_indices
        from strands_robots.policies.mock import MockPolicy
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        assets_root = _assets_root_path()
        root = str(tmp_path / "isaac_gpu_ds")

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=False, render_mode="rtx_realtime"))
        try:
            r = sim.create_world()
            assert r["status"] == "success", f"create_world: {r}"

            usd_path = f"{assets_root}/Isaac/Robots/Franka/franka.usd"
            r = sim.add_robot("franka", usd_path=usd_path)
            assert r["status"] == "success", f"add_robot: {r}"

            r = sim.add_camera("cam1")
            assert r["status"] == "success", f"add_camera cam1: {r}"
            r = sim.add_camera("cam2", position=[0.0, 2.5, 1.5], target=[0.0, 0.0, 0.5])
            assert r["status"] == "success", f"add_camera cam2: {r}"

            sim.reset()
            sim.step(2)

            # Post-reset joint state must be readable (#1895): on the pip
            # Isaac Sim 6.0.x wheels ``world.reset()`` invalidates the
            # articulation physics handles, and pre-fix this observation was
            # ``{}`` (get_observation's documented silent-empty mode) - the
            # exact read every reset_between recording flow makes after
            # episode 1.
            obs = sim.get_observation("franka", skip_images=True)
            joint_keys = [k for k, v in obs.items() if isinstance(v, float)]
            assert joint_keys, f"post-reset observation carries no joint state: {sorted(obs)}"

            r = sim.start_recording(repo_id="local/isaac_gpu_rec", root=root, fps=10, overwrite=True)
            assert r["status"] == "success", f"start_recording: {r}"

            # Keep control_frequency at or below 1/rendering_dt (default 30Hz)
            # so each recorded frame reflects a freshly rendered product.
            r = sim.run_policy(
                robot_name="franka",
                policy_object=MockPolicy(),
                n_steps=10,
                control_frequency=10.0,
                fast_mode=True,
            )
            assert r["status"] == "success", f"run_policy: {r}"

            r = sim.save_episode()
            assert r["status"] == "success", f"save_episode: {r}"

            r = sim.stop_recording()
            assert r["status"] == "success", f"stop_recording: {r}"

            r = sim.verify_dataset_episodes(1)
            assert r["status"] == "success", f"verify_dataset_episodes: {r}"

            # Round-trip: reopen the on-disk dataset and assert its truth.
            info = read_dataset_episode_indices(root)
            assert info["total_episodes"] == 1, info
            assert info["total_frames"] == 10, info
            assert (Path(root) / "meta" / "info.json").exists()

            mp4s = sorted(Path(root).glob("videos/**/*.mp4"))
            keys = {part for p in mp4s for part in p.parts if part.startswith("observation.images.")}
            assert keys == {"observation.images.cam1", "observation.images.cam2"}, keys
            assert all(p.stat().st_size > 0 for p in mp4s)

            # Train-shape == read-shape: the camera schema was declared from
            # the RTX native-resolution probe (which can differ from the
            # requested camera size - DLSS-safe render products), so the dims
            # declared in meta/info.json must match what stream_dataset
            # actually emits on read-back. A mismatch here means training
            # consumers see a different image shape than the schema promises.
            info_json = json.loads((Path(root) / "meta" / "info.json").read_text())
            declared = {
                key: tuple(int(d) for d in feat["shape"])
                for key, feat in info_json["features"].items()
                if key.startswith("observation.images.")
            }
            assert set(declared) == keys, declared
            reader = sim.stream_dataset("local/isaac_gpu_rec", root=root, shuffle=False)
            frame = next(iter(reader))
            for key, (c, h, w) in declared.items():
                emitted = tuple(int(d) for d in frame[key].shape[-3:])
                # DatasetRecorder declares (C, H, W); tolerate either layout
                # from the streaming decoder, but H/W/C must be the probed ones.
                assert emitted in ((c, h, w), (h, w, c)), (
                    f"{key}: schema declares CHW {(c, h, w)} but stream_dataset "
                    f"emitted {emitted} - RTX probe shape and read-back shape diverged"
                )
        finally:
            sim.destroy()

    def test_robot_factory_backend_swap_round_trip(self, tmp_path):
        """Robot("so100", backend="isaac") record-and-stream: the notebook pin.

        This is the exact entry point demonstrated by the optional Isaac step
        in examples/notebooks/05_streaming_data_loop.ipynb - the factory route
        (create_simulation("isaac") + add_robot(data_config="so100"),
        strands_robots/robot.py) rather than the raw IsaacSimulation + Franka
        USD form the tests above use. Pinning it here means the notebook's
        one-kwarg backend-swap story can never silently drift from reality
        (issue #1536: never ship example code CI cannot verify).

        Known risk: so100 registry resolution on Isaac has not been executed on
        a GPU host before this test (issue #1552 used a Franka USD). If this
        fails for asset/registry reasons (not recording reasons), the agreed
        fallback is to switch both this test and the notebook cell to the
        Franka-USD form and file a follow-up for so100-on-Isaac resolution.
        """
        from strands_robots import MockPolicy, Robot
        from strands_robots.dataset_recorder import read_dataset_episode_indices

        _skip_if_isaac_unavailable()
        root = str(tmp_path / "isaac_factory_ds")

        # Factory route: same call the notebook makes; returns the backend sim.
        sim = Robot("so100", backend="isaac", mesh=False)
        try:
            r = sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0.0, 0.05])
            assert r["status"] == "success", f"add_camera: {r}"

            r = sim.start_recording(
                repo_id="local/nb5_isaac",
                root=root,
                fps=10,
                task="pick up the red cube",
                cameras=["front"],
                overwrite=True,
            )
            assert r["status"] == "success", f"start_recording: {r}"

            # control_frequency <= 1/rendering_dt (Isaac default 30Hz) so each
            # recorded frame reflects a freshly rendered product - matches the
            # notebook cell's fps=10 / control_frequency=10.0 pacing.
            r = sim.run_policy(
                robot_name="so100",
                policy_object=MockPolicy(),
                n_steps=10,
                control_frequency=10.0,
                fast_mode=True,
                instruction="pick up the red cube",
            )
            assert r["status"] == "success", f"run_policy: {r}"

            r = sim.save_episode()
            assert r["status"] == "success", f"save_episode: {r}"

            r = sim.stop_recording()
            assert r["status"] == "success", f"stop_recording: {r}"

            r = sim.verify_dataset_episodes(1)
            assert r["status"] == "success", f"verify_dataset_episodes: {r}"

            # Round-trip: reopen the on-disk dataset and assert its truth.
            info = read_dataset_episode_indices(root)
            assert info["total_episodes"] == 1, info
            assert info["total_frames"] == 10, info
            assert (Path(root) / "meta" / "info.json").exists()

            mp4s = sorted(Path(root).glob("videos/**/*.mp4"))
            keys = {part for p in mp4s for part in p.parts if part.startswith("observation.images.")}
            assert keys == {"observation.images.front"}, keys
            assert all(p.stat().st_size > 0 for p in mp4s)

            # Train-shape == read-shape: the schema declared in meta/info.json
            # must match what stream_dataset actually emits on read-back, so
            # training consumers never see a different image shape than the
            # schema promises (mirrors the raw-API test above).
            info_json = json.loads((Path(root) / "meta" / "info.json").read_text())
            declared = {
                key: tuple(int(d) for d in feat["shape"])
                for key, feat in info_json["features"].items()
                if key.startswith("observation.images.")
            }
            assert set(declared) == keys, declared
            reader = sim.stream_dataset("local/nb5_isaac", root=root, shuffle=False)
            frame = next(iter(reader))
            for key, (c, h, w) in declared.items():
                emitted = tuple(int(d) for d in frame[key].shape[-3:])
                assert emitted in ((c, h, w), (h, w, c)), (
                    f"{key}: schema declares CHW {(c, h, w)} but stream_dataset "
                    f"emitted {emitted} - factory-path probe and read-back shape diverged"
                )
        finally:
            sim.destroy()

    def test_multi_camera_frames_are_distinct(self, tmp_path):
        """Two cameras with different vantages record distinct pixel content.

        Regression for the stale-render-product case: without the multi-camera
        refresh, the second camera's stream duplicates a stale buffer. Two
        cameras looking at different parts of the scene must yield frames that
        differ from each other within the same observation.
        """
        import numpy as np

        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        assets_root = _assets_root_path()

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=False, render_mode="rtx_realtime"))
        try:
            assert sim.create_world()["status"] == "success"
            usd_path = f"{assets_root}/Isaac/Robots/Franka/franka.usd"
            assert sim.add_robot("franka", usd_path=usd_path)["status"] == "success"
            assert sim.add_camera("cam1")["status"] == "success"
            assert sim.add_camera("cam2", position=[0.0, 2.5, 1.5], target=[0.0, 0.0, 0.5])["status"] == "success"
            sim.reset()
            sim.step(5)

            obs = sim.get_observation("franka")
            assert "cam1" in obs and "cam2" in obs, sorted(obs)
            assert obs["cam1"].ndim == 3 and obs["cam2"].ndim == 3
            assert not np.array_equal(obs["cam1"], obs["cam2"]), (
                "cameras at different vantages returned identical frames - stale render product"
            )
        finally:
            sim.destroy()
