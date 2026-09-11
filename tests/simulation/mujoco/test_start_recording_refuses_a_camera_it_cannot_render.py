"""``start_recording`` refuses a camera column no frame on this box can carry.

``get_observation`` skips every camera frame when MuJoCo offscreen rendering is
unavailable (headless Linux without EGL/OSMesa). The dataset schema used to be
declared from ``model.ncam`` regardless, so the opening block of
``docs/recording.md`` reported ``success ... 1 cameras`` and the rollout's first
``add_frame`` then failed with ``Missing features: {'observation.images.default'}``.
The refusal now lands in ``start_recording`` itself, before any dataset is
created, and names ``cameras=[]`` as the state-only path that does record.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.simulation.mujoco import backend as backend_mod  # noqa: E402


@pytest.fixture
def sim():
    from strands_robots.simulation import Simulation

    s = Simulation()
    s.create_world()
    s.add_robot("so100")
    yield s
    s.destroy()


def test_a_camera_the_box_cannot_render_is_refused_before_any_dataset_exists(sim, tmp_path, monkeypatch):
    # The cached probe result every render path consults; False is what a
    # headless box without EGL/OSMesa latches.
    monkeypatch.setattr(backend_mod, "_rendering_available", False)
    root = tmp_path / "ds"

    res = sim.start_recording(repo_id="local/headless", root=str(root), fps=30)

    assert res["status"] == "error", res
    text = res["content"][0]["text"]
    assert "default" in text and "cameras=[]" in text
    assert not root.exists()
    assert sim._world._backend_state.get("recording") is False
    assert "dataset_recorder" not in sim._world._backend_state


def test_state_only_recording_still_starts_where_rendering_is_unavailable(sim, tmp_path, monkeypatch):
    # The cached probe result every render path consults; False is what a
    # headless box without EGL/OSMesa latches.
    monkeypatch.setattr(backend_mod, "_rendering_available", False)

    res = sim.start_recording(repo_id="local/headless_state", root=str(tmp_path / "ds"), fps=30, cameras=[])

    assert res["status"] == "success", res
    feats = sim._world._backend_state["dataset_recorder"].dataset.features
    assert not any(k.startswith("observation.images.") for k in feats)
    sim.stop_recording()
