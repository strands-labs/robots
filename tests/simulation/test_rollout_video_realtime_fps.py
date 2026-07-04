"""Rollout video plays back at real time regardless of control_frequency.

A rollout renders at most one frame per applied control step, so the MP4 can
carry at most ``control_frequency`` unique frames per second of sim time. If the
writer used the requested ``fps`` when ``fps > control_frequency`` the video
would play back FASTER than real time (by ``fps / control_frequency``) - a
silent fidelity gap for a feature whose job is to faithfully show what the
policy did. ``_RolloutVideoWriter.open`` must cap the writer fps at
``control_frequency`` in that case, and leave it untouched when
``fps <= control_frequency`` (the capture cadence already down-samples to real
time there).
"""

from __future__ import annotations

import types
from typing import Any

from strands_robots.simulation import policy_runner
from strands_robots.simulation.policy_runner import VideoConfig, _RolloutVideoWriter


class _FakeWriter:
    def append_data(self, _frame: Any) -> None:  # pragma: no cover - unused here
        pass

    def close(self) -> None:  # pragma: no cover - unused here
        pass


class _FakeSim:
    """Minimal sim whose render probe succeeds so open() reaches get_writer."""

    def render(self, camera_name: str, width: int, height: int) -> dict[str, Any]:
        return {"status": "success", "content": [{"text": "ok"}]}


def _open_and_capture_fps(monkeypatch, tmp_path, requested_fps, control_frequency):
    """Call _RolloutVideoWriter.open and return the fps handed to get_writer."""
    captured: dict[str, Any] = {}

    def _fake_get_writer(path: str, fps: int, **kwargs: Any) -> _FakeWriter:
        captured["fps"] = fps
        return _FakeWriter()

    fake_imageio = types.SimpleNamespace(get_writer=_fake_get_writer)
    monkeypatch.setattr(policy_runner, "require_optional", lambda *a, **k: fake_imageio)

    video = VideoConfig(path=str(tmp_path / "out.mp4"), fps=requested_fps)
    writer, err = _RolloutVideoWriter.open(_FakeSim(), video, control_frequency)
    assert err is None, err
    assert writer is not None
    return captured["fps"]


def test_video_fps_capped_to_control_frequency_when_lower(monkeypatch, tmp_path):
    # cf < fps: requesting 30 fps at 15 Hz control would play back 2x too fast.
    # The writer must be capped to 15 so the video is real time.
    write_fps = _open_and_capture_fps(monkeypatch, tmp_path, requested_fps=30, control_frequency=15.0)
    assert write_fps == 15


def test_video_fps_capped_rounds_fractional_control_frequency(monkeypatch, tmp_path):
    write_fps = _open_and_capture_fps(monkeypatch, tmp_path, requested_fps=30, control_frequency=12.4)
    assert write_fps == 12


def test_video_fps_unchanged_when_control_frequency_higher(monkeypatch, tmp_path):
    # cf >= fps (the common default: cf=50, fps=30): the capture cadence
    # down-samples, so the requested fps is already real time - leave it alone.
    write_fps = _open_and_capture_fps(monkeypatch, tmp_path, requested_fps=30, control_frequency=50.0)
    assert write_fps == 30


def test_video_fps_unchanged_when_equal(monkeypatch, tmp_path):
    write_fps = _open_and_capture_fps(monkeypatch, tmp_path, requested_fps=30, control_frequency=30.0)
    assert write_fps == 30
