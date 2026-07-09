"""End-to-end round-trip for ``MuJoCoSimEngine.render(output_path=...)``.

The pure path-hardening helpers (``_validate_render_output_path``,
``_save_render_png``) are covered GL-free in ``test_render_output_path``.
These tests drive the full behaviour through a real GL render so the wiring
between ``render()`` and those helpers is pinned: a safe path renders, writes
the PNG, and surfaces the resolved location in both the ``json`` block
(``saved_path``) and the text summary; an unsafe path returns ``status=error``
and writes nothing.
"""

from __future__ import annotations

import io
import os

import pytest

pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl as _requires_mujoco  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Confine render writes to a temp sandbox; clear the abs/size opt-outs."""
    root = tmp_path / "renders"
    root.mkdir()
    monkeypatch.setenv("STRANDS_ROBOTS_RENDER_ROOT", str(root))
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_ALLOW_ABS", raising=False)
    monkeypatch.delenv("STRANDS_ROBOTS_RENDER_MAX_BYTES", raising=False)
    return root


def _png_block(result: dict) -> bytes:
    """Return the PNG bytes from a successful render result's image block."""
    for block in result["content"]:
        if isinstance(block, dict) and "image" in block:
            return block["image"]["source"]["bytes"]
    raise AssertionError(f"no image block in render result: {result}")


def _json_block(result: dict) -> dict:
    """Return the json block from a render result."""
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in render result: {result}")


@_requires_mujoco
def test_render_output_path_writes_file_and_reports_saved_path(sandbox) -> None:
    """A safe output_path renders, persists the PNG, and reports saved_path."""
    from PIL import Image

    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    try:
        target = sandbox / "shot.png"
        result = sim.render(camera_name="default", width=64, height=48, output_path=str(target))

        assert result["status"] == "success", result
        # The file was actually written and is a decodable PNG.
        assert target.exists()
        decoded_size = Image.open(io.BytesIO(target.read_bytes())).size
        assert decoded_size == (64, 48)
        # The bytes on disk match the inline image block (same PNG, not re-encoded).
        assert target.read_bytes() == _png_block(result)

        # saved_path is surfaced in the json block and points at our file.
        saved = _json_block(result)["saved_path"]
        assert os.path.realpath(saved) == os.path.realpath(str(target))
        # ... and echoed in the human-readable summary.
        summary = result["content"][0]["text"]
        assert "saved" in summary and saved in summary
    finally:
        sim.cleanup()


@_requires_mujoco
def test_render_rejects_unsafe_output_path_without_writing(sandbox) -> None:
    """A traversal output_path is rejected with status=error and no file write."""
    from strands_robots.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    try:
        result = sim.render(camera_name="default", width=64, height=48, output_path="../escape.png")

        assert result["status"] == "error", result
        assert "render:" in result["content"][0]["text"]
        # Nothing escaped the sandbox parent.
        assert not (sandbox.parent / "escape.png").exists()
    finally:
        sim.cleanup()
