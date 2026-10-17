"""The browser's vendored libraries are pinned: NOTICE, the hashes and twin_scene agree; its scene parses."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.dashboard import twin_scene  # noqa: E402

VENDOR = Path(twin_scene.__file__).parent / "static" / "vendor"


def test_notice_pins_the_vendored_mujoco_and_its_hashes() -> None:
    notice = (VENDOR / "NOTICE").read_text()
    assert f"@mujoco/mujoco {twin_scene.BROWSER_MUJOCO}" in notice
    assert f"onnxruntime-web {twin_scene.BROWSER_ORT}" in notice
    names = ("mujoco.js", "mujoco.wasm", "three.module.min.js", "OrbitControls.js")
    ort = ("ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm")
    for name in names + ort:
        pinned = re.search(rf"sha256 {re.escape(name)}\s+([0-9a-f]{{64}})", notice)
        assert pinned, name
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == pinned.group(1), name


def test_the_browser_scene_carries_no_attribute_newer_than_the_pinned_build() -> None:
    from strands_robots import Robot, Simulation
    from strands_robots.dashboard import Dashboard

    with Simulation() as sim:
        Robot("so101", sim=sim)
        xml = Dashboard(sim).scene_json()["xml"]
    assert ' input="' not in xml  # python mujoco 3.16 writes it; 3.14 WASM rejects it (journal iteration 1)
