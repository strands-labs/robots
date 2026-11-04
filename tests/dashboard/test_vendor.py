"""The browser's vendored libraries are pinned: NOTICE, the hashes and twin_scene agree; its scene parses."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.dashboard import twin_scene, vendor  # noqa: E402

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
        if name in vendor.FILES:  # fetched on first use, not shipped: the fetcher pins the same hash
            assert vendor.FILES[name][1] == pinned.group(1), name
        else:
            assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == pinned.group(1), name


def test_the_engines_stay_out_of_git_and_the_wheel() -> None:
    """hatch builds the wheel from what git keeps, so the gitignore line is what keeps 20 MB out of the package."""
    gitignore = (VENDOR.parents[3] / ".gitignore").read_text()  # the repository root
    assert "strands_robots/dashboard/static/vendor/*.wasm" in gitignore


def test_the_fetched_engines_are_the_pinned_versions() -> None:
    for name, (url, _sha) in vendor.FILES.items():
        version = twin_scene.BROWSER_MUJOCO if name.startswith("mujoco") else twin_scene.BROWSER_ORT
        assert url.startswith("https://") and f"@{version}/" in url, url


def test_a_cached_engine_is_served_without_the_network(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STRANDS_ROBOTS_CACHE", str(tmp_path))
    body = b"\0asm fake engine"
    monkeypatch.setitem(vendor.FILES, "mujoco.wasm", ("https://invalid.example/x", hashlib.sha256(body).hexdigest()))
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "dashboard" / "mujoco.wasm").write_bytes(body)
    assert vendor.path("mujoco.wasm").read_bytes() == body


def test_a_wrong_or_missing_engine_says_how_to_fetch_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STRANDS_ROBOTS_CACHE", str(tmp_path))
    monkeypatch.setitem(vendor.FILES, "mujoco.wasm", ("https://invalid.example/mujoco.wasm", "0" * 64))
    with pytest.raises(OSError, match="--fetch"):
        vendor.path("mujoco.wasm", timeout=2)
    assert not any((tmp_path / "dashboard").iterdir())  # no partial file left behind


@pytest.mark.network
def test_the_engines_download_and_match_their_hashes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STRANDS_ROBOTS_CACHE", str(tmp_path))
    for p in vendor.fetch_all():
        assert p.stat().st_size > 1_000_000


def test_the_browser_scene_carries_no_attribute_newer_than_the_pinned_build() -> None:
    from strands_robots import Robot, Simulation
    from strands_robots.dashboard import Dashboard

    with Simulation() as sim:
        Robot("so101", sim=sim)
        xml = Dashboard(sim).scene_json()["xml"]
    assert ' input="' not in xml  # python mujoco 3.16 writes it; 3.14 WASM rejects it
