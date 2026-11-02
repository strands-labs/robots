"""The page's Record panel: tick a robot and its cameras, REC, save an episode, stop -> a dataset LeRobot reads."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
pytest.importorskip("lerobot.datasets.lerobot_dataset")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

from .browser import page_on  # noqa: E402

LONG = 180_000  # LeRobot's first import (torch) and the first video encode, on a busy CI box


def _tap(page: object, selector: str) -> None:
    """Click the middle of an element: headless swiftshader paints ~1 fps, so Playwright's "stable" wait times out."""
    box = page.locator(selector).bounding_box()  # type: ignore[attr-defined]
    assert box is not None, selector
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)  # type: ignore[attr-defined]


def test_the_record_panel_records_two_episodes_of_the_slider_moves(tmp_path: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sim = Simulation()
    arm = Robot("so101", sim=sim)
    sim.camera("front", size=(128, 96), attach=arm)
    server = Dashboard(sim).start()
    target = tmp_path / "from-the-page"
    try:
        with page_on(server.url) as page:
            page.wait_for_function("!!document.querySelector('#record .rec-robot input')", timeout=LONG)
            assert "record" in page.evaluate("[...window.SR.panels.keys()]")  # a tab through the slot API
            _tap(page, '#tabs button[data-tab="record"]')
            page.wait_for_function("document.querySelector('#record').offsetParent !== null", timeout=LONG)
            box = page.locator('#record .rec-cam input[data-robot="so101"][value="front"]')
            assert box.is_checked()  # a robot's own camera is ticked for it
            page.fill("#record .rec-task", "sweep the pan")
            page.fill("#record .rec-target", str(target))
            page.fill("#record .rec-fps", "10")
            _tap(page, "#record .rec-go")
            page.wait_for_function("document.querySelector('#record').dataset.state === 'recording'", timeout=LONG)
            page.wait_for_function("!document.querySelector('.rec-bar').hidden", timeout=LONG)  # REC in the top bar
            for n, sweep in enumerate(((-0.6, -0.2, 0.2, 0.6), (0.6, 0.2, -0.2, -0.6)), start=1):
                for v in sweep:
                    page.evaluate(
                        """v => { const el = document.querySelector('input[type=range][data-robot="so101"]'
                        + '[data-label="shoulder_pan"]'); el.value = v; el.dispatchEvent(new Event('change',
                        { bubbles: true })); }""",
                        v,
                    )
                    page.wait_for_timeout(250)
                page.wait_for_function("window.__record.status && window.__record.status.frames >= 10", timeout=LONG)
                if n == 1:  # the first take is kept with save; stop keeps the second
                    _tap(page, "#record .rec-save")
                    page.wait_for_function(
                        "document.querySelectorAll('#record .rec-episodes li').length === 1 && !window.__record.busy",
                        timeout=LONG,
                    )
            _tap(page, "#record .rec-stop")
            page.wait_for_function(
                "document.querySelector('#record').dataset.state === 'idle' && !window.__record.busy", timeout=LONG
            )
            page.wait_for_function("document.querySelectorAll('#record .rec-episodes li').length === 2", timeout=LONG)
            listed = page.locator("#record .rec-episodes").inner_text()
            assert "ep 0" in listed and "ep 1" in listed
            assert page.locator(".rec-bar").is_hidden()
            assert not page.errors, page.errors
    finally:
        server.stop()
        sim.close()
    ds = LeRobotDataset("local/from-the-page", root=target)
    assert ds.meta.total_episodes == 2 and ds.fps == 10 and ds.meta.video_keys == ["observation.images.front"]
    assert ds.num_frames >= 20 and {"observation.state", "action"} <= set(ds.features)
    first, last = ds[0], ds[ds.num_frames - 1]
    assert first["task"] == "sweep the pan" and int(last["episode_index"]) == 1
    assert tuple(first["observation.images.front"].shape) == (3, 96, 128)  # the video decodes
    assert first["action"].shape == first["observation.state"].shape
    import torch

    pan = torch.stack([torch.as_tensor(a) for a in ds.hf_dataset["action"]])[:, 0]
    assert float(pan.max() - pan.min()) > 0.1  # the slider sweeps are in the actions
