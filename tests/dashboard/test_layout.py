"""The page's frame: no e-stop control covers the chat input, STOP ALL stops every robot, panels plug in by slot."""

from __future__ import annotations

import json
import urllib.request

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

from .browser import page_on  # noqa: E402

# Is the chat input the topmost element at 15 points across its box? (an overlay anywhere on it fails)
COVERED = """() => {
  const el = document.querySelector('#prompt'); const r = el.getBoundingClientRect();
  const pts = [];
  for (const fx of [0.02, 0.25, 0.5, 0.75, 0.98])
    for (const fy of [0.15, 0.5, 0.85]) pts.push([r.left + fx * r.width, r.top + fy * r.height]);
  const over = ([x, y]) => { const t = document.elementFromPoint(x, y); return t !== el && !el.contains(t); };
  return {w: r.width, covered: pts.filter(over).length};
}"""


def test_chat_input_is_never_covered_and_stop_all_latches_every_robot() -> None:
    sim = Simulation()
    arm, g1 = Robot("so101", sim=sim), Robot("so101", sim=sim, id="arm2", position=[0.6, 0, 0])
    server = Dashboard(sim).start()
    try:
        with page_on(server.url) as page:
            page.locator('.estop-ctl[data-robot="so101"]').wait_for(timeout=10_000)
            assert page.locator("#stop-all").is_visible()
            got = page.evaluate(COVERED)
            assert got["w"] > 100 and got["covered"] == 0, got
            # Mouse clicks at a point, no actionability wait: headless swiftshader renders the twin at a few fps, and
            # Playwright's "stable" check waits on animation frames.
            box = page.locator("#prompt").bounding_box()
            # the right end of the input, where the old floating e-stop tray sat
            page.mouse.click(box["x"] + box["width"] * 0.9, box["y"] + box["height"] / 2)
            assert page.evaluate("document.activeElement.id") == "prompt"

            box = page.locator("#stop-all").bounding_box()
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.wait_for_function("window.__safety.stopAll", timeout=10_000)
            assert page.evaluate("window.__safety.stopAll.ok") == 2, page.evaluate("window.__safety")
            assert arm.estopped and g1.estopped
            latched = "[...document.querySelectorAll('.card[data-id]')].every(c => c.dataset.latched === 'yes')"
            page.wait_for_function(latched, timeout=5_000)
            got = page.evaluate(COVERED)  # latched cards grow Resume buttons; still nothing over the input
            assert got["covered"] == 0, got

            box = page.locator(".card[data-id='so101'] .chev").bounding_box()  # collapsible cards
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            assert page.evaluate("document.querySelector(\".card[data-id='so101']\").dataset.collapsed") == "yes"
            assert not page.errors, page.errors
    finally:
        server.stop()
        sim.close()


def test_a_panel_registers_a_tab_and_gets_the_context() -> None:
    sim = Simulation()
    Robot("so101", sim=sim)
    server = Dashboard(sim).start()
    try:
        with urllib.request.urlopen(f"{server.url}/api/panels", timeout=5) as r:
            assert isinstance(json.loads(r.read())["panels"], list)
        with page_on(server.url) as page:
            got = page.evaluate(
                """async () => {
                  const seen = {};
                  window.SR.registerPanel({ id: 'demo', title: 'Demo', order: 10, mount(el, ctx) {
                    el.innerHTML = '<p class="muted">hello</p>';
                    seen.robots = ctx.robots.map(r => r.id);
                    ctx.on('frame', () => { seen.frames = (seen.frames || 0) + 1; });
                    ctx.setBadge(3);
                  }});
                  await new Promise(r => setTimeout(r, 500));
                  window.SR.select('demo');
                  const tab = document.querySelector('#tabs button[data-tab="demo"]');
                  let dup = null;
                  try { window.SR.registerPanel({ id: 'demo', mount() {} }); } catch (e) { dup = e.message; }
                  const scene = await window.SR.panels.get('demo').el.isConnected;
                  window.strandsSlots.register('chat-input', () => { const b = document.createElement('button');
                    b.id = 'mic'; b.textContent = 'mic'; return b; });
                  const mic = document.querySelector('#mic').getBoundingClientRect();
                  const input = document.querySelector('#prompt').getBoundingClientRect();
                  seen.micAbove = mic.bottom <= input.top;
                  return { seen, tab: tab.textContent, selected: tab.getAttribute('aria-selected'), dup, scene,
                           shown: !document.querySelector('[data-pane="demo"]').hidden,
                           robotsHidden: document.querySelector('[data-pane="robots"]').hidden };
                }"""
            )
            assert got["seen"]["micAbove"], got
            assert got["seen"]["robots"] == ["so101"] and got["seen"]["frames"] > 0, got
            assert got["tab"] == "Demo3" and got["selected"] == "true" and got["shown"] and got["robotsHidden"], got
            assert "taken" in got["dup"], got
            assert not page.errors, page.errors
    finally:
        server.stop()
        sim.close()


# Every robot's bounding box, projected by the twin's camera: all 8 corners of each must land on the canvas.
IN_FRAME = """() => {
  const t = window.__twin, out = {};
  const r = t.canvas.getBoundingClientRect();
  for (const o of t.geoms) {
    if (!o || !o.visible || o.userData.plane) continue;
    const id = (t.model.body_names[o.userData.body] || '').split('/')[0];
    o.geometry.computeBoundingBox();
    const b = o.geometry.boundingBox.clone().applyMatrix4(o.matrixWorld);
    for (const x of [b.min.x, b.max.x]) for (const y of [b.min.y, b.max.y]) for (const z of [b.min.z, b.max.z]) {
      const p = b.min.clone().set(x, y, z).project(t.camera);
      out[id] = Math.max(out[id] || 0, Math.abs(p.x), Math.abs(p.y));
    }
  }
  return {worst: out, h: r.height};
}"""


def test_a_phone_sees_every_robot_and_reaches_every_tab_from_the_dock() -> None:
    sim = Simulation()
    Robot("so101", sim=sim)
    Robot("g1", sim=sim, position=[1.0, 0, 0])
    server = Dashboard(sim).start()
    try:
        with page_on(server.url, viewport=(390, 844)) as page:
            page.wait_for_function("window.__twin.pose && window.__twin.framed", timeout=60_000)
            got = page.evaluate(IN_FRAME)
            assert set(got["worst"]) >= {"so101", "g1"} and max(got["worst"].values()) <= 1.0, got
            assert got["h"] > 600, got  # the sheet starts folded: the stage has the screen
            shown = "b => b.getBoundingClientRect().width > 0"
            tabs = page.evaluate(
                f"[...document.querySelectorAll('#tabs button')].filter({shown}).map(b => b.dataset.tab)"
            )
            assert {"robots", "chat"} <= set(tabs), tabs
            page.locator('#tabs button[data-tab="chat"]').click()
            page.wait_for_function("document.querySelector('#prompt').getBoundingClientRect().height > 0")
            assert page.evaluate(COVERED)["covered"] == 0
            page.wait_for_function(f"({IN_FRAME})().h < 600")  # the sheet takes its half, the camera re-frames
            got = page.evaluate(IN_FRAME)
            assert max(got["worst"].values()) <= 1.0, got
            page.locator('#tabs button[data-tab="chat"]').click()  # the open tab again folds the sheet
            page.wait_for_function("document.querySelector('#app').dataset.side === 'closed'")
            assert not page.errors, page.errors
    finally:
        server.stop()
