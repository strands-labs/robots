"""A server-side act() shows in the browser twin within 200 ms (p95 over 20)."""

from __future__ import annotations

import json
import statistics
import time

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

from .browser import page_on  # noqa: E402
from .test_twin_page import MEDIA  # noqa: E402

PROBE = """([adr, target]) => {
  const twin = window.__twin;
  if (!twin.__probed) {
    const inner = twin._onPose.bind(twin);
    twin._onPose = (p) => {
      const pr = window.__probe;
      if (pr && pr.hit === null && Math.abs(p.qpos[pr.adr] - pr.target) < 1e-6) {
        pr.hit = performance.timeOrigin + performance.now();
      }
      inner(p);
    };
    twin.__probed = true;
  }
  window.__probe = { adr, target, hit: null };
}"""


def test_server_act_reaches_the_twin_within_200ms() -> None:
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    Robot("unitree_g1", sim=sim, position=[1.0, 0.0, 0.0])
    server = Dashboard(sim).start()
    try:
        with page_on(server.url) as page:
            page.wait_for_function("window.__twin.frames > 5", timeout=20_000)
            key = arm.action_keys[0]
            joint = f"{arm.id}/{arm.description.unlabel(key)}"
            names = page.evaluate("window.__twin.model.jnt_names")
            adr = page.evaluate("window.__twin.model.jnt_qposadr")[names.index(joint)]
            latencies = []
            for i in range(20):
                goal = 0.4 if i % 2 == 0 else -0.4
                with sim.lock:
                    before = float(arm.observe(images=False).joints.positions[0])
                # the twin must not already show the value the act will produce
                t0 = time.time() * 1000.0
                obs = arm.act({key: goal})
                value = float(obs.joints.positions[0])
                assert abs(value - before) > 1e-6
                page.evaluate(PROBE, [adr, value])
                page.wait_for_function("window.__probe.hit !== null", timeout=5_000, polling=5)
                latencies.append(page.evaluate("window.__probe.hit") - t0)
            assert not page.errors, page.errors
    finally:
        server.stop()
        sim.close()
    q = statistics.quantiles(latencies, n=20)
    report = {
        "n": len(latencies),
        "p50_ms": statistics.median(latencies),
        "p95_ms": q[18],
        "max_ms": max(latencies),
        "samples_ms": [round(x, 1) for x in latencies],
    }
    MEDIA.mkdir(parents=True, exist_ok=True)
    (MEDIA / "row2-act-latency.json").write_text(json.dumps(report, indent=2))
    print(
        f"[measured] act->twin over {report['n']}: p50 {report['p50_ms']:.1f} ms, p95 {report['p95_ms']:.1f} ms, "
        f"max {report['max_ms']:.1f} ms"
    )
    assert report["p95_ms"] < 200.0
