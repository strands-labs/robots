"""Check 7: two processes on one mesh: discover, a state stream, an act whose result the device itself reads
back exactly, a refusal arriving verbatim, and an e-stop that latches until the operator resumes it.

The device is its own Python process holding a sim SO-101 mounted on ``Mesh``; the controller only dials it on
an explicit endpoint (no multicast) in a tenant of its own. Real Zenoh, no doubles.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from checks import needs_module

pytest.importorskip("zenoh")
pytest.importorskip("mujoco")
pytestmark = pytest.mark.timeout(120)
ROOT = Path(__file__).resolve().parents[2]

DEVICE = """
import json, sys
from strands_robots import Robot
from strands_robots.mesh import Mesh
arm = Robot("so101")
mesh = Mesh(sys.argv[1], tenant=sys.argv[2])
mesh.mount(arm, device_id="arm-a", labels={"role": "arm"}, state_hz=50.0, watchdog_ms=None)
print("ready", flush=True)
for line in sys.stdin:
    if line.strip() == "read":
        print(json.dumps({"q": arm.observe(images=False).joints.positions.tolist(), "estop": arm.estopped}), flush=True)
mesh.close()
"""


def _ask(dev) -> dict:
    dev.stdin.write("read\n")
    dev.stdin.flush()
    return json.loads(dev.stdout.readline())


def test_two_mesh_peers_exchange_state_rpc_and_estop():
    needs_module("strands_robots.mesh", "part 7: the mesh")
    from checks import free_endpoint

    from strands_robots.mesh import Mesh, RequestRefused

    endpoint, tenant = free_endpoint(), f"acceptance-{uuid.uuid4().hex[:8]}"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    dev = subprocess.Popen(
        [sys.executable, "-c", DEVICE, f"zenoh://?listen={endpoint}", tenant],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
    )  # fmt: skip
    try:
        assert dev.stdout.readline().strip() == "ready"
        with Mesh(f"zenoh://?connect={endpoint}", tenant=tenant) as ctl:
            (found,) = ctl.discover("device(role:arm)", min_count=1, timeout=5.0)
            assert found.id == "arm-a" and found.manifest["device_id"] == "arm-a"

            states: list = []
            ctl.subscribe("arm-a", "state", states.append)
            t = time.monotonic()
            while len(states) < 20 and time.monotonic() - t < 3.0:
                time.sleep(0.01)
            assert len(states) >= 20, len(states)

            obs = ctl.invoke("arm-a", "act", action={"shoulder_pan": 0.3}, steps=50)
            mine = _ask(dev)
            assert obs.joints.positions.tolist() == mine["q"]  # the exact floats the device read itself
            assert mine["q"][0] > 0.05

            with pytest.raises(RequestRefused, match="shoulder_pan"):
                ctl.invoke("arm-a", "act", action={"shoulder_pan": 99.0})

            assert ctl.broadcast("estop", selector="device(role:arm)") == ["arm-a"]
            assert _ask(dev)["estop"] is True
            before = _ask(dev)["q"]
            with pytest.raises(RequestRefused):
                ctl.invoke("arm-a", "act", action={"shoulder_pan": 0.0})
            np.testing.assert_array_equal(_ask(dev)["q"], before)  # nothing moved while latched
            ctl.invoke("arm-a", "resume", approved_by=ctl.operator)
            assert _ask(dev)["estop"] is False
    finally:
        dev.stdin.close()
        dev.kill()
        dev.wait(10)
