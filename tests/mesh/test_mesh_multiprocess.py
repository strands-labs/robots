"""Gate: separate processes find each other with zero config, and a killed device leaves discover()."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from strands_robots.mesh import Mesh

pytest.importorskip("zenoh")
ROOT = Path(__file__).resolve().parents[2]
PEER = Path(__file__).with_name("peer_device.py")


def _spawn(url, tenant, did):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    p = subprocess.Popen([sys.executable, str(PEER), url, tenant, did], stdout=subprocess.PIPE, env=env, text=True)
    assert p.stdout.readline().strip() == "ready"
    return p


@pytest.mark.timeout(60)
def test_two_devices_and_a_controller_discover_each_other_zero_config_then_a_kill_leaves(tenant):
    devices = [_spawn("zenoh://", tenant, f"arm-{i}") for i in range(2)]
    try:
        t0 = time.perf_counter()
        with Mesh("zenoh://", tenant=tenant) as controller:
            found = controller.discover("device(role:arm)", min_count=2, timeout=5.0)
            t_discover = time.perf_counter() - t0
            assert sorted(d.id for d in found) == ["arm-0", "arm-1"]
            devices[0].send_signal(signal.SIGKILL)
            t1 = time.perf_counter()
            while any(d.id == "arm-0" for d in controller.discover()) and time.perf_counter() - t1 < 10:
                time.sleep(0.01)
            t_leave = time.perf_counter() - t1
            assert [d.id for d in controller.discover()] == ["arm-1"]
        print(f"\nfull discover {t_discover * 1000:.0f} ms; SIGKILL leaves after {t_leave * 1000:.0f} ms")
        assert t_discover < 1.0
        assert t_leave <= controller.presence_ttl + 0.5
    finally:
        for p in devices:
            p.kill()
            p.wait(5)


def _port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("watchdog_ms", [300.0])
def test_controller_sigkill_mid_motion_makes_the_device_apply_on_disconnect(tenant, watchdog_ms):
    port = _port()
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    dev = subprocess.Popen(
        [
            sys.executable,
            str(PEER.with_name("peer_robot.py")),
            f"zenoh://?listen=tcp/127.0.0.1:{port}",
            tenant,
            "arm",
            "-",
            str(watchdog_ms),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    ctl = None
    try:
        assert dev.stdout.readline().strip() == "ready"
        ctl = subprocess.Popen(
            [
                sys.executable,
                str(PEER.with_name("peer_controller.py")),
                f"zenoh://?connect=tcp/127.0.0.1:{port}",
                tenant,
                "arm",
            ],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert ctl.stdout.readline().strip() == "moving"
        t_kill = time.monotonic()
        ctl.send_signal(signal.SIGKILL)
        line = dev.stdout.readline().strip()
        assert line.startswith("hold "), line
        dt = float(line.split()[1]) - t_kill
        print(
            f"\ncontroller SIGKILL → device on_disconnect(hold) after {dt * 1000:.1f} ms (watchdog_ms={watchdog_ms:g})"
        )
        assert 0 <= dt <= watchdog_ms / 1000 + 0.05
    finally:
        for p in (ctl, dev):
            if p is not None:
                p.kill()
                p.wait(5)


@pytest.mark.timeout(60)
def test_exiting_without_close_never_hangs_the_interpreter(tenant):
    """A session left open while its peer is dead used to hang exit (a non-daemon zenoh callback thread)."""
    port = _port()
    code = (
        "import sys, time\n"
        "from strands_robots.mesh import Mesh\n"
        "a = Mesh(sys.argv[1], tenant=sys.argv[3])\n"
        "b = Mesh(sys.argv[2], tenant=sys.argv[3])\n"
        "b.subscribe('x', 'state', print)\n"
        "time.sleep(0.3)\n"
        "print('exiting', flush=True)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    urls = [f"zenoh://?listen=tcp/127.0.0.1:{port}", f"zenoh://?connect=tcp/127.0.0.1:{port}"]
    t0 = time.monotonic()
    out = subprocess.run(
        [sys.executable, "-c", code, *urls, tenant], capture_output=True, text=True, env=env, timeout=30
    )
    took = time.monotonic() - t0
    print(f"\nexit without close: rc={out.returncode} in {took:.2f} s")
    assert out.returncode == 0 and "exiting" in out.stdout
    assert took < 15
