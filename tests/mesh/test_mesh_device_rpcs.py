"""#4939: a device adds its own rpcs at mount; the caller resolves them from the device's manifest, and each one goes
through the same gate as a built-in (observe-only, e-stop latch, watchdog, argument check, no shadowing)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from strands_robots import Robot
from strands_robots.mesh import Mesh, RequestRefused, device_rpc
from strands_robots.robot.safety import EStopped

ROOT = Path(__file__).resolve().parents[2]
PEER = Path(__file__).with_name("peer_rpcs.py")


def _rpcs(calls):
    def gripper(width: float, by: str | None = None) -> dict:
        """Open the gripper."""
        calls.append((width, by))
        return {"width": width}

    def battery() -> dict:
        return {"percent": 87}

    return {"gripper": gripper, "battery": device_rpc(battery, write=False)}


@pytest.fixture
def mounted(mesh_pair):
    a, b = mesh_pair
    calls: list = []
    arm = Robot("so101", id="arm")
    a.mount(arm, watchdog_ms=300, rpcs=_rpcs(calls))
    assert b.discover("arm", min_count=1, timeout=3.0)
    yield arm, a, b, calls
    arm.close()


def test_the_manifest_advertises_each_device_rpc_with_its_schema_and_kind(mounted):
    _, _, b, _ = mounted
    rows = {r["name"]: r for r in b.discover("arm")[0].manifest["rpcs"]}
    assert rows["gripper"]["write"] is True and rows["gripper"]["arms"] is True
    assert rows["gripper"]["params"]["required"] == ["width"] and rows["gripper"]["doc"] == "Open the gripper."
    assert rows["battery"]["write"] is False and rows["battery"]["arms"] is False
    assert {"act", "estop", "describe"} <= set(rows)


def test_a_caller_invokes_a_device_rpc_resolved_from_the_manifest(mounted):
    _, _, b, calls = mounted
    assert b.invoke("arm", "battery") == {"percent": 87}
    assert b.invoke("arm", "gripper", width=0.5) == {"width": 0.5}
    assert calls == [(0.5, b.operator)], "a write carries who asked, for the audit record and the handler"


def test_an_rpc_the_device_does_not_advertise_is_refused_at_the_caller(mounted):
    _, _, b, _ = mounted
    with pytest.raises(LookupError, match="gripper") as err:
        b.invoke("arm", "laser")
    assert "battery" in str(err.value), "the error lists the device's own rpcs too"


def test_bad_arguments_are_refused_by_the_device_before_the_handler_runs(mounted):
    _, _, b, calls = mounted
    for args, why in (({}, "needs"), ({"width": "wide"}, "number"), ({"width": 1, "laser": 1}, "laser")):
        with pytest.raises(TypeError, match=why):
            b.invoke("arm", "gripper", **args)
    with pytest.raises(TypeError, match="no arguments"):
        b.invoke("arm", "battery", level=1)
    assert calls == []


def test_a_write_device_rpc_arms_the_watchdog_for_its_caller(mounted):
    arm, _, b, _ = mounted
    holds = []
    arm.body.hold = lambda: holds.append(1)
    b.invoke("arm", "gripper", width=0.2)
    time.sleep(0.8)  # the caller goes quiet past watchdog_ms
    assert holds, "the watchdog armed on the device rpc and applied on_disconnect"


def test_a_read_device_rpc_never_arms_the_watchdog(mounted):
    arm, _, b, _ = mounted
    holds = []
    arm.body.hold = lambda: holds.append(1)
    b.invoke("arm", "battery")
    time.sleep(0.8)
    assert holds == []


def test_a_write_device_rpc_is_refused_while_e_stopped_and_a_read_still_answers(mounted):
    _, _, b, calls = mounted
    b.invoke("arm", "estop")
    with pytest.raises(EStopped, match="e-stopped"):
        b.invoke("arm", "gripper", width=0.1)
    assert b.invoke("arm", "battery") == {"percent": 87} and calls == []


@pytest.mark.parametrize("name", ["estop", "stop", "hold", "act", "describe", "skill", "run", "calibrate"])
def test_a_device_rpc_may_not_shadow_a_built_in(name):
    with Mesh("mem://shadow", tenant="t") as m, pytest.raises(ValueError, match="shadow"):
        m.mount(Robot("so101", id="arm"), rpcs={name: lambda: None})


@pytest.mark.parametrize(
    "rpc, why",
    [
        (lambda *a: None, r"\*args"),
        (lambda **kw: None, r"\*\*kw"),
        ("not a function", "callable"),
    ],
)
def test_a_handler_must_name_its_arguments_and_be_callable(rpc, why):
    with Mesh("mem://bad", tenant="t") as m, pytest.raises(TypeError, match=why):
        m.mount(Robot("so101", id="arm"), rpcs={"x": rpc})


def test_a_read_cannot_arm_and_a_bad_name_is_refused():
    with pytest.raises(ValueError, match="arms"):
        device_rpc(lambda: None, write=False, arms=True)
    with Mesh("mem://name", tenant="t") as m, pytest.raises(ValueError):
        m.mount(Robot("so101", id="arm"), rpcs={"a/b": lambda: None})


def _spawn(url, tenant, *extra):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    p = subprocess.Popen(
        [sys.executable, str(PEER), url, tenant, "arm", *extra], stdout=subprocess.PIPE, env=env, text=True
    )
    assert p.stdout is not None and p.stdout.readline().strip() == "ready"
    return p


@pytest.fixture
def router(port):
    zenohd = shutil.which("zenohd")
    if zenohd is None:
        pytest.skip("zenohd not installed")
    pytest.importorskip("zenoh")
    p = subprocess.Popen(
        [zenohd, "-l", f"tcp/127.0.0.1:{port}", "--no-multicast-scouting"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    yield f"zenoh://?connect=tcp/127.0.0.1:{port}"
    p.kill()
    p.wait(5)


@pytest.mark.timeout(60)
def test_two_processes_over_a_router_a_caller_that_never_saw_the_handler_invokes_it(router, tenant):
    dev = _spawn(router, tenant)
    try:
        with Mesh(router, tenant=tenant) as ctl:
            assert ctl.discover("arm", min_count=1, timeout=5.0)
            assert ctl.invoke("arm", "battery") == {"percent": 87}
            assert ctl.invoke("arm", "gripper", width=0.4) == {"width": 0.4}
            assert dev.stdout is not None and dev.stdout.readline().startswith("gripper 0.4 by=")
    finally:
        dev.kill()
        dev.wait(5)


@pytest.mark.timeout(60)
def test_two_processes_an_observe_only_device_refuses_a_write_device_rpc_and_answers_a_read(router, tenant):
    dev = _spawn(router, tenant, "real")
    try:
        with Mesh(router, tenant=tenant) as ctl:
            assert ctl.discover("arm", min_count=1, timeout=5.0)
            with pytest.raises(RequestRefused, match="observe-only"):
                ctl.invoke("arm", "gripper", width=0.4)
            assert ctl.invoke("arm", "battery") == {"percent": 87}
    finally:
        dev.kill()
        dev.wait(5)
