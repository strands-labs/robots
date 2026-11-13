"""A remote policy whose server goes away holds, reopens the session when the same deployment is back, and
faults when nobody (or a different deployment) comes back within ``reconnect_s``."""

import threading
import time

import numpy as np
import pytest
from conftest import free_endpoint
from test_lerobot import _arm
from tiny_act import save_tiny_act

from strands_robots import Description, Policy
from strands_robots.policy.base import PolicyFault
from strands_robots.policy.lerobot import serve, server_config
from strands_robots.policy.lerobot.redial import Redial


def _held_in_range(robot, q, slack=1e-3):
    """Finite and inside the joint limits, give or take a contact's settling in the simulated world."""
    lim = robot.description.limits
    lo, hi = (np.array([lim[k][i] for k in robot.action_keys]) for i in (0, 1))
    return bool(np.all(np.isfinite(q)) and np.all(q >= lo - slack) and np.all(q <= hi + slack))


def _flaky(cfg, down_at, back=None):
    """Serve ``cfg``; stop it at ``down_at`` s; serve ``back`` (a config, same endpoint) once it is down."""
    first = serve(cfg)
    url = first.__enter__()
    later = []

    def cycle():
        time.sleep(down_at)
        first.__exit__(None, None, None)
        if back is not None:
            time.sleep(0.5)
            later.append(serve(back))
            later[0].__enter__()

    threading.Thread(target=cycle, daemon=True).start()
    return url, later


def _cfg(checkpoint, endpoint, name="flaky"):
    return server_config(Description.load("so101"), checkpoint, deployment=name, endpoint=endpoint, device="cpu")


@pytest.mark.parametrize("execution", ["sequential", "splice"])
def test_the_run_carries_on_once_the_same_server_is_back(tiny_act, execution):
    cfg = _cfg(tiny_act, free_endpoint(), f"flaky-{execution}")
    url, later = _flaky(cfg, 1.0, back=cfg)
    arm, p = _arm(), Policy(url, execution=execution)
    try:
        r = arm.run(p, seconds=6)
    finally:
        for s in later:
            s.__exit__(None, None, None)
    inner = getattr(p, "inner", p)
    assert r.stopped_by == "seconds" and r.fault is None, r
    assert len(inner.reconnects) == 1 and inner.reconnects[0]["attempts"] >= 1
    assert r.holds > 0  # it held while the server was away
    assert len(inner.calls) > 4  # chunks came from the second server too
    assert _held_in_range(arm, r.final.joints.positions)


def test_nobody_back_within_the_window_is_a_fault_naming_it(tiny_act):
    url, _ = _flaky(_cfg(tiny_act, free_endpoint(), "gone"), 1.0)
    arm, t = _arm(), time.perf_counter()
    r = arm.run(Policy(url, reconnect_s=1.5, handshake_timeout_s=1.0), seconds=30)
    assert r.stopped_by == "fault" and "reconnect_s=1.5" in r.fault, r
    assert time.perf_counter() - t < 1.0 + 1.5 + 1.0 + 3.0  # drop + window + one handshake + slack
    assert _held_in_range(arm, r.final.joints.positions)


def test_a_different_checkpoint_back_at_the_address_is_refused(tiny_act, tmp_path):
    other = str(save_tiny_act(tmp_path / "other", 6, span=([-90.0] * 6, [90.0] * 6)))
    ep = free_endpoint()
    url, later = _flaky(_cfg(tiny_act, ep, "swap"), 1.0, back=_cfg(other, ep, "swap"))
    p = Policy(url)
    try:
        r = _arm().run(p, seconds=10)
    finally:
        for s in later:
            s.__exit__(None, None, None)
    inner = getattr(p, "inner", p)
    assert r.stopped_by == "fault" and "different deployment" in r.fault, r
    assert inner.reconnects == []  # never adopted: the arm took nothing from the second server


def test_reconnect_s_must_be_a_number_of_seconds():
    with pytest.raises(ValueError, match="reconnect_s"):
        Policy("zenoh://127.0.0.1:7447/d", reconnect_s=-1)


class _Session:
    closed = False

    def close(self):
        self.closed = True


def test_redial_backs_off_until_the_window_closes():
    tries = []

    def nobody():
        tries.append(time.monotonic())
        raise ConnectionError("no descriptor")

    r = Redial(nobody, 1.2, "p")
    assert r.poll() is None  # still trying: the loop holds
    r._thread.join(5)
    with pytest.raises(PolicyFault, match=r"reconnect_s=1.2 s \(\d+ attempts; last attempt: ConnectionError"):
        r.poll()
    gaps = np.diff(tries)
    assert 2 <= len(tries) <= 5 and gaps[0] >= 0.2  # 0.25 s, then doubling, cut short by the window
    assert len(gaps) < 2 or gaps[1] >= min(0.4, 1.2 - (tries[1] - tries[0]) - 0.05)


def test_redial_stops_at_a_refusal_and_closes_a_session_nobody_adopts():
    r = Redial(lambda: (_ for _ in ()).throw(PolicyFault("different deployment")), 10.0, "p")
    r._thread.join(2)
    with pytest.raises(PolicyFault, match="different deployment"):
        r.poll()
    assert r.attempts == 1
    gate, s = threading.Event(), _Session()
    late = Redial(lambda: (gate.wait(2), s)[1], 10.0, "p")
    late.stop()
    gate.set()
    late._thread.join(2)
    assert s.closed and late._result is None
