"""A Robot is a device: its manifest, rpcs, events and position hold."""

from __future__ import annotations

import json
import threading
import time

import pytest
from test_drivers import FakeSerial, make_driver

from strands_robots import Robot
from strands_robots.robot.device import Manifest
from strands_robots.robot.drivers import feetech as ft
from strands_robots.robot.safety import RequestRefused


def test_manifest_round_trips_and_describe_is_the_manifest():
    arm = Robot("so101")
    m = arm.manifest
    assert m is arm.manifest and m.device_id == "so101" and m.tenant == "default" and not m.real
    assert Manifest.from_dict(json.loads(json.dumps(m.to_dict()))) == m
    d = arm.describe()
    assert {k: v for k, v in d.items() if k not in ("task", "estop")} == m.to_dict()
    assert d["task"] is None and d["estop"] is False
    assert {r.name for r in m.rpcs} >= {"describe", "observe", "act", "run", "stop", "hold", "estop", "resume"}
    assert {e.name for e in m.events} == {"state", "fault", "estop", "run.started", "run.stopped"}


def test_one_control_per_actuated_joint_range_is_the_limit_and_safe_inside():
    arm = Robot("so101")
    controls = arm.manifest.controls
    assert [c.name for c in controls] == [f"{k}.pos" for k in arm.action_keys]
    for c, k in zip(controls, arm.action_keys, strict=True):
        assert c.range == tuple(arm.description.limits[k])
        assert c.range[0] <= c.safe[0] <= c.safe[1] <= c.range[1]
        assert c.on_disconnect == "hold"


def test_rpc_schema_is_generated_from_the_signature():
    schema = {r.name: r.params for r in Robot("so101").manifest.rpcs}
    assert schema["resume"]["properties"]["approved_by"]["type"] == "string"
    assert schema["act"]["required"] == ["action"]


def test_state_subscription_carries_the_policy_joints_dict():
    arm = Robot("so101")
    got: list[dict] = []
    unsubscribe = arm.subscribe("state", got.append, hz=100)
    time.sleep(0.12)
    unsubscribe()
    n = len(got)
    time.sleep(0.05)
    assert n >= 5 and len(got) == n
    assert [g["seq"] for g in got] == list(range(n)) and got[-1]["t"] > got[0]["t"]
    assert {f"{k}.pos" for k in arm.action_keys} <= set(got[0])
    with pytest.raises(ValueError, match="unknown event"):
        arm.subscribe("nope", print)


def test_rpc_schema_drops_the_receiver_of_a_run_loop_function():
    from strands_robots.robot.device import rpc_schema

    def run(robot, policy, *, hz: float | None = None):
        """A run-loop function bound onto Robot."""

    assert list(rpc_schema(run)["properties"]) == ["policy", "hz"]


def test_position_hold_is_declared_sim_always_holds_hardware_reads_the_registry():
    from strands_robots.description import registry
    from strands_robots.robot.run.bind import bind

    sim = Robot("so101")
    assert sim.supports_position_hold and sim.manifest.position_hold
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    assert real.supports_position_hold is registry()["so101"]["hardware"]["position_hold"] is True
    assert real.manifest.position_hold

    class NeedsHold:
        requires_position_hold = True

        def act(self, obs, instruction=None):
            return obs.joints.positions

        def reset(self):
            pass

    real.__dict__["supports_position_hold"] = False  # a bus that drops its goal
    with pytest.raises(RequestRefused, match="cannot hold position"):
        bind(real, NeedsHold(), None)
    bind(sim, NeedsHold(), None)
    real.close()


def test_a_manifest_that_omits_position_hold_reads_as_not_holding():
    d = Robot("so101").manifest.to_dict()
    del d["position_hold"]
    assert Manifest.from_dict(d).position_hold is False


def test_closing_a_robot_ends_its_state_stream_quietly():
    import threading as th

    arm = Robot("so101")
    errors: list[BaseException] = []
    old = th.excepthook
    th.excepthook = lambda a: errors.append(a.exc_value)
    try:
        got: list[dict] = []
        unsubscribe = arm.subscribe("state", got.append, hz=200)
        time.sleep(0.05)
        arm.close()
        time.sleep(0.05)
        n = len(got)
        time.sleep(0.05)
        unsubscribe()
    finally:
        th.excepthook = old
    assert errors == [] and len(got) == n and n > 0


def test_a_run_emits_run_started_and_run_stopped():
    arm = Robot("so101")
    got: list[tuple[str, dict]] = []
    for e in ("run.started", "run.stopped", "fault"):
        arm.subscribe(e, lambda p, e=e: got.append((e, p)))
    arm.run(lambda o: {"shoulder_pan": 0.1}, steps=3)
    assert [e for e, _ in got] == ["run.started", "run.stopped"]
    assert got[0][1]["device_id"] == "so101" and got[1][1] == {"device_id": "so101", "stopped_by": "steps", "steps": 3}


def test_a_policy_fault_emits_fault_then_run_stopped():
    from strands_robots.policy.base import PolicyFault

    def broken(o):
        raise PolicyFault("server lost")

    arm = Robot("so101")
    got: list[tuple[str, dict]] = []
    for e in ("fault", "run.stopped"):
        arm.subscribe(e, lambda p, e=e: got.append((e, p)))
    assert arm.run(broken, steps=5).stopped_by == "fault"
    assert [e for e, _ in got] == ["fault", "run.stopped"] and "server lost" in got[0][1]["reason"]


def test_lockstep_emits_per_robot():
    from strands_robots import Simulation

    sim = Simulation("mujoco")
    a, b = Robot("so101", sim=sim, id="a"), Robot("so101", sim=sim, id="b")
    stopped: list[str] = []
    for r in (a, b):
        r.subscribe("run.stopped", lambda p: stopped.append(p["device_id"]))
    sim.run({a: lambda o: {}, b: lambda o: {}}, steps=2)
    assert sorted(stopped) == ["a", "b"]
    sim.close()


class FlakySerial(FakeSerial):
    """The fake bus with trouble on the line: while ``noise`` is set every SYNC_READ reply's last checksum is flipped;
    while ``dead`` is set no servo answers, and a read waits out the bus's 20 ms timeout, as pyserial does."""

    def __init__(self):
        super().__init__()
        self.noise, self.dead, self.garbled = False, False, 0

    def write(self, data):
        if self.dead:
            self.frames.append(bytes(data))
            return
        super().write(data)
        if self.noise and data[4] == ft.SYNC_READ:
            self.rx[-1] ^= 0xFF
            self.garbled += 1

    def read(self, n):
        if self.dead and n > len(self.rx):
            time.sleep(0.02)
        return super().read(n)


def _until(cond, timeout=5.0):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < timeout:
        time.sleep(0.002)
    return cond()


def _streaming_arm(hz=200, **subscribers):
    """An SO-101 on a ``FlakySerial`` bus streaming state at ``hz`` into a list: (arm, serial, got, unsubscribe)."""
    d, s = make_driver(serial=FlakySerial())
    arm = Robot("so101", driver=d, twin=False)
    for event, cb in subscribers.items():
        arm.subscribe(event, cb)
    got: list[dict] = []
    return arm, s, got, arm.subscribe("state", got.append, hz=hz)


def _a_burst_of_noise(arm, s, got):
    """Garble every reply until three reads failed, then clear the line: a burst, since a driver that asks again for
    a garbled reply (``num_read_retries``) never lets a single one reach the stream. Returns "resumed", or where the
    stream stopped."""
    assert _until(lambda: len(got) >= 3)
    g, burst = s.garbled, 3 * (1 + getattr(arm.body, "num_read_retries", 0))
    s.noise = True
    if not _until(lambda: s.garbled >= g + burst):
        s.noise = False
        return f"stopped reading the bus after {s.garbled - g} of {burst} garbled replies"
    s.noise, n = False, len(got)
    return "resumed" if _until(lambda: len(got) >= n + 3) else "no sample after the noise"


def test_a_state_stream_keeps_running_through_a_burst_of_corrupted_replies_and_resumes_after_it():
    escaped: list[BaseException] = []
    old = threading.excepthook
    threading.excepthook = lambda a: escaped.append(a.exc_value)
    try:
        arm, s, got, unsubscribe = _streaming_arm()
        outcome = _a_burst_of_noise(arm, s, got)
        unsubscribe()
        arm.close()
    finally:
        threading.excepthook = old
    assert (outcome, escaped) == ("resumed", [])
    seqs = [g["seq"] for g in got]
    assert arm.events.errors == seqs[-1] + 1 - len(seqs) >= 3  # each failed read is counted, and is a gap in seq


def test_a_state_stream_reports_one_fault_event_per_run_of_failed_reads():
    faults: list[dict] = []
    arm, s, got, unsubscribe = _streaming_arm(fault=faults.append)
    arm.subscribe("fault", lambda f: 1 / 0)  # a fault subscriber's own bug does not end the stream either
    outcomes = [_a_burst_of_noise(arm, s, got) for _ in range(2)]
    unsubscribe()
    arm.close()
    assert outcomes == ["resumed", "resumed"] and len(faults) == 2, faults
    assert {f["device_id"] for f in faults} == {"so101"}
    assert all("state stream" in f["reason"] and "ProtocolError: servo 6: bad checksum" in f["reason"] for f in faults)


def test_a_state_subscriber_that_raises_keeps_its_stream_and_is_reported_once():
    d, _ = make_driver()
    arm = Robot("so101", driver=d, twin=False)
    faults: list[dict] = []
    arm.subscribe("fault", faults.append)
    calls: list[int] = []

    def broken(payload):
        calls.append(payload["seq"])
        raise RuntimeError("this subscriber cannot take the sample")

    unsubscribe = arm.subscribe("state", broken, hz=200)
    kept = _until(lambda: len(calls) >= 5)
    unsubscribe()
    arm.close()
    assert kept, f"the stream ended after {len(calls)} sample(s)"
    assert len(faults) == 1 and "RuntimeError: this subscriber cannot take the sample" in faults[0]["reason"]


def test_an_error_that_cannot_print_itself_is_reported_by_its_type_and_does_not_end_a_state_stream():
    class Unprintable(OSError):
        def __str__(self) -> str:
            raise RuntimeError("this error cannot format its own message")

    d, _ = make_driver()
    arm = Robot("so101", driver=d, twin=False)
    faults: list[dict] = []
    arm.subscribe("fault", faults.append)
    read, calls = d.read, []

    def read_failing_twice():
        calls.append(1)
        if len(calls) <= 2:
            raise Unprintable()
        return read()

    d.read = read_failing_twice  # type: ignore[method-assign]
    got: list[dict] = []
    unsubscribe = arm.subscribe("state", got.append, hz=200)
    resumed = _until(lambda: len(got) >= 3)
    unsubscribe()
    arm.close()
    assert resumed, f"the stream ended after {len(calls)} read(s)"
    assert arm.events.errors == 2
    assert len(faults) == 1 and "a sample failed (Unprintable)" in faults[0]["reason"], faults


def test_after_a_failed_read_a_state_stream_waits_a_period_so_it_never_bursts_when_the_bus_answers_again():
    arm, s, got, unsubscribe = _streaming_arm(hz=100)
    assert _until(lambda: len(got) >= 3)
    s.dead = True  # a read now fails after the 20 ms timeout, once per try: two periods or more at 100 Hz
    time.sleep(1.0)
    s.dead, n = False, len(got)
    t0 = time.monotonic()
    assert _until(lambda: len(got) > n), "no sample after the bus answered again"
    time.sleep(max(0.0, t0 + 0.25 - time.monotonic()))
    unsubscribe()
    arm.close()
    back = [g for g in got[n:] if g["t"] - t0 < 0.2]  # 20 periods: a late or retried read is made up, the outage not
    assert len(back) <= 40, f"{len(back)} samples in the first 200 ms back at 100 Hz"
