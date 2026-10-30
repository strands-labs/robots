"""Drivers against simulated buses: no hardware is touched here."""

import contextlib
import json
import math
import os
import socket
import tempfile
import threading

import numpy as np
import pytest

from strands_robots import Description, Robot, Simulation
from strands_robots.robot.drivers import Calibration
from strands_robots.robot.drivers import feetech as ft
from strands_robots.robot.drivers import microduck as md
from strands_robots.robot.drivers.unitree import G1_ORDER, H1_SLOTS, UnitreeDriver


class FakeSerial:
    """Six STS3215 servos on a half-duplex wire: answers PING/READ/WRITE/SYNC_READ, applies SYNC_WRITE."""

    def __init__(self, ids=(1, 2, 3, 4, 5, 6)):
        self.regs = {i: bytearray(0x50) for i in ids}
        self.actual = dict.fromkeys(ids, 2048)  # the encoder; the firmware reports Present = Actual - Homing_Offset
        self.rx = bytearray()
        self.frames = []
        self.closed = False
        self.waited = 0  # bytes asked for that never came: on a real wire each one costs the whole timeout

    def reset_input_buffer(self):
        self.rx.clear()

    def read(self, n):
        self.waited += max(0, n - len(self.rx))
        out, self.rx = bytes(self.rx[:n]), self.rx[n:]
        return out

    def close(self):
        self.closed = True

    def _status(self, mid, params=b""):
        payload = bytes([mid, len(params) + 2, 0]) + params
        return ft.HEADER + payload + bytes([ft.checksum(payload)])

    def _homing(self, i):
        return ft.decode_sm(bytes(self.regs[i][ft.REG_HOMING_OFFSET : ft.REG_HOMING_OFFSET + 2]), 11)

    def write(self, data):
        self.frames.append(bytes(data))
        for i, r in self.regs.items():
            present = (self.actual[i] - self._homing(i)) % 4096
            r[ft.REG_PRESENT_POSITION : ft.REG_PRESENT_POSITION + 2] = ft.word(present)
        mid, length, instr = data[2], data[3], data[4]
        params = data[5 : 5 + length - 2]
        assert data[5 + length - 2] == ft.checksum(data[2 : 5 + length - 2]), "host sent a bad checksum"
        if instr == ft.PING and mid in self.regs:
            self.rx += self._status(mid)
        elif instr == ft.READ and mid in self.regs:
            reg, n = params
            self.rx += self._status(mid, bytes(self.regs[mid][reg : reg + n]))
        elif instr == ft.WRITE and mid in self.regs:
            reg = params[0]
            self.regs[mid][reg : reg + len(params) - 1] = params[1:]
            self.rx += self._status(mid)
        elif instr == ft.SYNC_READ:
            reg, n, *ids = params
            for i in ids:
                if i in self.regs:
                    self.rx += self._status(i, bytes(self.regs[i][reg : reg + n]))
        elif instr == ft.SYNC_WRITE:
            reg, per = params[0], params[1]
            body = params[2:]
            for k in range(0, len(body), per + 1):
                i, chunk = body[k], body[k + 1 : k + 1 + per]
                self.regs[i][reg : reg + per] = chunk
                if reg == ft.REG_GOAL_POSITION:  # a perfect servo: it is where it was told to be
                    self.actual[i] = (int.from_bytes(chunk, "little") + self._homing(i)) % 4096


class SweepSerial(FakeSerial):
    """The operator's sweep: each SYNC_READ finds every joint but the ``still`` ones 200 counts below, then above,
    ``middle``."""

    def __init__(self, middle=2348, still=()):
        super().__init__()
        self.middle, self.still, self.side = middle, set(still), 200

    def write(self, data):
        if data[4] == ft.SYNC_READ:
            self.side = -self.side
            self.actual.update({i: self.middle + self.side for i in self.actual if i not in self.still})
        super().write(data)


CAL = {
    n: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 1024, "range_max": 3072}
    for i, n in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))
}
CAL["wrist_roll"]["drive_mode"] = 1


def make_driver(cal=CAL, serial=None, max_speed=None):
    """``max_speed=None``: these tests write goals and read them back at once; the speed cap has its own tests."""
    serial = serial or FakeSerial()
    bus = ft.FeetechBus("/dev/fake", serial=serial)
    drv = ft.FeetechDriver(Description.load("so101"), port="/dev/fake", calibration=cal, bus=bus, max_speed=max_speed)
    return drv, serial


def test_protocol_codec():
    assert ft.frame(1, ft.PING) == bytes.fromhex("ff ff 01 02 01 fb")
    assert ft.decode_sm(ft.word(0x8064), 15) == -100 and ft.decode_sm(ft.word(100), 15) == 100
    assert ft.encode_sm(-5, 11) == ft.word(0x805) and ft.encode_sm(5, 11) == ft.word(5)
    with pytest.raises(ft.ProtocolError, match="checksum"):
        ft.parse_status(bytes.fromhex("ff ff 01 02 00 00"), 1, 0)
    with pytest.raises(ft.ProtocolError, match="expected 1"):
        ft.parse_status(bytes.fromhex("ff ff 02 02 00 fb"), 1, 0)


class EchoSerial(FakeSerial):
    """An adapter that hears its own request before the servo answers."""

    def write(self, data):
        self.rx += bytes(data)
        super().write(data)


class NoisySerial(FakeSerial):
    """A stray byte on the line ahead of every reply."""

    def write(self, data):
        self.rx += b"\x00"
        super().write(data)


BUS_CALLS = {
    "ping": lambda bus: bus.ping(1) is True,
    "read": lambda bus: bus.read(1, ft.REG_PRESENT_POSITION, 2),
    "write": lambda bus: bus.write(1, ft.REG_TORQUE_ENABLE, b"\x01") is None,
    "sync_read": lambda bus: bus.sync_read(ft.REG_PRESENT_POSITION, 6, [1, 2, 3, 4, 5, 6]),
}


@pytest.mark.parametrize("wire", [FakeSerial, EchoSerial, NoisySerial])
@pytest.mark.parametrize("call", BUS_CALLS)
def test_bus_reads_exactly_the_reply_and_never_waits(wire, call):
    serial = wire()
    bus = ft.FeetechBus("/dev/fake", serial=serial)
    for _ in range(3):
        assert BUS_CALLS[call](bus)
    assert serial.waited == 0 and not serial.rx


@pytest.mark.parametrize("wire", [FakeSerial, EchoSerial, NoisySerial])
def test_a_servo_that_never_answers_is_absent_on_every_wire(wire):
    bus = ft.FeetechBus("/dev/fake", serial=wire())
    for _ in range(2):  # before and after the bus has learned whether the adapter echoes
        assert bus.ping(7) is False
        with pytest.raises(ft.ProtocolError, match="no FF FF header"):
            bus.read(7, ft.REG_PRESENT_POSITION, 2)
        assert bus.ping(1) is True


class VoltageErrorSerial(FakeSerial):
    """No echo; every status carries ERR 0x01, so a PING reply is byte-identical to the PING request."""

    def _status(self, mid, params=b""):
        payload = bytes([mid, len(params) + 2, 1]) + params
        return ft.HEADER + payload + bytes([ft.checksum(payload)])


def test_a_reply_identical_to_the_request_stands_once_the_adapter_is_known_not_to_echo():
    bus = ft.FeetechBus("/dev/fake", serial=VoltageErrorSerial())
    assert bus.read(1, ft.REG_PRESENT_POSITION, 2)  # a reply unlike the request: this adapter does not echo
    assert all(bus.ping(i) for i in range(1, 7))


class BreakSerial(FakeSerial):
    """A line held low (a pinched cable): the adapter delivers 0x00 for as long as anyone reads."""

    def read(self, n):
        return b"\x00" * n


@pytest.mark.parametrize("call", BUS_CALLS)
def test_a_line_that_never_sends_a_header_is_refused_and_frees_the_wire(call):
    bus = ft.FeetechBus("/dev/fake", serial=BreakSerial())
    if call == "ping":
        assert bus.ping(1) is False
    else:
        with pytest.raises(ft.ProtocolError):
            BUS_CALLS[call](bus)
    freed = []
    t = threading.Thread(target=lambda: freed.append(bus.wire.acquire(timeout=0.5)))
    t.start()
    t.join()
    assert freed == [True]  # an e-stop from another thread still gets the bus


def test_driver_cycle_never_waits_on_the_wire():
    d, serial = make_driver(serial=EchoSerial())
    d.write(d.read().positions)
    assert serial.waited == 0


class CorruptingSerial(FakeSerial):
    """Noise on the line: the next ``corrupt`` SYNC_READ replies arrive with the last servo's checksum flipped."""

    def __init__(self):
        super().__init__()
        self.corrupt = 0

    def write(self, data):
        super().write(data)
        if self.corrupt and data[4] == ft.SYNC_READ:
            self.rx[-1] ^= 0xFF
            self.corrupt -= 1


def test_a_corrupted_status_packet_is_read_again_at_most_twice():
    d, s = make_driver(serial=CorruptingSerial())
    s.corrupt = 2
    assert d.read().names == d.names  # the third reading of the same request lands
    s.corrupt = 3
    with pytest.raises(ft.ProtocolError, match="servo 6: bad checksum"):
        d.read()


def test_one_corrupted_status_packet_in_twenty_ticks_does_not_abort_a_run():
    d, s = make_driver(serial=CorruptingSerial())
    real = Robot("so101", driver=d, twin=False)
    ticks: list[int] = []

    def policy(obs):
        ticks.append(len(ticks))
        if len(ticks) == 10:
            s.corrupt = 1  # the next status packet on the wire arrives garbled
        return {"shoulder_pan": 0.3}

    res = real.run(policy, hz=100, steps=20)
    assert (res.steps, res.stopped_by, len(ticks), s.corrupt) == (20, "steps", 20, 0)
    real.close()


def test_driver_reads_through_calibration_and_writes_goals():
    d, s = make_driver()
    js = d.read()
    assert js.names == Description.load("so101").action_keys
    lo, hi = d.description.limits["shoulder_pan"]
    assert js["shoulder_pan"] == pytest.approx((lo + hi) / 2)  # 2048 is the middle of [1024, 3072]
    d.write(np.array([hi, 0, 0, 0, 0, 0.5]))
    assert d.extras()["torque"] is True
    assert s.regs[1][ft.REG_TORQUE_ENABLE] == 1
    assert int.from_bytes(s.regs[1][ft.REG_GOAL_POSITION : ft.REG_GOAL_POSITION + 2], "little") == 3072
    assert d.read()["shoulder_pan"] == pytest.approx(hi, abs=1e-3)
    assert d.read()["gripper"] == pytest.approx(0.5, abs=2e-3)
    # drive_mode=1 reverses: a target at the joint's low end goes to range_max
    rlo, rhi = d.description.limits["wrist_roll"]
    assert d.rad_to_counts("wrist_roll", rlo) == 3072 and d.rad_to_counts("wrist_roll", rhi) == 1024
    d.close()
    assert s.regs[1][ft.REG_TORQUE_ENABLE] == 0 and s.closed


def test_write_computes_goals_before_torque_comes_on():
    """A non-finite target is refused before torque is enabled: a refused write leaves a limp arm limp."""
    d, s = make_driver()
    s.frames.clear()
    targets = np.zeros(6)
    targets[2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        d.write(targets)
    assert s.frames == []  # nothing on the wire: torque was never enabled, no stale Goal_Position
    assert d.extras()["torque"] is False


def test_uncalibrated_driver_uses_full_turn_counts():
    d, _ = make_driver(cal=None)
    assert d.calibration is None and "raw" in d.extras()["calibration"]
    assert d.read()["elbow_flex"] == pytest.approx(0.0)  # 2048 → 0 rad
    assert d.rad_to_counts("elbow_flex", math.pi / 2) == 3072


def test_missing_servo_is_a_connection_error():
    with pytest.raises(ConnectionError, match=r"servo id\(s\) \[6\]"):
        make_driver(serial=FakeSerial(ids=(1, 2, 3, 4, 5)))


def test_calibration_loads_from_lerobot_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
    p = tmp_path / "robots" / "so101_follower" / "so101_follower.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(CAL))
    d, _ = make_driver(cal=None)
    assert d.calibration_source == str(p) and d.calibration == Calibration.from_json(CAL)


def test_calibrate_writes_lerobot_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
    d, s = make_driver(cal=None, serial=SweepSerial(middle=2100))
    with pytest.raises(RuntimeError, match="interactive=True"):
        d.calibrate(interactive=False)  # no file yet: recording one needs a human
    for i in s.regs:  # the encoder reads 2100 at "middle"; the EEPROM still carries an old calibration's offset
        s.actual[i] = 2100
        s.regs[i][ft.REG_HOMING_OFFSET : ft.REG_HOMING_OFFSET + 2] = ft.encode_sm(-300, 11)
    cal = d.calibrate(interactive=True, prompt=lambda msg: None, sweep_seconds=0.1)
    s.actual.update(dict.fromkeys(s.actual, 2100))  # the sweep over, the arm is back at the middle
    # LeRobot's Feetech convention: Homing_Offset = middle - 2047, so the firmware then reports the middle at 2047
    assert cal.joints["shoulder_pan"].homing_offset == 2100 - 2047 == s._homing(1)
    assert ft.decode_sm(d.bus.read(1, ft.REG_PRESENT_POSITION, 2), 15) == 2047
    assert (cal.joints["shoulder_pan"].range_min, cal.joints["shoulder_pan"].range_max) == (2047 - 200, 2047 + 200)
    saved = json.loads((tmp_path / "robots" / "so101_follower" / "so101_follower.json").read_text())
    assert saved["gripper"]["id"] == 6
    assert set(saved["gripper"]) == {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
    assert d.calibrate(interactive=False) == cal  # reads what was saved, moves nothing
    assert make_driver(cal=None)[0].calibration == cal  # the next connect loads the file it wrote


def test_hardware_robot_gets_a_twin_in_the_default_world():
    d, s = make_driver()
    real = Robot("so101", driver=d)
    assert real.is_real and real.sim is None and real.twin is not None
    assert real.twin.sim is Simulation.default() and real.twin.id == "so101_twin"
    hi = d.description.limits["shoulder_pan"][1]
    real.act({"shoulder_pan": hi})
    assert real.observe()["shoulder_pan"] == pytest.approx(hi, abs=1e-3)
    # the twin was told the same target and now shows the measured joint
    for _ in range(30):
        real.twin.sim.step(10)
    assert real.twin.observe()["shoulder_pan"] == pytest.approx(hi, abs=0.05)
    with pytest.raises(NotImplementedError, match="hardware"):
        real.set_pose(position=[1, 0, 0])
    real.twin.set_pose(position=[1, 0, 0])
    assert real.calibrate(interactive=False) == Calibration.from_json(
        CAL
    )  # the loaded file; True walks a human through it
    real.close()
    assert "so101_twin" not in Simulation.default().robots


@pytest.mark.parametrize("manual", [False, True])
def test_a_real_act_advances_the_twins_world_one_control_step(manual):
    # a real act() steps the twin's world as a simulated act() steps its own; manual_time() still wins
    d, _ = make_driver()
    real = Robot("so101", driver=d)
    world = real.twin.sim
    t0, start = world.time, real.twin.observe()["shoulder_pan"]
    with world.manual_time() if manual else contextlib.nullcontext():
        for _ in range(60):
            real.act({"shoulder_pan": 0.5})
            real.observe()  # a read mirrors the measurement but is not time
    moved = real.twin.observe()["shoulder_pan"]
    if manual:  # no time passed, yet the twin stands where the arm measured (observe places it)
        assert world.time == t0 and start == 0.0 and moved == pytest.approx(0.5, abs=1e-3)
    else:
        assert world.time == pytest.approx(t0 + 60 * world.physics.control_dt) and moved == pytest.approx(0.5, abs=0.05)
    real.close()


def test_a_real_observe_places_the_twin_at_the_measurement_with_no_step():
    # the twin mirrors a MEASUREMENT: after observe() it shows the real pose, before its world steps once
    d, _ = make_driver()
    real = Robot("so101", driver=d)
    world = real.twin.sim
    with world.manual_time():
        t0 = world.time
        real.act({"shoulder_pan": 0.5, "shoulder_lift": -0.6, "elbow_flex": 0.4})
        measured = real.observe().joints.positions
        twin = real.twin.observe().joints
    assert world.time == t0 and np.abs(measured).max() > 0.3
    assert np.allclose(twin.positions, measured, atol=1e-9) and np.allclose(twin.velocities, 0.0)
    real.close()


def test_hardware_robot_without_twin_and_bad_kwargs():
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    assert real.twin is None and not Simulation.default().robots
    with pytest.raises(TypeError, match="not sim="):
        Robot("so101", driver=d, sim=Simulation.default())
    with pytest.raises(TypeError, match="hardware argument"):
        Robot("so101", cameras={"wrist": 0})
    with pytest.raises(RuntimeError, match="for hardware"):
        Robot("so101").calibrate()


class FakeMotor:
    def __init__(self, q):
        self.q, self.dq, self.tau_est = q, 0.1, 0.5


class FakeImu:
    quaternion = (1.0, 0.0, 0.0, 0.0)
    gyroscope = (0.0, 0.0, 0.0)
    accelerometer = (0.0, 0.0, 9.81)


class FakeLowState:
    def __init__(self):
        self.motor_state = [FakeMotor(i * 0.01) for i in range(35)]
        self.imu_state = FakeImu()
        self.power_v = 52.0
        self.mode_machine = 4


class FakeSub:
    def __init__(self, msg=None):
        self.msg = msg
        self.closed = False

    def read(self):
        return self.msg

    def close(self):
        self.closed = True


def test_unitree_read_path_maps_sdk_order_to_action_keys():
    desc = Description.load("g1")
    d = UnitreeDriver(desc, port="en7", subscriber=FakeSub(FakeLowState()))
    js = d.read()
    assert js.names == desc.action_keys and len(js) == 29
    assert js["right_elbow_joint"] == pytest.approx(G1_ORDER.index("right_elbow") * 0.01)
    assert js.efforts[0] == 0.5
    ex = d.extras()
    assert ex["imu"]["quaternion"] == [1, 0, 0, 0] and ex["power_v"] == 52.0 and ex["mode_machine"] == 4
    with pytest.raises(NotImplementedError, match="observe-only"):
        d.write(np.zeros(29))
    g1 = Robot("g1", driver=d, twin=False)
    assert g1.is_real and g1.observe().joints.names[0] == "left_hip_pitch_joint"
    g1.close()
    assert d.sub.closed


def test_unitree_without_samples_fails_fast():
    with pytest.raises(ConnectionError, match="no rt/lowstate"):
        UnitreeDriver(Description.load("g1"), port="en7", subscriber=FakeSub(None), timeout=0.05)


def test_unitree_h1_reads_the_go_wire_in_h1_slot_order():
    desc = Description.load("unitree_h1")
    d = UnitreeDriver(desc, port="en7", subscriber=FakeSub(FakeLowState()))
    js = d.read()
    assert js.names == desc.action_keys and len(js) == 19
    for joint in ("left_hip_yaw", "right_hip_roll", "torso", "left_elbow"):
        assert js[joint] == pytest.approx(H1_SLOTS[joint] * 0.01)
    assert 9 not in d.index  # the H1 has no motor in slot 9


class FakeRobotd:
    """robotd on a real unix socket: answers hello, then streams robot.state after robot.subscribe."""

    def __init__(self, joints):
        self.path = os.path.join(tempfile.mkdtemp(), "robotd.sock")
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen(1)
        self.joints = joints
        self.seen: list[dict] = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.server.accept()
        for raw in conn.makefile("rb"):
            msg = json.loads(raw)
            self.seen.append(msg)
            if msg["method"] == "hello":
                conn.sendall(md.line({"jsonrpc": "2.0", "id": msg["id"], "result": {"api_version": 31}}))
            elif msg["method"] == "robot.subscribe":
                conn.sendall(md.line({"jsonrpc": "2.0", "id": msg["id"], "result": {}}))
                state = {"t": 1.5, "joints": self.joints, "odom": {"position": [0.1, 0, 0], "yaw": 0.2}}
                conn.sendall(md.line({"jsonrpc": "2.0", "method": "robot.state", "params": state}))


def test_microduck_reads_robotd_state_and_drops_the_mouth():
    joints = [i * 0.1 for i in range(15)]  # robotd's wire order, mouth at index 9
    robotd = FakeRobotd(joints)
    duck = Robot("microduck", port=robotd.path, twin=False)
    obs = duck.observe()
    assert len(obs.joints) == 14 and md.HARDWARE_JOINT_NAMES[md.MOUTH_INDEX] == "mouth"
    assert obs.joints["right_hip_yaw"] == pytest.approx(1.0) and obs.joints["head_roll"] == pytest.approx(0.8)
    assert obs.extras["odom"]["yaw"] == 0.2
    assert [m["method"] for m in robotd.seen] == ["hello", "robot.subscribe"]
    assert robotd.seen[0]["params"] == {"api_version": md.API_VERSION}
    with pytest.raises(NotImplementedError, match="observe-only"):
        duck.act(obs.joints.positions)
    duck.close()


def test_microduck_without_robotd_names_the_forward():
    with pytest.raises(ConnectionError, match="ssh -N -L"):
        Robot("microduck", port="/nonexistent/robotd.sock", twin=False)


# The readers' findings: a NaN, a mislabelled driver, a LeRobot file under its 0.6 name, a calibrate with no human


def writes(serial):
    return [f for f in serial.frames if f[4] in (ft.WRITE, ft.SYNC_WRITE)]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_target_never_reaches_the_bus(bad):
    from strands_robots.robot.safety import NonFiniteTarget

    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)
    with pytest.raises(NonFiniteTarget, match="shoulder_pan"):
        real.act({"shoulder_pan": bad})  # NaN used to round to range_min: a full-speed move to the end stop
    assert writes(s) == []  # no torque, no goal
    with pytest.raises(ValueError, match="non-finite"):
        d.rad_to_counts("shoulder_pan", bad)  # the driver's own last line of defence
    real.close()


class Renamed:
    """A third-party Driver: the Body verbs over a FeetechDriver, with its own ``names``."""

    def __init__(self, inner, names):
        self.inner, self.names = inner, names
        self.supports_position_hold, self.calibration, self.resolution = True, None, None
        for verb in ("read", "write", "pose", "images", "extras", "hold", "estop", "calibrate", "close"):
            setattr(self, verb, getattr(inner, verb))


@pytest.mark.parametrize(
    "names, refused",
    [
        (lambda keys: keys, None),
        (lambda keys: keys[::-1], "in order"),  # the shoulder command would close the gripper
        (lambda keys: keys[:2], "in order"),
    ],
)
def test_a_driver_whose_names_are_not_the_action_keys_is_refused(names, refused):
    d, _ = make_driver()
    drv = Renamed(d, names(d.description.action_keys))
    if refused is None:
        Robot("so101", driver=drv, twin=False).close()
        return
    with pytest.raises(ValueError, match=refused):
        Robot("so101", driver=drv, twin=False)


def test_the_driver_protocol_names_every_attribute_a_caller_reads():
    from strands_robots.robot.drivers import Driver

    d, _ = make_driver()
    drv = Renamed(d, d.description.action_keys)
    assert isinstance(drv, Driver)
    del drv.resolution  # bind() reads it to convert LeRobot degrees exactly: a Driver must say
    assert not isinstance(drv, Driver)
    with pytest.raises(TypeError, match="resolution"):
        Robot("so101", driver=drv, twin=False)


@pytest.mark.parametrize(
    "where, robot_id",
    [
        ("so_follower/so101_follower.json", None),  # LeRobot 0.6: every SO-10x arm is so_follower
        ("so_follower/my_arm.json", None),  # --robot.id=my_arm: the only file in the directory
        ("so_follower/my_arm.json", "my_arm"),
        ("so101_follower/so101_follower.json", None),  # LeRobot before 0.5
    ],
)
def test_a_lerobot_calibration_is_found_under_every_name_lerobot_writes(tmp_path, monkeypatch, where, robot_id):
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
    p = tmp_path / "robots" / where
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(CAL))
    d = ft.FeetechDriver(
        Description.load("so101"),
        port="/dev/fake",
        robot_id=robot_id,
        bus=ft.FeetechBus("/dev/fake", serial=FakeSerial()),
    )
    assert d.calibration_source == str(p) and d.calibration == Calibration.from_json(CAL)


def test_robot_takes_calibration_and_a_missing_file_names_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
    from strands_robots.robot.drivers.feetech import driver as ftd

    monkeypatch.setattr(ftd, "FeetechBus", lambda port, baud: ft.FeetechBus(port, serial=FakeSerial()))
    p = tmp_path / "anywhere.json"
    p.write_text(json.dumps(CAL))
    real = Robot("so101", port="/dev/fake", calibration=p, twin=False)
    assert real.calibrate(interactive=False) == Calibration.from_json(CAL)
    real.close()
    bare = Robot("so101", port="/dev/fake", twin=False)
    with pytest.raises(RuntimeError, match=r"so_follower/so101_follower\.json.*calibration="):
        bare.calibrate(interactive=False)
    bare.close()
    with pytest.raises(TypeError, match="hardware argument"):
        Robot("so101", calibration=p)


def test_calibrate_with_no_human_at_a_terminal_moves_nothing(monkeypatch):
    import io
    import sys

    d, s = make_driver()
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # an agent's tool thread: no tty to press Enter on
    with pytest.raises(RuntimeError, match="human at a terminal"):
        d.calibrate(interactive=True)
    assert writes(s) == []  # no torque-off, no homing-offset write
    d.close()


def _goal(s, mid):
    return int.from_bytes(s.regs[mid][ft.REG_GOAL_POSITION : ft.REG_GOAL_POSITION + 2], "little")


def test_a_far_goal_is_a_ramp_at_max_speed_by_default_and_trained_motion_is_untouched():
    """The first write from a read ramps from the measured pose; each write continues from the last goal; a step the
    cap allows passes as written. Default: 2 rad/s, well below the servo, well above a trained chunk."""
    from strands_robots.robot.drivers.slew import MAX_SPEED

    clock = [100.0]
    d, s = make_driver(max_speed=MAX_SPEED)
    d.slew._now = lambda: clock[0]
    mid = d.read()["shoulder_pan"]  # 2048: the middle of the calibrated range
    lo, hi = d.description.limits["shoulder_pan"]
    far = np.array([hi, mid, mid, mid, mid, 0.5])
    clock[0] += 1 / 30
    d.write(far)  # one period after the read: the goal moved by at most 2 rad/s * 1/30 s
    assert d.read()["shoulder_pan"] == pytest.approx(mid + MAX_SPEED / 30, abs=2e-3)
    assert d.extras()["slewed"] == 1 and d.extras()["max_speed"] == MAX_SPEED
    for _ in range(60):  # a run keeps asking for the far goal: it arrives at 2 rad/s, never in one write
        clock[0] += 1 / 30
        d.write(far)
    assert d.read()["shoulder_pan"] == pytest.approx(hi, abs=2e-3)
    q = d.read().positions.copy()
    q[0] -= 0.01  # a trained-motion sized step (0.3 rad/s at 30 Hz): written as given, not counted
    clock[0] += 1 / 30
    n = d.extras()["slewed"]
    d.write(q)
    assert d.read()["shoulder_pan"] == pytest.approx(hi - 0.01, abs=2e-3) and d.extras()["slewed"] == n


def test_the_cap_anchors_on_a_read_again_after_torque_goes_off_and_max_speed_none_opts_out():
    clock = [100.0]
    d, s = make_driver(max_speed=1.0)
    d.slew._now = lambda: clock[0]
    d.read()
    lo, hi = d.description.limits["shoulder_pan"]
    clock[0] += 2.0  # a long pause before the write still moves at most half a second's worth (DT_MAX)
    d.write(np.array([hi, 0, 0, 0, 0, 0.5]))
    assert d.read()["shoulder_pan"] == pytest.approx(d.counts_to_rad("shoulder_pan", 2048) + 0.5, abs=2e-3)
    d.torque(False)  # the arm may be moved by hand now
    s.actual[1] = 1024  # ... and is
    assert d.read()["shoulder_pan"] == pytest.approx(lo, abs=1e-3)
    clock[0] += 0.1
    d.write(np.array([hi, 0, 0, 0, 0, 0.5]))
    assert d.read()["shoulder_pan"] == pytest.approx(lo + 0.1, abs=2e-3)  # ramps from the NEW read, not the old goal
    raw, s2 = make_driver(max_speed=None)
    raw.read()
    raw.write(np.array([hi, 0, 0, 0, 0, 0.5]))
    assert raw.read()["shoulder_pan"] == pytest.approx(hi, abs=1e-3) and raw.extras()["max_speed"] is None
    with pytest.raises(ValueError, match="max_speed must be > 0"):
        make_driver(max_speed=0)


@pytest.mark.parametrize("before", ["nothing", "torque_off_then_moved_by_hand"])
def test_a_write_with_no_fresh_read_reads_first_so_the_cap_holds(before):
    """The cap must not depend on the caller reading first: a fresh driver, or one whose torque went off (the arm
    may have been moved by hand, ``calibrate()`` sweeps with raw reads), anchors on a read taken by the write."""
    from strands_robots.robot.drivers.slew import MAX_SPEED

    clock = [100.0]
    d, s = make_driver(max_speed=MAX_SPEED)
    d.slew._now = lambda: clock[0]
    lo, hi = d.description.limits["shoulder_pan"]
    if before != "nothing":
        d.read()
        d.torque(False)
        s.actual[1] = 1024  # moved by hand to the low end; no read follows
        clock[0] += 5.0
    present = d.counts_to_rad("shoulder_pan", s.actual[1])
    d.write(np.array([hi, 0, 0, 0, 0, 0.5]))
    goal = d.counts_to_rad("shoulder_pan", _goal(s, 1))
    assert abs(goal - present) <= MAX_SPEED * 0.5 + 2e-3, f"goal {goal:.3f} rad, body at {present:.3f}"
    assert d.extras()["slewed"] == 1


def test_a_hold_and_the_estop_write_the_measured_pose_through_no_cap_even_when_a_joint_is_blocked():
    """A blocked servo never reaches its goal. Through the cap, ``hold()`` would have moved the goal toward the
    present pose by only ``max_speed * dt`` and left the servo pushing; it must write Present_Position itself."""
    from strands_robots.robot.drivers.slew import MAX_SPEED

    class Blocked(FakeSerial):
        def write(self, data):
            keep = self.actual[1]
            super().write(data)
            self.actual[1] = keep  # servo 1 is held in place: the goal moves, the encoder does not

    clock = [100.0]
    d, s = make_driver(serial=Blocked(), max_speed=MAX_SPEED)
    d.slew._now = lambda: clock[0]
    present = d.read().positions.copy()
    lo, hi = d.description.limits["shoulder_pan"]
    far = present.copy()
    far[0] = hi
    for _ in range(30):  # one second of a run commanding a far goal: the goal ramps away, the joint stays
        clock[0] += 1 / 30
        d.write(far)
    goal_rad = d.counts_to_rad("shoulder_pan", _goal(s, 1))
    assert goal_rad - present[0] > 0.9 * MAX_SPEED, "the ramp should have carried the goal ~2 rad away"
    clock[0] += 1 / 30
    d.estop()  # -> hold(): Goal_Position := Present_Position, in ONE write
    assert d.counts_to_rad("shoulder_pan", _goal(s, 1)) == pytest.approx(present[0], abs=2e-3)
    assert abs(d.counts_to_rad("shoulder_pan", _goal(s, 1)) - d.read()["shoulder_pan"]) < 2e-3  # not pushing
    clock[0] += 1 / 30
    d.write(far)  # the cap continues from the held pose, not from the pre-hold goal
    assert d.counts_to_rad("shoulder_pan", _goal(s, 1)) == pytest.approx(present[0] + MAX_SPEED / 30, abs=2e-3)


def test_robot_id_names_the_lerobot_calibration_file_and_an_uncalibrated_arm_warns(tmp_path, monkeypatch):
    # agi-agent-min.py on the real bench: Robot(..., id="blue_follower") read calibration_source None, so the arm ran
    # uncalibrated with nothing said. The Robot's id is LeRobot's --robot.id, the calibration file's name.
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
    from strands_robots.robot.drivers.feetech import driver as ftd

    monkeypatch.setattr(ftd, "FeetechBus", lambda port, baud: ft.FeetechBus(port, serial=FakeSerial()))
    p = tmp_path / "robots" / "so_follower" / "blue_follower.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(CAL))
    (p.parent / "other.json").write_text(json.dumps(CAL))  # two files: no "only file" fallback
    arm = Robot("so101", port="/dev/fake", id="blue_follower", twin=False)
    assert arm.body.calibration_source == str(p) and arm.body.calibration == Calibration.from_json(CAL)
    arm.close()
    named = Robot("so101", port="/dev/fake", id="blue_follower", robot_id="other", twin=False)
    assert named.body.calibration_source == str(p.parent / "other.json")  # an explicit robot_id= still wins
    named.close()
    alias = Robot("so101_follower", port="/dev/fake", twin=False)  # the id is the name as written ...
    assert alias.id == "so101_follower" and alias.body.calibration_source is None  # ... but only id= names a file
    alias.close()
    with pytest.warns(UserWarning, match=r"no calibration found.*so_follower/nobody\.json"):
        bare = Robot("so101", port="/dev/fake", id="nobody", twin=False)
    assert bare.body.calibration is None
    bare.close()
