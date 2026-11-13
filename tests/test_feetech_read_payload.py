"""A Feetech read asks each servo for Present_Position only (2 bytes); velocity and load only with read_motion."""

import numpy as np

from strands_robots import Description, Robot
from strands_robots.mesh import wire as mesh_wire
from strands_robots.mesh.codec import observation_from_dict, observation_to_dict
from strands_robots.robot.drivers import feetech as ft
from tests.test_drivers import CAL, FakeSerial, make_driver


class CountingSerial(FakeSerial):
    """Records the per-servo byte count of every SYNC_READ, and moves servo 1 so velocity has something to say."""

    def __init__(self):
        super().__init__()
        self.sync_reads: list[int] = []
        for i in self.regs:  # a reading the firmware would report mid-move: 40 counts/s, 30% load against motion
            self.regs[i][0x3A:0x3C] = ft.word(40)
            self.regs[i][0x3C:0x3E] = ft.encode_sm(-300, 10)

    def write(self, data):
        if data[4] == ft.SYNC_READ:
            self.sync_reads.append(data[6])
        super().write(data)


def _driver(**kw):
    serial = CountingSerial()
    bus = ft.FeetechBus("/dev/fake", serial=serial)
    drv = ft.FeetechDriver(Description.load("so101"), port="/dev/fake", calibration=CAL, bus=bus, max_speed=None, **kw)
    return drv, serial


def test_default_read_is_position_only_and_reports_motion_as_unmeasured():
    d, s = _driver()
    js = d.read()
    assert s.sync_reads == [2]
    assert np.isfinite(js.positions).all()
    assert np.isnan(js.velocities).all() and np.isnan(js.efforts).all()
    assert d.extras()["read_motion"] is False


def test_read_motion_reads_velocity_and_load_with_the_same_positions():
    full, s = _driver(read_motion=True)
    lean, _ = _driver()
    js = full.read()
    assert s.sync_reads == [6]
    assert np.allclose(js.positions, lean.read().positions)
    assert np.isfinite(js.velocities).all() and (js.velocities != 0).any()
    assert np.allclose(js.efforts, -0.3)


def test_a_run_reads_two_bytes_per_servo_on_every_tick():
    d, s = _driver()
    s.sync_reads.clear()
    res = Robot("so101", driver=d, twin=False).run(lambda obs: {"shoulder_pan": 0.1}, hz=100, steps=20)
    assert res.steps == 20 and len(s.sync_reads) >= 20
    assert set(s.sync_reads) == {2}  # 12 bytes per tick of replies' parameters at 6 servos, not 36


def test_read_motion_reaches_the_driver_from_robot():
    d, s = _driver(read_motion=True)
    Robot("so101", driver=d, twin=False).observe(images=False)
    assert s.sync_reads[-1] == 6


def test_unmeasured_motion_crosses_the_mesh_wire_as_null():
    d, _ = make_driver(serial=FakeSerial())
    obs = Robot("so101", driver=d, twin=False).observe(images=False)
    back = observation_from_dict(mesh_wire.decode(mesh_wire.encode(observation_to_dict(obs)))).joints  # NaN refused
    assert np.allclose(back.positions, obs.joints.positions) and np.isnan(back.velocities).all()


def test_an_agent_tool_result_with_unmeasured_motion_is_strict_json():
    import json

    from strands_robots.tools.wire import json_safe

    d, _ = make_driver(serial=FakeSerial())
    out = json.loads(json.dumps(json_safe(Robot("so101", driver=d, twin=False).observe(images=False)), allow_nan=False))
    assert set(out["velocities"].values()) == {None} and all(isinstance(v, float) for v in out["joints"].values())
