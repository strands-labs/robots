"""A servo that does not answer a sync_read is named by its own ID, never by the neighbour whose reply came next."""

import pytest

from strands_robots.robot.drivers import feetech as ft
from tests.test_drivers import EchoSerial, FakeSerial, make_driver

IDS = [1, 2, 3, 4, 5, 6]


def silent(base, quiet, times=None):
    """``base`` with the ``quiet`` servos left out of every SYNC_READ reply (or of the first ``times`` replies)."""

    class Silent(base):
        left = times

        def write(self, data):
            if data[4] != ft.SYNC_READ or self.left == 0:
                return super().write(data)
            if self.left is not None:
                self.left -= 1
            self.frames.append(bytes(data))
            if isinstance(self, EchoSerial):
                self.rx += bytes(data)
            reg, n, *ids = data[5 : 5 + data[3] - 2]
            for i in ids:
                if i not in quiet:
                    self.rx += self._status(i, bytes(self.regs[i][reg : reg + n]))

    return Silent()


@pytest.mark.parametrize("wire", [FakeSerial, EchoSerial])
@pytest.mark.parametrize(("quiet", "named"), [({3}, r"\[3\]"), ({2, 5}, r"\[2, 5\]"), ({1}, r"\[1\]"), ({6}, r"\[6\]")])
def test_the_silent_servo_is_the_one_named(wire, quiet, named):
    bus = ft.FeetechBus("/dev/fake", serial=silent(wire, quiet))
    bus.ping(1)  # the bus learns whether this adapter echoes
    with pytest.raises(ft.ProtocolError, match=rf"no reply from servo\(s\) {named} \(got {6 - len(quiet)} of 6\)"):
        bus.sync_read(ft.REG_PRESENT_POSITION, 2, IDS)


def test_a_silent_middle_servo_costs_no_timeout():
    serial = silent(FakeSerial, {3})
    bus = ft.FeetechBus("/dev/fake", serial=serial)
    with pytest.raises(ft.ProtocolError, match=r"\[3\]"):
        bus.sync_read(ft.REG_PRESENT_POSITION, 2, IDS)
    assert serial.waited == 0  # servo 6 answered last: nothing was asked for that never came


def test_a_servo_silent_once_is_read_on_the_retry_and_one_silent_every_time_reaches_the_driver():
    d, _ = make_driver(serial=silent(FakeSerial, {4}, times=1))
    assert d.read().names == d.names
    d, _ = make_driver(serial=silent(FakeSerial, {4}))
    with pytest.raises(ft.ProtocolError, match=r"servo\(s\) \[4\]"):
        d.read()


class Stranger(FakeSerial):
    """A seventh device on the bus answers every SYNC_READ after servo 2, though nobody asked it."""

    def write(self, data):
        super().write(data)
        if data[4] == ft.SYNC_READ:
            first, rest = self.rx[: 2 * 8], self.rx[2 * 8 :]  # two 8-byte replies (2 params each)
            self.rx = bytearray(first + self._status(9, b"\x00\x08") + rest)


def test_a_reply_from_a_servo_not_asked_is_refused_not_filed():
    bus = ft.FeetechBus("/dev/fake", serial=Stranger())
    with pytest.raises(ft.ProtocolError, match="servo 9, which was not asked"):
        bus.sync_read(ft.REG_PRESENT_POSITION, 2, IDS)
