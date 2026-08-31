"""A scripted Robotiq 2F-85 speaking Modbus TCP, shared by the driver tests.

A hardware-shaped fake rather than a mock: it accepts a real TCP connection,
reads real MBAP-framed requests and answers with real register payloads, and it
enforces the two behaviours that make a 2F-85 different from a register file -
it ignores a position command until it has been activated, and it only moves on
a frame that sets ``rGTO``. A driver that skipped the activation sequence would
pass against a mock that just stored bytes; against this it reports a gripper
that never moves, which is exactly what the hardware does.

The fake is the oracle for the codec too: it decodes with :mod:`struct` off the
manual's byte layout rather than by calling the codec under test, so a frame the
driver builds wrongly is not un-built by the same mistake.
"""

from __future__ import annotations

import socket
import struct
import threading
from collections.abc import Callable, Iterator
from types import TracebackType

import pytest

from strands_robots.drivers.robotiq import RobotiqDriver

INPUT_BASE = 0x07D0
OUTPUT_BASE = 0x03E8
READ_INPUT_REGISTERS = 0x04
WRITE_MULTIPLE_REGISTERS = 0x10


class FakeGripper:
    """A 2F-85 that answers Modbus TCP on an ephemeral localhost port.

    Attributes:
        port: The port the server bound to; pass it to the driver as ``tcp_port``.
        commanded: Every ``rPR`` the gripper was told to go to, in order, for a
            test to assert what actually reached the wire.
        writes: Every decoded output payload, for asserting the flag bits.
    """

    def __init__(
        self,
        *,
        activation_reads: int = 1,
        object_status: int = 3,
        fault: int = 0,
        current: int = 0,
        exception_code: int | None = None,
        starts_activated: bool = False,
        never_activates: bool = False,
    ) -> None:
        """Configure the gripper's scripted behaviour.

        Args:
            activation_reads: Status reads the calibration stroke takes before
                ``gSTA`` reports ACTIVE.
            object_status: ``gOBJ`` to report - ``2`` is a grasp, ``3`` is "at
                the requested position, holding nothing".
            fault: ``gFLT`` to report.
            current: ``gCU`` to report, in units of 10 mA.
            exception_code: When set, every request is answered with a Modbus
                exception carrying this code.
            starts_activated: Begin already activated, as a gripper that was
                left powered would.
            never_activates: Stay in ACTIVATING forever, so a driver's
                activation timeout can be exercised.
        """
        self._activation_reads = activation_reads
        self._object_status = object_status
        self._fault = fault
        self._current = current
        self._exception_code = exception_code
        self._never_activates = never_activates

        self.activated = starts_activated
        self.go_to = False
        self.activation = 3 if starts_activated else 0
        self.position = 0
        self.requested = 0
        self.speed = 0
        self.force = 0
        self.commanded: list[int] = []
        self.writes: list[dict[str, int]] = []
        self._reads_since_activate = 0

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.port: int = self._server.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="fake-2f85", daemon=True)
        self._thread.start()

    def __enter__(self) -> FakeGripper:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Stop serving and release the listening socket."""
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)

    # ---------------------------------------------------------------- #
    # Wire handling.                                                    #
    # ---------------------------------------------------------------- #

    def _serve(self) -> None:
        try:
            conn, _addr = self._server.accept()
        except OSError:
            return  # closed before a client arrived
        with conn:
            while not self._stop.is_set():
                header = self._read_exactly(conn, 7)
                if header is None:
                    return
                (length,) = struct.unpack(">H", header[4:6])
                body = self._read_exactly(conn, length - 1)
                if body is None:
                    return
                transaction, _protocol, _length, unit = struct.unpack(">HHHB", header)
                try:
                    conn.sendall(self._answer(transaction, unit, body))
                except OSError:
                    return

    @staticmethod
    def _read_exactly(conn: socket.socket, count: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = count
        while remaining > 0:
            try:
                chunk = conn.recv(remaining)
            except OSError:
                return None
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _answer(self, transaction: int, unit: int, body: bytes) -> bytes:
        function = body[0]
        if self._exception_code is not None:
            return _frame(transaction, unit, struct.pack(">BB", function | 0x80, self._exception_code))
        if function == WRITE_MULTIPLE_REGISTERS:
            address, count, _byte_count = struct.unpack(">HHB", body[1:6])
            self._apply(body[6 : 6 + count * 2])
            return _frame(transaction, unit, struct.pack(">BHH", function, address, count))
        if function == READ_INPUT_REGISTERS:
            address, count = struct.unpack(">HH", body[1:5])
            assert address == INPUT_BASE, f"read at {address:#06x}, expected {INPUT_BASE:#06x}"
            payload = self._status_payload()[: count * 2]
            return _frame(transaction, unit, struct.pack(">BB", function, len(payload)) + payload)
        return _frame(transaction, unit, struct.pack(">BB", function | 0x80, 0x01))

    def _apply(self, payload: bytes) -> None:
        """Absorb an output payload the way the gripper's firmware would."""
        action, _r1, _r2, position, speed, force = struct.unpack(">BBBBBB", payload)
        activate = bool(action & 0x01)
        self.go_to = bool(action & 0x08)
        self.writes.append(
            {
                "activate": int(activate),
                "go_to": int(self.go_to),
                "auto_release": int(bool(action & 0x10)),
                "position": position,
                "speed": speed,
                "force": force,
            }
        )
        if activate and not self.activated:
            # rACT rising edge starts the calibration stroke.
            self.activation = 1
            self._reads_since_activate = 0
        if not activate:
            self.activation = 0
        self.activated = activate
        self.requested = position
        self.speed = speed
        self.force = force
        # THE BEHAVIOUR THAT MATTERS: a position only lands when the gripper is
        # activated and the frame set rGTO. Anything else is silently dropped,
        # which is what the hardware does and why a driver must activate first.
        if self.activated and self.activation == 3 and self.go_to:
            self.position = position
            self.commanded.append(position)

    def _status_payload(self) -> bytes:
        if self.activation == 1 and not self._never_activates:
            self._reads_since_activate += 1
            if self._reads_since_activate >= self._activation_reads:
                self.activation = 3
        status = (
            (0x01 if self.activated else 0x00)
            | (0x08 if self.go_to else 0x00)
            | ((self.activation & 0x03) << 4)
            | ((self._object_status & 0x03) << 6)
        )
        return struct.pack(">BBBBBB", status, 0, self._fault, self.requested, self.position, self._current)


def _frame(transaction: int, unit: int, pdu: bytes) -> bytes:
    return struct.pack(">HHHB", transaction, 0, len(pdu) + 1, unit) + pdu


@pytest.fixture
def gripper() -> Iterator[Callable[..., FakeGripper]]:
    """Yield a factory for scripted grippers, each closed at teardown.

    A factory rather than a single instance because most behaviours here need a
    gripper configured differently - already activated, reporting a grasp,
    answering only exceptions - and a test that had to remember to close its own
    server would leak a listening socket on every failure.

    Yields:
        A callable taking :class:`FakeGripper`'s keyword arguments.
    """
    built: list[FakeGripper] = []

    def _build(**kwargs: object) -> FakeGripper:
        made = FakeGripper(**kwargs)  # type: ignore[arg-type]
        built.append(made)
        return made

    yield _build
    for made in built:
        made.close()


@pytest.fixture
def connected(gripper: Callable[..., FakeGripper]) -> Iterator[Callable[..., tuple[RobotiqDriver, FakeGripper]]]:
    """Yield a factory for drivers already connected to a scripted gripper.

    Yields:
        A callable taking :class:`FakeGripper`'s keyword arguments and returning
        the connected driver paired with the gripper it drives. Asserts the
        connection succeeded, so a test body starts from a working gripper.
    """
    drivers: list[RobotiqDriver] = []

    def _build(**kwargs: object) -> tuple[RobotiqDriver, FakeGripper]:
        fake = gripper(**kwargs)
        driver = RobotiqDriver(tool_name="robotiq_2f85", port="127.0.0.1", tcp_port=fake.port, timeout=2.0)
        drivers.append(driver)
        assert driver.connect_eagerly() is None
        return driver, fake

    yield _build
    for driver in drivers:
        driver.cleanup()
