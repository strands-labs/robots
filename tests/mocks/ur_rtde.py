"""A UR controller double, shaped like the RTDE interfaces the driver imports.

``ur_rtde`` is a compiled SDK that is not installed on CI, and
:func:`strands_robots.drivers.ur._resolve_rtde` reaches it through
``importlib.import_module``. The ``fake_rtde`` fixture installs these doubles as
the ``rtde_control`` / ``rtde_receive`` importables, so the driver's *real*
resolution path finds them: the same code that would find the SDK on a
workstation finds these here, and a refactor that stopped resolving the SDK
correctly fails the tests too.

The doubles answer the register reads a real controller answers, and record the
motion commands, so a test can assert on the vector that reached the wire.
"""

from __future__ import annotations

from typing import Any

#: A plausible measured pose, in the driver's joint order. Not all zeros: a
#: transposition or a dropped element in the wire vector is invisible against a
#: vector of identical values.
MEASURED_Q: tuple[float, ...] = (0.1, -1.2, 1.4, -0.3, 0.7, 0.05)

#: The controller's own TCP readings, distinct per element for the same reason.
MEASURED_TCP_POSE: tuple[float, ...] = (0.4, -0.15, 0.32, 1.2, -0.4, 0.03)
MEASURED_TCP_SPEED: tuple[float, ...] = (0.01, 0.0, -0.02, 0.0, 0.0, 0.0)
MEASURED_WRENCH: tuple[float, ...] = (1.5, -0.5, 9.81, 0.2, -0.1, 0.0)


class FakeReceive:
    """The RTDE receive interface: register reads, and the two mode words."""

    def __init__(self, host: str, frequency: float | None = None) -> None:
        self.host = host
        self.frequency = frequency
        self.q = list(MEASURED_Q)
        self.robot_mode = 7  # RUNNING
        self.safety_mode = 1  # NORMAL
        self.disconnected = False

    def getActualQ(self) -> list[float]:  # noqa: N802 - the SDK's own spelling
        return list(self.q)

    def getActualQd(self) -> list[float]:  # noqa: N802
        return [0.0] * len(self.q)

    def getActualTCPPose(self) -> list[float]:  # noqa: N802
        return list(MEASURED_TCP_POSE)

    def getActualTCPSpeed(self) -> list[float]:  # noqa: N802
        return list(MEASURED_TCP_SPEED)

    def getActualTCPForce(self) -> list[float]:  # noqa: N802
        return list(MEASURED_WRENCH)

    def getRobotMode(self) -> int:  # noqa: N802
        return self.robot_mode

    def getSafetyMode(self) -> int:  # noqa: N802
        return self.safety_mode

    def disconnect(self) -> None:
        self.disconnected = True


class FakeControl:
    """The RTDE control interface: records ``servoJ``, reports acceptance."""

    def __init__(self, host: str, frequency: float | None = None) -> None:
        self.host = host
        self.frequency = frequency
        self.servoj_calls: list[tuple[list[float], tuple[float, ...]]] = []
        self.servo_stops = 0
        self.accepts = True
        self.disconnected = False

    def servoJ(  # noqa: N802 - the SDK's own spelling
        self,
        q: list[float],
        speed: float,
        acceleration: float,
        time: float,
        lookahead_time: float,
        gain: float,
    ) -> bool:
        self.servoj_calls.append((list(q), (speed, acceleration, time, lookahead_time, gain)))
        return self.accepts

    def servoStop(self) -> None:  # noqa: N802
        self.servo_stops += 1

    def disconnect(self) -> None:
        self.disconnected = True


class FakeRTDE:
    """The pair of doubles plus the interfaces each side actually constructed."""

    def __init__(self) -> None:
        self.receives: list[FakeReceive] = []
        self.controls: list[FakeControl] = []

    def make_receive(self, host: str, frequency: float | None = None) -> FakeReceive:
        """Build a receive double, recording that the driver asked for one."""
        interface = FakeReceive(host, frequency)
        self.receives.append(interface)
        return interface

    def make_control(self, host: str, frequency: float | None = None) -> FakeControl:
        """Build a control double, recording that the driver asked for one."""
        interface = FakeControl(host, frequency)
        self.controls.append(interface)
        return interface

    @property
    def receive(self) -> FakeReceive:
        """The single receive interface, asserting exactly one was built."""
        assert len(self.receives) == 1, f"expected one receive interface, got {len(self.receives)}"
        return self.receives[0]

    @property
    def control(self) -> FakeControl:
        """The single control interface, asserting exactly one was built."""
        assert len(self.controls) == 1, f"expected one control interface, got {len(self.controls)}"
        return self.controls[0]


def json_of(envelope: dict[str, Any]) -> dict[str, Any]:
    """Read the single JSON block out of a driver envelope."""
    content = envelope["content"][0]
    assert "json" in content, f"expected a json block, got {content}"
    return dict(content["json"])


def text_of(envelope: dict[str, Any]) -> str:
    """Read the refusal text out of an error envelope."""
    return str(envelope["content"][0]["text"])
