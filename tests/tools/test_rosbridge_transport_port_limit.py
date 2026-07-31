"""The rosbridge surfaces refuse a port their transport cannot address.

``65535`` is a legal TCP port. The shared owner of the 16-bit domain
(:func:`strands_robots.utils.tcp_port_error`) accepts it, the kernel binds it,
and ``gr00t_inference`` connects to it - but the WebSocket transport behind
roslibpy builds its URL with

    assert port is None or (type(port) == int and port in range(0, 65535))

in ``autobahn/websocket/util.py``, and ``range(0, 65535)`` stops one short. So a
value inside the tool's own accepted domain used to leave ``use_rosbridge`` as a
bare ``AssertionError`` with an empty message, raised out of a function
annotated ``-> dict[str, Any]``: an agent got an exception where every other
refusal is a result dict, naming neither the tool nor the parameter.

The bound is a property of one transport, not of the port space, so it is
declared beside that transport and applied ahead of the backend probe - the
refusal reads the same whether or not roslibpy is installed, and no socket is
dialed to discover it. These tests pin three things that have to stay true
together: the refusal reaches the caller through the envelope, the shared 16-bit
domain is *not* narrowed to match, and the transport still has the quirk that
justifies the divergence. If a future autobahn accepts 65535, the premise class
at the bottom fails and says so, which is the signal to widen
``_TRANSPORT_MAX_PORT`` rather than to delete a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.mesh.rosbridge_robot import RosbridgeRobot
from strands_robots.tools import use_rosbridge as ur
from strands_robots.tools.use_rosbridge import use_rosbridge
from strands_robots.utils import tcp_port_error

# The one port that the shared domain accepts and the transport cannot carry.
UNADDRESSABLE_PORT = 65535

# Its neighbour, which both accept: every assertion about the refusal is paired
# with this control so a blanket refusal cannot pass as a targeted one.
ADDRESSABLE_PORT = 65534


def _robot(port: int) -> RosbridgeRobot:
    """A bridge that differs from a usable one only in its port."""
    return RosbridgeRobot(
        node_name="probe",
        cmd_vel_topic="/cmd_vel",
        odom_topic="/odom",
        host="127.0.0.1",
        port=port,
    )


def _texts(result: dict) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


class TestTheToolReportsItThroughTheEnvelope:
    """The failure channel is the result dict, not an exception."""

    def test_the_top_of_the_range_is_an_error_result(self) -> None:
        result = use_rosbridge(action="status", host="127.0.0.1", port=UNADDRESSABLE_PORT, timeout=0.05)

        assert result["status"] == "error"

    def test_the_message_names_the_parameter_the_transport_and_the_range(self) -> None:
        text = _texts(use_rosbridge(action="status", host="127.0.0.1", port=UNADDRESSABLE_PORT, timeout=0.05))

        assert "use_rosbridge" in text
        assert str(UNADDRESSABLE_PORT) in text
        assert "rosbridge WebSocket transport" in text
        assert f"1-{ADDRESSABLE_PORT}" in text

    @pytest.mark.parametrize("action", sorted(ur._ACTIONS))
    def test_every_action_refuses_it(self, action: str) -> None:
        """The port is read by whichever action dials, so none may accept it."""
        result = use_rosbridge(
            action=action,
            host="127.0.0.1",
            port=UNADDRESSABLE_PORT,
            topic="/chatter",
            service="/rosout/get_loggers",
            type="std_msgs/String",
            timeout=0.05,
        )

        assert result["status"] == "error"
        assert "cannot address" in _texts(result)

    def test_the_refusal_precedes_the_backend_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing is imported, probed or dialed to learn the port is unusable."""

        def fail(*args: object, **kwargs: object) -> bool:
            raise AssertionError("the backend must not be probed for a port refused up front")

        monkeypatch.setattr(ur._backend, "available", fail)
        monkeypatch.setattr(ur._backend, "connect", fail)

        result = use_rosbridge(action="status", host="127.0.0.1", port=UNADDRESSABLE_PORT, timeout=0.05)

        assert result["status"] == "error"

    def test_the_neighbour_below_is_not_refused(self) -> None:
        """The control: 65534 reaches the transport, so the guard is targeted."""
        text = _texts(use_rosbridge(action="status", host="127.0.0.1", port=ADDRESSABLE_PORT, timeout=0.05))

        assert "cannot address" not in text

    @pytest.mark.parametrize("port", [1, 5555, ADDRESSABLE_PORT, UNADDRESSABLE_PORT])
    def test_no_port_in_the_accepted_domain_raises(self, port: int) -> None:
        """The whole point: a dict-returning tool returns a dict for each of these."""
        result = use_rosbridge(action="status", host="127.0.0.1", port=port, timeout=0.05)

        assert result["status"] in {"success", "error"}


class TestTheRobotRefusesItAtConstruction:
    """``RosbridgeRobot`` forwards every call through the tool, so it shares the bound."""

    def test_the_constructor_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot address"):
            _robot(UNADDRESSABLE_PORT)

    def test_the_message_names_the_class(self) -> None:
        with pytest.raises(ValueError, match="RosbridgeRobot"):
            _robot(UNADDRESSABLE_PORT)

    def test_the_neighbour_below_constructs(self) -> None:
        robot = _robot(ADDRESSABLE_PORT)

        assert robot.port == ADDRESSABLE_PORT


class TestTheSharedDomainIsNotNarrowedToMatch:
    """The divergence is deliberate: the bound belongs to a transport, not to TCP."""

    def test_the_shared_owner_still_accepts_the_whole_port_space(self) -> None:
        assert tcp_port_error(UNADDRESSABLE_PORT, "port", "ctx") is None

    def test_the_transport_bound_is_one_below_the_shared_ceiling(self) -> None:
        assert ur._TRANSPORT_MAX_PORT == UNADDRESSABLE_PORT - 1

    def test_the_helper_accepts_everything_the_transport_can_carry(self) -> None:
        assert ur._transport_port_error(ADDRESSABLE_PORT, "port", "ctx") is None
        assert ur._transport_port_error(1, "port", "ctx") is None

    def test_only_the_rosbridge_surfaces_carry_the_narrower_bound(self) -> None:
        """A transport bound that leaked onto another surface would be a bug.

        Same scan shape as the shared-owner check in
        ``test_gr00t_numeric_option_guards.py``: read the caller-facing modules
        that take a port and assert which of them apply this bound. The
        inference-service tool talks to a plain socket and must keep the whole
        port space.
        """
        root = Path(ur.__file__).resolve().parent.parent
        sources = {
            path.name: path.read_text(encoding="utf-8")
            for glob in ("tools/*.py", "mesh/*_robot.py")
            for path in sorted(root.glob(glob))
        }
        carriers = {name for name, text in sources.items() if "_transport_port_error" in text}

        assert carriers == {"use_rosbridge.py", "rosbridge_robot.py"}


class TestTheTransportPremise:
    """Why the bound exists. If autobahn is fixed, these fail and say to widen it.

    Constructing ``roslibpy.Ros`` builds the URL and opens no socket, so the
    premise is checked without a rosbridge server.
    """

    def test_the_transport_still_rejects_the_top_of_the_range(self) -> None:
        roslibpy = pytest.importorskip("roslibpy")

        with pytest.raises(AssertionError):
            roslibpy.Ros(host="127.0.0.1", port=UNADDRESSABLE_PORT)

    def test_the_transport_accepts_the_neighbour_below(self) -> None:
        """Pins that the assert is about that one value, not about the call."""
        roslibpy = pytest.importorskip("roslibpy")

        assert roslibpy.Ros(host="127.0.0.1", port=ADDRESSABLE_PORT) is not None

    def test_the_rejection_carries_no_message_of_its_own(self) -> None:
        """Why translating it is necessary: the raw failure names nothing."""
        roslibpy = pytest.importorskip("roslibpy")

        with pytest.raises(AssertionError) as excinfo:
            roslibpy.Ros(host="127.0.0.1", port=UNADDRESSABLE_PORT)

        assert str(excinfo.value) == ""
