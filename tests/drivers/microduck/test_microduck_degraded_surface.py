"""The Microduck driver's degraded surface: what it does when it cannot comply.

:mod:`tests.drivers.microduck.test_microduck_driver_over_socket` grades the happy
path over a real robotd socket. This file grades the other half - every path
where the driver refuses, degrades or gives up - because a native hardware
driver whose failure surface is ungraded is one that can report success about a
robot that did not move.

Three properties, each of them a promise the module makes about itself:

* **Delegate-only.** The module docstring's central claim ("THE DECISIVE FACT -
  robotd exposes **no per-joint write**") is the whole justification for
  :meth:`~strands_robots.drivers.microduck.MicroduckDriver.start_task` and
  :meth:`~strands_robots.drivers.microduck.MicroduckDriver.run_policy` refusing
  rather than streaming joint targets. Those two refusals are the driver's entire
  answer to "why does ``mode='real'`` not run a policy", so a refusal that
  silently became a success - or stopped naming the intent path a caller is
  supposed to use instead - would leave a caller with no route at all.
* **Fail loud, and name the cause.** The connect ladder distinguishes "nothing is
  listening", "robotd refused the handshake" and "robotd hung up mid-handshake";
  a discrete intent over a dead connection refuses under its own label; a read
  taken before any frame arrived refuses instead of publishing an empty pose as
  though it were a measurement.
* **Degrade, do not crash.** The reader thread survives a blank line and an
  unparseable line and keeps delivering frames; ``close`` is idempotent; a call
  robotd never answers reports the method and the budget, and leaves no pending
  slot behind.

Every socket cell drives a genuine ``AF_UNIX`` server - :class:`MockRobotd` for
the faithful protocol, :func:`_scripted_robotd` for the misbehaviours a faithful
server by definition does not exhibit - so the NDJSON framing and the id
correlation are exercised rather than mocked away.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

from strands_robots.drivers.microduck import (
    HARDWARE_JOINT_NAMES,
    LOCOMOTION_JOINT_NAMES,
    MICRODUCK_API_VERSION,
    MOUTH_INDEX,
    SKILLS,
    MicroduckDriver,
    _RobotdClient,
    map_hardware_joints,
)
from tests.mocks.microduck_robotd import STATE_PARAMS, MockRobotd

#: A path no daemon answers, for the cells that never intend to connect.
ABSENT_SOCKET = "/tmp/microduck-degraded-no-such.sock"

#: How long to wait for a streamed frame before calling the reader stuck.
FRAME_DEADLINE_S = 3.0


def _send(conn: socket.socket, obj: Mapping[str, object]) -> None:
    conn.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))


def _scripted_robotd(script: Callable[[socket.socket], None]) -> str:
    """Serve exactly one connection with ``script`` and return the socket path.

    :class:`MockRobotd` is faithful, so it cannot produce the frames this file
    needs: a refused handshake, a hang-up mid-handshake, a blank line, a line
    that is not JSON. ``script`` writes those bytes directly.
    """
    directory = tempfile.mkdtemp(prefix="scripted-robotd-")
    path = os.path.join(directory, "robotd.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    server.settimeout(FRAME_DEADLINE_S * 2)

    def serve() -> None:
        with contextlib.suppress(OSError):
            conn, _ = server.accept()
            try:
                script(conn)
            finally:
                conn.close()
        with contextlib.suppress(OSError):
            server.close()

    threading.Thread(target=serve, name="scripted-robotd", daemon=True).start()
    return path


def _requests(reader: Any) -> Iterator[int]:
    """Yield the id of each request the client sends, ending at EOF.

    Notifications (no ``id``) and any unparseable line are skipped, and the
    generator returns when the client hangs up - so a script never raises on the
    server thread after the driver under test has closed its socket.
    """
    while True:
        line = reader.readline()
        if not line:
            return
        try:
            request_id = json.loads(line).get("id")
        except json.JSONDecodeError:
            continue
        if request_id is not None:
            yield int(request_id)


def _answer(conn: socket.socket, request_id: int, result: Mapping[str, object]) -> None:
    _send(conn, {"jsonrpc": "2.0", "id": request_id, "result": result})


def _connected(server: MockRobotd) -> MicroduckDriver:
    driver = MicroduckDriver(tool_name="microduck", port=server.path, timeout=2.0)
    reason = driver.connect_eagerly()
    assert reason is None, reason
    return driver


def _robotd_that_errors_after_connect() -> str:
    """Connects cleanly, then answers every intent with a JSON-RPC error.

    A robotd that is up and *refusing* is the case that separates "the driver
    could not reach the daemon" from "the daemon declined the intent" - the two
    have different remedies, so they must not read the same.
    """

    def script(conn: socket.socket) -> None:
        handshake = ({"api_version": MICRODUCK_API_VERSION}, {"accepted": True}, {"healthy": True})
        for index, request_id in enumerate(_requests(conn.makefile("rb"))):
            if index < len(handshake):
                _answer(conn, request_id, handshake[index])
                continue
            _send(
                conn,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": "estop engaged"},
                },
            )

    return _scripted_robotd(script)


def _reason(envelope: dict[str, object]) -> str:
    blocks = envelope["content"]
    assert isinstance(blocks, list) and blocks, envelope
    return str(blocks[0].get("text", ""))


class TestTheDelegateOnlyRefusalsNameTheIntentPath:
    """robotd owns the policy, so the two host-rollout verbs refuse.

    This is the module docstring's headline claim. A refusal that stopped naming
    ``send_action`` would leave a caller told "no" with no route to yes.
    """

    def test_start_task_refuses_and_names_the_intent_path(self) -> None:
        driver = MicroduckDriver(port=ABSENT_SOCKET)
        envelope = driver.start_task("walk forward for a bit")
        assert envelope["status"] == "error"
        reason = _reason(envelope)
        assert reason.startswith("start_task:")
        assert "send_action" in reason, reason
        # The reason has to say *why*, or it reads as a transport failure.
        assert "per-joint write" in reason, reason

    def test_run_policy_refuses_and_names_both_the_intent_and_the_sim_route(self) -> None:
        driver = MicroduckDriver(port=ABSENT_SOCKET)
        envelope = driver.run_policy(policy_object=object())  # type: ignore[arg-type]
        assert envelope["status"] == "error"
        reason = _reason(envelope)
        assert reason.startswith("run_policy:")
        assert "send_action" in reason, reason
        assert 'mode="sim"' in reason, reason

    def test_neither_refusal_depends_on_being_disconnected(self) -> None:
        """The refusal is about robotd's wire, not about this connection.

        Connected or not, there is no per-joint write to stream to, so both
        verbs must refuse identically - otherwise a caller could conclude the
        rollout would work once the socket came up.
        """
        with MockRobotd() as server:
            driver = MicroduckDriver(port=server.path, timeout=2.0)
            assert driver.connect_eagerly() is None
            try:
                assert driver.is_connected
                assert driver.start_task("go")["status"] == "error"
                assert driver.run_policy(policy_object=object())["status"] == "error"  # type: ignore[arg-type]
            finally:
                driver.cleanup()


class TestTheConnectLadderNamesWhichStepFailed:
    """Three different failures, three different reasons.

    An operator reading "could not connect" cannot tell a missing daemon from a
    version mismatch, and the two have different remedies.
    """

    def test_nothing_listening_names_the_socket(self) -> None:
        driver = MicroduckDriver(port=ABSENT_SOCKET)
        reason = driver.connect_eagerly()
        assert reason is not None
        assert ABSENT_SOCKET in reason, reason
        assert driver.is_connected is False

    def test_a_refused_handshake_carries_robotds_own_error(self) -> None:
        def refuse(conn: socket.socket) -> None:
            for request_id in _requests(conn.makefile("rb")):
                _send(
                    conn,
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32602, "message": "api_version 16 is not supported"},
                    },
                )

        driver = MicroduckDriver(port=_scripted_robotd(refuse), timeout=2.0)
        reason = driver.connect_eagerly()
        assert reason is not None
        assert "Hello" in reason, reason
        # robotd's own message survives to the operator, not just "handshake failed".
        assert "api_version 16 is not supported" in reason, reason
        assert driver.is_connected is False

    def test_a_hangup_mid_handshake_is_a_reason_not_a_traceback(self) -> None:
        def hang_up(conn: socket.socket) -> None:
            conn.close()

        driver = MicroduckDriver(port=_scripted_robotd(hang_up), timeout=2.0)
        reason = driver.connect_eagerly()
        assert reason is not None
        assert "Hello" in reason, reason
        assert driver.is_connected is False
        # And the driver stays usable: the next call refuses rather than raising.
        assert driver.send_action({"vx": 0.1})["status"] == "error"

    def test_a_refused_subscribe_names_that_step_rather_than_the_handshake(self) -> None:
        """The handshake succeeded; it is the state stream that was declined."""

        def refuse_subscribe(conn: socket.socket) -> None:
            for index, request_id in enumerate(_requests(conn.makefile("rb"))):
                if index == 0:
                    _answer(conn, request_id, {"api_version": MICRODUCK_API_VERSION})
                    continue
                _send(
                    conn,
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32000, "message": "already subscribed"},
                    },
                )

        driver = MicroduckDriver(port=_scripted_robotd(refuse_subscribe), timeout=2.0)
        reason = driver.connect_eagerly()
        assert reason is not None
        assert "subscribe" in reason, reason
        assert "Hello" not in reason, "a declined subscribe was reported as a handshake failure"
        assert driver.is_connected is False

    def test_an_unavailable_battery_read_does_not_stop_the_driver_coming_up(self) -> None:
        """Battery rides on ``robot.health``, which is explicitly best-effort.

        A robotd that cannot read its bus still walks, so a missing battery must
        cost the ``battery_pct`` field and nothing else.
        """

        def no_health(conn: socket.socket) -> None:
            for index, request_id in enumerate(_requests(conn.makefile("rb"))):
                if index < 2:
                    _answer(
                        conn, request_id, {"api_version": MICRODUCK_API_VERSION} if index == 0 else {"accepted": True}
                    )
                    continue
                _send(
                    conn,
                    {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": "bus unreadable"}},
                )

        driver = MicroduckDriver(port=_scripted_robotd(no_health), timeout=2.0)
        try:
            assert driver.connect_eagerly() is None, "a best-effort health read blocked the connect"
            assert driver.is_connected is True
            status = asyncio.run(driver.get_status())["content"][0]["json"]
            assert status["battery_pct"] is None
        finally:
            driver.cleanup()

    def test_a_second_connect_on_a_live_connection_is_a_no_op_success(self) -> None:
        """Documented as idempotent, so a re-connect must not re-handshake."""
        with MockRobotd() as server:
            driver = _connected(server)
            try:
                before = list(server.methods)
                assert driver.connect_eagerly() is None
                assert server.methods == before, "a second connect_eagerly re-ran the handshake"
            finally:
                driver.cleanup()


class TestTheShutdownHookIsIdempotentAndDoesNotOverclaim:
    """``stop()`` is the ``-> None`` HardwareDriver hook, so it cannot report.

    What it *can* do is not lie by omission: ``motion_stopped`` may only become
    true once robotd has actually accepted ``robot.stop``.
    """

    def test_stop_without_a_connection_leaves_motion_stopped_false(self) -> None:
        driver = MicroduckDriver(port=ABSENT_SOCKET)
        asyncio.run(driver.stop())
        status = asyncio.run(driver.get_status())["content"][0]["json"]
        assert status["motion_stopped"] is False
        assert status["connected"] is False

    def test_stop_sends_robot_stop_and_records_it(self) -> None:
        with MockRobotd() as server:
            driver = MicroduckDriver(port=server.path, timeout=2.0)
            assert driver.connect_eagerly() is None
            try:
                asyncio.run(driver.stop())
                assert "robot.stop" in server.methods
                status = asyncio.run(driver.get_status())["content"][0]["json"]
                assert status["motion_stopped"] is True
            finally:
                driver.cleanup()

    def test_stop_after_cleanup_is_a_no_op_rather_than_a_raise(self) -> None:
        with MockRobotd() as server:
            driver = MicroduckDriver(port=server.path, timeout=2.0)
            assert driver.connect_eagerly() is None
            driver.cleanup()
            asyncio.run(driver.stop())  # the hook has no envelope; it must not raise
            driver.cleanup()  # and cleanup itself is idempotent
            assert driver.is_connected is False


class TestADegradedReadRefusesRatherThanInventing:
    """A read with nothing behind it is a refusal, not an empty measurement."""

    def test_read_state_before_any_frame_refuses(self) -> None:
        driver = MicroduckDriver(port=ABSENT_SOCKET)
        envelope = driver.read_state()
        assert envelope["status"] == "error"
        assert "no robot.state" in _reason(envelope)

    @pytest.mark.parametrize("verb", ["emergency_stop", "relax", "enable_torque"])
    def test_a_discrete_intent_refuses_under_its_own_label(self, verb: str) -> None:
        """The label matters: an operator needs to know *which* intent was refused."""
        driver = MicroduckDriver(port=ABSENT_SOCKET)
        envelope = getattr(driver, verb)()
        assert envelope["status"] == "error"
        reason = _reason(envelope)
        assert reason.startswith(f"{verb}:"), reason
        assert "not connected" in reason, reason

    def test_task_status_reports_no_policy_rather_than_claiming_one_runs(self) -> None:
        driver = MicroduckDriver(port=ABSENT_SOCKET)
        report = driver.get_task_status()
        assert report["status"] == "success"
        body = report["content"][0]["json"]
        assert body["running"] is False
        assert body["policy"] is None


class TestTheReaderSurvivesAMalformedFrame:
    """One bad line must not take the state stream down with it.

    The reader thread is the only route by which joints, IMU and pose reach the
    mesh, so a robotd that emits a blank keepalive or a truncated line has to
    cost one frame rather than every future frame.
    """

    def test_a_blank_line_and_a_non_json_line_do_not_stop_the_stream(self) -> None:
        def noisy(conn: socket.socket) -> None:
            """hello, then subscribe answered around two unreadable lines.

            The order matters: the reader thread only starts after the handshake,
            so the malformed lines have to arrive once the subscribe request has
            been seen, or ``hello``'s own synchronous read would consume them.
            """
            for index, request_id in enumerate(_requests(conn.makefile("rb"))):
                if index == 0:
                    _answer(conn, request_id, {"api_version": MICRODUCK_API_VERSION})
                    continue
                if index == 1:
                    conn.sendall(b"\n")  # a blank keepalive line
                    conn.sendall(b'{"jsonrpc":"2.0",trunc\n')  # a line that is not JSON
                    _answer(conn, request_id, {"accepted": True})
                    continue
                _answer(conn, request_id, {"healthy": True})
                break
            while True:  # the client's close ends this with an OSError the server suppresses
                _send(conn, {"jsonrpc": "2.0", "method": "robot.state", "params": STATE_PARAMS})
                time.sleep(0.01)

        driver = MicroduckDriver(port=_scripted_robotd(noisy), timeout=2.0)
        assert driver.connect_eagerly() is None
        try:
            deadline = time.time() + FRAME_DEADLINE_S
            while driver.read_state()["status"] != "success" and time.time() < deadline:
                time.sleep(0.01)
            # The frames after the malformed lines still arrive and still parse.
            assert driver.read_state()["status"] == "success", "the reader stopped at the bad line"
            assert len(driver.get_observation()) == len(LOCOMOTION_JOINT_NAMES)
        finally:
            driver.cleanup()


class TestAnUnansweredCallNamesTheMethodAndLeavesNoSlot:
    """The id-correlation table must not grow by one per unanswered request."""

    def test_a_timeout_names_the_method_and_the_budget(self) -> None:
        def silent(conn: socket.socket) -> None:
            for index, request_id in enumerate(_requests(conn.makefile("rb"))):
                if index == 0:
                    _answer(conn, request_id, {"api_version": MICRODUCK_API_VERSION})
                # every later request is read and deliberately left unanswered

        client = _RobotdClient(_scripted_robotd(silent), timeout=0.3)
        client.connect()
        client.hello(MICRODUCK_API_VERSION)
        client.start_reader(lambda _params: None)
        try:
            with pytest.raises(TimeoutError, match=r"robot\.stop"):
                client.call("robot.stop", {})
            assert not client._pending, "an unanswered request left its slot behind"
        finally:
            client.close()

    def test_a_jsonrpc_error_reply_is_raised_with_the_method_named(self) -> None:
        def erroring(conn: socket.socket) -> None:
            for index, request_id in enumerate(_requests(conn.makefile("rb"))):
                if index == 0:
                    _answer(conn, request_id, {"api_version": MICRODUCK_API_VERSION})
                    continue
                _send(
                    conn,
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32000, "message": "estop engaged"},
                    },
                )

        client = _RobotdClient(_scripted_robotd(erroring), timeout=2.0)
        client.connect()
        client.hello(MICRODUCK_API_VERSION)
        client.start_reader(lambda _params: None)
        try:
            with pytest.raises(ConnectionError, match="estop engaged"):
                client.call("robot.stop", {})
            # The reply resolved the slot even though it was an error.
            assert not client._pending
        finally:
            client.close()

    def test_close_is_idempotent_and_survives_a_peer_that_is_already_gone(self) -> None:
        with MockRobotd() as server:
            client = _RobotdClient(server.path, timeout=1.0)
            client.connect()
            client.hello(MICRODUCK_API_VERSION)
            client.start_reader(lambda _params: None)
        # The server is gone; closing must still not raise, twice over.
        client.close()
        client.close()
        assert client.alive is False


class TestTheFifteenToFourteenMapIsPositional:
    """The mouth-drop contract, and what the positional fallback really does."""

    def test_a_fifteen_wide_vector_drops_the_mouth_exactly(self) -> None:
        values = [float(index) for index in range(len(HARDWARE_JOINT_NAMES))]
        joints = map_hardware_joints(values)
        assert list(joints) == list(LOCOMOTION_JOINT_NAMES)
        # Every locomotion joint carries its own hardware index, mouth removed.
        expected = {name: float(index) for index, name in enumerate(HARDWARE_JOINT_NAMES) if index != MOUTH_INDEX}
        assert joints == expected
        assert HARDWARE_JOINT_NAMES[MOUTH_INDEX] not in joints

    def test_a_short_vector_degrades_to_a_partial_read(self) -> None:
        joints = map_hardware_joints([1.0, 2.0, 3.0])
        assert joints == dict(zip(LOCOMOTION_JOINT_NAMES[:3], [1.0, 2.0, 3.0], strict=True))

    def test_the_docstring_states_that_the_two_directions_differ(self) -> None:
        """The fallback is asymmetric, and the docstring has to say so.

        A shorter vector yields a partial read; a longer one yields a full
        14-joint read whose joints after the mouth are named one position early.
        Describing both as "a partial read" would tell a reader the grown case is
        safe to ignore.
        """
        doc = " ".join((map_hardware_joints.__doc__ or "").split())
        assert "not symmetric" in doc, doc
        assert "partial read" in doc, doc
        assert "one position early" in doc, doc


class TestTheCommandPathRefusesBeforeItSends:
    """``send_action`` is the only path that moves the robot, so it gates first.

    Each of these refusals has to reach the caller *instead of* a wire write:
    a twist the driver could not read, or a key it does not recognise, must not
    become a partial command with the readable half applied.
    """

    def test_a_wrong_robot_name_is_refused_and_names_both(self) -> None:
        driver = MicroduckDriver(tool_name="microduck", port=ABSENT_SOCKET)
        envelope = driver.send_action({"vx": 0.1}, robot_name="unitree_g1")
        assert envelope["status"] == "error"
        reason = _reason(envelope)
        assert "unitree_g1" in reason and "microduck" in reason, reason

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_a_non_finite_twist_is_refused_and_nothing_reaches_the_wire(self, value: float) -> None:
        with MockRobotd() as server:
            driver = _connected(server)
            try:
                envelope = driver.send_action({"vx": value})
                assert envelope["status"] == "error"
                assert "vx" in _reason(envelope), _reason(envelope)
                assert "robot.move" not in server.methods, "a refused twist still reached robotd"
            finally:
                driver.cleanup()

    def test_an_action_naming_no_intent_lists_what_it_would_accept(self) -> None:
        with MockRobotd() as server:
            driver = _connected(server)
            try:
                envelope = driver.send_action({"elbow": 0.2})
                assert envelope["status"] == "error"
                reason = _reason(envelope)
                assert "nothing to send" in reason, reason
                # A refusal that does not list the accepted keys is a dead end.
                assert "vx" in reason, reason
            finally:
                driver.cleanup()


class TestADaemonThatDeclinesAnIntentIsReportedNotSwallowed:
    """robotd up and refusing reads differently from robotd unreachable."""

    def test_send_action_reports_the_method_the_daemon_declined(self) -> None:
        driver = MicroduckDriver(port=_robotd_that_errors_after_connect(), timeout=2.0)
        assert driver.connect_eagerly() is None
        try:
            # Derived from the module's own vocabulary, so a renamed skill
            # cannot leave this cell silently grading a refusal it did not mean.
            envelope = driver.send_action({"skill": sorted(SKILLS)[0]})
            assert envelope["status"] == "error"
            reason = _reason(envelope)
            assert "robot.do" in reason, reason
            assert "estop engaged" in reason, reason
        finally:
            driver.cleanup()

    def test_a_declined_emergency_stop_refuses_under_its_own_label(self) -> None:
        driver = MicroduckDriver(port=_robotd_that_errors_after_connect(), timeout=2.0)
        assert driver.connect_eagerly() is None
        try:
            envelope = driver.emergency_stop()
            assert envelope["status"] == "error"
            reason = _reason(envelope)
            assert reason.startswith("emergency_stop:"), reason
            assert "estop engaged" in reason, reason
        finally:
            driver.cleanup()

    def test_a_declined_stop_hook_does_not_claim_the_motion_stopped(self) -> None:
        """The hook returns ``None``, so this is the only thing it can get wrong.

        ``motion_stopped`` is what an operator reads to decide whether the robot
        is safe to approach. A hook that set it after robotd declined the stop
        would be an affirmative lie on the safety path.
        """
        driver = MicroduckDriver(port=_robotd_that_errors_after_connect(), timeout=2.0)
        assert driver.connect_eagerly() is None
        try:
            asyncio.run(driver.stop())  # declined, and the hook has no envelope to say so
            status = asyncio.run(driver.get_status())["content"][0]["json"]
            assert status["motion_stopped"] is False, "a declined stop was reported as a halt"
            assert status["connected"] is True, "the connection is fine; only the intent was declined"
        finally:
            driver.cleanup()
