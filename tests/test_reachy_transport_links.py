"""Behavior tests for the Reachy Mini hardware-link transports.

Exercises the real-time I/O abstractions in
``strands_robots.device_connect.reachy_transport``:

- ``ZenohLink`` -- wireless variant that bridges Device Connect's Zenoh
  pub/sub to the driver's joint/IMU callbacks.
- ``WebSocketLink`` -- lite variant that talks to the daemon's ``/ws/sdk``
  WebSocket, including the command-type mapping and the read loop.
- ``api`` REST helper error handling (HTTP error -> structured body, generic
  failure -> ``{"error": ...}``).
- ``resolve_host`` -- both branches of hostname resolution, including the
  lookup-failure fallback that passes an unresolvable name through unchanged.

Each of those surfaces also carries a *degradation* branch, exercised here for
the same reason: a link that dies on one imperfect input is indistinguishable
from a disconnected robot. A malformed sensor frame is dropped so the
subscription survives, the WebSocket header keyword falls back to the legacy
spelling when ``websockets.connect`` cannot be introspected (so a configured
bearer credential is not lost), and an unresolvable hostname is passed through
rather than refused.

All transports are mocked; no hardware, daemon, or network is touched.
"""

import asyncio
import inspect
import json
import socket
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from strands_robots.device_connect import reachy_transport
from strands_robots.device_connect.reachy_transport import (
    WebSocketLink,
    ZenohLink,
    api,
)


def _run(coro):
    return asyncio.run(coro)


class TestZenohLink(unittest.TestCase):
    """ZenohLink bridges Zenoh pub/sub to driver callbacks."""

    def test_start_subscribes_both_topics_with_prefix(self):
        transport = MagicMock()
        transport.subscribe = AsyncMock()
        link = ZenohLink(transport, prefix="dev/reachy")

        _run(link.start(on_joints=lambda d: None, on_imu=lambda d: None))

        subscribed = [c.args[0] for c in transport.subscribe.call_args_list]
        self.assertEqual(
            subscribed,
            ["dev/reachy/joint_positions", "dev/reachy/imu_data"],
        )

    def test_subscription_callbacks_decode_and_forward(self):
        transport = MagicMock()
        transport.subscribe = AsyncMock()
        link = ZenohLink(transport, prefix="p")
        joints_seen: list = []
        imu_seen: list = []

        _run(link.start(on_joints=joints_seen.append, on_imu=imu_seen.append))

        # Grab the wrapper callbacks the link registered with the transport.
        on_joints_cb = transport.subscribe.call_args_list[0].args[1]
        on_imu_cb = transport.subscribe.call_args_list[1].args[1]

        _run(on_joints_cb(json.dumps({"pos": [1, 2, 3]}).encode()))
        _run(on_imu_cb(json.dumps({"accel": [0, 0, 9.8]}).encode()))

        self.assertEqual(joints_seen, [{"pos": [1, 2, 3]}])
        self.assertEqual(imu_seen, [{"accel": [0, 0, 9.8]}])

    def test_malformed_joints_frame_is_dropped_without_raising(self):
        """Covers the joints topic; its IMU twin is pinned by the test below."""
        transport = MagicMock()
        transport.subscribe = AsyncMock()
        link = ZenohLink(transport, prefix="p")
        joints_seen: list = []

        _run(link.start(on_joints=joints_seen.append, on_imu=lambda d: None))
        on_joints_cb = transport.subscribe.call_args_list[0].args[1]

        # Invalid JSON must be swallowed so the subscription stays alive.
        _run(on_joints_cb(b"not-json{"))
        self.assertEqual(joints_seen, [])

    def test_malformed_imu_frame_is_dropped_and_subscription_survives(self):
        """The IMU topic carries the same tolerance as the joints topic.

        ``start`` registers two byte-identical wrapper callbacks and its
        docstring states the contract for both topics, but only the joints
        wrapper had ever been driven with a malformed frame. A dropped frame
        must also leave the subscription usable, so a good frame arriving
        afterwards is still forwarded -- swallowing the error is not the same
        as staying alive.
        """
        transport = MagicMock()
        transport.subscribe = AsyncMock()
        link = ZenohLink(transport, prefix="p")
        imu_seen: list = []

        _run(link.start(on_joints=lambda d: None, on_imu=imu_seen.append))
        on_imu_cb = transport.subscribe.call_args_list[1].args[1]

        _run(on_imu_cb(b"not-json{"))
        self.assertEqual(imu_seen, [])

        # The subscription must still deliver after the bad frame.
        _run(on_imu_cb(json.dumps({"accel": [0, 0, 9.8]}).encode()))
        self.assertEqual(imu_seen, [{"accel": [0, 0, 9.8]}])

    def test_a_raising_consumer_does_not_kill_either_subscription(self):
        """The tolerance is ``except Exception``, not a JSON-decode guard.

        A well-formed frame whose consumer raises must be dropped just like a
        malformed one: the wrapper is the only thing between a buggy driver
        callback and a dead sensor topic.
        """
        transport = MagicMock()
        transport.subscribe = AsyncMock()

        def boom(_frame):
            raise RuntimeError("consumer blew up")

        _run(ZenohLink(transport, prefix="p").start(on_joints=boom, on_imu=boom))
        good = json.dumps({"pos": [1]}).encode()

        for index in (0, 1):
            callback = transport.subscribe.call_args_list[index].args[1]
            # Must not raise out of the wrapper for either topic.
            _run(callback(good))

    def test_send_cmd_publishes_encoded_json_to_command_topic(self):
        transport = MagicMock()
        transport.publish = AsyncMock()
        link = ZenohLink(transport, prefix="dev/r")

        _run(link.send_cmd({"body_yaw": 0.5}))

        topic, payload = transport.publish.call_args.args
        self.assertEqual(topic, "dev/r/command")
        self.assertEqual(json.loads(payload.decode()), {"body_yaw": 0.5})

    def test_stop_is_noop(self):
        link = ZenohLink(MagicMock(), prefix="p")
        # Should not raise; teardown is owned by DeviceRuntime.
        _run(link.stop())


class _FakeWS:
    """Minimal async-iterable WebSocket double."""

    def __init__(self, incoming=None):
        self._incoming = list(incoming or [])
        self.sent: list = []
        self.closed = False

    def __aiter__(self):
        async def _gen():
            for item in self._incoming:
                yield item

        return _gen()

    async def send(self, data):
        # Shaped like the real socket: websockets raises ConnectionClosed on a
        # send after close rather than accepting it, so a link that offers a
        # closed socket as connected fails here instead of looking healthy.
        if self.closed:
            raise ConnectionError("cannot send on a closed WebSocket")
        self.sent.append(data)

    async def close(self):
        self.closed = True


class TestWebSocketLink(unittest.TestCase):
    """WebSocketLink talks to the daemon /ws/sdk endpoint."""

    def setUp(self):
        # Ensure an unauthenticated, plaintext posture for deterministic URLs.
        for var in ("REACHY_DAEMON_TOKEN", "REACHY_DAEMON_TLS", "REACHY_DAEMON_TLS_INSECURE"):
            self.addCleanup(lambda v=var, old=__import__("os").environ.get(var): _restore_env(v, old))
            __import__("os").environ.pop(var, None)

    def test_start_connects_plaintext_url_and_spawns_read_task(self):
        fake_ws = _FakeWS()
        fake_websockets = MagicMock()
        fake_websockets.connect = AsyncMock(return_value=fake_ws)

        async def scenario():
            with patch.dict(sys.modules, {"websockets": fake_websockets}):
                link = WebSocketLink("reachy.local", 8000)
                await link.start(on_joints=lambda m: None, on_imu=lambda m: None)
                self.assertIs(link._ws, fake_ws)
                self.assertIsNotNone(link._read_task)
                await link.stop()

        _run(scenario())
        url = fake_websockets.connect.call_args.args[0]
        self.assertEqual(url, "ws://reachy.local:8000/ws/sdk")

    def test_start_with_token_sends_authorization_header(self):
        import os

        os.environ["REACHY_DAEMON_TOKEN"] = "secret-token"
        fake_ws = _FakeWS()
        fake_websockets = MagicMock()
        fake_websockets.connect = AsyncMock(return_value=fake_ws)

        async def scenario():
            with patch.dict(sys.modules, {"websockets": fake_websockets}):
                link = WebSocketLink("h", 1)
                await link.start(on_joints=lambda m: None, on_imu=lambda m: None)
                await link.stop()

        _run(scenario())
        kwargs = fake_websockets.connect.call_args.kwargs
        headers = kwargs.get("additional_headers") or kwargs.get("extra_headers")
        self.assertEqual(headers, {"Authorization": "Bearer secret-token"})

    def test_modern_connect_receives_the_additional_headers_keyword(self):
        """Control: with an introspectable ``connect``, the modern keyword wins.

        ``start`` picks the keyword by reading ``websockets.connect``'s
        signature. Pinning that choice is what makes the fallback below a
        distinguishable outcome rather than a coincidence of the default.
        """
        import os

        os.environ["REACHY_DAEMON_TOKEN"] = "secret-token"
        fake_ws = _FakeWS()
        seen: dict = {}

        async def _connect(url, *, additional_headers=None, extra_headers=None, ssl=None):
            seen["additional_headers"] = additional_headers
            seen["extra_headers"] = extra_headers
            return fake_ws

        fake_websockets = MagicMock()
        fake_websockets.connect = _connect

        async def scenario():
            with patch.dict(sys.modules, {"websockets": fake_websockets}):
                link = WebSocketLink("h", 1)
                await link.start(on_joints=lambda m: None, on_imu=lambda m: None)
                await link.stop()

        _run(scenario())
        self.assertEqual(seen["additional_headers"], {"Authorization": "Bearer secret-token"})
        self.assertIsNone(seen["extra_headers"])

    def test_authorization_survives_when_the_connect_signature_is_unreadable(self):
        """A C-implemented ``connect`` must not cost the bearer credential.

        ``inspect.signature`` raises for a builtin, and the header keyword is
        chosen from that signature. The fallback exists so the legacy spelling
        is used instead of the headers being dropped -- but nothing pinned that
        the token still reaches ``connect`` on that path, so a mutation that
        dropped the headers there would connect unauthenticated in silence.
        """
        import os

        os.environ["REACHY_DAEMON_TOKEN"] = "secret-token"
        fake_ws = _FakeWS()
        seen: dict = {}

        async def _connect(url, *, additional_headers=None, extra_headers=None, ssl=None):
            seen["additional_headers"] = additional_headers
            seen["extra_headers"] = extra_headers
            return fake_ws

        fake_websockets = MagicMock()
        fake_websockets.connect = _connect
        real_signature = inspect.signature
        probed: list = []

        def _unreadable_signature(obj, *args, **kwargs):
            if obj is _connect:
                probed.append(obj)
                raise ValueError("no signature found for builtin <built-in function connect>")
            return real_signature(obj, *args, **kwargs)

        async def scenario():
            with (
                patch.dict(sys.modules, {"websockets": fake_websockets}),
                patch("inspect.signature", _unreadable_signature),
            ):
                link = WebSocketLink("h", 1)
                await link.start(on_joints=lambda m: None, on_imu=lambda m: None)
                await link.stop()

        _run(scenario())

        # Non-vacuity: the signature probe was attempted and really did fail.
        self.assertEqual(len(probed), 1)
        # The credential reaches connect under the legacy spelling, not dropped.
        self.assertEqual(seen["extra_headers"], {"Authorization": "Bearer secret-token"})
        self.assertIsNone(seen["additional_headers"])

    def test_read_loop_routes_messages_by_type(self):
        joints_seen: list = []
        imu_seen: list = []
        link = WebSocketLink("h", 1)
        link._ws = _FakeWS(
            incoming=[
                json.dumps({"type": "joint_positions", "pos": [1]}),
                json.dumps({"type": "imu_data", "accel": [2]}),
                json.dumps({"type": "other"}),  # ignored
                "broken{",  # malformed -> skipped
            ]
        )

        _run(link._read_loop(joints_seen.append, imu_seen.append))

        self.assertEqual(joints_seen, [{"type": "joint_positions", "pos": [1]}])
        self.assertEqual(imu_seen, [{"type": "imu_data", "accel": [2]}])

    def test_send_cmd_maps_each_command_type(self):
        cases = {
            "head_pose": {"head_pose": [[1, 0], [0, 1]]},
            "antennas_joint_positions": {"antennas_joint_positions": [0.1, 0.2]},
            "body_yaw": {"body_yaw": 0.3},
            "torque": {"torque": True, "ids": [1, 2]},
        }
        expected_types = {
            "head_pose": "set_target",
            "antennas_joint_positions": "set_antennas",
            "body_yaw": "set_body_yaw",
            "torque": "set_torque",
        }
        for key, cmd in cases.items():
            link = WebSocketLink("h", 1)
            link._ws = _FakeWS()
            _run(link.send_cmd(cmd))
            self.assertEqual(len(link._ws.sent), 1, key)
            self.assertEqual(json.loads(link._ws.sent[0])["type"], expected_types[key])

    def test_send_cmd_head_pose_flattens_matrix(self):
        link = WebSocketLink("h", 1)
        link._ws = _FakeWS()
        _run(link.send_cmd({"head_pose": [[1, 2], [3, 4]]}))
        self.assertEqual(json.loads(link._ws.sent[0])["head"], [1, 2, 3, 4])

    def test_send_cmd_noop_when_not_connected(self):
        link = WebSocketLink("h", 1)
        # _ws is None -> must return silently, not raise.
        _run(link.send_cmd({"body_yaw": 0.1}))

    def test_send_cmd_after_stop_is_the_documented_noop(self):
        """A stopped link refuses the send instead of using the closed socket.

        ``send_cmd`` reads ``_ws`` as its "is the socket connected?" test, so a
        stop that closes the socket but leaves the handle in place keeps that
        guard unreachable: the send goes out on a closed connection.
        """
        link = WebSocketLink("h", 1)
        fake_ws = _FakeWS()
        link._ws = fake_ws

        async def scenario():
            link._read_task = asyncio.create_task(asyncio.sleep(60))
            await asyncio.sleep(0)
            await link.stop()
            await link.send_cmd({"body_yaw": 0.1})  # documented no-op

        _run(scenario())
        self.assertEqual(fake_ws.sent, [])

    def test_a_stop_whose_close_fails_still_leaves_the_link_disconnected(self):
        """The socket is gone whether or not its close succeeded.

        The handle is therefore dropped before the close is awaited, so a
        failing close cannot leave the link still offering a dead socket.
        """
        link = WebSocketLink("h", 1)
        fake_ws = _FakeWS()

        async def _boom():
            fake_ws.closed = True
            raise ConnectionError("socket teardown raced with close")

        fake_ws.close = _boom
        link._ws = fake_ws

        async def scenario():
            with self.assertRaises(ConnectionError):
                await link.stop()
            await link.send_cmd({"body_yaw": 0.1})  # documented no-op

        _run(scenario())
        self.assertEqual(fake_ws.sent, [])

    def test_stop_cancels_read_task_and_closes_socket(self):
        link = WebSocketLink("h", 1)
        fake_ws = _FakeWS()
        link._ws = fake_ws

        async def scenario():
            link._read_task = asyncio.create_task(asyncio.sleep(60))
            await asyncio.sleep(0)  # let the task start before cancelling
            await link.stop()
            # Allow the cancellation to propagate to the awaited task.
            with self.assertRaises(asyncio.CancelledError):
                await link._read_task
            self.assertTrue(link._read_task.cancelled())
            self.assertTrue(fake_ws.closed)

        _run(scenario())


class TestResolveHost(unittest.TestCase):
    """``resolve_host`` answers with an address, or with the name it was given."""

    def test_a_resolvable_name_is_translated_to_its_address(self):
        with patch("socket.gethostbyname", return_value="10.1.2.3"):
            self.assertEqual(reachy_transport.resolve_host("reachy-mini.local"), "10.1.2.3")

    def test_an_unresolvable_name_is_returned_verbatim(self):
        """A lookup failure degrades to the name rather than refusing the host.

        mDNS ``.local`` names are the common case: the stdlib resolver may not
        answer for a name the link's own dialer can still reach, so the caller
        gets a host it can try instead of an exception.
        """
        with patch("socket.gethostbyname", side_effect=socket.gaierror(-2, "Name or service not known")):
            self.assertEqual(reachy_transport.resolve_host("reachy-mini.local"), "reachy-mini.local")


class TestRestApiErrorHandling(unittest.TestCase):
    """The REST helper returns structured error dicts, never raises."""

    def test_http_error_returns_body_and_code(self):
        err = urllib.error.HTTPError(
            url="http://h/api",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )
        err.read = lambda: b"missing"
        with patch("urllib.request.urlopen", side_effect=err):
            result = api("h", 8000, "/api/x")
        self.assertEqual(result, {"error": "missing", "code": 404})

    def test_generic_exception_returns_error_string(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = api("h", 8000, "/api/x")
        self.assertEqual(result, {"error": "connection refused"})

    def test_success_decodes_json_body(self):
        fake_resp = MagicMock()
        fake_resp.read.return_value = b'{"ok": true}'
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = lambda *a: False
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = api("h", 8000, "/status")
        self.assertEqual(result, {"ok": True})


class TestWebSocketLinkConnectKwargsTyping(unittest.TestCase):
    """The WebSocket connect-kwargs dict must be statically well-typed.

    ``WebSocketLink.start`` builds an untyped ``_connect_kwargs`` mapping and
    ``**``-unpacks it into ``websockets.connect``. With the inline type stubs
    shipped by recent ``websockets`` releases, an unannotated ``{}`` literal is
    inferred as ``dict[str, dict[str, str]]`` (from its first assignment) and
    fails every ``connect`` overload on the unpack. Annotating the dict as
    ``dict[str, Any]`` keeps the call type-correct. This pins that contract so a
    future edit cannot silently reintroduce the over-narrow inference.
    """

    def test_module_typechecks_against_websockets_stubs(self):
        import subprocess

        module_path = Path(reachy_transport.__file__)
        result = subprocess.run(
            [sys.executable, "-m", "mypy", str(module_path)],
            capture_output=True,
            text=True,
        )
        arg_type_errors = [
            line for line in result.stdout.splitlines() if "reachy_transport.py" in line and "[arg-type]" in line
        ]
        self.assertEqual(
            arg_type_errors,
            [],
            msg=(
                "websockets.connect call must type-check cleanly; got arg-type errors:\n" + "\n".join(arg_type_errors)
            ),
        )


def _restore_env(name, old):
    import os

    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


if __name__ == "__main__":
    unittest.main()
