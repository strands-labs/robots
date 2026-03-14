"""Unit tests for the robot_mesh tool — Device Connect dispatch path + fallback.

All external dependencies are mocked. No Docker or real connections needed.
"""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch


# ── Mock heavy dependencies before importing ──────────────────────

# Mock strands @tool decorator as passthrough
mock_strands = MagicMock()
mock_strands.tool = lambda fn: fn
sys.modules.setdefault("strands", mock_strands)

# Mock device_connect_agent_tools
mock_dc_connection = MagicMock()
sys.modules.setdefault("device_connect_agent_tools", MagicMock())
sys.modules.setdefault("device_connect_agent_tools.connection", mock_dc_connection)


class _FakeConnection:
    """Fake _DeviceConnectConnection with all methods the dispatch uses."""

    def __init__(self, devices=None):
        self.zone = "default"
        self._devices = devices or []
        self._invoke_results = {}
        self._inbox = {}
        self._sync_subs = {}

    def list_devices(self, device_type=None):
        if device_type:
            return [d for d in self._devices if d.get("device_type") == device_type]
        return list(self._devices)

    def invoke(self, device_id, function, params=None, timeout=30.0):
        key = (device_id, function)
        if key in self._invoke_results:
            return self._invoke_results[key]
        return {"result": {"status": "ok"}}

    def broadcast(self, function, params=None, timeout=5.0):
        results = []
        for d in self._devices:
            try:
                r = self.invoke(d["device_id"], function, params, timeout=timeout)
                results.append({"device_id": d["device_id"], "result": r})
            except Exception as e:
                results.append({"device_id": d["device_id"], "error": str(e)})
        return results

    def subscribe_buffered(self, subject, name=None):
        name = name or subject
        self._inbox[name] = []
        self._sync_subs[name] = True
        return name

    def get_inbox(self, name=None):
        if name is not None:
            return {name: list(self._inbox.get(name, []))}
        return {k: list(v) for k, v in self._inbox.items()}


SAMPLE_DEVICES = [
    {
        "device_id": "so100-lab-1",
        "device_type": "strands_robot",
        "status": {"availability": "idle"},
        "functions": [{"name": "execute"}, {"name": "stop"}, {"name": "getStatus"}],
        "events": ["taskStarted", "taskComplete"],
    },
    {
        "device_id": "panda-sim-1",
        "device_type": "strands_sim",
        "status": {"availability": "idle"},
        "functions": [{"name": "execute"}, {"name": "step"}, {"name": "reset"}],
        "events": ["stateUpdate"],
    },
]


class TestDeviceConnectDispatch(unittest.TestCase):
    """Test _device_connect_dispatch handles all 10 actions."""

    def setUp(self):
        self.conn = _FakeConnection(devices=SAMPLE_DEVICES)
        # Patch get_connection to return our fake
        self.patcher = patch(
            "device_connect_agent_tools.connection.get_connection",
            return_value=self.conn,
        )
        self.patcher.start()

        # Import after mocking
        from strands_robots.tools.robot_mesh import _device_connect_dispatch
        self.dispatch = _device_connect_dispatch

    def tearDown(self):
        self.patcher.stop()

    def _call(self, action, **kwargs):
        defaults = dict(
            target="", instruction="", command="",
            policy_provider="mock", policy_port=0,
            duration=30.0, timeout=5.0,
        )
        defaults.update(kwargs)
        return self.dispatch(action, **{k: defaults[k] for k in [
            "target", "instruction", "command",
            "policy_provider", "policy_port", "duration", "timeout",
        ]})

    def test_peers(self):
        result = self._call("peers")
        self.assertEqual(result["status"], "success")
        text = result["content"][0]["text"]
        self.assertIn("so100-lab-1", text)
        self.assertIn("panda-sim-1", text)
        self.assertIn("2 device(s)", text)

    def test_tell(self):
        result = self._call("tell", target="so100-lab-1", instruction="pick up cube")
        self.assertEqual(result["status"], "success")
        text = result["content"][0]["text"]
        self.assertIn("so100-lab-1", text)
        self.assertIn("pick up cube", text)

    def test_tell_missing_args(self):
        result = self._call("tell", target="", instruction="")
        self.assertEqual(result["status"], "error")

    def test_send(self):
        result = self._call("send", target="so100-lab-1")
        self.assertEqual(result["status"], "success")

    def test_send_with_command(self):
        cmd = json.dumps({"action": "getFeatures"})
        result = self._call("send", target="so100-lab-1", command=cmd)
        self.assertEqual(result["status"], "success")

    def test_stop(self):
        result = self._call("stop", target="so100-lab-1")
        self.assertEqual(result["status"], "success")
        self.assertIn("Stop", result["content"][0]["text"])

    def test_stop_missing_target(self):
        result = self._call("stop", target="")
        self.assertEqual(result["status"], "error")

    def test_emergency_stop(self):
        result = self._call("emergency_stop")
        self.assertEqual(result["status"], "success")
        self.assertIn("E-STOP", result["content"][0]["text"])
        self.assertIn("2/2", result["content"][0]["text"])

    def test_broadcast(self):
        result = self._call("broadcast")
        self.assertEqual(result["status"], "success")
        text = result["content"][0]["text"]
        self.assertIn("2 response(s)", text)
        self.assertIn("so100-lab-1", text)
        self.assertIn("panda-sim-1", text)

    def test_broadcast_with_command(self):
        cmd = json.dumps({"function": "getStatus"})
        result = self._call("broadcast", command=cmd)
        self.assertEqual(result["status"], "success")

    def test_subscribe(self):
        result = self._call("subscribe", target="device-connect.default.*.event.>")
        self.assertEqual(result["status"], "success")
        text = result["content"][0]["text"]
        self.assertIn("Subscribed", text)
        self.assertIn("inbox", text)
        # Verify subscription was created
        self.assertIn("device-connect.default.*.event.>", self.conn._sync_subs)

    def test_subscribe_missing_target(self):
        result = self._call("subscribe", target="")
        self.assertEqual(result["status"], "error")

    def test_watch(self):
        result = self._call("watch", target="so100-lab-1")
        self.assertEqual(result["status"], "success")
        text = result["content"][0]["text"]
        self.assertIn("Watching", text)
        self.assertIn("so100-lab-1", text)
        # Verify subscription uses correct subject pattern
        self.assertIn("stream:so100-lab-1", self.conn._sync_subs)

    def test_watch_missing_target(self):
        result = self._call("watch", target="")
        self.assertEqual(result["status"], "error")

    def test_inbox_empty(self):
        result = self._call("inbox")
        self.assertEqual(result["status"], "success")
        self.assertIn("No subscriptions", result["content"][0]["text"])

    def test_inbox_with_messages(self):
        # Create a subscription and add messages
        self.conn.subscribe_buffered("test-subject", name="test")
        self.conn._inbox["test"] = [
            ("device-connect.default.so100-lab-1.event.taskStarted", {"event_name": "taskStarted", "device_id": "so100-lab-1"}),
            ("device-connect.default.so100-lab-1.event.taskComplete", {"event_name": "taskComplete", "device_id": "so100-lab-1"}),
        ]
        result = self._call("inbox")
        self.assertEqual(result["status"], "success")
        text = result["content"][0]["text"]
        self.assertIn("test", text)
        self.assertIn("2 messages", text)

    def test_status(self):
        result = self._call("status")
        self.assertEqual(result["status"], "success")
        text = result["content"][0]["text"]
        self.assertIn("2 device(s)", text)

    def test_unknown_action(self):
        result = self._call("nonexistent")
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown action", result["content"][0]["text"])
        # Verify all valid actions are listed in the error message
        for a in ["peers", "tell", "send", "broadcast", "stop",
                   "emergency_stop", "status", "subscribe", "watch", "inbox"]:
            self.assertIn(a, result["content"][0]["text"])


class TestFallbackToZenoh(unittest.TestCase):
    """Test that robot_mesh falls back to Zenoh when Device Connect fails."""

    def test_fallback_on_dispatch_error(self):
        """When _device_connect_dispatch raises, _mesh_dispatch should be called."""
        with patch("strands_robots.tools.robot_mesh._ensure_connected", side_effect=Exception("no DC")), \
             patch("strands_robots.tools.robot_mesh._mesh_dispatch") as mock_mesh:
            mock_mesh.return_value = {"status": "success", "content": [{"text": "zenoh fallback"}]}

            from strands_robots.tools.robot_mesh import robot_mesh
            result = robot_mesh(action="peers")

            mock_mesh.assert_called_once()
            self.assertEqual(result["status"], "success")
            self.assertIn("zenoh fallback", result["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
