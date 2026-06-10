"""Unit tests for ShadowMirror and presence-shadow auto-wiring.

No real AWS — uses MagicMock-backed transports.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from strands_robots.mesh.iot.shadow import (
    ShadowMirror,
    enable_for_mesh,
    shadow_get_topic,
    shadow_update_topic,
)

# Topic helpers


class TestShadowTopics:
    def test_default_shadow_name(self):
        assert shadow_update_topic("so100-01") == "$aws/things/so100-01/shadow/name/presence/update"

    def test_custom_shadow_name(self):
        assert shadow_update_topic("so100-01", "task") == "$aws/things/so100-01/shadow/name/task/update"

    def test_get_topic(self):
        assert shadow_get_topic("so100-01") == "$aws/things/so100-01/shadow/name/presence/get"


# ShadowMirror.update


class TestShadowMirrorUpdate:
    def test_publishes_with_envelope(self):
        transport = MagicMock()
        transport.is_alive = MagicMock(return_value=True)
        m = ShadowMirror("so100-01")
        m.update(transport, {"connected": True, "robot_type": "so100"})
        transport.put.assert_called_once()
        topic, payload = transport.put.call_args.args
        assert topic == "$aws/things/so100-01/shadow/name/presence/update"
        assert payload == {"state": {"reported": {"connected": True, "robot_type": "so100"}}}

    def test_noop_when_transport_none(self):
        m = ShadowMirror("so100-01")
        m.update(None, {"connected": True})  # No exception

    def test_noop_when_transport_dead(self):
        transport = MagicMock()
        transport.is_alive = MagicMock(return_value=False)
        m = ShadowMirror("so100-01")
        m.update(transport, {"k": 1})
        transport.put.assert_not_called()

    def test_swallows_put_errors(self):
        transport = MagicMock()
        transport.is_alive = MagicMock(return_value=True)
        transport.put.side_effect = RuntimeError("network")
        m = ShadowMirror("so100-01")
        # Should not raise — shadow updates are best-effort.
        m.update(transport, {"k": 1})


# enable_for_mesh — auto-wiring


class TestEnableForMesh:
    def test_noop_for_zenoh_backend(self):
        """Shadow mirror is irrelevant on plain Zenoh."""
        mesh = MagicMock()
        with patch(
            "strands_robots.mesh.transport.factory.current_backend",
            return_value="zenoh",
        ):
            result = enable_for_mesh(mesh)
        assert result is None

    def test_wraps_build_presence_for_iot_backend(self):
        mesh = MagicMock()
        mesh.peer_id = "test-thing"
        original_build = MagicMock(return_value={"robot_id": "test-thing", "v": 1})
        mesh._build_presence = original_build

        transport = MagicMock()
        transport.is_alive = MagicMock(return_value=True)

        with (
            patch(
                "strands_robots.mesh.transport.factory.current_backend",
                return_value="iot",
            ),
            patch(
                "strands_robots.mesh.transport.factory.current_transport",
                return_value=transport,
            ),
        ):
            mirror = enable_for_mesh(mesh)

        assert mirror is not None
        assert mirror.thing_name == "test-thing"

        # The wrapper now calls original AND publishes a shadow update.
        result = mesh._build_presence()
        assert result == {"robot_id": "test-thing", "v": 1}
        original_build.assert_called_once()
        transport.put.assert_called_once()
        # Topic must be the shadow update path.
        topic, _payload = transport.put.call_args.args
        assert topic == "$aws/things/test-thing/shadow/name/presence/update"

    def test_wraps_build_presence_for_bridge_backend(self):
        mesh = MagicMock()
        mesh.peer_id = "bridge-thing"
        mesh._build_presence = MagicMock(return_value={"k": 1})
        with (
            patch(
                "strands_robots.mesh.transport.factory.current_backend",
                return_value="bridge",
            ),
            patch(
                "strands_robots.mesh.transport.factory.current_transport",
                return_value=MagicMock(is_alive=MagicMock(return_value=True)),
            ),
        ):
            assert enable_for_mesh(mesh) is not None

    def test_shadow_update_failure_does_not_break_heartbeat(self):
        """If shadow update raises, the original presence payload is still returned."""
        mesh = MagicMock()
        mesh.peer_id = "x"
        mesh._build_presence = MagicMock(return_value={"v": 1})

        transport = MagicMock()
        transport.is_alive = MagicMock(return_value=True)
        transport.put.side_effect = RuntimeError("shadow broken")

        with (
            patch(
                "strands_robots.mesh.transport.factory.current_backend",
                return_value="iot",
            ),
            patch(
                "strands_robots.mesh.transport.factory.current_transport",
                return_value=transport,
            ),
        ):
            enable_for_mesh(mesh)

        result = mesh._build_presence()  # must not raise
        assert result == {"v": 1}
