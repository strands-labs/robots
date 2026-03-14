"""Tests for strands_robots.factory — Robot(), list_robots()."""

from strands_robots.factory import list_robots
from strands_robots.registry import (
    get_robot,
    list_aliases,
    resolve_name,
)
from strands_robots.registry import (
    list_robots as registry_list_robots,
)


class TestResolveNames:
    def test_canonical(self):
        assert resolve_name("so100") == "so100"

    def test_alias(self):
        assert resolve_name("franka") == "panda"
        assert resolve_name("g1") == "unitree_g1"
        assert resolve_name("h1") == "unitree_h1"

    def test_case_insensitive(self):
        assert resolve_name("SO100") == "so100"
        assert resolve_name("Panda") == "panda"

    def test_hyphen_to_underscore(self):
        assert resolve_name("reachy-mini") == "reachy_mini"


class TestListRobots:
    def test_list_all(self):
        robots = list_robots("all")
        assert len(robots) > 0
        names = [r["name"] for r in robots]
        assert "so100" in names
        assert "panda" in names

    def test_list_sim(self):
        robots = list_robots("sim")
        for r in robots:
            assert r["has_sim"] is True

    def test_list_real(self):
        robots = list_robots("real")
        for r in robots:
            assert r["has_real"] is True

    def test_list_both(self):
        robots = list_robots("both")
        for r in robots:
            assert r["has_sim"] is True
            assert r["has_real"] is True

    def test_robot_has_fields(self):
        robots = list_robots()
        for r in robots:
            assert "name" in r
            assert "description" in r
            assert "has_sim" in r
            assert "has_real" in r


class TestRobotRegistry:
    def test_so100_exists(self):
        info = get_robot("so100")
        assert info is not None
        assert "asset" in info
        assert info["asset"]["dir"] == "trs_so_arm100"

    def test_all_aliases_point_to_valid_robots(self):
        aliases = list_aliases()
        for alias, canonical in aliases.items():
            info = get_robot(canonical)
            assert info is not None, f"Alias '{alias}' points to unknown robot '{canonical}'"

    def test_robot_count(self):
        """Ensure we have a reasonable number of robots."""
        robots = registry_list_robots()
        assert len(robots) >= 30

    def test_all_robots_have_description(self):
        robots = registry_list_robots()
        for r in robots:
            assert "description" in r, f"Robot '{r['name']}' missing description"
            assert len(r["description"]) > 0


class TestRobotRunMethod:
    """Test Robot.run() foreground server lifecycle."""

    def _make_robot(self):
        """Create a Robot instance with mocked Simulation backend."""
        from unittest.mock import MagicMock, patch

        mock_sim = MagicMock()
        mock_sim._dispatch_action = MagicMock(return_value={"status": "success"})

        with patch("strands_robots.simulation.Simulation", return_value=mock_sim):
            from strands_robots.factory import Robot
            return Robot("so100", mode="sim")

    def test_run_method_exists(self):
        """Robot instance should have a callable .run() method."""
        instance = self._make_robot()
        assert hasattr(instance, "run")
        assert callable(instance.run)

    def test_peer_id_metadata(self):
        """Robot instance should have _peer_id set."""
        instance = self._make_robot()
        assert hasattr(instance, "_peer_id")
        assert instance._peer_id.startswith("so100-")
        assert len(instance._peer_id) > len("so100-")

    def test_custom_peer_id(self):
        """Robot with peer_id should use it."""
        from unittest.mock import MagicMock, patch

        mock_sim = MagicMock()
        mock_sim._dispatch_action = MagicMock(return_value={"status": "success"})

        with patch("strands_robots.simulation.Simulation", return_value=mock_sim):
            from strands_robots.factory import Robot
            instance = Robot("so100", mode="sim", peer_id="my-robot-1")
            assert instance._peer_id == "my-robot-1"

    def test_peer_type_sim(self):
        """Sim robot should have _peer_type='sim'."""
        instance = self._make_robot()
        assert instance._peer_type == "sim"

    def test_device_connect_runtime_initially_none(self):
        """_device_connect_runtime should be None before .run()."""
        instance = self._make_robot()
        assert instance._device_connect_runtime is None

    def test_run_foreground_calls_event_wait(self):
        """_run_foreground should block on threading.Event.wait()."""
        import threading
        from unittest.mock import MagicMock, patch

        from strands_robots.factory import _run_foreground

        instance = MagicMock()
        instance._peer_id = "test-robot"
        instance._peer_type = "sim"
        instance._device_connect_runtime = None

        mock_event = MagicMock(spec=threading.Event)
        mock_dc_module = MagicMock(
            init_device_connect_sync=MagicMock(side_effect=ImportError("no DC"))
        )

        with patch.dict("sys.modules", {"strands_robots.device_connect": mock_dc_module}), \
             patch("threading.Event", return_value=mock_event), \
             patch("signal.signal"):
            _run_foreground(instance)
            mock_event.wait.assert_called_once()

    def test_run_foreground_inits_device_connect(self):
        """_run_foreground should call init_device_connect_sync."""
        import threading
        from unittest.mock import MagicMock, patch

        from strands_robots.factory import _run_foreground

        instance = MagicMock()
        instance._peer_id = "test-robot"
        instance._peer_type = "sim"
        instance._device_connect_runtime = None

        mock_dc_init = MagicMock(return_value=MagicMock())
        mock_dc_module = MagicMock(init_device_connect_sync=mock_dc_init)

        mock_event = MagicMock(spec=threading.Event)

        with patch.dict("sys.modules", {"strands_robots.device_connect": mock_dc_module}), \
             patch("threading.Event", return_value=mock_event), \
             patch("signal.signal"):
            _run_foreground(instance)

            mock_dc_init.assert_called_once_with(
                instance, peer_id="test-robot", peer_type="sim",
            )
            assert instance._device_connect_runtime is not None

    def test_run_foreground_survives_dc_failure(self):
        """_run_foreground should not crash if Device Connect init fails."""
        import threading
        from unittest.mock import MagicMock, patch

        from strands_robots.factory import _run_foreground

        instance = MagicMock()
        instance._peer_id = "test-robot"
        instance._peer_type = "sim"
        instance._device_connect_runtime = None

        mock_dc_module = MagicMock(
            init_device_connect_sync=MagicMock(side_effect=RuntimeError("DC unavailable"))
        )
        mock_event = MagicMock(spec=threading.Event)

        with patch.dict("sys.modules", {"strands_robots.device_connect": mock_dc_module}), \
             patch("threading.Event", return_value=mock_event), \
             patch("signal.signal"):
            # Should not raise
            _run_foreground(instance)
            mock_event.wait.assert_called_once()

    def test_run_foreground_registers_signal_handlers(self):
        """_run_foreground should register SIGINT and SIGTERM handlers."""
        import signal
        import threading
        from unittest.mock import MagicMock, call, patch

        from strands_robots.factory import _run_foreground

        instance = MagicMock()
        instance._peer_id = "test-robot"
        instance._peer_type = "sim"
        instance._device_connect_runtime = None

        mock_dc_module = MagicMock(
            init_device_connect_sync=MagicMock(side_effect=ImportError("no DC"))
        )
        mock_event = MagicMock(spec=threading.Event)
        mock_signal = MagicMock()

        with patch.dict("sys.modules", {"strands_robots.device_connect": mock_dc_module}), \
             patch("threading.Event", return_value=mock_event), \
             patch("signal.signal", mock_signal):
            _run_foreground(instance)

            signal_calls = [c[0][0] for c in mock_signal.call_args_list]
            assert signal.SIGINT in signal_calls
            assert signal.SIGTERM in signal_calls
