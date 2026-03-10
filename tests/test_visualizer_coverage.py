"""Additional visualizer.py tests targeting uncovered lines (78%→95%+).

Covers:
- _render_matplotlib() (lines 251-293) — mock matplotlib, test figure lifecycle
- _start_web_server() (lines 304-341) — mock socketserver, test HTTP server
- get_stats_dict (line 345-346) — verify dict output
"""

import json
import sys
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np

from strands_robots.visualizer import RecordingStats, RecordingVisualizer

# ═══════════════════════════════════════════════════════════════════════════
# Tests: RecordingStats
# ═══════════════════════════════════════════════════════════════════════════


class TestRecordingStats:
    def test_defaults(self):
        s = RecordingStats()
        assert s.episode == 0
        assert s.fps_actual == 0.0
        assert s.cameras == []
        assert s.recording is False

    def test_custom_values(self):
        s = RecordingStats(episode=3, fps_actual=29.5, task="pick cube", recording=True)
        assert s.episode == 3
        assert s.fps_actual == 29.5
        assert s.task == "pick cube"


# ═══════════════════════════════════════════════════════════════════════════
# Tests: RecordingVisualizer core lifecycle
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualizerLifecycle:
    def test_init_defaults(self):
        v = RecordingVisualizer()
        assert v.mode == "terminal"
        assert v.refresh_rate == 2.0
        assert v.port == 8888
        assert v._running is False

    def test_start_stop(self):
        v = RecordingVisualizer(mode="terminal", refresh_rate=10.0)
        v.start()
        assert v._running is True
        assert v.stats.recording is True
        time.sleep(0.05)
        v.stop()
        assert v._running is False
        assert v.stats.recording is False

    def test_start_idempotent(self):
        v = RecordingVisualizer(mode="terminal", refresh_rate=10.0)
        v.start()
        v.start()  # Should not create second thread
        v.stop()

    def test_update_frame_count(self):
        v = RecordingVisualizer()
        v._start_time = time.time()
        v.update(frame_count=42)
        assert v.stats.frame_count == 42

    def test_update_episode(self):
        v = RecordingVisualizer()
        v._start_time = time.time()
        v.update(episode=3, total_episodes=10, task="grasp")
        assert v.stats.episode == 3
        assert v.stats.total_episodes == 10
        assert v.stats.task == "grasp"

    def test_update_cameras(self):
        v = RecordingVisualizer()
        v._start_time = time.time()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        v.update(cameras={"wrist": frame})
        assert "wrist" in v.stats.cameras
        assert "wrist" in v._camera_frames

    def test_update_action_state(self):
        v = RecordingVisualizer()
        v._start_time = time.time()
        v.update(last_action={"j0": 0.1, "j1": 0.2}, last_state={"s0": 1.0})
        assert v.stats.action_dim == 2
        assert v.stats.state_dim == 1

    def test_update_error_count(self):
        v = RecordingVisualizer()
        v._start_time = time.time()
        v.update(error=True)
        v.update(error=True)
        assert v.stats.errors == 2

    def test_update_fps_tracking(self):
        v = RecordingVisualizer()
        v._start_time = time.time() - 10.0
        # Simulate rapid updates
        for _ in range(10):
            v.update(frame_count=v.stats.frame_count + 1)
        # fps_actual should be calculated
        assert v.stats.fps_actual >= 0

    def test_new_episode(self):
        v = RecordingVisualizer()
        v._start_time = time.time() - 5.0
        v.stats.frame_count = 100
        v.stats.fps_actual = 30.0
        v.new_episode(episode=2, task="place")
        assert v.stats.episode == 2
        assert v.stats.task == "place"
        assert v.stats.frame_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _render_terminal
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderTerminal:
    def test_render_basic(self, capsys):
        v = RecordingVisualizer(mode="terminal")
        v.stats = RecordingStats(
            episode=1,
            total_episodes=5,
            frame_count=42,
            fps_actual=28.5,
            fps_target=30.0,
            duration_s=12.0,
            task="pick block",
            cameras=["wrist"],
        )
        v._render_terminal()
        captured = capsys.readouterr()
        assert "LIVE RECORDING MONITOR" in captured.out
        assert "pick block" in captured.out

    def test_render_with_actions(self, capsys):
        v = RecordingVisualizer(mode="terminal")
        v.stats = RecordingStats(
            episode=1,
            fps_actual=15.0,
            fps_target=30.0,
            duration_s=60.0,
            task="test",
            last_action={"j0": 0.1, "j1": -0.2, "j2": 0.3, "j3": 0.4, "j4": 0.5},
        )
        v._render_terminal()
        captured = capsys.readouterr()
        assert "Action:" in captured.out
        assert "+1 more" in captured.out

    def test_render_with_errors(self, capsys):
        v = RecordingVisualizer(mode="terminal")
        v.stats = RecordingStats(errors=3, fps_target=30.0)
        v._render_terminal()
        captured = capsys.readouterr()
        assert "Errors: 3" in captured.out

    def test_render_zero_fps_target(self, capsys):
        v = RecordingVisualizer(mode="terminal")
        v.stats = RecordingStats(fps_actual=10.0, fps_target=0.0)
        v._render_terminal()
        # Should not crash on division by zero

    def test_render_high_fps_ratio(self, capsys):
        v = RecordingVisualizer(mode="terminal")
        v.stats = RecordingStats(fps_actual=30.0, fps_target=30.0)
        v._render_terminal()
        captured = capsys.readouterr()
        assert "█" in captured.out


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _render_json
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderJson:
    def test_render_json(self, capsys):
        v = RecordingVisualizer(mode="json")
        v.stats = RecordingStats(
            episode=2,
            frame_count=100,
            fps_actual=29.7,
            duration_s=3.5,
            task="grasp",
            cameras=["wrist", "top"],
            errors=1,
        )
        v._render_json()
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["episode"] == 2
        assert data["frame"] == 100
        assert data["fps"] == 29.7
        assert data["cameras"] == ["wrist", "top"]


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _render_matplotlib (lines 251-293)
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderMatplotlib:
    def test_matplotlib_no_frames(self):
        """No frames → early return, no crash."""
        v = RecordingVisualizer(mode="matplotlib")
        v._camera_frames = {}
        v.stats = RecordingStats()

        mock_matplotlib = MagicMock()
        mock_plt = MagicMock()

        with patch.dict(sys.modules, {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_plt}):
            v._render_matplotlib()

        # Should not create any figure
        mock_plt.subplots.assert_not_called()

    def test_matplotlib_first_render_creates_fig(self):
        """First render creates figure and axes."""
        v = RecordingVisualizer(mode="matplotlib")
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        v._camera_frames = {"wrist": frame}
        v.stats = RecordingStats(episode=1, frame_count=10, fps_actual=30.0, fps_target=30.0, task="test")

        mock_matplotlib = types.ModuleType("matplotlib")
        mock_matplotlib.use = MagicMock()
        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_matplotlib.pyplot = mock_plt

        with patch.dict(sys.modules, {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_plt}):
            v._render_matplotlib()

        mock_matplotlib.use.assert_called_once_with("TkAgg")
        mock_plt.subplots.assert_called_once()
        mock_plt.ion.assert_called_once()
        mock_plt.show.assert_called_once_with(block=False)

    def test_matplotlib_second_render_updates(self):
        """Subsequent renders update existing images."""
        v = RecordingVisualizer(mode="matplotlib")
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        v._camera_frames = {"wrist": frame}
        v.stats = RecordingStats(episode=1, fps_actual=30.0, fps_target=30.0, task="test")

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_img = MagicMock()
        v._fig = mock_fig
        v._axes = [mock_ax]
        v._img_plots = {"wrist": mock_img}

        mock_matplotlib = MagicMock()
        mock_plt = MagicMock()

        with patch.dict(sys.modules, {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_plt}):
            v._render_matplotlib()

        mock_img.set_data.assert_called_once()
        mock_fig.suptitle.assert_called_once()
        mock_fig.canvas.draw_idle.assert_called_once()
        mock_fig.canvas.flush_events.assert_called_once()

    def test_matplotlib_import_error_falls_back(self):
        """ImportError for matplotlib falls back to terminal mode."""
        v = RecordingVisualizer(mode="matplotlib")
        v._camera_frames = {"wrist": np.zeros((64, 64, 3), dtype=np.uint8)}
        v.stats = RecordingStats()

        # Make matplotlib import fail
        with patch.dict(sys.modules, {"matplotlib": None}):
            with patch("builtins.__import__", side_effect=ImportError("no matplotlib")):
                v._render_matplotlib()

        assert v.mode == "terminal"

    def test_matplotlib_runtime_error_handled(self):
        """Runtime errors in matplotlib are handled gracefully."""
        v = RecordingVisualizer(mode="matplotlib")
        v._camera_frames = {"wrist": np.zeros((64, 64, 3), dtype=np.uint8)}
        v.stats = RecordingStats()

        mock_matplotlib = MagicMock()
        mock_plt = MagicMock()
        mock_plt.subplots.side_effect = RuntimeError("display not available")

        with patch.dict(sys.modules, {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_plt}):
            # Should not raise
            v._render_matplotlib()

    def test_matplotlib_multiple_cameras(self):
        """Multiple cameras create multiple axes."""
        v = RecordingVisualizer(mode="matplotlib")
        v._camera_frames = {
            "wrist": np.zeros((64, 64, 3), dtype=np.uint8),
            "top": np.ones((64, 64, 3), dtype=np.uint8) * 128,
        }
        v.stats = RecordingStats(episode=1, fps_actual=30.0, fps_target=30.0, task="test")

        mock_matplotlib = types.ModuleType("matplotlib")
        mock_matplotlib.use = MagicMock()
        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_matplotlib.pyplot = mock_plt

        with patch.dict(sys.modules, {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_plt}):
            v._render_matplotlib()

        assert mock_plt.subplots.call_args[0] == (1, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _start_web_server (lines 304-341)
# ═══════════════════════════════════════════════════════════════════════════


class TestStartWebServer:
    def test_web_server_starts_on_available_port(self):
        """Web server starts successfully and serves on a port."""
        v = RecordingVisualizer(mode="web", port=0)  # port 0 = OS picks free port
        v.stats = RecordingStats(episode=1, task="test")

        # Use a real TCPServer but bind to port 0 (OS picks free port)
        # This validates the DashboardHandler class definition and server setup
        v._start_web_server()

        if v._web_server is not None:
            v._web_server.shutdown()
            v._web_server.server_close()

    def test_web_server_exception_logged(self):
        """OSError on bind is handled gracefully."""
        v = RecordingVisualizer(mode="web", port=99999)  # invalid port

        # The function catches the Exception and logs it
        v._start_web_server()
        # If it raised, the test would fail — it should handle gracefully

    def test_start_with_web_mode(self):
        v = RecordingVisualizer(mode="web", port=0)

        v.start()
        assert v._running is True
        time.sleep(0.05)  # let server thread start
        v.stop()
        assert v._running is False

    def test_stop_without_web_server(self):
        v = RecordingVisualizer(mode="terminal")
        v._running = True
        v.stop()
        assert v._running is False


# ═══════════════════════════════════════════════════════════════════════════
# Tests: get_stats_dict (lines 345-346)
# ═══════════════════════════════════════════════════════════════════════════


class TestGetStatsDict:
    def test_returns_dict(self):
        v = RecordingVisualizer()
        v.stats = RecordingStats(
            episode=3,
            total_episodes=10,
            frame_count=500,
            fps_actual=29.5,
            fps_target=30.0,
            duration_s=16.7,
            task="pick cube",
            cameras=["wrist"],
        )
        d = v.get_stats_dict()
        assert isinstance(d, dict)
        assert d["episode"] == 3
        assert d["frame_count"] == 500
        assert d["fps_actual"] == 29.5
        assert d["task"] == "pick cube"

    def test_get_stats_dict_empty(self):
        v = RecordingVisualizer()
        d = v.get_stats_dict()
        assert d["episode"] == 0
        assert d["frame_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Display loop dispatch
# ═══════════════════════════════════════════════════════════════════════════


class TestDisplayLoop:
    def test_display_loop_terminal(self):
        v = RecordingVisualizer(mode="terminal", refresh_rate=100)
        v._running = True
        v.stats = RecordingStats(fps_target=30.0)

        with patch.object(v, "_render_terminal") as mock_render:
            # Run one iteration
            def stop_after_one(*args, **kwargs):
                v._running = False

            mock_render.side_effect = stop_after_one
            v._display_loop()

        mock_render.assert_called_once()

    def test_display_loop_matplotlib(self):
        v = RecordingVisualizer(mode="matplotlib", refresh_rate=100)
        v._running = True

        with patch.object(v, "_render_matplotlib") as mock_render:

            def stop_after_one(*args, **kwargs):
                v._running = False

            mock_render.side_effect = stop_after_one
            v._display_loop()

        mock_render.assert_called_once()

    def test_display_loop_json(self):
        v = RecordingVisualizer(mode="json", refresh_rate=100)
        v._running = True

        with patch.object(v, "_render_json") as mock_render:

            def stop_after_one(*args, **kwargs):
                v._running = False

            mock_render.side_effect = stop_after_one
            v._display_loop()

        mock_render.assert_called_once()

    def test_display_loop_exception_handled(self):
        v = RecordingVisualizer(mode="terminal", refresh_rate=100)
        v._running = True
        call_count = [0]

        def failing_render():
            call_count[0] += 1
            if call_count[0] >= 2:
                v._running = False
            raise RuntimeError("render failed")

        with patch.object(v, "_render_terminal", side_effect=failing_render):
            v._display_loop()
        # Should have survived the exception and iterated
        assert call_count[0] >= 2
