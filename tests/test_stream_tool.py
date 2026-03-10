"""
Tests for the stream AgentTool wrapper (strands_robots/tools/stream.py).

Covers all 7 actions: start, stop, emit, status, flush, start_trace, end_trace.
Also covers error paths, unknown actions, and the OTel transport integration.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Pre-mock strands before importing any strands_robots modules
_mock_strands = MagicMock()
_mock_strands.tool = lambda f: f
sys.modules.setdefault("strands", _mock_strands)

# Pre-mock cv2 to prevent OpenCV 4.12 crash
import importlib.machinery as _im_fix  # noqa: E402

_mock_cv2 = MagicMock()
_mock_cv2.__spec__ = _im_fix.ModuleSpec("cv2", None)
_mock_cv2.dnn = MagicMock()
sys.modules.setdefault("cv2", _mock_cv2)
sys.modules.setdefault("cv2.dnn", _mock_cv2.dnn)

from strands_robots.tools.stream import _STREAMS  # noqa: E402
from strands_robots.tools.stream import stream as stream_tool  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_streams():
    """Ensure _STREAMS is clean before and after each test."""
    _STREAMS.clear()
    yield
    # Stop any running streams to prevent background thread leaks
    for stream in list(_STREAMS.values()):
        try:
            stream.stop()
        except Exception:
            pass
    _STREAMS.clear()


# ============================================================================
# Action: start
# ============================================================================


class TestActionStart:
    """Tests for the 'start' action."""

    def test_start_basic(self, tmp_path):
        """Start a telemetry stream with default settings."""
        result = stream_tool(
            action="start",
            robot_id="test_robot",
            wal_dir=str(tmp_path / "wal"),
        )
        assert result["status"] == "success"
        assert "test_robot" in result["content"][0]["text"]
        assert "test_robot" in _STREAMS

    def test_start_with_stdout(self, tmp_path):
        """Start with stdout transport enabled."""
        result = stream_tool(
            action="start",
            robot_id="stdout_bot",
            wal_dir=str(tmp_path / "wal"),
            enable_stdout=True,
            enable_wal=True,
        )
        assert result["status"] == "success"
        assert "stdout_bot" in _STREAMS
        # Should have both transports
        transports = [t.name for t in _STREAMS["stdout_bot"]._transports]
        assert "local_wal" in transports
        assert "stdout" in transports

    def test_start_with_otel(self, tmp_path):
        """Start with OTel transport enabled (covers lines 170-182)."""
        result = stream_tool(
            action="start",
            robot_id="otel_bot",
            wal_dir=str(tmp_path / "wal"),
            enable_otel=True,
            enable_wal=False,
            enable_stdout=False,
        )
        assert result["status"] == "success"
        assert "otel_bot" in _STREAMS
        transports = [t.name for t in _STREAMS["otel_bot"]._transports]
        assert "otel" in transports

    def test_start_all_transports(self, tmp_path):
        """Start with all 3 transports enabled."""
        result = stream_tool(
            action="start",
            robot_id="all_bot",
            wal_dir=str(tmp_path / "wal"),
            enable_wal=True,
            enable_stdout=True,
            enable_otel=True,
        )
        assert result["status"] == "success"
        transports = [t.name for t in _STREAMS["all_bot"]._transports]
        assert "local_wal" in transports
        assert "stdout" in transports
        assert "otel" in transports

    def test_start_no_transports(self, tmp_path):
        """Start with no transports (valid — stream still works)."""
        result = stream_tool(
            action="start",
            robot_id="bare_bot",
            wal_dir=str(tmp_path / "wal"),
            enable_wal=False,
            enable_stdout=False,
            enable_otel=False,
        )
        assert result["status"] == "success"
        assert "bare_bot" in _STREAMS
        assert len(_STREAMS["bare_bot"]._transports) == 0

    def test_start_duplicate(self, tmp_path):
        """Starting a stream for the same robot_id twice fails."""
        stream_tool(
            action="start",
            robot_id="dup_bot",
            wal_dir=str(tmp_path / "wal"),
        )
        result = stream_tool(
            action="start",
            robot_id="dup_bot",
            wal_dir=str(tmp_path / "wal"),
        )
        assert result["status"] == "error"
        assert "already running" in result["content"][0]["text"]

    def test_start_custom_flush_interval(self, tmp_path):
        """Custom flush interval is passed through."""
        result = stream_tool(
            action="start",
            robot_id="fast_bot",
            wal_dir=str(tmp_path / "wal"),
            flush_interval_s=0.1,
        )
        assert result["status"] == "success"
        assert "0.1s" in result["content"][0]["text"]


# ============================================================================
# Action: stop
# ============================================================================


class TestActionStop:
    """Tests for the 'stop' action."""

    def test_stop_running_stream(self, tmp_path):
        """Stop a running stream returns stats."""
        stream_tool(action="start", robot_id="s1", wal_dir=str(tmp_path / "wal"))
        result = stream_tool(action="stop", robot_id="s1")

        assert result["status"] == "success"
        assert "s1" in result["content"][0]["text"]
        assert len(result["content"]) == 2  # text + json stats
        stats = result["content"][1]["json"]
        assert "emitted" in stats
        assert "flushed" in stats
        assert "s1" not in _STREAMS

    def test_stop_nonexistent(self):
        """Stopping a non-existent stream returns error (covers line 204)."""
        result = stream_tool(action="stop", robot_id="ghost")
        assert result["status"] == "error"
        assert "No stream running" in result["content"][0]["text"]


# ============================================================================
# Action: emit
# ============================================================================


class TestActionEmit:
    """Tests for the 'emit' action."""

    def test_emit_basic(self, tmp_path):
        """Emit a basic telemetry event."""
        stream_tool(action="start", robot_id="e1", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        result = stream_tool(
            action="emit",
            robot_id="e1",
            category="joint_state",
            data='{"q": [0.1, 0.2]}',
        )
        assert result["status"] == "success"
        assert "joint_state" in result["content"][0]["text"]

    def test_emit_custom_category(self, tmp_path):
        """Emit with a custom category."""
        stream_tool(action="start", robot_id="e2", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        result = stream_tool(
            action="emit",
            robot_id="e2",
            category="custom",
            data='{"key": "value"}',
        )
        assert result["status"] == "success"

    def test_emit_invalid_json_data(self, tmp_path):
        """Invalid JSON data falls back to raw string."""
        stream_tool(action="start", robot_id="e3", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        result = stream_tool(
            action="emit",
            robot_id="e3",
            category="custom",
            data="not valid json",
        )
        assert result["status"] == "success"
        assert "raw" in result["content"][0]["text"]  # data_keys contains 'raw'

    def test_emit_no_data(self, tmp_path):
        """Emit with no data payload."""
        stream_tool(action="start", robot_id="e4", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        result = stream_tool(
            action="emit",
            robot_id="e4",
            category="heartbeat",
        )
        assert result["status"] == "success"

    def test_emit_nonexistent_stream(self):
        """Emitting to non-existent stream returns error."""
        result = stream_tool(
            action="emit",
            robot_id="ghost",
            category="test",
        )
        assert result["status"] == "error"
        assert "No stream running" in result["content"][0]["text"]

    def test_emit_sim_or_real(self, tmp_path):
        """sim_or_real parameter is passed through."""
        stream_tool(action="start", robot_id="e5", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        result = stream_tool(
            action="emit",
            robot_id="e5",
            category="joint_state",
            data='{"q": [0.1]}',
            sim_or_real="sim",
        )
        assert result["status"] == "success"


# ============================================================================
# Action: status
# ============================================================================


class TestActionStatus:
    """Tests for the 'status' action."""

    def test_status_no_streams(self):
        """Status with no streams running."""
        result = stream_tool(action="status", robot_id="default")
        assert result["status"] == "success"
        assert "No telemetry streams running" in result["content"][0]["text"]

    def test_status_specific_robot(self, tmp_path):
        """Status for a specific running stream."""
        stream_tool(action="start", robot_id="s1", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        result = stream_tool(action="status", robot_id="s1")
        assert result["status"] == "success"
        assert "s1" in result["content"][0]["text"]
        stats = result["content"][1]["json"]
        assert "emitted" in stats
        assert "running" in stats

    def test_status_all_streams(self, tmp_path):
        """Status for all streams when specific robot_id not found (covers lines 283-286)."""
        stream_tool(action="start", robot_id="a1", wal_dir=str(tmp_path / "wal_a"), enable_wal=False)
        stream_tool(action="start", robot_id="a2", wal_dir=str(tmp_path / "wal_b"), enable_wal=False)
        result = stream_tool(action="status", robot_id="unknown_robot")

        assert result["status"] == "success"
        assert "a1" in result["content"][0]["text"]
        assert "a2" in result["content"][0]["text"]
        stats = result["content"][1]["json"]
        assert "a1" in stats
        assert "a2" in stats


# ============================================================================
# Action: flush
# ============================================================================


class TestActionFlush:
    """Tests for the 'flush' action."""

    def test_flush_running_stream(self, tmp_path):
        """Flush a running stream."""
        stream_tool(action="start", robot_id="f1", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        # Emit some events first
        stream_tool(action="emit", robot_id="f1", category="joint_state", data='{"q": [0.1]}')
        result = stream_tool(action="flush", robot_id="f1")

        assert result["status"] == "success"
        assert "Flush complete" in result["content"][0]["text"]
        stats = result["content"][1]["json"]
        assert "flushed" in stats

    def test_flush_nonexistent(self):
        """Flushing a non-existent stream returns error (covers line 311)."""
        result = stream_tool(action="flush", robot_id="ghost")
        assert result["status"] == "error"
        assert "No stream running" in result["content"][0]["text"]


# ============================================================================
# Action: start_trace
# ============================================================================


class TestActionStartTrace:
    """Tests for the 'start_trace' action."""

    def test_start_trace(self, tmp_path):
        """Start a correlation trace."""
        stream_tool(action="start", robot_id="t1", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        result = stream_tool(
            action="start_trace",
            robot_id="t1",
            trace_name="pick_and_place",
        )
        assert result["status"] == "success"
        assert "pick_and_place" in result["content"][0]["text"]
        assert "trace_id=" in result["content"][0]["text"]

    def test_start_trace_nonexistent(self):
        """Starting trace on non-existent stream returns error (covers line 335)."""
        result = stream_tool(
            action="start_trace",
            robot_id="ghost",
            trace_name="test",
        )
        assert result["status"] == "error"
        assert "No stream running" in result["content"][0]["text"]

    def test_start_trace_no_name(self, tmp_path):
        """Starting trace without name returns error."""
        stream_tool(action="start", robot_id="t2", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        result = stream_tool(
            action="start_trace",
            robot_id="t2",
            trace_name="",
        )
        assert result["status"] == "error"
        assert "trace_name is required" in result["content"][0]["text"]


# ============================================================================
# Action: end_trace
# ============================================================================


class TestActionEndTrace:
    """Tests for the 'end_trace' action."""

    def test_end_trace(self, tmp_path):
        """End a running correlation trace."""
        stream_tool(action="start", robot_id="et1", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        stream_tool(action="start_trace", robot_id="et1", trace_name="pick")
        result = stream_tool(action="end_trace", robot_id="et1")

        assert result["status"] == "success"
        assert "pick" in result["content"][0]["text"]
        assert "Trace ended" in result["content"][0]["text"]

    def test_end_trace_nonexistent(self):
        """Ending trace on non-existent stream returns error (covers line 370)."""
        result = stream_tool(action="end_trace", robot_id="ghost")
        assert result["status"] == "error"
        assert "No stream running" in result["content"][0]["text"]

    def test_end_trace_no_active_trace(self, tmp_path):
        """Ending trace when no trace is active returns error."""
        stream_tool(action="start", robot_id="et2", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        result = stream_tool(action="end_trace", robot_id="et2")
        assert result["status"] == "error"
        assert "No active trace" in result["content"][0]["text"]


# ============================================================================
# Unknown action and error handling
# ============================================================================


class TestErrorHandling:
    """Tests for error paths and unknown actions."""

    def test_unknown_action(self):
        """Unknown action returns error with valid actions list."""
        result = stream_tool(action="invalid_action")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]
        assert "start, stop, emit, status, flush, start_trace, end_trace" in result["content"][0]["text"]

    def test_exception_in_action(self):
        """Exception during action execution is caught (covers lines 114-116)."""
        with patch("strands_robots.tools.stream._action_start", side_effect=RuntimeError("boom")):
            result = stream_tool(action="start", robot_id="crash_bot")
        assert result["status"] == "error"
        assert "boom" in result["content"][0]["text"]

    def test_full_lifecycle(self, tmp_path):
        """Full lifecycle: start → emit → flush → start_trace → end_trace → stop."""
        # Start
        r = stream_tool(action="start", robot_id="lc1", wal_dir=str(tmp_path / "wal"), enable_wal=False)
        assert r["status"] == "success"

        # Emit
        r = stream_tool(action="emit", robot_id="lc1", category="joint_state", data='{"q": [0.1, 0.2, 0.3]}')
        assert r["status"] == "success"

        # Flush
        r = stream_tool(action="flush", robot_id="lc1")
        assert r["status"] == "success"

        # Status
        r = stream_tool(action="status", robot_id="lc1")
        assert r["status"] == "success"

        # Start trace
        r = stream_tool(action="start_trace", robot_id="lc1", trace_name="test_task")
        assert r["status"] == "success"

        # Emit during trace
        r = stream_tool(action="emit", robot_id="lc1", category="observation", data='{"obs": [1, 2, 3]}')
        assert r["status"] == "success"

        # End trace
        r = stream_tool(action="end_trace", robot_id="lc1")
        assert r["status"] == "success"

        # Stop
        r = stream_tool(action="stop", robot_id="lc1")
        assert r["status"] == "success"
        assert "lc1" not in _STREAMS
