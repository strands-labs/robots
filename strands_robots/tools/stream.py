"""Telemetry Stream — Strands AgentTool for robotics telemetry management.

Natural-language interface for the telemetry streaming system.
Wraps TelemetryStream lifecycle and provides 7 actions:

Actions:
    - start: Start telemetry streaming for a robot
    - stop: Stop telemetry streaming
    - emit: Manually emit a telemetry event
    - status: Get streaming status and stats
    - flush: Force flush all buffered events
    - start_trace: Begin a correlation span
    - end_trace: End a correlation span

Usage with Strands Agent:
    agent = Agent(tools=[stream])
    agent("Start telemetry for robot so100 with local WAL")
    agent("What's the telemetry status?")
    agent("Flush all telemetry buffers")
"""

import json
import logging
from typing import Any, Dict, Optional

try:
    from strands import tool
except ImportError:
    # Fallback for environments without strands SDK
    def tool(fn):  # type: ignore[misc]
        return fn


logger = logging.getLogger(__name__)

# Global stream registry (persists across tool calls)
_STREAMS: Dict[str, Any] = {}


@tool
def stream(
    action: str,
    robot_id: str = "default",
    category: str = "custom",
    data: Optional[str] = None,
    wal_dir: str = "./telemetry_wal",
    enable_stdout: bool = False,
    enable_wal: bool = True,
    enable_otel: bool = False,
    trace_name: str = "",
    sim_or_real: str = "real",
    flush_interval_s: float = 0.5,
) -> Dict[str, Any]:
    """Telemetry streaming for robotics.

    Manage real-time telemetry with strategy-based tier routing,
    auto-batching, gzip compression, and correlation tracking.

    Args:
        action: One of: start, stop, emit, status, flush, start_trace, end_trace.
        robot_id: Robot identifier (default: "default").
        category: Event category for emit action (e.g., "joint_state", "task_start").
        data: JSON string payload for emit action.
        wal_dir: Directory for Write-Ahead Log files.
        enable_stdout: Enable stdout/logging transport for debugging.
        enable_wal: Enable local WAL transport for persistence.
        enable_otel: Enable OpenTelemetry bridge transport.
        trace_name: Span name for start_trace/end_trace actions.
        sim_or_real: "real" or "sim" for event tagging.
        flush_interval_s: Background flush interval in seconds.

    Returns:
        Dict with status and result content.
    """
    try:
        if action == "start":
            return _action_start(
                robot_id=robot_id,
                wal_dir=wal_dir,
                enable_stdout=enable_stdout,
                enable_wal=enable_wal,
                enable_otel=enable_otel,
                flush_interval_s=flush_interval_s,
            )
        elif action == "stop":
            return _action_stop(robot_id=robot_id)
        elif action == "emit":
            return _action_emit(
                robot_id=robot_id,
                category=category,
                data=data,
                sim_or_real=sim_or_real,
            )
        elif action == "status":
            return _action_status(robot_id=robot_id)
        elif action == "flush":
            return _action_flush(robot_id=robot_id)
        elif action == "start_trace":
            return _action_start_trace(robot_id=robot_id, trace_name=trace_name)
        elif action == "end_trace":
            return _action_end_trace(robot_id=robot_id)
        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. "
                        f"Valid actions: start, stop, emit, status, flush, start_trace, end_trace"
                    }
                ],
            }
    except Exception as e:
        logger.error("stream error: %s", e)
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}


def _action_start(
    robot_id: str,
    wal_dir: str,
    enable_stdout: bool,
    enable_wal: bool,
    enable_otel: bool,
    flush_interval_s: float,
) -> Dict[str, Any]:
    """Start telemetry stream for a robot."""
    from strands_robots.telemetry.stream import TelemetryStream, TransportConfig
    from strands_robots.telemetry.types import StreamTier

    if robot_id in _STREAMS:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Stream already running for robot_id={robot_id}. Stop it first."
                }
            ],
        }

    stream = TelemetryStream(
        robot_id=robot_id,
        flush_interval_s=flush_interval_s,
    )

    # Add transports
    all_tiers = [StreamTier.STREAM, StreamTier.BATCH, StreamTier.STORAGE]

    if enable_wal:
        from strands_robots.telemetry.transports.local import LocalWALTransport

        wal = LocalWALTransport(wal_dir=wal_dir)
        stream.add_transport(
            TransportConfig(
                name="local_wal",
                tiers=all_tiers,
                handler=wal.send,
            )
        )

    if enable_stdout:
        from strands_robots.telemetry.transports.stdout import StdoutTransport

        stdout = StdoutTransport()
        stream.add_transport(
            TransportConfig(
                name="stdout",
                tiers=all_tiers,
                handler=stdout.send,
            )
        )

    if enable_otel:
        try:
            from strands_robots.telemetry.transports.otel import OTelTransport

            otel = OTelTransport()
            stream.add_transport(
                TransportConfig(
                    name="otel",
                    tiers=all_tiers,
                    handler=otel.send,
                )
            )
        except ImportError:
            logger.warning("OpenTelemetry not available, skipping OTel transport")

    stream.start()
    _STREAMS[robot_id] = stream

    transports = [t.name for t in stream._transports]
    return {
        "status": "success",
        "content": [
            {
                "text": f"Telemetry streaming started for robot_id={robot_id}. "
                f"Transports: {transports}. "
                f"Flush interval: {flush_interval_s}s. "
                f"WAL dir: {wal_dir if enable_wal else 'disabled'}."
            }
        ],
    }


def _action_stop(robot_id: str) -> Dict[str, Any]:
    """Stop telemetry stream."""
    if robot_id not in _STREAMS:
        return {
            "status": "error",
            "content": [{"text": f"No stream running for robot_id={robot_id}."}],
        }

    stream = _STREAMS.pop(robot_id)
    stats = stream.get_stats()
    stream.stop()

    return {
        "status": "success",
        "content": [
            {
                "text": f"Telemetry streaming stopped for robot_id={robot_id}. "
                f"Total emitted: {stats['emitted']}, flushed: {stats['flushed']}, "
                f"dropped: {stats['dropped']}, errors: {stats['errors']}."
            },
            {"json": stats},
        ],
    }


def _action_emit(
    robot_id: str,
    category: str,
    data: Optional[str],
    sim_or_real: str,
) -> Dict[str, Any]:
    """Manually emit a telemetry event."""
    if robot_id not in _STREAMS:
        return {
            "status": "error",
            "content": [
                {"text": f"No stream running for robot_id={robot_id}. Start one first."}
            ],
        }

    from strands_robots.telemetry.types import EventCategory

    # Resolve category
    cat_enum = EventCategory.CUSTOM
    for ec in EventCategory:
        if ec.category_name == category:
            cat_enum = ec
            break

    # Parse data
    event_data = {}
    if data:
        try:
            event_data = json.loads(data)
        except json.JSONDecodeError:
            event_data = {"raw": data}

    stream = _STREAMS[robot_id]
    stream.emit(cat_enum, event_data, sim_or_real=sim_or_real)

    return {
        "status": "success",
        "content": [
            {
                "text": f"Event emitted: category={category}, robot_id={robot_id}, "
                f"data_keys={list(event_data.keys())}."
            }
        ],
    }


def _action_status(robot_id: str) -> Dict[str, Any]:
    """Get telemetry status and stats."""
    if robot_id == "default" and not _STREAMS:
        return {
            "status": "success",
            "content": [
                {"text": "No telemetry streams running."},
                {"json": {"active_streams": [], "total_streams": 0}},
            ],
        }

    if robot_id not in _STREAMS:
        # Return status for all streams
        all_stats = {}
        for rid, stream in _STREAMS.items():
            all_stats[rid] = stream.get_stats()
        return {
            "status": "success",
            "content": [
                {"text": f"Active telemetry streams: {list(_STREAMS.keys())}"},
                {"json": all_stats},
            ],
        }

    stats = _STREAMS[robot_id].get_stats()
    return {
        "status": "success",
        "content": [
            {
                "text": f"Telemetry status for robot_id={robot_id}: "
                f"emitted={stats['emitted']}, flushed={stats['flushed']}, "
                f"running={stats['running']}, transports={stats['transports']}."
            },
            {"json": stats},
        ],
    }


def _action_flush(robot_id: str) -> Dict[str, Any]:
    """Force flush all buffered events."""
    if robot_id not in _STREAMS:
        return {
            "status": "error",
            "content": [{"text": f"No stream running for robot_id={robot_id}."}],
        }

    stream = _STREAMS[robot_id]
    stream._flush_all()

    stats = stream.get_stats()
    return {
        "status": "success",
        "content": [
            {
                "text": f"Flush complete for robot_id={robot_id}. "
                f"Flushed: {stats['flushed']}, buffers: {stats['buffer_sizes']}."
            },
            {"json": stats},
        ],
    }


def _action_start_trace(robot_id: str, trace_name: str) -> Dict[str, Any]:
    """Start a correlation span."""
    if robot_id not in _STREAMS:
        return {
            "status": "error",
            "content": [{"text": f"No stream running for robot_id={robot_id}."}],
        }

    if not trace_name:
        return {
            "status": "error",
            "content": [{"text": "trace_name is required for start_trace action."}],
        }

    import uuid

    stream = _STREAMS[robot_id]
    trace_id = uuid.uuid4().hex[:16]
    stream._correlation.trace_id = trace_id
    stream._correlation.span_name = trace_name

    from strands_robots.telemetry.types import EventCategory

    stream.emit(
        EventCategory.TASK_START, {"span_name": trace_name, "trace_id": trace_id}
    )

    return {
        "status": "success",
        "content": [
            {
                "text": f"Trace started: name={trace_name}, trace_id={trace_id}, robot_id={robot_id}."
            }
        ],
    }


def _action_end_trace(robot_id: str) -> Dict[str, Any]:
    """End a correlation span."""
    if robot_id not in _STREAMS:
        return {
            "status": "error",
            "content": [{"text": f"No stream running for robot_id={robot_id}."}],
        }

    stream = _STREAMS[robot_id]
    trace_id = getattr(stream._correlation, "trace_id", None)
    span_name = getattr(stream._correlation, "span_name", None)

    if not trace_id:
        return {
            "status": "error",
            "content": [{"text": f"No active trace for robot_id={robot_id}."}],
        }

    from strands_robots.telemetry.types import EventCategory

    stream.emit(EventCategory.TASK_END, {"span_name": span_name, "trace_id": trace_id})
    stream._correlation.trace_id = None
    stream._correlation.span_name = None

    return {
        "status": "success",
        "content": [
            {
                "text": f"Trace ended: name={span_name}, trace_id={trace_id}, robot_id={robot_id}."
            }
        ],
    }
