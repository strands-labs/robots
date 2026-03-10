"""
Telemetry Streaming for Robotics.

Key patterns:
- **Strategy tier routing** — STREAM/BATCH/STORAGE based on event urgency
- **Auto-batching** — count/size/age triple-trigger flush thresholds
- **Gzip compression** — at batch level when payload exceeds threshold
- **Correlation tracking** — W3C trace-context compatible span IDs
- **Exponential backoff** — retry with jitter on transport failures
- **Non-blocking emit** — collections.deque for O(1) append in 50Hz loops

Usage:
    from strands_robots.telemetry import TelemetryStream, EventCategory

    stream = TelemetryStream(robot_id="so100_left")
    stream.start()

    # In 50Hz control loop (non-blocking, <0.05ms)
    stream.emit(EventCategory.JOINT_STATE, {"q": [0.1, 0.2, 0.3]})

    # Correlation tracking
    with stream.span("pick_and_place") as trace_id:
        stream.emit(EventCategory.TASK_START, {"instruction": "pick cube"})
        # ... control loop ...
        stream.emit(EventCategory.TASK_END, {"success": True})

    stream.stop()

See Also:
    - Design doc: docs/telemetry.md
"""

from strands_robots.telemetry.stream import TelemetryStream
from strands_robots.telemetry.types import (
    BatchConfig,
    EventCategory,
    StreamTier,
    TelemetryEvent,
)

__all__ = [
    "TelemetryStream",
    "TelemetryEvent",
    "EventCategory",
    "StreamTier",
    "BatchConfig",
]
