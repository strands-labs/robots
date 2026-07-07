"""Telemetry event types and tier routing.

Defines the core data model for robotics telemetry events:
- numpy array serialization
- Robot/sim identity tracking
- Frame counter for temporal ordering
- Auto tier selection by event category

Patterns adopted:
- Payload envelope with correlation tracking
- Tier routing (STREAM/BATCH/STORAGE)
- Gzip-ready serialization
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class StreamTier(Enum):
    """Strategy-based tier selection.

    Each tier maps to a different transport backend with different
    latency/throughput/cost characteristics.

    - STREAM: lowest-latency transport; flushed on the background flush interval
      (small count/age thresholds). Safety events, e-stops.
    - BATCH:  buffered, moderate latency. Joint states at 50Hz.
    - STORAGE: best-effort, high-throughput. Camera frames, point clouds.

    Note: delivery latency is bounded by the ``TelemetryStream`` flush interval,
    not guaranteed sub-50ms; size the interval to your latency budget.
    """

    STREAM = auto()
    BATCH = auto()
    STORAGE = auto()


class EventCategory(Enum):
    """Robotics event categories with default tier routing.

    Each category encodes its default tier - the routing table is defined
    once here, not scattered across if/elif chains. Callers can override
    per-event when needed.

    Format: (category_name, default_tier)
    """

    # --- Safety (STREAM tier - lowest-latency transport, flushed first) ---
    EMERGENCY_STOP = ("emergency_stop", StreamTier.STREAM)
    COLLISION = ("collision", StreamTier.STREAM)
    JOINT_LIMIT = ("joint_limit", StreamTier.STREAM)
    SAFETY_ALERT = ("safety_alert", StreamTier.STREAM)

    # --- Control loop (BATCH tier - high frequency, batch-friendly) ---
    JOINT_STATE = ("joint_state", StreamTier.BATCH)
    VELOCITY_COMMAND = ("velocity_command", StreamTier.BATCH)
    ACTION_SENT = ("action_sent", StreamTier.BATCH)
    OBSERVATION = ("observation", StreamTier.BATCH)
    IMU = ("imu", StreamTier.BATCH)
    END_EFFECTOR = ("end_effector", StreamTier.BATCH)

    # --- Task lifecycle (BATCH tier - moderate frequency) ---
    TASK_START = ("task_start", StreamTier.BATCH)
    TASK_END = ("task_end", StreamTier.BATCH)
    POLICY_INFERENCE = ("policy_inference", StreamTier.BATCH)
    REWARD = ("reward", StreamTier.BATCH)
    EPISODE_START = ("episode_start", StreamTier.BATCH)
    EPISODE_END = ("episode_end", StreamTier.BATCH)

    # --- Heavy data (STORAGE tier - large payloads) ---
    CAMERA_FRAME = ("camera_frame", StreamTier.STORAGE)
    POINT_CLOUD = ("point_cloud", StreamTier.STORAGE)
    DEPTH_MAP = ("depth_map", StreamTier.STORAGE)
    EPISODE_CHECKPOINT = ("episode_checkpoint", StreamTier.STORAGE)

    # --- System (BATCH tier) ---
    SYSTEM_INFO = ("system_info", StreamTier.BATCH)
    HEARTBEAT = ("heartbeat", StreamTier.BATCH)
    CONNECTION = ("connection", StreamTier.BATCH)
    ERROR = ("error", StreamTier.STREAM)

    # --- Custom ---
    CUSTOM = ("custom", StreamTier.BATCH)

    def __init__(self, category_name: str, default_tier: StreamTier):
        self.category_name = category_name
        self.default_tier = default_tier


@dataclass
class BatchConfig:
    """Auto-batching thresholds.

    Flush triggers when the count or age threshold is exceeded:
    - max_count: Maximum events before flush
    - max_age_ms: Maximum time since oldest event before flush
    - max_bytes: advisory only - size-based flushing is not currently a trigger
    """

    max_count: int = 100
    max_bytes: int = 256 * 1024  # 256 KB
    max_age_ms: float = 500.0  # 500ms


@dataclass
class TelemetryEvent:
    """Robotics telemetry event.

    Features:
    - numpy array serialization
    - Robot/sim identity tracking
    - Frame counter for temporal ordering
    - Auto tier selection by event category

    Attributes:
        category: Event category with default tier routing.
        robot_id: Robot identifier (e.g., "so100_left", "unitree_g1_sim").
        data: Event payload dict (supports numpy arrays).
        timestamp_ms: Epoch milliseconds (auto-generated).
        event_id: Unique event ID (auto-generated UUID).
        correlation_id: Trace context for correlation tracking.
        frame_id: Frame counter for temporal ordering.
        tier: Override tier (None = use category default).
        sim_or_real: Whether this is simulation or real hardware.
        metadata: Optional additional metadata.
    """

    category: EventCategory
    robot_id: str
    data: dict[str, Any]
    timestamp_ms: int = field(default_factory=lambda: round(time.time() * 1000))
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    correlation_id: str | None = None
    frame_id: int = 0
    tier: StreamTier | None = None
    sim_or_real: str = "real"
    metadata: dict[str, Any] | None = None

    @property
    def effective_tier(self) -> StreamTier:
        """Resolve tier: explicit override > category default."""
        return self.tier if self.tier is not None else self.category.default_tier

    # (Serialization lives in TelemetryStream._serialize_event, the single path
    # used on the flush hot-path; TelemetryEvent stays a plain data record.)
