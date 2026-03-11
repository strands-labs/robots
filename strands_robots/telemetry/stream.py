"""Core telemetry stream for robotics.

This is the main entry point for the telemetry subsystem. It provides:
- Non-blocking emit() for the 50Hz control loop
- Per-tier buffering with auto-batching
- Background flush thread for transport delivery
- Correlation context (span) management
- Graceful start/stop lifecycle

Patterns adopted:
- Strategy pattern for tier selection
- Auto-batching with count/size/age thresholds
- Gzip compression at batch level
- Correlation tracking with trace/span IDs
- Exponential backoff retry on transport failure

Key design decision: Uses collections.deque (not asyncio.Queue) for the
emit-side buffer. The 50Hz robot control loop is synchronous — asyncio.Queue
would crash with RuntimeError due to cross-event-loop access. deque with
maxlen gives O(1), GIL-protected, bounded append.
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional

from strands_robots.telemetry.types import (
    BatchConfig,
    EventCategory,
    StreamTier,
    TelemetryEvent,
)

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


@dataclass
class TransportConfig:
    """Configuration for a transport backend.

    Attributes:
        name: Transport identifier (e.g., "local_wal", "otel", "stdout").
        tiers: Which tiers this transport handles.
        handler: Callable that receives a list of serialized event dicts.
        retry_max: Max retry attempts on failure.
        retry_base_ms: Base retry delay in ms (exponential backoff).
    """

    name: str
    tiers: list[StreamTier]
    handler: Callable[[List[Dict[str, Any]]], bool]
    retry_max: int = 3
    retry_base_ms: float = 100.0


class TelemetryStream:
    """Non-blocking telemetry stream for robotics control loops.

    Designed for real-time robotics constraints (50Hz control loop,
    edge devices, sim-to-real parity).

    Thread-safety: emit() is safe to call from any thread (uses deque).
    The background flush thread drains buffers and delivers to transports.

    Args:
        robot_id: Robot identifier for all emitted events.
        batch_config: Auto-batching thresholds per tier.
        buffer_maxlen: Max events per tier buffer (ring buffer — drops oldest).
        flush_interval_s: How often the flush thread checks buffers.
        compression_threshold: Min bytes before gzip compression kicks in.
    """

    def __init__(
        self,
        robot_id: str,
        batch_config: Optional[Dict[StreamTier, BatchConfig]] = None,
        buffer_maxlen: int = 10_000,
        flush_interval_s: float = 0.5,
        compression_threshold: int = 1024,
    ):
        self.robot_id = robot_id
        self._flush_interval = flush_interval_s
        self._compression_threshold = compression_threshold
        self._buffer_maxlen = buffer_maxlen

        # Per-tier buffers (deque for GIL-protected O(1) append)
        self._buffers: Dict[StreamTier, Deque[TelemetryEvent]] = {
            tier: deque(maxlen=buffer_maxlen) for tier in StreamTier
        }

        # Per-tier batch configs with sensible defaults
        self._batch_configs: Dict[StreamTier, BatchConfig] = batch_config or {
            StreamTier.STREAM: BatchConfig(
                max_count=10, max_bytes=64 * 1024, max_age_ms=50.0
            ),
            StreamTier.BATCH: BatchConfig(
                max_count=100, max_bytes=256 * 1024, max_age_ms=500.0
            ),
            StreamTier.STORAGE: BatchConfig(
                max_count=10, max_bytes=1024 * 1024, max_age_ms=2000.0
            ),
        }

        # Per-tier oldest event timestamp for age-based flush
        self._buffer_first_ts: Dict[StreamTier, Optional[float]] = {
            tier: None for tier in StreamTier
        }

        # Transports
        self._transports: List[TransportConfig] = []

        # Correlation context (thread-local for concurrent operations)
        self._correlation = threading.local()

        # Frame counter
        self._frame_counter = 0
        self._frame_lock = threading.Lock()

        # Lifecycle
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Stats
        self._stats = {
            "emitted": 0,
            "flushed": 0,
            "dropped": 0,
            "errors": 0,
            "compressed_bytes": 0,
            "raw_bytes": 0,
        }
        self._stats_lock = threading.Lock()

    # --- Lifecycle ---

    def start(self) -> None:
        """Start the background flush thread."""
        if self._running:
            logger.warning("TelemetryStream already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name=f"telemetry-flush-{self.robot_id}",
            daemon=True,
        )
        self._flush_thread.start()
        logger.info("TelemetryStream started for robot=%s", self.robot_id)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the flush thread and drain remaining events."""
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=timeout)

        # Final flush
        self._flush_all()
        logger.info(
            f"TelemetryStream stopped for robot={self.robot_id} | stats={self.get_stats()}"
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def add_transport(self, transport: TransportConfig) -> None:
        """Register a transport backend."""
        self._transports.append(transport)
        logger.info(
            f"Transport added: {transport.name} for tiers={[t.name for t in transport.tiers]}"
        )

    # --- Emit (hot path — must be <1ms) ---

    def emit(
        self,
        category: EventCategory,
        data: Dict[str, Any],
        tier: Optional[StreamTier] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sim_or_real: str = "real",
    ) -> None:
        """Emit a telemetry event (non-blocking, <0.05ms typical).

        Safe to call from any thread, including the 50Hz sync control loop.
        Events are buffered and flushed by the background thread.

        Args:
            category: Event category (determines default tier).
            data: Event payload (supports numpy arrays, dicts, lists).
            tier: Override tier (None = use category default).
            metadata: Optional additional metadata.
            sim_or_real: "real" or "sim".
        """
        # Get frame counter (atomic increment)
        with self._frame_lock:
            frame_id = self._frame_counter
            self._frame_counter += 1

        # Get correlation context
        correlation_id = getattr(self._correlation, "trace_id", None)

        event = TelemetryEvent(
            category=category,
            robot_id=self.robot_id,
            data=data,
            correlation_id=correlation_id,
            frame_id=frame_id,
            tier=tier,
            sim_or_real=sim_or_real,
            metadata=metadata,
        )

        effective_tier = event.effective_tier
        buf = self._buffers[effective_tier]

        # deque.append with maxlen is O(1) and GIL-protected
        # If buffer is full, oldest event is silently dropped (ring buffer)
        was_full = len(buf) >= self._buffer_maxlen
        buf.append(event)

        # Track first event timestamp for age-based flush
        if self._buffer_first_ts[effective_tier] is None:
            self._buffer_first_ts[effective_tier] = time.time()

        with self._stats_lock:
            self._stats["emitted"] += 1
            if was_full:
                self._stats["dropped"] += 1

    # --- Correlation Context ---

    @contextmanager
    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a correlation span for tracing event chains.

        All events emitted within this context share the same trace_id,
        enabling end-to-end tracing from agent decision to robot action.

        Args:
            name: Span name (e.g., "pick_and_place").
            metadata: Optional span metadata.

        Yields:
            trace_id: The correlation ID for this span.
        """
        trace_id = uuid.uuid4().hex[:16]
        parent_id = getattr(self._correlation, "trace_id", None)

        self._correlation.trace_id = trace_id
        self._correlation.span_name = name
        self._correlation.parent_id = parent_id

        # Emit span start
        self.emit(
            EventCategory.TASK_START,
            {
                "span_name": name,
                "trace_id": trace_id,
                "parent_id": parent_id,
                **(metadata or {}),
            },
        )

        try:
            yield trace_id
        finally:
            # Emit span end
            self.emit(
                EventCategory.TASK_END,
                {
                    "span_name": name,
                    "trace_id": trace_id,
                    "parent_id": parent_id,
                },
            )
            # Restore parent context
            if parent_id:
                self._correlation.trace_id = parent_id
            else:
                self._correlation.trace_id = None
                self._correlation.span_name = None
                self._correlation.parent_id = None

    # --- Stats ---

    def get_stats(self) -> Dict[str, Any]:
        """Get current telemetry stats."""
        with self._stats_lock:
            stats = dict(self._stats)
        stats["buffer_sizes"] = {
            tier.name: len(buf) for tier, buf in self._buffers.items()
        }
        stats["running"] = self._running
        stats["robot_id"] = self.robot_id
        stats["transports"] = [t.name for t in self._transports]
        stats["frame_counter"] = self._frame_counter
        return stats

    # --- Background Flush ---

    def _flush_loop(self) -> None:
        """Background thread: periodically check and flush buffers."""
        while not self._shutdown_event.is_set():
            try:
                self._flush_all()
            except Exception as e:
                logger.error("Flush error: %s", e)
                with self._stats_lock:
                    self._stats["errors"] += 1
            self._shutdown_event.wait(timeout=self._flush_interval)

    def _flush_all(self) -> None:
        """Check all tier buffers and flush if thresholds exceeded."""
        for tier in StreamTier:
            config = self._batch_configs[tier]
            buf = self._buffers[tier]

            if not buf:
                continue

            # Check flush triggers (count OR size OR age)
            count = len(buf)
            age_ms = 0.0
            first_ts = self._buffer_first_ts[tier]
            if first_ts is not None:
                age_ms = (time.time() - first_ts) * 1000.0

            should_flush = (
                count >= config.max_count
                or age_ms >= config.max_age_ms
                # Size check is deferred — estimating without serializing
            )

            if should_flush:
                self._drain_and_send(tier)

    def _drain_and_send(self, tier: StreamTier) -> None:
        """Drain buffer and send to transports."""
        buf = self._buffers[tier]
        events: List[TelemetryEvent] = []

        # Drain buffer (deque.popleft is O(1) and GIL-protected)
        while buf:
            try:
                events.append(buf.popleft())
            except IndexError:
                break

        if not events:
            return

        # Reset age tracking
        self._buffer_first_ts[tier] = None

        # Serialize batch
        serialized = [self._serialize_event(e) for e in events]
        batch_json = json.dumps(serialized, separators=(",", ":")).encode("utf-8")
        raw_bytes = len(batch_json)

        # Compress if above threshold
        batch_payload: Any
        compressed = False
        if raw_bytes > self._compression_threshold:
            batch_payload = gzip.compress(batch_json)
            compressed = True
            with self._stats_lock:
                self._stats["compressed_bytes"] += len(batch_payload)
                self._stats["raw_bytes"] += raw_bytes
        else:
            batch_payload = serialized
            with self._stats_lock:
                self._stats["raw_bytes"] += raw_bytes

        # Send to transports for this tier
        for transport in self._transports:
            if tier in transport.tiers:
                self._send_with_retry(transport, batch_payload, tier, compressed)

        with self._stats_lock:
            self._stats["flushed"] += len(events)

    def _serialize_event(self, event: TelemetryEvent) -> Dict[str, Any]:
        """Serialize a single event to dict (numpy-aware)."""

        def _convert(obj: Any) -> Any:
            if _HAS_NUMPY:
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
            if isinstance(obj, bytes):
                return f"<bytes:{len(obj)}>"
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj

        return {
            "event_id": event.event_id,
            "category": event.category.category_name,
            "robot_id": event.robot_id,
            "tier": event.effective_tier.name,
            "sim_or_real": event.sim_or_real,
            "frame_id": event.frame_id,
            "timestamp_ms": event.timestamp_ms,
            "correlation_id": event.correlation_id,
            "data": _convert(event.data),
            "metadata": _convert(event.metadata) if event.metadata else None,
        }

    def _send_with_retry(
        self,
        transport: TransportConfig,
        payload: Any,
        tier: StreamTier,
        compressed: bool,
    ) -> bool:
        """Send payload with exponential backoff retry.

        Args:
            transport: Transport configuration with handler.
            payload: Serialized (and optionally compressed) batch.
            tier: Tier for logging.
            compressed: Whether payload is gzip-compressed.

        Returns:
            True if send succeeded, False if all retries exhausted.
        """
        import random

        for attempt in range(transport.retry_max):
            try:
                success = transport.handler(payload)
                if success:
                    return True
            except Exception as e:
                logger.warning(
                    f"Transport {transport.name} error (tier={tier.name}, "
                    f"attempt={attempt + 1}/{transport.retry_max}): {e}"
                )

            # Exponential backoff with jitter
            if attempt < transport.retry_max - 1:
                delay_ms = transport.retry_base_ms * (2**attempt)
                jitter_ms = random.uniform(0, delay_ms * 0.1)
                time.sleep((delay_ms + jitter_ms) / 1000.0)

        with self._stats_lock:
            self._stats["errors"] += 1

        logger.error(
            f"Transport {transport.name} failed after {transport.retry_max} attempts "
            f"(tier={tier.name})"
        )
        return False
