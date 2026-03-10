"""
Comprehensive tests for the OpenTelemetry transport bridge.

Tests cover all functional paths through OTelTransport:
- Constructor: with OTel SDK available vs unavailable (no-op mode)
- send(): list payloads, gzip-compressed bytes, raw JSON bytes,
          single dict auto-wrap, error handling
- _emit_span(): span creation, attribute mapping, data filtering
                (only str/int/float/bool), missing field defaults
- get_stats(): return structure verification
- Integration: OTelTransport wired into TelemetryStream

Also covers bonus lines in:
- transports/__init__.py: OTelTransport import/export path
- stdout.py: non-compact mode, single dict payload
- types.py: HAS_NUMPY=False path, bytes size estimate, no-metadata serialize
"""

import gzip
import json
import sys
import time
from unittest.mock import MagicMock

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

# Detect whether the OpenTelemetry SDK is installed (needed for span capture tests).
# The opentelemetry *API* may be installed without the SDK — the API provides the
# trace interface used by OTelTransport in production, but tests that create a real
# TracerProvider + InMemoryExporter need the SDK package.
try:
    import opentelemetry.sdk.trace  # noqa: F401

    HAS_OTEL_SDK = True
except ImportError:
    HAS_OTEL_SDK = False

requires_otel_sdk = pytest.mark.skipif(
    not HAS_OTEL_SDK,
    reason="opentelemetry-sdk not installed",
)


# ============================================================================
# Helpers: In-memory OTel span capture via direct tracer injection
# ============================================================================


def _create_capturing_tracer():
    """Create an OTel TracerProvider + InMemory exporter and return (tracer, exporter).

    Instead of setting a global TracerProvider (which can only be done once),
    we create a local provider and get a tracer from it, then inject that
    tracer directly into the OTelTransport instance.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )

    class InMemoryExporter(SpanExporter):
        def __init__(self):
            self.spans = []

        def export(self, spans):
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    exporter = InMemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-tracer")

    return tracer, exporter, provider


def _make_transport_with_capture(service_name="strands-robots"):
    """Create an OTelTransport with a capturing tracer injected.

    Returns (transport, exporter) where exporter.spans has captured spans.
    """
    from strands_robots.telemetry.transports.otel import OTelTransport

    tracer, exporter, provider = _create_capturing_tracer()
    transport = OTelTransport(service_name=service_name)
    transport._tracer = tracer  # Inject our capturing tracer
    return transport, exporter


# ============================================================================
# OTelTransport — Constructor Tests
# ============================================================================


class TestOTelTransportConstructor:
    """Tests for OTelTransport initialization paths."""

    @requires_otel_sdk
    def test_constructor_with_otel_available(self):
        """When OTel SDK is available, tracer is initialized."""
        from strands_robots.telemetry.transports.otel import HAS_OTEL, OTelTransport

        assert HAS_OTEL is True
        transport = OTelTransport()
        assert transport._tracer is not None
        assert transport.tracer_name == "strands-robots-telemetry"
        assert transport.service_name == "strands-robots"

    @requires_otel_sdk
    def test_constructor_custom_names(self):
        """Custom tracer_name and service_name are stored."""
        from strands_robots.telemetry.transports.otel import OTelTransport

        transport = OTelTransport(
            tracer_name="custom-tracer",
            service_name="custom-service",
        )
        assert transport.tracer_name == "custom-tracer"
        assert transport.service_name == "custom-service"
        assert transport._tracer is not None

    def test_constructor_without_otel(self):
        """When HAS_OTEL=False, tracer is None (no-op mode).

        Covers lines 26-27 (HAS_OTEL = False) and line 53 (logger.debug fallback).
        """
        from strands_robots.telemetry.transports import otel as otel_module

        original_has_otel = otel_module.HAS_OTEL
        try:
            otel_module.HAS_OTEL = False
            transport = otel_module.OTelTransport()
            assert transport._tracer is None
        finally:
            otel_module.HAS_OTEL = original_has_otel


# ============================================================================
# OTelTransport — send() Tests
# ============================================================================


@requires_otel_sdk
class TestOTelTransportSend:
    """Tests for OTelTransport.send() — the core method."""

    def test_send_list_payload(self):
        """Send a list of event dicts — the standard path."""
        transport, exporter = _make_transport_with_capture()
        events = [
            {
                "category": "joint_state",
                "robot_id": "so100",
                "tier": "BATCH",
                "frame_id": 42,
                "sim_or_real": "sim",
                "correlation_id": "trace-abc",
                "data": {"q": [0.1, 0.2], "speed": 1.5, "active": True},
            },
        ]
        result = transport.send(events)

        assert result is True
        assert len(exporter.spans) == 1

        span = exporter.spans[0]
        assert span.name == "robot.joint_state"
        attrs = dict(span.attributes)
        assert attrs["robot.id"] == "so100"
        assert attrs["robot.category"] == "joint_state"
        assert attrs["robot.tier"] == "BATCH"
        assert attrs["robot.frame_id"] == 42
        assert attrs["robot.sim_or_real"] == "sim"
        assert attrs["robot.correlation_id"] == "trace-abc"
        assert attrs["service.name"] == "strands-robots"
        # Data items that are str/int/float/bool should be span attributes
        assert attrs["robot.data.speed"] == 1.5
        assert attrs["robot.data.active"] is True

    def test_send_multiple_events(self):
        """Send multiple events in a single batch."""
        transport, exporter = _make_transport_with_capture()
        events = [
            {"category": "joint_state", "robot_id": "r1", "data": {}},
            {"category": "observation", "robot_id": "r2", "data": {}},
            {"category": "emergency_stop", "robot_id": "r3", "data": {"reason": "collision"}},
        ]
        result = transport.send(events)

        assert result is True
        assert len(exporter.spans) == 3
        assert exporter.spans[0].name == "robot.joint_state"
        assert exporter.spans[1].name == "robot.observation"
        assert exporter.spans[2].name == "robot.emergency_stop"

    def test_send_gzip_compressed_bytes(self):
        """Send gzip-compressed bytes payload (Covers lines 72-74)."""
        transport, exporter = _make_transport_with_capture()
        events = [
            {"category": "camera_frame", "robot_id": "cam0", "data": {"size": 1024}},
        ]
        compressed = gzip.compress(json.dumps(events).encode("utf-8"))

        result = transport.send(compressed)

        assert result is True
        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "robot.camera_frame"
        assert dict(exporter.spans[0].attributes)["robot.data.size"] == 1024

    def test_send_raw_json_bytes(self):
        """Send raw (non-compressed) JSON bytes — fallback path (lines 75-76)."""
        transport, exporter = _make_transport_with_capture()
        events = [{"category": "heartbeat", "robot_id": "test", "data": {}}]
        raw_bytes = json.dumps(events).encode("utf-8")

        result = transport.send(raw_bytes)

        assert result is True
        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "robot.heartbeat"

    def test_send_single_dict_auto_wrap(self):
        """Send bytes that decode to a single dict (not list) — auto-wrapped (line 80)."""
        transport, exporter = _make_transport_with_capture()
        single_event = {"category": "imu", "robot_id": "g1", "data": {"accel": 9.8}}
        raw_bytes = json.dumps(single_event).encode("utf-8")

        result = transport.send(raw_bytes)

        assert result is True
        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "robot.imu"

    def test_send_error_handling(self):
        """Send with an event that causes _emit_span to fail (lines 87-88)."""
        transport, exporter = _make_transport_with_capture()

        # Replace tracer with one that raises on span creation
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.side_effect = RuntimeError("Tracer exploded")
        transport._tracer = mock_tracer

        result = transport.send([{"category": "test", "robot_id": "x", "data": {}}])
        assert result is False  # Error path returns False

    def test_send_noop_without_tracer(self):
        """When tracer is None, send() returns True immediately (line 66)."""
        transport, exporter = _make_transport_with_capture()
        transport._tracer = None  # Simulate no-op mode

        result = transport.send([{"category": "test", "data": {}}])
        assert result is True
        assert len(exporter.spans) == 0  # Nothing emitted

    def test_send_empty_list(self):
        """Sending an empty list produces no spans."""
        transport, exporter = _make_transport_with_capture()
        result = transport.send([])
        assert result is True
        assert len(exporter.spans) == 0

    def test_send_compressed_single_dict(self):
        """Gzip-compressed single dict (not list) is auto-wrapped."""
        transport, exporter = _make_transport_with_capture()
        single = {"category": "reward", "robot_id": "test", "data": {"value": 0.99}}
        compressed = gzip.compress(json.dumps(single).encode("utf-8"))

        result = transport.send(compressed)
        assert result is True
        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "robot.reward"


# ============================================================================
# OTelTransport — _emit_span() Tests
# ============================================================================


@requires_otel_sdk
class TestOTelTransportEmitSpan:
    """Tests for OTelTransport._emit_span() — span creation and attributes."""

    def test_emit_span_all_attributes(self):
        """All expected attributes are set on the span."""
        transport, exporter = _make_transport_with_capture(service_name="test-service")
        event = {
            "category": "policy_inference",
            "robot_id": "unitree_g1",
            "tier": "BATCH",
            "frame_id": 100,
            "sim_or_real": "real",
            "correlation_id": "corr-xyz",
            "data": {
                "latency_ms": 12,
                "model": "gr00t-n1",
                "confidence": 0.95,
                "success": True,
            },
        }
        transport._emit_span(event)

        assert len(exporter.spans) == 1
        span = exporter.spans[0]
        attrs = dict(span.attributes)

        assert span.name == "robot.policy_inference"
        assert attrs["robot.id"] == "unitree_g1"
        assert attrs["robot.category"] == "policy_inference"
        assert attrs["robot.tier"] == "BATCH"
        assert attrs["robot.frame_id"] == 100
        assert attrs["robot.sim_or_real"] == "real"
        assert attrs["robot.correlation_id"] == "corr-xyz"
        assert attrs["service.name"] == "test-service"
        assert attrs["robot.data.latency_ms"] == 12
        assert attrs["robot.data.model"] == "gr00t-n1"
        assert attrs["robot.data.confidence"] == 0.95
        assert attrs["robot.data.success"] is True

    def test_emit_span_missing_fields_defaults(self):
        """Missing fields use defaults (lines 97-98)."""
        transport, exporter = _make_transport_with_capture()
        # Minimal event — missing category, robot_id, tier, etc.
        event = {"data": {"x": 1}}
        transport._emit_span(event)

        assert len(exporter.spans) == 1
        span = exporter.spans[0]
        attrs = dict(span.attributes)

        assert span.name == "robot.unknown"
        assert attrs["robot.id"] == "unknown"
        assert attrs["robot.category"] == "unknown"
        assert attrs["robot.tier"] == ""
        assert attrs["robot.frame_id"] == 0
        assert attrs["robot.sim_or_real"] == ""
        assert attrs["robot.correlation_id"] == ""
        assert attrs["robot.data.x"] == 1

    def test_emit_span_data_filtering(self):
        """Only str/int/float/bool values from data become span attributes.

        Lists, dicts, None, bytes, etc. are filtered out (too large for attrs).
        """
        transport, exporter = _make_transport_with_capture()
        event = {
            "category": "observation",
            "robot_id": "test",
            "data": {
                "name": "joint_1",  # str ✓
                "position": 0.5,  # float ✓
                "steps": 42,  # int ✓
                "active": False,  # bool ✓
                "q_array": [0.1, 0.2],  # list ✗
                "nested": {"a": 1},  # dict ✗
                "raw": b"bytes",  # bytes ✗
            },
        }
        transport._emit_span(event)

        assert len(exporter.spans) == 1
        attrs = dict(exporter.spans[0].attributes)

        # These should be present
        assert attrs["robot.data.name"] == "joint_1"
        assert attrs["robot.data.position"] == 0.5
        assert attrs["robot.data.steps"] == 42
        assert attrs["robot.data.active"] is False

        # These should NOT be present (filtered out)
        assert "robot.data.q_array" not in attrs
        assert "robot.data.nested" not in attrs
        assert "robot.data.raw" not in attrs

    def test_emit_span_empty_data(self):
        """Empty data dict doesn't crash."""
        transport, exporter = _make_transport_with_capture()
        event = {"category": "heartbeat", "robot_id": "test", "data": {}}
        transport._emit_span(event)
        assert len(exporter.spans) == 1

    def test_emit_span_no_data_key(self):
        """Missing 'data' key doesn't crash."""
        transport, exporter = _make_transport_with_capture()
        event = {"category": "connection", "robot_id": "test"}
        transport._emit_span(event)
        assert len(exporter.spans) == 1

    def test_emit_span_data_is_not_dict(self):
        """Non-dict 'data' field doesn't crash (guard on line 111)."""
        transport, exporter = _make_transport_with_capture()
        event = {"category": "custom", "robot_id": "test", "data": "just a string"}
        transport._emit_span(event)

        assert len(exporter.spans) == 1
        attrs = dict(exporter.spans[0].attributes)
        data_attrs = {k: v for k, v in attrs.items() if k.startswith("robot.data.")}
        assert len(data_attrs) == 0

    def test_emit_span_noop_without_tracer(self):
        """_emit_span with no tracer is a no-op (line 93)."""
        transport, exporter = _make_transport_with_capture()
        transport._tracer = None
        transport._emit_span({"category": "test", "data": {}})
        assert len(exporter.spans) == 0

    def test_emit_span_none_correlation_id(self):
        """None correlation_id is coerced to empty string (line 104 'or' clause)."""
        transport, exporter = _make_transport_with_capture()
        event = {
            "category": "joint_state",
            "robot_id": "test",
            "correlation_id": None,
            "data": {},
        }
        transport._emit_span(event)

        assert len(exporter.spans) == 1
        attrs = dict(exporter.spans[0].attributes)
        assert attrs["robot.correlation_id"] == ""

    def test_emit_span_with_none_values_in_data(self):
        """None values in data dict are filtered out (not str/int/float/bool)."""
        transport, exporter = _make_transport_with_capture()
        event = {
            "category": "test",
            "robot_id": "test",
            "data": {"valid": 1, "null_val": None},
        }
        transport._emit_span(event)

        attrs = dict(exporter.spans[0].attributes)
        assert attrs["robot.data.valid"] == 1
        assert "robot.data.null_val" not in attrs


# ============================================================================
# OTelTransport — get_stats() Tests
# ============================================================================


class TestOTelTransportStats:
    """Tests for OTelTransport.get_stats()."""

    @requires_otel_sdk
    def test_get_stats_with_otel(self):
        """Stats reflect OTel availability and tracer name."""
        from strands_robots.telemetry.transports.otel import OTelTransport

        transport = OTelTransport(tracer_name="my-tracer")
        stats = transport.get_stats()

        assert stats["otel_available"] is True
        assert stats["tracer_name"] == "my-tracer"

    def test_get_stats_without_otel(self):
        """Stats with OTel unavailable."""
        from strands_robots.telemetry.transports import otel as otel_module

        original = otel_module.HAS_OTEL
        try:
            otel_module.HAS_OTEL = False
            transport = otel_module.OTelTransport()
            stats = transport.get_stats()
            assert stats["otel_available"] is False
            assert stats["tracer_name"] == "strands-robots-telemetry"
        finally:
            otel_module.HAS_OTEL = original

    def test_get_stats_returns_dict(self):
        """get_stats always returns a dict with expected keys."""
        from strands_robots.telemetry.transports.otel import OTelTransport

        transport = OTelTransport()
        stats = transport.get_stats()
        assert isinstance(stats, dict)
        assert "otel_available" in stats
        assert "tracer_name" in stats


# ============================================================================
# OTelTransport — Integration with TelemetryStream
# ============================================================================


@requires_otel_sdk
class TestOTelTransportIntegration:
    """Integration tests: OTelTransport wired into TelemetryStream."""

    def test_end_to_end_with_telemetry_stream(self):
        """OTelTransport receives events from TelemetryStream flush cycle."""
        from strands_robots.telemetry.stream import TelemetryStream, TransportConfig
        from strands_robots.telemetry.types import EventCategory, StreamTier

        transport, exporter = _make_transport_with_capture(service_name="e2e-test")
        stream = TelemetryStream(robot_id="e2e_bot")

        stream.add_transport(
            TransportConfig(
                name="otel",
                tiers=[StreamTier.STREAM, StreamTier.BATCH, StreamTier.STORAGE],
                handler=transport.send,
            )
        )

        # Emit events
        stream.emit(EventCategory.JOINT_STATE, {"q": [0.1, 0.2]})
        stream.emit(EventCategory.EMERGENCY_STOP, {"reason": "test"})
        stream.emit(EventCategory.CAMERA_FRAME, {"size": 1024})

        # Force flush by setting old timestamps
        for tier in StreamTier:
            stream._buffer_first_ts[tier] = time.time() - 10.0
        stream._flush_all()

        # OTel should have received spans
        assert len(exporter.spans) >= 3

        categories = {s.name for s in exporter.spans}
        assert "robot.joint_state" in categories
        assert "robot.emergency_stop" in categories
        assert "robot.camera_frame" in categories

    def test_correlation_preserved_through_otel(self):
        """Correlation IDs from TelemetryStream spans reach OTel spans."""
        from strands_robots.telemetry.stream import TelemetryStream, TransportConfig
        from strands_robots.telemetry.types import BatchConfig, EventCategory, StreamTier

        transport, exporter = _make_transport_with_capture()
        stream = TelemetryStream(
            robot_id="corr_test",
            compression_threshold=1024 * 1024,  # High threshold to avoid compression
        )

        # Small batch count to ensure flush happens
        stream._batch_configs[StreamTier.BATCH] = BatchConfig(
            max_count=1,
            max_bytes=10 * 1024 * 1024,
            max_age_ms=50000.0,
        )

        stream.add_transport(
            TransportConfig(
                name="otel",
                tiers=[StreamTier.BATCH],
                handler=transport.send,
            )
        )

        with stream.span("pick_and_place") as trace_id:
            stream.emit(EventCategory.JOINT_STATE, {"q": [0.1]})
            stream._flush_all()

        # Find the joint_state span
        joint_spans = [s for s in exporter.spans if s.name == "robot.joint_state"]
        assert len(joint_spans) >= 1
        attrs = dict(joint_spans[0].attributes)
        assert attrs["robot.correlation_id"] == trace_id


# ============================================================================
# Bonus Coverage: transports/__init__.py
# ============================================================================


class TestTransportsInit:
    """Cover the OTelTransport import path in transports/__init__.py (lines 13-14)."""

    def test_otel_in_all(self):
        """OTelTransport is exported in __all__ when OTel is available."""
        from strands_robots.telemetry import transports

        assert "OTelTransport" in transports.__all__

    def test_otel_importable(self):
        """OTelTransport can be imported from the transports package."""
        from strands_robots.telemetry.transports import OTelTransport

        assert OTelTransport is not None


# ============================================================================
# Bonus Coverage: stdout.py — non-compact mode + single dict
# ============================================================================


class TestStdoutTransportBonusLines:
    """Cover uncovered lines in StdoutTransport."""

    def test_non_compact_mode(self):
        """Non-compact mode prints full JSON (line 80)."""
        import logging

        from strands_robots.telemetry.transports.stdout import StdoutTransport

        transport = StdoutTransport(compact=False, log_level=logging.DEBUG)
        events = [
            {"category": "joint_state", "robot_id": "test", "data": {"q": [0.1]}},
        ]
        result = transport.send(events)
        assert result is True
        assert transport._total_events == 1

    def test_single_dict_payload(self):
        """Single dict (not list) is auto-wrapped (line 53-54)."""
        from strands_robots.telemetry.transports.stdout import StdoutTransport

        transport = StdoutTransport()
        # Send bytes that decode to a single dict
        single_event = {"category": "heartbeat", "robot_id": "test"}
        raw_bytes = json.dumps(single_event).encode("utf-8")

        result = transport.send(raw_bytes)
        assert result is True
        assert transport._total_events == 1

    def test_non_compact_multiple_events(self):
        """Non-compact mode with multiple events."""
        import logging

        from strands_robots.telemetry.transports.stdout import StdoutTransport

        transport = StdoutTransport(compact=False, log_level=logging.DEBUG)
        events = [
            {"category": "joint_state", "robot_id": "r1", "data": {"q": [0.1]}},
            {"category": "observation", "robot_id": "r2", "data": {"vel": 1.0}},
        ]
        result = transport.send(events)
        assert result is True
        assert transport._total_events == 2


# ============================================================================
# Bonus Coverage: types.py — HAS_NUMPY=False, bytes size, no metadata
# ============================================================================


class TestTelemetryEventBonusLines:
    """Cover uncovered lines in TelemetryEvent."""

    def test_serialize_without_metadata(self):
        """Serialization without metadata omits the key (line 204-212)."""
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent

        event = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="test",
            data={"q": [0.1]},
            metadata=None,
        )
        raw = event.serialize()
        parsed = json.loads(raw)
        assert "metadata" not in parsed

    def test_serialize_with_metadata(self):
        """Serialization with metadata includes it."""
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent

        event = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="test",
            data={"q": [0.1]},
            metadata={"source": "test"},
        )
        raw = event.serialize()
        parsed = json.loads(raw)
        assert parsed["metadata"]["source"] == "test"

    def test_size_bytes_with_bytes_data(self):
        """Size estimate for bytes data uses length reference (line 175)."""
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent

        event = TelemetryEvent(
            category=EventCategory.CAMERA_FRAME,
            robot_id="cam0",
            data={"frame": b"\x00" * 5000, "label": "test"},
        )
        est = event.size_bytes()
        assert est > 0
        # Bytes aren't counted at full size — just a reference
        assert est < 5000

    def test_size_bytes_with_dict_data(self):
        """Size estimate for nested dict data."""
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent

        event = TelemetryEvent(
            category=EventCategory.CUSTOM,
            robot_id="test",
            data={"nested": {"a": 1, "b": 2}},
        )
        est = event.size_bytes()
        assert est > 0

    def test_size_bytes_with_scalar_data(self):
        """Size estimate for scalar values."""
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent

        event = TelemetryEvent(
            category=EventCategory.HEARTBEAT,
            robot_id="test",
            data={"value": 42, "name": "test"},
        )
        est = event.size_bytes()
        assert est > 200  # Base overhead

    def test_has_numpy_false_serialize(self):
        """When HAS_NUMPY=False, serialize still works for basic types."""
        from strands_robots.telemetry import types as types_module

        original = types_module.HAS_NUMPY
        try:
            types_module.HAS_NUMPY = False

            from strands_robots.telemetry.types import EventCategory, TelemetryEvent

            event = TelemetryEvent(
                category=EventCategory.JOINT_STATE,
                robot_id="test",
                data={"q": [0.1, 0.2], "name": "joint"},
            )
            raw = event.serialize()
            parsed = json.loads(raw)
            assert parsed["data"]["q"] == [0.1, 0.2]
            assert parsed["data"]["name"] == "joint"
        finally:
            types_module.HAS_NUMPY = original

    def test_has_numpy_false_size_bytes(self):
        """HAS_NUMPY=False path in size_bytes with various data types."""
        from strands_robots.telemetry import types as types_module

        original = types_module.HAS_NUMPY
        try:
            types_module.HAS_NUMPY = False

            from strands_robots.telemetry.types import EventCategory, TelemetryEvent

            event = TelemetryEvent(
                category=EventCategory.OBSERVATION,
                robot_id="test",
                data={
                    "values": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "raw": b"\x00" * 100,
                    "nested": {"a": 1},
                    "scalar": 42,
                },
            )
            est = event.size_bytes()
            assert est > 0
        finally:
            types_module.HAS_NUMPY = original
