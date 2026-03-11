"""OpenTelemetry bridge transport.

Forwards telemetry events as OTel spans, bridging into the existing
Strands SDK telemetry infrastructure (Langfuse, Jaeger, X-Ray, etc.).

This allows robot telemetry to appear alongside agent decision traces
in the same observability backend.

Requires: opentelemetry-api, opentelemetry-sdk (optional).
"""

from __future__ import annotations

import gzip
import json
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace as trace_api

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


class OTelTransport:
    """OpenTelemetry span transport for telemetry events.

    Converts telemetry events into OTel spans, enabling them to
    appear in the same trace as Strands agent operations.

    Args:
        tracer_name: OTel tracer name.
        service_name: Service name for span attributes.
    """

    def __init__(
        self,
        tracer_name: str = "strands-robots-telemetry",
        service_name: str = "strands-robots",
    ):
        self.tracer_name = tracer_name
        self.service_name = service_name
        self._tracer = None

        if HAS_OTEL:
            self._tracer = trace_api.get_tracer(tracer_name)
        else:
            logger.debug("OpenTelemetry not available — OTelTransport is a no-op")

    def send(self, payload: Union[List[Dict[str, Any]], bytes]) -> bool:
        """Convert events to OTel spans.

        Args:
            payload: Either a list of event dicts or gzip-compressed bytes.

        Returns:
            True on success, False on failure.
        """
        if not self._tracer:
            return True  # No-op if OTel not available

        try:
            # Handle compressed payloads
            if isinstance(payload, bytes):
                try:
                    decompressed = gzip.decompress(payload)
                    events = json.loads(decompressed)
                except Exception:
                    events = json.loads(payload)
            else:
                events = payload

            if not isinstance(events, list):
                events = [events]

            for event in events:
                self._emit_span(event)

            return True

        except Exception as e:
            logger.error("OTel transport error: %s", e)
            return False

    def _emit_span(self, event: Dict[str, Any]) -> None:
        """Create an OTel span from a telemetry event."""
        if not self._tracer:
            return

        category = event.get("category", "unknown")
        robot_id = event.get("robot_id", "unknown")

        with self._tracer.start_as_current_span(
            name=f"robot.{category}",
            attributes={
                "robot.id": robot_id,
                "robot.category": category,
                "robot.tier": event.get("tier", ""),
                "robot.frame_id": event.get("frame_id", 0),
                "robot.sim_or_real": event.get("sim_or_real", ""),
                "robot.correlation_id": event.get("correlation_id", "") or "",
                "service.name": self.service_name,
            },
        ) as span:
            # Add data as span events (not attributes — data can be large)
            data = event.get("data", {})
            if data and isinstance(data, dict):
                # Only add small data items as attributes
                for k, v in data.items():
                    if isinstance(v, (str, int, float, bool)):
                        span.set_attribute(f"robot.data.{k}", v)

    def get_stats(self) -> Dict[str, Any]:
        """Get OTel transport stats."""
        return {
            "otel_available": HAS_OTEL,
            "tracer_name": self.tracer_name,
        }
