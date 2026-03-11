"""Stdout/logging transport for debugging and development.

Prints telemetry events to stdout or Python logging.
Useful for development, debugging, and CI testing.
"""

from __future__ import annotations

import gzip
import json
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class StdoutTransport:
    """Debug transport that logs events to stdout/logging.

    Args:
        log_level: Python logging level (default INFO).
        compact: If True, print compact one-line summaries.
        max_events_per_batch: Max events to print per batch (avoid flooding).
    """

    def __init__(
        self,
        log_level: int = logging.INFO,
        compact: bool = True,
        max_events_per_batch: int = 10,
    ):
        self.log_level = log_level
        self.compact = compact
        self.max_events_per_batch = max_events_per_batch
        self._total_events = 0

    def send(self, payload: Union[List[Dict[str, Any]], bytes]) -> bool:
        """Log events to stdout.

        Args:
            payload: Either a list of event dicts or gzip-compressed bytes.

        Returns:
            Always True (stdout never fails).
        """
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

            for i, event in enumerate(events):
                if i >= self.max_events_per_batch:
                    logger.log(
                        self.log_level,
                        f"  ... and {len(events) - i} more events",
                    )
                    break

                if self.compact:
                    cat = event.get("category", "?")
                    robot = event.get("robot_id", "?")
                    frame = event.get("frame_id", "?")
                    corr = event.get("correlation_id", "")
                    corr_str = f" trace={corr}" if corr else ""
                    logger.log(
                        self.log_level,
                        f"📡 [{cat}] robot={robot} frame={frame}{corr_str}",
                    )
                else:
                    logger.log(
                        self.log_level,
                        f"📡 {json.dumps(event, indent=2)}",
                    )

            self._total_events += len(events)
            return True

        except Exception as e:
            logger.error("Stdout transport error: %s", e)
            return True  # Never fail

    def get_stats(self) -> Dict[str, Any]:
        """Get stdout transport stats."""
        return {"total_events_logged": self._total_events}
