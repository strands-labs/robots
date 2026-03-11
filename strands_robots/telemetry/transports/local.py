"""Local Write-Ahead Log (WAL) transport for edge devices.

Writes telemetry events to local JSONL files with automatic rotation.
Designed for edge deployment (Thor Jetson AGX, Isaac Sim EC2) where
network connectivity may be intermittent.

Adapted from S3 writer concept for edge persistence.

Features:
- JSONL append-only format (one JSON object per line)
- Automatic file rotation by size
- Gzip compression of rotated files
- Ring-buffer file count limit (oldest rotated files deleted)
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class LocalWALTransport:
    """Write-ahead log transport for local telemetry persistence.

    Args:
        wal_dir: Directory for WAL files.
        max_file_bytes: Max size before rotation (default 10MB).
        max_files: Max rotated files to keep (oldest deleted).
        compress_rotated: Gzip compress rotated files.
    """

    def __init__(
        self,
        wal_dir: str = "./telemetry_wal",
        max_file_bytes: int = 10 * 1024 * 1024,
        max_files: int = 50,
        compress_rotated: bool = True,
    ):
        self.wal_dir = Path(wal_dir)
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.compress_rotated = compress_rotated

        self._current_file: Optional[Any] = None
        self._current_path: Optional[Path] = None
        self._current_size: int = 0
        self._total_written: int = 0
        self._file_counter: int = 0  # Monotonic counter for unique filenames

        # Ensure directory exists
        self.wal_dir.mkdir(parents=True, exist_ok=True)

    def send(self, payload: Union[List[Dict[str, Any]], bytes]) -> bool:
        """Write payload to WAL file.

        Args:
            payload: Either a list of event dicts or gzip-compressed bytes.

        Returns:
            True on success, False on failure.
        """
        try:
            # Handle compressed payloads
            if isinstance(payload, bytes):
                try:
                    decompressed = gzip.decompress(payload)
                    events = json.loads(decompressed)
                except Exception:
                    # Not gzip — treat as raw JSON
                    events = json.loads(payload)
            else:
                events = payload

            if not isinstance(events, list):
                events = [events]

            # Write each event as a JSONL line
            for event in events:
                line = json.dumps(event, separators=(",", ":")) + "\n"
                line_bytes = line.encode("utf-8")
                self._write_line(line_bytes)

            return True

        except Exception as e:
            logger.error("WAL write error: %s", e)
            return False

    def _write_line(self, line_bytes: bytes) -> None:
        """Write a single line, rotating file if needed."""
        # Check rotation
        if (
            self._current_file is None
            or self._current_size + len(line_bytes) > self.max_file_bytes
        ):
            self._rotate()

        self._current_file.write(line_bytes)
        self._current_file.flush()
        self._current_size += len(line_bytes)
        self._total_written += len(line_bytes)

    def _rotate(self) -> None:
        """Close current file, compress if configured, open new file."""
        if self._current_file:
            self._current_file.close()
            # Clear handle immediately so a failed open() below
            # doesn't leave us pointing at a closed file.
            self._current_file = None

            # Compress the rotated file
            if self.compress_rotated and self._current_path:
                try:
                    gz_path = self._current_path.with_suffix(".jsonl.gz")
                    with open(self._current_path, "rb") as f_in:
                        with gzip.open(gz_path, "wb") as f_out:
                            f_out.write(f_in.read())
                    self._current_path.unlink()
                except Exception as e:
                    logger.warning("Failed to compress %s: %s", self._current_path, e)

        # Clean up old files if over limit
        self._cleanup_old_files()

        # Open new file — use monotonic counter to guarantee unique names
        # even when rotations happen within the same millisecond
        timestamp = int(time.time() * 1000)
        self._file_counter += 1
        filename = f"telemetry_{timestamp}_{self._file_counter:04d}.jsonl"
        self._current_path = self.wal_dir / filename
        self._current_file = open(self._current_path, "ab")
        self._current_size = 0

    def _cleanup_old_files(self) -> None:
        """Remove oldest rotated files if over limit.

        This is called just before opening a new file, so we must leave
        room for the file about to be created.  Use ``>=`` instead of ``>``
        to ensure the total on-disk count (existing + new) never exceeds
        ``max_files``.
        """
        files = sorted(
            self.wal_dir.glob("telemetry_*"), key=lambda p: p.stat().st_mtime
        )
        while len(files) >= self.max_files:
            oldest = files.pop(0)
            try:
                oldest.unlink()
                logger.debug("Deleted old WAL file: %s", oldest)
            except OSError:
                pass

    def close(self) -> None:
        """Close the current WAL file."""
        if self._current_file:
            self._current_file.close()
            self._current_file = None

    def get_stats(self) -> Dict[str, Any]:
        """Get WAL transport stats."""
        files = list(self.wal_dir.glob("telemetry_*"))
        total_size = sum(f.stat().st_size for f in files if f.exists())
        return {
            "wal_dir": str(self.wal_dir),
            "file_count": len(files),
            "total_size_bytes": total_size,
            "total_written_bytes": self._total_written,
            "current_file": str(self._current_path) if self._current_path else None,
            "current_size": self._current_size,
        }
