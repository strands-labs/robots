"""A small bounded TTL cache - the type-ahead's memory, with an actual end to it. Both Hub searches
(checkpoints and datasets) memoised their answers in a plain dict keyed by the query, with a TTL
checked on read.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

#: Enough for a long type-ahead session (each keystroke is a key) without letting a
#: day-long page keep every prefix ever typed.
DEFAULT_MAX_ENTRIES = 64


class TTLCache[V]:
    """Thread-safe, size-bounded, self-pruning cache of values with an age."""

    def __init__(
        self,
        ttl_s: float,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._ttl_s = float(ttl_s)
        self._max = int(max_entries)
        self._clock = clock
        self._lock = threading.Lock()
        # dicts preserve insertion order, which is exactly the eviction order wanted
        self._data: dict[str, tuple[float, V]] = {}

    def get(self, key: str) -> V | None:
        """The value if it is still fresh, else None - and an expired entry is dropped."""
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            if self._clock() - hit[0] >= self._ttl_s:
                del self._data[key]
                return None
            return hit[1]

    def put(self, key: str, value: V) -> None:
        with self._lock:
            now = self._clock()
            # a re-written key must move to the END of the eviction order: it is the
            # most recently useful, and leaving it in place would evict it first
            self._data.pop(key, None)
            self._data = {k: v for k, v in self._data.items() if now - v[0] < self._ttl_s}
            while len(self._data) >= self._max:
                self._data.pop(next(iter(self._data)))
            self._data[key] = (now, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
