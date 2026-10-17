"""Who is commanding: the developer's own code, or an agent/dashboard (whose every write passes the gate).

``with from_agent():`` marks the calls made inside it; the run loop then writes each step through the safety
chain and the operator gate instead of acting directly. A background ``start`` carries the mark to its thread.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_AGENT: ContextVar[bool] = ContextVar("strands_robots_agent_source", default=False)


@contextmanager
def from_agent() -> Iterator[None]:
    token = _AGENT.set(True)
    try:
        yield
    finally:
        _AGENT.reset(token)


def is_agent() -> bool:
    return _AGENT.get()
