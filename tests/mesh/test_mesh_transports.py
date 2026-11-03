"""Every transport honours the same contract: publish/subscribe, query/queryable, liveliness."""

from __future__ import annotations

import threading
import time

import pytest

from strands_robots.mesh.transports import key_matches, open_transport


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_key_expressions():
    assert key_matches("strands/a/*/state", "strands/a/d1/state")
    assert not key_matches("strands/a/*/state", "strands/a/d1/event/x")
    assert key_matches("strands/a/**", "strands/a/d1/event/x")
    assert not key_matches("strands/b/**", "strands/a/d1")


def test_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match="unknown mesh transport"):
        open_transport("carrier-pigeon://x")


def test_publish_subscribe(pair, tenant):
    a, b = pair
    got = []
    unsub = b.subscribe(f"strands/{tenant}/*/state", lambda k, p: got.append((k, p)))
    time.sleep(0.2)
    a.publish(f"strands/{tenant}/d1/state", b"x")
    assert _wait(lambda: got)
    assert got[0] == (f"strands/{tenant}/d1/state", b"x")
    unsub()
    time.sleep(0.2)
    a.publish(f"strands/{tenant}/d1/state", b"y")
    time.sleep(0.2)
    assert len(got) == 1


def test_query_reaches_the_queryable(pair, tenant):
    a, b = pair
    a.queryable(f"strands/{tenant}/d1/rpc/*", lambda k, p: k.encode() + b"|" + p)
    time.sleep(0.2)
    assert b.query(f"strands/{tenant}/d1/rpc/act", b"go", 2.0) == [f"strands/{tenant}/d1/rpc/act|go".encode()]
    assert b.query(f"strands/{tenant}/nobody/rpc/act", b"go", 0.3) == []


def test_liveliness_appears_and_disappears(pair, tenant):
    a, b = pair
    seen: dict[str, bool] = {}
    lock = threading.Lock()

    def on(k, alive):
        with lock:
            seen[k] = alive

    key = f"strands/{tenant}/@presence/d1"
    drop = a.liveliness(key)
    b.watch_liveliness(f"strands/{tenant}/@presence/*", on)
    assert _wait(lambda: seen.get(key) is True)
    drop()
    assert _wait(lambda: seen.get(key) is False)
