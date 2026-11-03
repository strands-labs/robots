"""The ``control`` shadow reply, kept under IoT's document cap whatever the delta was. AWS accepts any operator
``desired`` under its own 8 KB cap and delivers it; the device's answer carries one refusal line per key plus a
``desired: {key: null}`` clear per key, 2-3x the delta, so a large or hostile delta would push the reply over the cap
and the write would fail (killing the control path if nothing caught it). Here the per-key refusals collapse into one
line when they would not fit, and the clears (which AWS needs, or it re-delivers the delta) go out in as many writes
as it takes."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any


def encode(reported: Mapping[str, Any], clear: Mapping[str, None]) -> bytes:
    body = {"state": {"reported": dict(reported), **({"desired": dict(clear)} if clear else {})}}
    return json.dumps(body, separators=(",", ":"), allow_nan=False).encode()


def control_writes(reported: Mapping[str, Any], clear: Mapping[str, None], cap: int) -> Iterator[dict[str, Any]]:
    """The ``state`` documents to write, in order, each at most ``cap`` bytes; the first says everything the reply
    has to say, later ones only carry the clears that did not fit."""
    refused = reported.get("refused") or {}
    if refused and len(encode(reported, clear)) > cap:
        key, why = next(iter(refused.items()))
        word = f"{len(refused)} keys refused; reply over {cap} B collapsed into one line; first: {key}: {why}"
        reported = {**reported, "refused": {"*": word[:512]}}
    keys = list(clear)
    while True:
        n = len(keys)
        while n and len(encode(reported, dict.fromkeys(keys[:n]))) > cap:
            n //= 2
        if not (n or reported):  # a clear too big even alone stays in desired; AWS re-delivers it, refused again
            return
        yield {"reported": dict(reported), **({"desired": dict.fromkeys(keys[:n])} if n else {})}
        keys = keys[n:]
        if not keys:
            return
        reported = {}
