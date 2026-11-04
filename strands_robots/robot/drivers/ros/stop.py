"""The profile's STOP sequence: zero velocity + release, nav stop. Never gated, never a damp."""

from __future__ import annotations

import json
import logging
from typing import Any

from ....ros.bus import RosBus
from ....ros.convert import from_dict, message_class
from ....ros.profile import Profile

LOG = logging.getLogger(__name__)


def _fill(fields: Any, sender: str) -> Any:
    raw = json.dumps(fields).replace("{sender_id}", sender)
    return json.loads(raw)


def send_stop(bus: RosBus, profile: Profile, sender: str) -> list[str]:
    """Publish every STOP step; one failing step never skips the next. Returns what was sent."""
    sent = []
    for step in profile.stop:
        try:
            msg = from_dict(message_class(step["type"]), _fill(step.get("fields") or {}, sender))
            for _ in range(int(step.get("repeat", 1))):
                bus.publish(step["topic"], step["type"], msg, qos=step.get("qos", "reliable"))
            sent.append(step["topic"])
        except Exception:  # noqa: BLE001 - a stop must try every step
            LOG.exception("ros stop step on %s failed", step.get("topic"))
    return sent
