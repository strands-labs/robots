"""Record topics for a while: a real rosbag2 (MCAP) when rosbag2_py is here, else JSON lines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .bus import RosBus
from .convert import to_data
from .names import canonical, positive

LEROBOT_HINT = (
    "For a LeRobot dataset, record through strands-robots, not a bag: "
    "robot.run(policy, seconds=..., record='<dir>', push='<hf repo>') writes observation.state, "
    "action and camera frames in LeRobot format from the same run loop."
)


def record(bus: RosBus, topics: dict[str, str], seconds: float, path: str | Path) -> dict[str, Any]:
    """``topics``: {name: type}. Returns per-topic message counts and the file written."""
    seconds = positive(seconds, "seconds")
    if not 0 < seconds <= 600:
        raise ValueError("record seconds must be in (0, 600]")
    topics = {canonical(t): ty for t, ty in topics.items()}
    path = Path(path).expanduser()
    counts = {t: 0 for t in topics}
    try:
        import rosbag2_py
        from rclpy.serialization import serialize_message
    except ImportError:
        rosbag2_py = None
    w, fmt = None, "jsonl"
    for storage in ("mcap", "sqlite3") if rosbag2_py is not None else ():
        try:  # Humble's ros-base has no mcap plugin; sqlite3 is always there
            w = rosbag2_py.SequentialWriter()
            w.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=storage), rosbag2_py.ConverterOptions("", ""))
            fmt = storage
            break
        except RuntimeError:
            w = None
    if w is not None:
        for i, (t, ty) in enumerate(topics.items()):
            meta = {"name": t, "type": ty, "serialization_format": "cdr"}
            try:
                w.create_topic(rosbag2_py.TopicMetadata(id=i, **meta))
            except TypeError:  # Humble's TopicMetadata has no id
                w.create_topic(rosbag2_py.TopicMetadata(**meta))

        def sink(t: str, m: Any) -> None:
            w.write(t, serialize_message(m), time.time_ns())
            counts[t] += 1

    else:
        path = path.with_suffix(".jsonl")
        fh = path.open("w")

        def sink(t: str, m: Any) -> None:
            fh.write(json.dumps({"t": time.time(), "topic": t, "msg": to_data(m)}) + "\n")
            counts[t] += 1

    listeners = {t: (lambda m, t=t: sink(t, m)) for t in topics}
    for t, ty in topics.items():
        bus.subscribe(t, ty, listener=listeners[t])
    time.sleep(seconds)
    for t in topics:
        bus.unlisten(t, listeners[t])
    if fmt == "jsonl":
        fh.close()
    else:
        del w
    return {"path": str(path), "format": fmt, "counts": counts, "lerobot": LEROBOT_HINT}
