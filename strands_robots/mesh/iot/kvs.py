"""Kinesis Video Streams for a device's cameras (MAPPING rows 6.1, 6.2, 6.5, 6.6).

One KVS stream and one WebRTC signaling channel per camera sensor, named ``strands-<tenant>-<device>-<camera>``;
the names are written into the ``manifest`` shadow as ``cameras[]`` so a dashboard finds the live view and the
recording of a camera it has never met. Frames go out through GStreamer: ``appsrc → videoconvert → H.264 →
kvssink`` on one thread per camera, fed from the same ``observe(images=True)`` the mesh already pays for (measured
on an L40S: 640×480 at 15 fps, x264 +0.15 ms per frame to push, nvh264enc fragment-head → consumer p50 36 ms;
GetMedia first byte 0.33 s; WebRTC media path ready 1.25 s). There is no PutMedia in boto3 and a hand-rolled
chunked POST has no ACK or backpressure, so without ``kvssink`` this module refuses with the install hint rather
than pretend.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from typing import Any

from ..._extras import hint
from .. import wire

_NAME = re.compile(r"[^a-zA-Z0-9_.-]")
ENCODERS = {
    "x264": "x264enc tune=zerolatency speed-preset=ultrafast bitrate={kbps} key-int-max={fps} bframes=0",
    "nvh264": "nvh264enc preset=low-latency-hq zerolatency=true bitrate={kbps} gop-size={fps} rc-mode=cbr",
}


def stream_name(tenant: str, device_id: str, camera: str) -> str:
    """``strands-<tenant>-<device>-<camera>`` in KVS's ``[a-zA-Z0-9_.-]{1,256}``."""
    raw = f"strands-{wire.check_id(tenant, 'tenant')}-{wire.check_id(device_id, 'device_id')}-{camera}"
    return _NAME.sub("-", raw)[:256]


def cameras_of(manifest: Mapping[str, Any]) -> list[str]:
    """The bare camera names (``wrist``), keys of ``observe(images=True).images``; the manifest says ``wrist.image``."""
    sensors = manifest.get("sensors") or ()
    return [
        str(s["name"]).removesuffix(".image") for s in sensors if isinstance(s, Mapping) and s.get("kind") == "camera"
    ]


def camera_entries(record: Mapping[str, Any], region: str) -> list[dict[str, str]]:
    """What goes into ``manifest.cameras[]``: per camera the stream, the signaling channel and the region."""
    t, did = str(record["tenant"]), str(record["id"])
    return [
        {"camera": c, "stream": stream_name(t, did, c), "channel": stream_name(t, did, c), "region": region}
        for c in cameras_of(record.get("manifest") or {})
    ]


def pipeline(
    name: str, width: int, height: int, fps: int, *, region: str, encoder: str = "x264", kbps: int = 1500
) -> str:
    """The measured GStreamer description for one camera (RGB frames in, 1 s fragments out)."""
    if encoder not in ENCODERS:
        raise ValueError(f"encoder {encoder!r} is not one of {sorted(ENCODERS)}")
    enc = ENCODERS[encoder].format(kbps=kbps, fps=fps)
    return (
        f"appsrc name=src is-live=true format=time do-timestamp=true "
        f"caps=video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1 "
        f"! videoconvert ! {enc} ! h264parse ! video/x-h264,stream-format=avc,alignment=au "
        f"! kvssink stream-name={name} aws-region={region} storage-size=128 fragment-duration=1000"
    )


def ensure_stream(kv: Any, name: str, *, retention_hours: int = 24) -> str:
    """Create the stream and the signaling channel of ``name`` if missing (``kv``: boto3 ``kinesisvideo``)."""
    from botocore.exceptions import ClientError

    for call, kw in (
        (kv.create_stream, {"StreamName": name, "DataRetentionInHours": retention_hours, "MediaType": "video/h264"}),
        (kv.create_signaling_channel, {"ChannelName": name, "ChannelType": "SINGLE_MASTER"}),
    ):
        try:
            call(**kw)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") != "ResourceInUseException":  # only "it exists" is fine
                raise
    return name


def delete_stream(kv: Any, name: str) -> None:
    arn = kv.describe_stream(StreamName=name)["StreamInfo"]["StreamARN"]
    kv.delete_stream(StreamARN=arn)
    ch = kv.describe_signaling_channel(ChannelName=name)["ChannelInfo"]["ChannelARN"]
    kv.delete_signaling_channel(ChannelARN=ch)


def clip_window(run: Mapping[str, Any], pad_s: float = 1.0) -> dict[str, Any]:
    """``GetClip`` arguments for a recorded run (``started``/``stopped`` epoch seconds, part 9's event payload)."""
    t0, t1 = float(run["started"]) - pad_s, float(run["stopped"]) + pad_s
    return {"ClipFragmentSelector": {"FragmentSelectorType": "PRODUCER_TIMESTAMP",
                                     "TimestampRange": {"StartTimestamp": t0, "EndTimestamp": t1}}}  # fmt: skip


class CameraPublisher:
    """Feed one camera's frames into ``kvssink`` on its own thread (sample once, any number of viewers)."""

    def __init__(self, name: str, frame: Callable[[], Any], fps: int, *, region: str, encoder: str = "x264") -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except (ImportError, ValueError) as exc:
            raise ImportError(
                f"KVS cameras need GStreamer (python3-gi) and the kvssink plugin: {hint('iot-edge')}"
            ) from exc
        Gst.init(None)
        if Gst.ElementFactory.find("kvssink") is None:
            raise ImportError(f"GStreamer has no kvssink element (build the KVS producer SDK): {hint('iot-edge')}")
        first = frame()
        h, w = int(first.shape[0]), int(first.shape[1])
        self._gst, self.name, self.frame, self.fps = Gst, name, frame, int(fps)
        self.pipe = Gst.parse_launch(pipeline(name, w, h, self.fps, region=region, encoder=encoder))
        self.src = self.pipe.get_by_name("src")
        self.frames = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"kvs-{name}")

    def start(self) -> CameraPublisher:
        self.pipe.set_state(self._gst.State.PLAYING)
        self._thread.start()
        return self

    def _loop(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.wait(period):
            data = self.frame().tobytes()
            buf = self._gst.Buffer.new_wrapped(data)
            if self.src.emit("push-buffer", buf) == self._gst.FlowReturn.OK:
                self.frames += 1

    def stop(self) -> None:
        self._stop.set()
        self.src.emit("end-of-stream")
        self.pipe.set_state(self._gst.State.NULL)


def publish_cameras(
    record: Mapping[str, Any], robot: Any, *, region: str = "us-west-2", fps: int = 15
) -> Callable[[], None]:
    """Start a ``CameraPublisher`` per camera sensor of ``record``; returns the undo. Refuses (``ImportError``)
    when no producer is available, before any thread starts."""
    names = cameras_of(record.get("manifest") or {})
    if not names:
        return lambda: None
    pubs: list[CameraPublisher] = []
    for cam in names:

        def frame(c: str = cam) -> Any:
            return robot.observe(images=True).images[c]

        pubs.append(
            CameraPublisher(stream_name(str(record["tenant"]), str(record["id"]), cam), frame, fps, region=region)
        )
    for p in pubs:
        p.start()

    def undo() -> None:
        for p in pubs:
            p.stop()

    return undo
