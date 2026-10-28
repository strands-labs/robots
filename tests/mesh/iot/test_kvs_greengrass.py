"""KVS (MAPPING 6.1, 6.2, 6.5, 6.6) and Greengrass (5.1, 5.2, 5.5) as data; the producer refuses honestly."""

from __future__ import annotations

import json
import re

import pytest

from strands_robots.mesh.iot import greengrass, kvs

RECORD = {
    "id": "arm-1",
    "device_type": "robot_arm",
    "tenant": "lab",
    "labels": {},
    "manifest": {"sensors": [{"name": "shoulder_pan.pos", "unit": "rad", "kind": "joint"},
                             {"name": "camera1.image", "unit": "rgb8", "kind": "camera"},
                             {"name": "wrist.image", "unit": "rgb8", "kind": "camera"}]},
}  # fmt: skip


def test_one_stream_and_channel_per_camera_sensor_with_legal_names():
    entries = kvs.camera_entries(RECORD, "us-west-2")
    assert [e["camera"] for e in entries] == ["camera1", "wrist"]
    assert entries[0]["stream"] == entries[0]["channel"] == "strands-lab-arm-1-camera1"
    assert all(re.fullmatch(r"[a-zA-Z0-9_.-]{1,256}", e["stream"]) for e in entries)
    assert kvs.stream_name("lab", "arm", "cam:top") == "strands-lab-arm-cam-top"
    assert kvs.cameras_of({}) == []


def test_camera_names_are_the_keys_observe_returns_for_the_record_the_mesh_builds():
    from strands_robots import Robot, Simulation
    from strands_robots.mesh.record import record

    sim = Simulation()
    arm = Robot("so101", sim=sim)
    sim.camera("wrist", size=(32, 24), attach=arm)
    rec = record(arm, "arm-1", {}, "lab", ())
    assert kvs.cameras_of(rec["manifest"]) == sorted(arm.observe(images=True).images) == ["wrist"]
    assert [e["stream"] for e in kvs.camera_entries(rec, "us-west-2")] == ["strands-lab-arm-1-wrist"]


@pytest.mark.parametrize(("code", "raises"), [("ResourceInUseException", False),
                                              ("AccountStreamLimitExceededException", True),
                                              ("AccountChannelLimitExceededException", True)])  # fmt: skip
def test_ensure_stream_passes_only_an_existing_resource(code, raises):
    from botocore.exceptions import ClientError

    class Kv:
        def _refuse(self, **kw):
            raise ClientError({"Error": {"Code": code, "Message": code}}, "Create")

        create_stream = create_signaling_channel = _refuse

    if raises:
        with pytest.raises(ClientError, match=code):
            kvs.ensure_stream(Kv(), "s")
    else:
        assert kvs.ensure_stream(Kv(), "s") == "s"


def test_pipeline_is_the_measured_one_and_clip_window_pads_the_run():
    p = kvs.pipeline("strands-lab-arm-1-camera1", 640, 480, 15, region="us-west-2")
    assert p.startswith("appsrc name=src is-live=true")
    assert "width=640,height=480,framerate=15/1" in p and "x264enc tune=zerolatency" in p
    assert p.endswith(
        "kvssink stream-name=strands-lab-arm-1-camera1 aws-region=us-west-2 storage-size=128 fragment-duration=1000"
    )
    assert "nvh264enc" in kvs.pipeline("s", 256, 256, 15, region="us-west-2", encoder="nvh264")
    with pytest.raises(ValueError, match="encoder"):
        kvs.pipeline("s", 1, 1, 1, region="us-west-2", encoder="vp9")
    w = kvs.clip_window({"started": 100.0, "stopped": 130.0})
    assert w["ClipFragmentSelector"]["TimestampRange"] == {"StartTimestamp": 99.0, "EndTimestamp": 131.0}


def test_ensure_stream_is_idempotent_on_moto(aws):
    import boto3

    kv = boto3.client("kinesisvideo", region_name="us-west-2")
    try:
        kvs.ensure_stream(kv, "strands-lab-arm-1-camera1")
    except Exception as exc:  # moto 5.2 has streams but not signaling channels
        pytest.skip(f"moto kinesisvideo: {type(exc).__name__}")
    kvs.ensure_stream(kv, "strands-lab-arm-1-camera1")
    assert kv.describe_stream(StreamName="strands-lab-arm-1-camera1")["StreamInfo"]["MediaType"] == "video/h264"


def test_publish_cameras_refuses_without_a_producer_before_any_thread_starts():
    class Arm:
        def observe(self, images=False):
            raise AssertionError("observe must not be called when the producer is missing")

    pytest.importorskip("numpy")
    try:
        import gi  # noqa: F401

        pytest.skip("GStreamer present on this host; the refusal path is for hosts without it")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="iot-edge"):
        kvs.publish_cameras(RECORD, Arm())
    assert kvs.publish_cameras({**RECORD, "manifest": {}}, Arm())() is None  # no cameras: a no-op undo


def _resolve(value: str, config: dict) -> str:
    """The nucleus' ``{configuration:<ptr>}`` interpolation (RFC 6901, as Jackson's ``JsonNode.at``): text as is,
    numbers and containers as JSON, a missing node leaves the variable untouched."""

    def one(m: re.Match) -> str:
        node: object = config
        for part in m.group(1).split("/")[1:] if m.group(1) else []:
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict) and key in node:
                node = node[key]
            elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
                node = node[int(key)]
            else:
                return m.group(0)
        return node if isinstance(node, str) else json.dumps(node)

    return re.sub(r"\{configuration:([^}]*)\}", one, value)


def _component_env(r: dict) -> dict[str, str]:
    cfg = r["ComponentConfiguration"]["DefaultConfiguration"]
    return {k: _resolve(v, cfg) for k, v in r["Manifests"][0]["Lifecycle"]["Run"]["Setenv"].items()}


def test_resolver_follows_rfc6901_so_slash_alone_is_not_the_root():
    assert _resolve("{configuration:/}", {"tenant": "lab"}) == "{configuration:/}"
    assert _resolve("{configuration:/tenant}", {"tenant": "lab"}) == "lab"
    assert _resolve("{configuration:/robots}", {"robots": [{"name": "so101"}]}) == '[{"name": "so101"}]'


def test_the_run_step_environment_resolves_to_the_default_configuration():
    r = greengrass.recipe("1.0.0", "s3://b/w.whl", tenant="lab", robots=[{"name": "so101", "device_id": "arm-1"}])
    env = _component_env(r)
    assert not any("{configuration:" in v for v in env.values())  # every pointer names a node
    assert greengrass.from_env(env) == r["ComponentConfiguration"]["DefaultConfiguration"]
    with pytest.raises(ValueError, match="STRANDS_GG_ROBOTS"):
        greengrass.from_env({"STRANDS_GG_ROBOTS": "{configuration:/robots}"})


def test_recipe_and_deployment_are_small_and_point_at_the_module():
    r = greengrass.recipe("1.0.0", "s3://bucket/strands_robots-1.0.0-py3-none-any.whl", tenant="lab",
                          robots=[{"name": "so101", "device_id": "arm-1"}])  # fmt: skip
    assert r["ComponentName"] == "com.strands.robots" and len(json.dumps(r)) <= 8 * 1024
    life = r["Manifests"][0]["Lifecycle"]
    assert life["Run"]["Script"] == "python3 -m strands_robots.mesh.iot.greengrass"
    assert life["Install"]["Script"].endswith("strands_robots-1.0.0-py3-none-any.whl")
    cfg = r["ComponentConfiguration"]["DefaultConfiguration"]
    assert cfg["tenant"] == "lab" and cfg["robots"][0]["device_id"] == "arm-1" and cfg["shadow_hz"] == 0.5
    d = greengrass.deployment("arn:aws:iot:us-west-2:123456789012:thinggroup/strands-lab", "1.0.0")
    assert d["components"] == {"com.strands.robots": {"componentVersion": "1.0.0"}}
    assert d["deploymentPolicies"]["componentUpdatePolicy"]["action"] == "NOTIFY_COMPONENTS"


def test_component_main_mounts_the_configured_robots_on_a_mem_mesh(monkeypatch):
    import threading
    import uuid

    from strands_robots.mesh import Mesh

    name = f"mem://gg-{uuid.uuid4().hex[:8]}"
    r = greengrass.recipe("1.0.0", "s3://b/w.whl", tenant="lab", robots=[{"name": "so101", "device_id": "arm-1"}],
                          transports=[name], state_hz=5)  # fmt: skip
    for k, v in _component_env(r).items():  # what the nucleus hands the Run step
        monkeypatch.setenv(k, v)
    mounted = []
    real = Mesh.mount

    def spy(self, robot, **kw):
        mounted.append((kw.get("device_id"), kw.get("iot")))
        return real(self, robot, **kw)

    monkeypatch.setattr(Mesh, "mount", spy)
    stop = threading.Event()
    stop.set()  # "stay up" returns at once
    greengrass.main(stop=stop)
    assert mounted == [("arm-1", None)]  # no iot:// transport → no IotOptions
    assert Mesh(name, tenant="lab").discover() == []  # closed cleanly on the way out
