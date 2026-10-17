"""ONE real run against AWS IoT Core, only with ``STRANDS_IOT_LIVE=1``: provision a thing from the record with the
fleet policy, mount a simulated SO-101 over ``iot://`` with shadows on, read the manifest shadow, push a desired
target and watch the arm move through ``confirm_act``, then delete everything it created (and prove it).

Resources: ``strands-v1-iot-implement-<run>`` (thing, policy, thing group ``strands-<tenant>``), tag
``Project=strands-v1-iot``. Run: ``STRANDS_IOT_LIVE=1 pytest tests/mesh/iot/test_live.py -s``.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STRANDS_IOT_LIVE") != "1", reason="set STRANDS_IOT_LIVE=1 for the live run"
)

REGION = os.environ.get("STRANDS_IOT_REGION", "us-west-2")
TAGS = {"Project": "strands-v1-iot"}


def _wait(pred, timeout, what):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def test_live_provision_mount_shadow_round_trip_and_cleanup(tmp_path):
    import boto3

    from strands_robots import Robot, Simulation
    from strands_robots.mesh import Mesh
    from strands_robots.mesh.iot import IotOptions, provision
    from strands_robots.mesh.record import record
    from strands_robots.mesh.serve import ARGS

    run = uuid.uuid4().hex[:6]
    tenant, did = f"implement-{run}", f"strands-v1-iot-implement-{run}"
    policy = f"strands-v1-iot-implement-{run}-policy"
    iot = boto3.client("iot", region_name=REGION)
    data = boto3.client("iot-data", region_name=REGION)
    account = boto3.client("sts").get_caller_identity()["Account"]
    endpoint = iot.describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"]
    _sweep_types(iot)
    arm = Robot("so101", sim=Simulation())
    rec = record(arm, did, {"role": "arm"}, tenant, tuple(ARGS))
    rec["device_type"] = f"{did}-type"  # run-scoped: a thing type can only be deprecated, then deleted 5 min later
    device = controller = None
    t: dict[str, float] = {}
    try:
        t0 = time.perf_counter()
        provision.setup_tenant(iot, tenant, region=REGION, account=account, policy=policy, tags=TAGS)
        out = provision.register(iot, rec, tmp_path, policy=policy)
        t["provision_s"] = round(time.perf_counter() - t0, 2)
        print(f"\ncreated thing {did}, policy {policy}, cert {out['certificate_id']} in {t['provision_s']} s")
        time.sleep(2.0)  # policy propagation (measured 0.57 s)

        t0 = time.perf_counter()
        device = Mesh(f"iot://{REGION}?endpoint={endpoint}&{out['query']}", tenant=tenant)
        device.mount(arm, device_id=did, watchdog_ms=None, iot=IotOptions(shadow_hz=1.0))
        t["mount_s"] = round(time.perf_counter() - t0, 2)

        def manifest_shadow():
            try:
                return json.loads(data.get_thing_shadow(thingName=did, shadowName="manifest")["payload"].read())
            except data.exceptions.ResourceNotFoundException:
                return None

        _wait(lambda: manifest_shadow() is not None, 10, "the manifest shadow")
        doc = manifest_shadow()
        assert doc["state"]["reported"]["manifest"]["device_id"] == did
        t["manifest_bytes"] = len(json.dumps(doc["state"]["reported"]))

        def state_shadow():
            try:
                return json.loads(data.get_thing_shadow(thingName=did, shadowName="state")["payload"].read())
            except data.exceptions.ResourceNotFoundException:
                return None

        _wait(lambda: state_shadow() is not None, 10, "the state shadow")
        v0 = state_shadow()["version"]
        time.sleep(3.0)
        t["state_updates_in_3s"] = state_shadow()["version"] - v0
        assert 2 <= t["state_updates_in_3s"] <= 4, t  # 1 Hz, bounded

        controller = Mesh(f"iot://{REGION}?endpoint={endpoint}", tenant=tenant)
        assert controller.discover(did, min_count=1, timeout=10.0)
        before = float(arm.observe(images=False).joints.positions[0])
        t0 = time.perf_counter()
        target = json.dumps({"state": {"desired": {"shoulder_pan.pos": 0.3}}}).encode()
        data.update_thing_shadow(thingName=did, shadowName="control", payload=target)

        def moved():
            return abs(float(arm.observe(images=False).joints.positions[0]) - before) > 1e-3

        _wait(moved, 10, "the arm to move")
        t["desired_to_act_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        def control_reported():
            d = json.loads(data.get_thing_shadow(thingName=did, shadowName="control")["payload"].read())
            return d["state"].get("reported") or {}

        _wait(lambda: control_reported().get("shoulder_pan.pos") == 0.3, 10, "the reported echo")
        bad = json.dumps({"state": {"desired": {"elbow_flex.pos": 3.1, "estop": True}}}).encode()
        data.update_thing_shadow(thingName=did, shadowName="control", payload=bad)
        _wait(lambda: (control_reported().get("refused") or {}).get("estop"), 10, "the refusal")
        refused = control_reported()["refused"]
        assert refused["estop"] == "estop is an rpc, not a shadow field"
        assert "outside its range" in refused["elbow_flex.pos"]
        assert not arm.estopped
        d = json.loads(data.get_thing_shadow(thingName=did, shadowName="control")["payload"].read())
        assert "estop" not in (d["state"].get("desired") or {}), "refused targets are cleared from desired"
        assert np.isfinite(before)

        # the same rpc two ways: operator Direct Message (IAM) vs a certificate-only peer on MQTT request/response
        t0 = time.perf_counter()
        for _ in range(10):
            controller.invoke(did, "describe")
        t["dm_describe_ms_avg10"] = round((time.perf_counter() - t0) * 100, 1)
        prec = {
            **rec,
            "id": f"{did}-peer",
            "labels": {"role": "peer"},
        }  # its own thing: the policy pins client == thing
        pout = provision.register(iot, prec, tmp_path, policy=policy, controller=True)
        time.sleep(1.5)
        peer = Mesh(f"iot://{REGION}?endpoint={endpoint}&{pout['query']}", tenant=tenant)
        assert peer.transports[0].rpc == "mqtt"
        assert peer.discover(did, min_count=1, timeout=10.0)
        t0 = time.perf_counter()
        for _ in range(10):
            peer.invoke(did, "describe")
        t["rr_describe_ms_avg10"] = round((time.perf_counter() - t0) * 100, 1)
        peer.close()

        device.unmount(did)
        _wait(lambda: manifest_shadow() is None, 10, "the manifest shadow to be deleted at unmount")
        print("numbers:", json.dumps(t))
    finally:
        for m in (controller, device):
            if m is not None:
                m.close()
        arm.sim.close()
        steps = [
            lambda: [_quiet(data.delete_thing_shadow, thingName=did, shadowName=n) for n in ("state", "control")],
            lambda: provision.teardown(iot, did),
            lambda: provision.teardown(iot, f"{did}-peer"),
            lambda: provision.teardown_tenant(iot, tenant, policy=policy),
            lambda: iot.deprecate_thing_type(thingTypeName=rec["device_type"]),
        ]
        for step in steps:  # every step runs even when an earlier one fails
            try:
                step()
            except Exception as exc:  # noqa: BLE001
                print("cleanup step failed:", type(exc).__name__, str(exc)[:300])
        left = [x["thingName"] for x in iot.list_things().get("things", []) if x["thingName"].startswith(did)]
        pols = [p["policyName"] for p in iot.list_policies().get("policies", []) if p["policyName"] == policy]
        grp = iot.list_thing_groups(namePrefixFilter=f"strands-{tenant}").get("thingGroups", [])
        groups = [g["groupName"] for g in grp]
        print(
            "after cleanup: things", left, "policies", pols, "groups", groups, "(thing type deprecated, swept next run)"
        )
        assert left == [] and pols == [] and groups == []


def _quiet(call, **kw):
    try:
        call(**kw)
    except Exception:  # noqa: BLE001 — a shadow that was never written
        pass


def _sweep_types(iot) -> None:
    """Delete our deprecated thing types once AWS allows it (5 min after deprecation)."""
    import datetime as dt

    for tt in iot.list_thing_types().get("thingTypes", []):
        name, meta = tt["thingTypeName"], tt.get("thingTypeMetadata", {})
        if name.startswith("strands-v1-iot-implement-") and meta.get("deprecated"):
            age = dt.datetime.now(dt.timezone.utc) - meta["deprecationDate"]
            if age.total_seconds() > 300:
                iot.delete_thing_type(thingTypeName=name)
                print("swept thing type", name)
