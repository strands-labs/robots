"""Shared bits for the IoT experiments: naming, tagging, stats, the manifest, cleanup, MQTT5 links.

Every resource is named ``strands-v1-iot-exp-<exp>-<run>`` and tagged ``Project=strands-v1-iot`` so anyone can
list leftovers with one call. Nothing here prints an ARN, an account id or key material.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REGION = os.environ.get("STRANDS_IOT_REGION", "us-west-2")
PREFIX = "strands-v1-iot-exp"
TAGS = [{"Key": "Project", "Value": "strands-v1-iot"}]
TAG_MAP = {"Project": "strands-v1-iot"}


def run_id() -> str:
    return uuid.uuid4().hex[:6]


def name(exp: str, run: str, suffix: str = "") -> str:
    return f"{PREFIX}-{exp}-{run}" + (f"-{suffix}" if suffix else "")


def pct(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    return {
        "n": len(s),
        "p50": round(q(0.50), 2),
        "p95": round(q(0.95), 2),
        "p99": round(q(0.99), 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "mean": round(statistics.fmean(s), 2),
    }


def so101_manifest() -> dict[str, Any]:
    """The SO-101 manifest exactly as the reference's Mesh publishes it (sim, no hardware)."""
    from strands_robots import Robot
    from strands_robots.mesh import Mesh

    with Mesh(f"mem://exp-{uuid.uuid4().hex[:6]}", tenant="exp") as m:
        m.mount(Robot("so101"), device_id="arm", watchdog_ms=None)
        d = m.discover("arm", min_count=1, timeout=3.0)[0]
        return {"id": d.id, "device_type": d.device_type, "labels": dict(d.labels), "manifest": d.manifest}


def account_and_endpoint(iot) -> tuple[str, str]:
    import boto3

    acct = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    return acct, iot.describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"]


def device_policy(region: str, account: str, tenant: str, device_ids: list[str]) -> dict[str, Any]:
    """Least privilege for ONE device certificate, by policy variable: connect only as its own thing name, beat
    presence only for the device ids it serves, talk only inside its tenant, read its own shadows only."""
    arn = f"arn:aws:iot:{region}:{account}"
    thing = "${iot:Connection.Thing.ThingName}"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": "iot:Connect", "Resource": f"{arn}:client/{thing}",
             "Condition": {"Bool": {"iot:Connection.Thing.IsAttached": "true"}}},
            {"Effect": "Allow", "Action": "iot:Publish",
             "Resource": [f"{arn}:topic/strands/{tenant}/@presence/{d}" for d in device_ids]},
            {"Effect": "Allow", "Action": "iot:Publish",
             "Resource": [f"{arn}:topic/strands/{tenant}/{d}/*" for d in device_ids]
             + [f"{arn}:topic/strands/{tenant}/@reply/*", f"{arn}:topic/strands/{tenant}/@estop",
                f"{arn}:topic/$aws/things/{thing}/shadow/*"]},
            {"Effect": "Allow", "Action": "iot:RetainPublish",  # retained messages need this on top of iot:Publish
             "Resource": [f"{arn}:topic/strands/{tenant}/@estop"] + [f"{arn}:topic/strands/{tenant}/{d}/@manifest" for d in device_ids]},
            {"Effect": "Allow", "Action": "iot:Receive", "Resource": [f"{arn}:topic/strands/{tenant}/*",
                                                                     f"{arn}:topic/$aws/things/{thing}/shadow/*"]},
            {"Effect": "Allow", "Action": "iot:Subscribe",
             "Resource": [f"{arn}:topicfilter/strands/{tenant}/*", f"{arn}:topicfilter/$aws/things/{thing}/shadow/*"]},
        ],
    }  # fmt: skip


class Provisioned:
    """thing type + thing group + thing + certificate + policy for one device; ``delete()`` removes all of it."""

    def __init__(self, iot, *, exp: str, run: str, tenant: str, device_type: str, device_id: str, workdir: Path):
        self.iot, self.exp, self.run, self.tenant, self.workdir = iot, exp, run, tenant, workdir
        self.thing = name(exp, run, device_id)
        self.thing_type = name(exp, run, f"type-{device_type}")
        self.group = name(exp, run, f"tenant-{tenant}")
        self.policy = name(exp, run, f"policy-{device_id}")
        self.cert_id = self.cert_arn = None
        self.timings: dict[str, float] = {}
        self.cert_path = workdir / f"{self.thing}.cert.pem"
        self.key_path = workdir / f"{self.thing}.key.pem"

    def _t(self, k: str, fn, **kw):
        t = time.perf_counter()
        out = fn(**kw)
        self.timings[k] = round((time.perf_counter() - t) * 1e3, 1)
        return out

    def create(self, account: str, device_ids: list[str], manifest: dict[str, Any]) -> "Provisioned":
        iot = self.iot
        self._t("create_thing_type", iot.create_thing_type, thingTypeName=self.thing_type, tags=TAGS,
                thingTypeProperties={"thingTypeDescription": "strands-robots v1 device type (experiment)"})
        self._t("create_thing_group", iot.create_thing_group, thingGroupName=self.group, tags=TAGS,
                thingGroupProperties={"thingGroupDescription": f"tenant {self.tenant} (experiment)"})
        attrs = {"tenant": self.tenant, "device_type": manifest["device_type"], "n_controls": str(len(manifest["manifest"].get("controls", [])))}
        self._t("create_thing", iot.create_thing, thingName=self.thing, thingTypeName=self.thing_type,
                attributePayload={"attributes": attrs})
        self._t("add_thing_to_group", iot.add_thing_to_thing_group, thingName=self.thing, thingGroupName=self.group)
        self._t("create_policy", iot.create_policy, policyName=self.policy, tags=TAGS,
                policyDocument=json.dumps(device_policy(REGION, account, self.tenant, device_ids)))
        cert = self._t("create_keys_and_certificate", iot.create_keys_and_certificate, setAsActive=True)
        self.cert_id, self.cert_arn = cert["certificateId"], cert["certificateArn"]
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cert_path.write_text(cert["certificatePem"])
        self.key_path.write_text(cert["keyPair"]["PrivateKey"])
        os.chmod(self.key_path, 0o600)
        self._t("attach_policy", iot.attach_policy, policyName=self.policy, target=self.cert_arn)
        self._t("attach_thing_principal", iot.attach_thing_principal, thingName=self.thing, principal=self.cert_arn)
        return self

    def delete(self) -> list[str]:
        iot, done = self.iot, []
        if self.cert_arn:
            for fn, kw in (
                (iot.detach_thing_principal, {"thingName": self.thing, "principal": self.cert_arn}),
                (iot.detach_policy, {"policyName": self.policy, "target": self.cert_arn}),
                (iot.update_certificate, {"certificateId": self.cert_id, "newStatus": "INACTIVE"}),
                (iot.delete_certificate, {"certificateId": self.cert_id, "forceDelete": True}),
            ):
                try:
                    fn(**kw)
                except Exception as exc:  # keep going; the leftovers scan is the judge
                    done.append(f"FAILED {fn.__name__}: {type(exc).__name__}")
            done.append("certificate")
        for fn, kw, what in (
            (iot.delete_policy, {"policyName": self.policy}, "policy"),
            (iot.remove_thing_from_thing_group, {"thingName": self.thing, "thingGroupName": self.group}, "group member"),
            (iot.delete_thing, {"thingName": self.thing}, "thing"),
            (iot.delete_thing_group, {"thingGroupName": self.group}, "thing group"),
            (iot.deprecate_thing_type, {"thingTypeName": self.thing_type}, "thing type (deprecated)"),
        ):
            try:
                fn(**kw)
                done.append(what)
            except Exception as exc:
                done.append(f"FAILED {what}: {type(exc).__name__}")
        for p in (self.cert_path, self.key_path):
            p.unlink(missing_ok=True)
        return done


def leftovers(iot) -> dict[str, list[str]]:
    """Everything of ours still alive, by type: the proof of cleanup is this returning only empty lists
    (deprecated thing types cannot be deleted for 5 minutes; they are listed under 'thing_types_deprecated')."""
    out: dict[str, list[str]] = {"things": [], "policies": [], "thing_groups": [], "thing_types": [],
                                 "thing_types_deprecated": [], "topic_rules": [], "templates": [], "certificates_active": []}
    for t in iot.get_paginator("list_things").paginate():
        out["things"] += [x["thingName"] for x in t["things"] if x["thingName"].startswith(PREFIX)]
    for p in iot.get_paginator("list_policies").paginate():
        out["policies"] += [x["policyName"] for x in p["policies"] if x["policyName"].startswith(PREFIX)]
    for g in iot.get_paginator("list_thing_groups").paginate():
        out["thing_groups"] += [x["groupName"] for x in g["thingGroups"] if x["groupName"].startswith(PREFIX)]
    for tt in iot.get_paginator("list_thing_types").paginate():
        for x in tt["thingTypes"]:
            if x["thingTypeName"].startswith(PREFIX):
                key = "thing_types_deprecated" if x.get("thingTypeMetadata", {}).get("deprecated") else "thing_types"
                out[key].append(x["thingTypeName"])
    out["topic_rules"] = [r["ruleName"] for r in iot.list_topic_rules()["rules"] if r["ruleName"].startswith(PREFIX.replace("-", "_"))]
    out["templates"] = [t["templateName"] for t in iot.list_provisioning_templates()["templates"] if t["templateName"].startswith(PREFIX)]
    for c in iot.get_paginator("list_certificates").paginate():
        for x in c["certificates"]:
            if x["status"] == "ACTIVE":
                try:
                    pr = iot.list_principal_things(principal=x["certificateArn"])["things"]
                except Exception:
                    pr = []
                if any(n.startswith(PREFIX) for n in pr):
                    out["certificates_active"].append(x["certificateId"][:12] + "…")
    return out


def sweep_deprecated_thing_types(iot) -> list[str]:
    """Delete deprecated experiment thing types that are past AWS's 5-minute grace; returns what was deleted."""
    gone = []
    for tt in iot.get_paginator("list_thing_types").paginate():
        for x in tt["thingTypes"]:
            md = x.get("thingTypeMetadata", {})
            if x["thingTypeName"].startswith(PREFIX) and md.get("deprecated"):
                try:
                    iot.delete_thing_type(thingTypeName=x["thingTypeName"])
                    gone.append(x["thingTypeName"])
                except Exception:  # best-effort cleanup: a type still in use is reported by the next sweep
                    pass
    return gone


class Mqtt5:
    """One MQTT5 connection (mTLS as a thing, or SigV4 WebSocket as an operator) with timing and an inbox."""

    def __init__(self, endpoint: str, client_id: str, *, cert: Path | None = None, key: Path | None = None,
                 clean_session: bool = True, timeout: float = 15.0, session_expiry: int = 0, keep_alive: int = 30):
        from awscrt import auth, mqtt5
        from awsiot import mqtt5_client_builder

        self.m, self.client_id = mqtt5, client_id
        self.up, self.down, self.why = threading.Event(), threading.Event(), ""
        self.handlers: list = []
        self.disconnects: list[tuple[float, str]] = []
        kw: dict[str, Any] = dict(
            endpoint=endpoint, client_id=client_id, clean_session=clean_session, keep_alive_secs=keep_alive,
            on_publish_received=self._on_pub,
            on_lifecycle_connection_success=lambda *_: self.up.set(),
            on_lifecycle_connection_failure=lambda d: setattr(self, "why", str(getattr(d, "exception", d))),
            on_lifecycle_disconnection=lambda d: self.disconnects.append((time.perf_counter(), str(getattr(getattr(d, "disconnect_packet", None), "reason_code", getattr(d, "exception", ""))))),
            on_lifecycle_stopped=lambda *_: self.down.set(),
        )
        if session_expiry:
            kw["session_expiry_interval_sec"] = session_expiry
        t = time.perf_counter()
        if cert and key:
            self.client = mqtt5_client_builder.mtls_from_path(cert_filepath=str(cert), pri_key_filepath=str(key), **kw)
            self.kind = "mTLS"
        else:
            creds = auth.AwsCredentialsProvider.new_default_chain()
            self.client = mqtt5_client_builder.websockets_with_default_aws_signing(region=REGION, credentials_provider=creds, **kw)
            self.kind = "SigV4-WS"
        self.client.start()
        if not self.up.wait(timeout):
            self.client.stop()
            raise TimeoutError(f"{client_id} ({self.kind}) did not connect in {timeout}s: {self.why}")
        self.connect_ms = round((time.perf_counter() - t) * 1e3, 1)

    def _on_pub(self, d) -> None:
        pkt = d.publish_packet
        for h in list(self.handlers):
            try:
                h(pkt)
            except Exception:  # one handler's error must not stop the others from seeing the packet
                pass

    def on(self, handler) -> None:
        self.handlers.append(handler)

    def subscribe(self, topic_filter: str, qos: int = 1) -> tuple[list[int], float]:
        m = self.m
        t = time.perf_counter()
        r = self.client.subscribe(m.SubscribePacket([m.Subscription(topic_filter=topic_filter, qos=m.QoS(qos))])).result(10.0)
        return [int(c) for c in r.reason_codes or []], round((time.perf_counter() - t) * 1e3, 1)

    def publish(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False, **kw) -> Any:
        m = self.m
        pkt = m.PublishPacket(topic=topic, payload=payload, qos=m.QoS(qos), retain=retain, **kw)
        fut = self.client.publish(pkt)
        return fut.result(10.0) if qos else None

    def close(self) -> None:
        try:
            self.client.stop()
            self.down.wait(5.0)
        except Exception:  # best-effort teardown: nobody left to report to
            pass
