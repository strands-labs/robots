"""A real round trip over AWS IoT Core: ``python tests/mesh/aws_iot_round_trip.py [--region us-west-2]``.

Creates ONE thing, certificate and policy named ``strands-v1-local-mesh-<id>``, runs a simulated SO-101 as that
thing in a child process (mTLS, client id = thing name), drives it from here with IAM credentials (SigV4 WebSocket +
SendDirectMessage), then deletes all three. Prints every id it created and every id it deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEVICE = """
import sys, time
from strands_robots import Robot
from strands_robots.mesh import Mesh
url, tenant = sys.argv[1:3]
mesh = Mesh(url, tenant=tenant)
mesh.mount(Robot("so101"), device_id="arm", labels={"role": "arm"}, watchdog_ms=None)
print("ready", flush=True)
while True:
    time.sleep(1)
"""


def policy(region: str, account: str, thing: str, tenant: str) -> str:
    arn = f"arn:aws:iot:{region}:{account}"
    return json.dumps({"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "iot:Connect", "Resource": f"{arn}:client/{thing}"},
        {"Effect": "Allow", "Action": ["iot:Publish", "iot:Receive"], "Resource": f"{arn}:topic/strands/{tenant}/*"},
        {"Effect": "Allow", "Action": "iot:Subscribe", "Resource": f"{arn}:topicfilter/strands/{tenant}/*"},
    ]})  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    import boto3

    sys.path.insert(0, str(ROOT))
    from strands_robots.mesh import Mesh
    from strands_robots.mesh.transports import Offline

    iot = boto3.client("iot", region_name=a.region)
    account = boto3.client("sts").get_caller_identity()["Account"]
    endpoint = iot.describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"]
    thing = f"strands-v1-local-mesh-{uuid.uuid4().hex[:8]}"
    tenant = f"rt{uuid.uuid4().hex[:6]}"
    tmp = Path(tempfile.mkdtemp(prefix="strands-mesh-iot-"))
    cert_id = cert_arn = None
    proc = None
    try:
        iot.create_thing(thingName=thing)
        print("created thing", thing)
        iot.create_policy(policyName=thing, policyDocument=policy(a.region, account, thing, tenant))
        print("created policy", thing)
        cert = iot.create_keys_and_certificate(setAsActive=True)
        cert_id, cert_arn = cert["certificateId"], cert["certificateArn"]
        print("created certificate", cert_id)
        iot.attach_policy(policyName=thing, target=cert_arn)
        iot.attach_thing_principal(thingName=thing, principal=cert_arn)
        (tmp / "cert.pem").write_text(cert["certificatePem"])
        (tmp / "key.pem").write_text(cert["keyPair"]["PrivateKey"])
        q = f"endpoint={endpoint}&client={thing}&cert={tmp / 'cert.pem'}&key={tmp / 'key.pem'}"
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        time.sleep(8)  # IoT Core policy propagation
        proc = subprocess.Popen([sys.executable, "-c", DEVICE, f"iot://{a.region}?{q}", tenant],
                                stdout=subprocess.PIPE, text=True, env=env)  # fmt: skip
        assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
        with Mesh(f"iot://{a.region}?endpoint={endpoint}", tenant=tenant) as m:
            t0 = time.perf_counter()
            found = m.discover("device(role:arm)", min_count=1, timeout=15.0)
            print(f"discover over IoT: {[d.id for d in found]} in {time.perf_counter() - t0:.2f} s")
            ts = []
            for _ in range(a.n):
                t = time.perf_counter()
                m.invoke("arm", "act", action={"shoulder_pan": 0.2}, timeout=10.0)
                ts.append((time.perf_counter() - t) * 1e3)
            arr = np.array(ts)
            p50, p99 = np.percentile(arr, 50), np.percentile(arr, 99)
            print(f"invoke(act) over IoT Direct Messaging, n={a.n}: p50={p50:.1f} ms p99={p99:.1f} ms")
            print("broadcast estop acked:", m.broadcast("estop", deadline=10.0))
            proc.kill()
            proc.wait()
            time.sleep(2)
            try:
                m.invoke("arm", "estop", timeout=5.0)
                print("estop to the killed device: delivered?!")
            except Offline as exc:
                print("estop to the killed device:", exc)
        return 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
        if cert_arn:
            iot.detach_thing_principal(thingName=thing, principal=cert_arn)
            iot.detach_policy(policyName=thing, target=cert_arn)
            iot.update_certificate(certificateId=cert_id, newStatus="INACTIVE")
            iot.delete_certificate(certificateId=cert_id, forceDelete=True)
            print("deleted certificate", cert_id)
        for fn, kw, what in ((iot.delete_policy, {"policyName": thing}, "policy"),
                             (iot.delete_thing, {"thingName": thing}, "thing")):  # fmt: skip
            try:
                fn(**kw)
                print("deleted", what, thing)
            except Exception as exc:
                print("delete", what, "failed:", exc)


if __name__ == "__main__":
    raise SystemExit(main())
