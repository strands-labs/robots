"""Experiment 1: thing type + thing group + thing + certificate + least-privilege policy FROM the so101 manifest,
then connect as the device (MQTT5 mTLS) and as an operator (SigV4 WebSocket); prove the policy is tight.

    python scripts/iot/exp1_provision.py [--keep] [--cleanup-only]

Measures: each control-plane call (ms), policy propagation (first successful mTLS connect after attach), connect
time for both auth kinds, SUBACK/PUBACK reason codes for allowed vs forbidden topics. Deletes everything unless
--keep. Prints a JSON summary; nothing in it is an ARN, account id or key material.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from _common import REGION, Mqtt5, Provisioned, account_and_endpoint, leftovers, run_id, so101_manifest, sweep_deprecated_thing_types


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="leave the resources alive (another experiment uses them)")
    ap.add_argument("--cleanup-only", action="store_true", help="only sweep deprecated thing types and list leftovers")
    ap.add_argument("--tenant", default="lab")
    a = ap.parse_args()
    import boto3

    iot = boto3.client("iot", region_name=REGION)
    if a.cleanup_only:
        print(json.dumps({"swept_thing_types": sweep_deprecated_thing_types(iot), "leftovers": leftovers(iot)}, indent=1))
        return 0

    run = run_id()
    dev = so101_manifest()
    account, endpoint = account_and_endpoint(iot)
    work = Path(tempfile.mkdtemp(prefix="strands-iot-exp1-"))
    p = Provisioned(iot, exp="e1", run=run, tenant=a.tenant, device_type=dev["device_type"], device_id=dev["id"], workdir=work)
    out: dict = {"experiment": 1, "run": run, "region": REGION, "manifest": {
        "device_type": dev["device_type"], "sensors": len(dev["manifest"].get("sensors", [])),
        "controls": len(dev["manifest"].get("controls", [])), "rpcs": len(dev["manifest"].get("rpcs", [])),
        "bytes": len(json.dumps(dev["manifest"]).encode())}}
    try:
        t0 = time.perf_counter()
        p.create(account, [dev["id"]], dev)
        out["provision_ms"] = {**p.timings, "total": round((time.perf_counter() - t0) * 1e3, 1)}
        out["policy_statements"] = 5

        # policy propagation: retry the mTLS connect until IoT Core accepts the new cert+policy
        t_attach = time.perf_counter()
        last = ""
        for attempt in range(40):
            try:
                dev_link = Mqtt5(endpoint, p.thing, cert=p.cert_path, key=p.key_path, timeout=6.0)
                break
            except TimeoutError as exc:
                last = str(exc)[-80:]
                time.sleep(0.5)
        else:
            raise SystemExit(f"device never connected: {last}")
        out["propagation_s"] = round(time.perf_counter() - t_attach, 2)
        out["connect_attempts"] = attempt + 1
        out["connect_ms"] = {"mTLS_device": dev_link.connect_ms}

        # as the device: allowed vs forbidden subscribe / publish (MQTT5 reason codes: 0=granted, 135=not authorized)
        own, other = f"strands/{a.tenant}", "strands/other-tenant"
        codes, ms = dev_link.subscribe(f"{own}/{dev['id']}/+")
        out["device_sub_own_tenant"] = {"reason_codes": codes, "ms": ms}
        codes, ms = dev_link.subscribe(f"{other}/#")
        out["device_sub_other_tenant"] = {"reason_codes": codes, "ms": ms}
        r = dev_link.publish(f"{own}/@presence/{dev['id']}", b'{"alive":true}', qos=1)
        out["device_pub_own_presence"] = {"puback": int(r.puback.reason_code) if hasattr(r, "puback") and r.puback else 0}
        try:
            r = dev_link.publish(f"{own}/@presence/someone-else", b'{"alive":true}', qos=1)
            out["device_pub_other_presence"] = {"puback": int(r.puback.reason_code) if hasattr(r, "puback") and r.puback else 0}
        except Exception as exc:
            out["device_pub_other_presence"] = {"error": type(exc).__name__}
        time.sleep(1.0)
        out["device_disconnects_after_forbidden_pub"] = len(dev_link.disconnects)

        # as an operator: SigV4 WebSocket with the role in the env
        op = Mqtt5(endpoint, f"strands-v1-iot-exp-e1-{run}-operator")
        out["connect_ms"]["SigV4_ws_operator"] = op.connect_ms
        codes, ms = op.subscribe(f"{own}/#")
        out["operator_sub_tenant"] = {"reason_codes": codes, "ms": ms}

        # connect-twice-as-the-same-thing: the second connection must evict the first (IoT Core rule, worth knowing)
        dev2 = Mqtt5(endpoint, p.thing, cert=p.cert_path, key=p.key_path, timeout=10.0)
        time.sleep(1.5)
        out["same_client_id_twice"] = {"first_link_disconnects": len(dev_link.disconnects), "second_connect_ms": dev2.connect_ms,
                                       "reason": dev_link.disconnects[-1][1] if dev_link.disconnects else None}
        dev2.close()
        dev_link.close()
        op.close()
        out["thing"] = p.thing
        out["thing_type"] = p.thing_type
        out["thing_group"] = p.group
        if a.keep:
            out["kept"] = True
            out["cert_dir"] = str(work)
    finally:
        if not a.keep:
            t = time.perf_counter()
            out["deleted"] = p.delete()
            out["delete_ms"] = round((time.perf_counter() - t) * 1e3, 1)
            out["swept_thing_types"] = sweep_deprecated_thing_types(iot)
            out["leftovers"] = leftovers(iot)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
