"""Experiment 2: device shadows for one so101 device.

    python scripts/iot/exp2_shadow.py [--rates 10,30] [--seconds 10] [--n 100]

(a) ``manifest`` named shadow = the device's manifest document (does it fit the 8 KB limit; get latency as an operator)
(b) ``state`` named shadow, ``state.reported`` at each rate for N seconds over MQTT: accepted vs rejected, bytes/update,
    accept latency (publish → update/accepted) p50/p95
(c) a reported document over 8 KB → the rejection code
(d) operator sets ``state.desired`` (MQTT update) → device receives ``update/delta``: round trip p50/p95 ×N; the same via
    the HTTPS data-plane API (boto3 iot-data update_thing_shadow) ×N
Everything is deleted at the end; the leftovers list is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from _common import REGION, Mqtt5, Provisioned, account_and_endpoint, leftovers, pct, run_id, so101_manifest, sweep_deprecated_thing_types


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="10,30")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    import boto3

    iot = boto3.client("iot", region_name=REGION)
    run, dev = run_id(), so101_manifest()
    account, endpoint = account_and_endpoint(iot)
    work = Path(tempfile.mkdtemp(prefix="strands-iot-exp2-"))
    p = Provisioned(iot, exp="e2", run=run, tenant="lab", device_type=dev["device_type"], device_id=dev["id"], workdir=work)
    out: dict = {"experiment": 2, "run": run}
    try:
        p.create(account, [dev["id"]], dev)
        time.sleep(1.0)
        link = Mqtt5(endpoint, p.thing, cert=p.cert_path, key=p.key_path, timeout=10.0)
        base = f"$aws/things/{p.thing}/shadow/name"
        acc, rej, delta, got = [], [], [], []
        lock = threading.Lock()

        def on(pkt):
            t = time.perf_counter()
            topic = str(pkt.topic)
            body = json.loads(bytes(pkt.payload or b"{}") or b"{}")
            with lock:
                if topic.endswith("/update/accepted"):
                    acc.append((t, body))
                elif topic.endswith("/update/rejected"):
                    rej.append((t, body))
                elif topic.endswith("/update/delta"):
                    delta.append((t, body))
                elif topic.endswith("/get/accepted"):
                    got.append((t, body))

        link.on(on)
        for sfx in ("update/accepted", "update/rejected", "update/delta", "get/accepted"):
            for shadow in ("manifest", "state"):
                codes, _ = link.subscribe(f"{base}/{shadow}/{sfx}")
                assert codes and codes[0] < 0x80, (shadow, sfx, codes)

        # (a) the manifest as a named shadow
        doc = json.dumps({"state": {"reported": {"manifest": dev["manifest"], "device_type": dev["device_type"], "labels": dev["labels"]}}}).encode()
        t = time.perf_counter()
        link.publish(f"{base}/manifest/update", doc, qos=1)
        for _ in range(100):
            if acc or rej:
                break
            time.sleep(0.02)
        out["manifest_shadow"] = {"bytes": len(doc), "accepted": len(acc), "rejected": [r[1].get("code") for r in rej],
                                  "accept_ms": round((acc[0][0] - t) * 1e3, 1) if acc else None}
        acc.clear(); rej.clear()
        data = boto3.client("iot-data", region_name=REGION, endpoint_url=f"https://{endpoint}")
        gets = []
        for _ in range(20):
            t = time.perf_counter()
            r = data.get_thing_shadow(thingName=p.thing, shadowName="manifest")
            gets.append((time.perf_counter() - t) * 1e3)
        body = json.loads(r["payload"].read())
        out["manifest_shadow"]["operator_get_ms"] = pct(gets)
        out["manifest_shadow"]["get_bytes"] = len(json.dumps(body))
        out["manifest_shadow"]["controls_seen_by_operator"] = len(body["state"]["reported"]["manifest"]["controls"])

        # (b) reported state at each rate
        joints = [c["name"].split(".")[0] for c in dev["manifest"]["controls"]]
        def reported(i: int) -> bytes:
            st = {f"{j}.{k}": round(0.001 * i * (n + 1), 4) for n, j in enumerate(joints) for k in ("pos", "vel", "effort")}
            return json.dumps({"state": {"reported": {"obs": st, "seq": i, "t": time.time()}}, "clientToken": f"r{i}"}).encode()

        out["reported"] = {}
        for rate in [float(x) for x in a.rates.split(",")]:
            acc.clear(); rej.clear()
            sent, period, lat = [], 1.0 / rate, {}
            t_end = time.perf_counter() + a.seconds
            i = 0
            while time.perf_counter() < t_end:
                t = time.perf_counter()
                body = reported(i)
                lat[f"r{i}"] = t
                link.publish(f"{base}/state/update", body, qos=0)
                sent.append(len(body))
                i += 1
                nxt = t + period
                while time.perf_counter() < nxt:
                    time.sleep(0.0005)
            time.sleep(2.0)
            with lock:
                lats = [(ta - lat[b["clientToken"]]) * 1e3 for ta, b in acc if b.get("clientToken") in lat]
                codes = [r[1].get("code") for r in rej]
            out["reported"][f"{rate:g}hz"] = {"sent": len(sent), "bytes_per_update": sent[0], "accepted": len(acc), "rejected": len(rej),
                                               "reject_codes": sorted(set(codes)), "reject_msgs": sorted({r[1].get("message", "")[:60] for r in rej}),
                                               "achieved_hz": round(len(sent) / a.seconds, 1), "accept_latency_ms": pct(lats)}
        # (c) too big
        acc.clear(); rej.clear()
        big = json.dumps({"state": {"reported": {"blob": "x" * 9000}}}).encode()
        link.publish(f"{base}/state/update", big, qos=1)
        time.sleep(1.5)
        out["over_8kb"] = {"bytes": len(big), "accepted": len(acc), "rejected": [(r[1].get("code"), r[1].get("message", "")[:80]) for r in rej]}

        # (d) desired → delta round trip, operator over MQTT (SigV4) and over HTTPS
        op = Mqtt5(endpoint, f"strands-v1-iot-exp-e2-{run}-operator")
        acc.clear(); rej.clear(); delta.clear()
        rtt_mqtt = []
        for i in range(a.n):
            delta.clear()
            t = time.perf_counter()
            op.publish(f"{base}/state/update", json.dumps({"state": {"desired": {"target": {"shoulder_pan.pos": 0.001 * i}}}}).encode(), qos=0)
            while not delta and time.perf_counter() - t < 5:
                time.sleep(0.0005)
            if delta:
                rtt_mqtt.append((delta[0][0] - t) * 1e3)
        out["desired_to_delta_mqtt_ms"] = pct(rtt_mqtt)
        rtt_http, api_ms = [], []
        for i in range(a.n):
            delta.clear()
            t = time.perf_counter()
            data.update_thing_shadow(thingName=p.thing, shadowName="state", payload=json.dumps({"state": {"desired": {"target": {"shoulder_pan.pos": 0.5 + 0.001 * i}}}}).encode())
            api_ms.append((time.perf_counter() - t) * 1e3)
            while not delta and time.perf_counter() - t < 5:
                time.sleep(0.0005)
            if delta:
                rtt_http.append((delta[0][0] - t) * 1e3)
        out["desired_to_delta_https_ms"] = pct(rtt_http)
        out["https_update_call_ms"] = pct(api_ms)
        out["delta_payload_example_bytes"] = len(json.dumps(delta[0][1])) if delta else None
        # the device acknowledges by reporting the same value → delta clears
        link.publish(f"{base}/state/update", json.dumps({"state": {"reported": {"target": {"shoulder_pan.pos": 0.5 + 0.001 * (a.n - 1)}}}}).encode(), qos=1)
        time.sleep(1.0)
        r = json.loads(data.get_thing_shadow(thingName=p.thing, shadowName="state")["payload"].read())
        out["delta_cleared_after_report"] = "delta" not in r.get("state", {})
        out["state_shadow_version"] = r.get("version")
        op.close(); link.close()
        for s in ("manifest", "state"):
            data.delete_thing_shadow(thingName=p.thing, shadowName=s)
        out["shadows_deleted"] = 2
    finally:
        out["deleted"] = p.delete()
        out["swept_thing_types"] = sweep_deprecated_thing_types(iot)
        out["leftovers"] = leftovers(iot)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
