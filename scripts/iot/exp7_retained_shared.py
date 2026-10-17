"""Experiment 7: the manifest as a RETAINED document, and shared subscriptions for N operators.

    python scripts/iot/exp7_retained_shared.py [--n 100]

(a) the device publishes its manifest once with RETAIN on `strands/<tenant>/<id>/@manifest`; an operator that connects LATER gets it on
    subscribe (ms after the subscribe call, bytes, retain flag); a second publish replaces it; an empty retained publish clears it.
(b) `$share/<group>/strands/<tenant>/<id>/@state`: two operators in one share group vs two plain subscribers; the device publishes n state
    messages at 10 Hz (QoS 1): how the shared copies split, that each message reaches exactly one member, and the per-message latency.
Everything deleted at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from _common import REGION, Mqtt5, Provisioned, account_and_endpoint, leftovers, name, pct, run_id, so101_manifest, sweep_deprecated_thing_types


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    import boto3

    iot = boto3.client("iot", region_name=REGION)
    run, dev, tenant = run_id(), so101_manifest(), "lab"
    account, endpoint = account_and_endpoint(iot)
    work = Path(tempfile.mkdtemp(prefix="strands-iot-exp7-"))
    p = Provisioned(iot, exp="e7", run=run, tenant=tenant, device_type=dev["device_type"], device_id=dev["id"], workdir=work)
    out: dict = {"experiment": 7, "run": run}
    try:
        p.create(account, [dev["id"]], dev)
        time.sleep(1.0)
        link = Mqtt5(endpoint, p.thing, cert=p.cert_path, key=p.key_path, timeout=10.0)
        mtopic = f"strands/{tenant}/{dev['id']}/@manifest"
        doc = json.dumps({"device_type": dev["device_type"], "labels": dev["labels"], "manifest": dev["manifest"]}).encode()
        r = link.publish(mtopic, doc, qos=1, retain=True)
        out["retained_manifest"] = {"bytes": len(doc), "puback": int(r.puback.reason_code) if r and r.puback else None}
        time.sleep(0.5)
        # a late operator
        got: list = []
        late = Mqtt5(endpoint, name("e7", run, "late"))
        late.on(lambda pkt: got.append((time.perf_counter(), bool(pkt.retain), len(bytes(pkt.payload or b"")))))
        t = time.perf_counter()
        codes, sub_ms = late.subscribe(mtopic)
        while not got and time.perf_counter() - t < 5:
            time.sleep(0.001)
        out["retained_manifest"].update({"late_operator_ms_after_subscribe_call": round((got[0][0] - t) * 1e3, 1) if got else None,
                                         "suback_ms": sub_ms, "retain_flag": got[0][1] if got else None, "bytes_received": got[0][2] if got else None})
        got.clear()
        doc2 = json.dumps({"device_type": dev["device_type"], "labels": {**dev["labels"], "rev": 2}, "manifest": dev["manifest"]}).encode()
        link.publish(mtopic, doc2, qos=1, retain=True)
        time.sleep(0.8)
        out["retained_manifest"]["replaced_live_update_received_by_late_operator"] = len(got)
        data = boto3.client("iot-data", region_name=REGION, endpoint_url=f"https://{endpoint}")
        rt = data.list_retained_messages()["retainedTopics"]
        out["retained_manifest"]["list_retained_messages"] = [{"topic": x["topic"], "bytes": x["payloadSize"]} for x in rt if x["topic"] == mtopic]
        got_r = data.get_retained_message(topic=mtopic)
        out["retained_manifest"]["https_get_retained_bytes"] = len(got_r["payload"])
        link.publish(mtopic, b"", qos=1, retain=True)
        time.sleep(0.5)
        out["retained_manifest"]["cleared"] = not any(x["topic"] == mtopic for x in data.list_retained_messages()["retainedTopics"])
        late.close()

        # (b) shared vs plain subscriptions
        stopic = f"strands/{tenant}/{dev['id']}/@state"
        members: dict[str, list] = {"shared-A": [], "shared-B": [], "plain-A": [], "plain-B": []}
        links = {}
        for mname in members:
            l = Mqtt5(endpoint, name("e7", run, mname))
            l.on(lambda pkt, m=mname: members[m].append((time.perf_counter(), json.loads(bytes(pkt.payload or b"{}")).get("seq"))))
            filt = f"$share/ops-{run}/{stopic}" if mname.startswith("shared") else stopic
            codes, _ = l.subscribe(filt, qos=1)
            assert codes[0] < 0x80, (filt, codes)
            links[mname] = l
        sent: dict[int, float] = {}
        for i in range(a.n):
            sent[i] = time.perf_counter()
            link.publish(stopic, json.dumps({"seq": i, "obs": {"shoulder_pan.pos": 0.001 * i}}).encode(), qos=1)
            time.sleep(0.1)
        time.sleep(2.0)
        sa, sb = members["shared-A"], members["shared-B"]
        seqs_shared = sorted(s for _, s in sa) + sorted(s for _, s in sb)
        out["shared_subscription"] = {
            "sent": a.n, "member_A": len(sa), "member_B": len(sb), "total_shared_copies": len(sa) + len(sb),
            "each_message_exactly_once": sorted(seqs_shared) == list(range(a.n)),
            "plain_A": len(members["plain-A"]), "plain_B": len(members["plain-B"]),
            "latency_ms_shared": pct([(t - sent[s]) * 1e3 for t, s in sa + sb if s in sent]),
            "latency_ms_plain": pct([(t - sent[s]) * 1e3 for t, s in members["plain-A"] if s in sent]),
        }
        # one member leaves: the other must get everything
        links["shared-A"].close()
        time.sleep(0.5)
        sb_before = len(sb)
        for i in range(a.n, a.n + 20):
            sent[i] = time.perf_counter()
            link.publish(stopic, json.dumps({"seq": i}).encode(), qos=1)
            time.sleep(0.05)
        time.sleep(1.5)
        out["shared_subscription"]["after_A_left_B_got"] = f"{len(sb) - sb_before}/20"
        for l in links.values():
            l.close()
        link.close()
    finally:
        out["deleted"] = p.delete()
        out["swept_thing_types"] = sweep_deprecated_thing_types(iot)
        out["leftovers"] = leftovers(iot)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
