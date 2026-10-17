"""Experiment 4: conversations — MQTT5 request/response vs AWS IoT Direct Messaging on the v1 link.

    python scripts/iot/exp4_request_response.py [--n 200] [--no-ref]

The device (mTLS, client id = thing) serves `strands/<tenant>/<id>/@rpc/+`: every request is answered on the request's
MQTT5 ``response_topic`` with its ``correlation_data`` (the broker carries both). The operator (SigV4 WS) measures:
(a) MQTT5 request/response RTT, QoS 0 and QoS 1, ×n
(b) Direct Message RTT: boto3 iot-data ``send_direct_message(confirmation=True)`` to the device's client id, reply over MQTT, ×n;
    the HTTPS call alone; ``confirmation=False`` variant
(c) offline: the device link is closed → ``send_direct_message`` → 404, how long it takes, ×20; and what an MQTT5 request gets (nothing)
(d) the reference's own round trip, ``tests/mesh/aws_iot_round_trip.py --n <n>`` (Mesh.discover + Mesh.invoke(act) over the v1 IotTransport)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from _common import REGION, ROOT, Mqtt5, Provisioned, account_and_endpoint, leftovers, name, pct, run_id, so101_manifest, sweep_deprecated_thing_types


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-ref", action="store_true")
    a = ap.parse_args()
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    iot = boto3.client("iot", region_name=REGION)
    run, dev, tenant = run_id(), so101_manifest(), "lab"
    account, endpoint = account_and_endpoint(iot)
    work = Path(tempfile.mkdtemp(prefix="strands-iot-exp4-"))
    p = Provisioned(iot, exp="e4", run=run, tenant=tenant, device_type=dev["device_type"], device_id=dev["id"], workdir=work)
    out: dict = {"experiment": 4, "run": run, "n": a.n}
    try:
        p.create(account, [dev["id"]], dev)
        time.sleep(1.0)
        link = Mqtt5(endpoint, p.thing, cert=p.cert_path, key=p.key_path, timeout=10.0)
        m = link.m
        served = {"n": 0}

        def serve(pkt) -> None:  # the device side: answer where the request says, echo the correlation data
            topic = str(pkt.topic)
            if __import__("os").environ.get("EXP_DEBUG"):
                print("DEV got", topic, pkt.response_topic, pkt.correlation_data, file=sys.stderr)
            if "/@rpc/" not in topic:
                return
            served["n"] += 1
            body = json.loads(bytes(pkt.payload or b"{}"))
            reply_topic = pkt.response_topic or body.get("reply")
            if not reply_topic:
                return
            cd = pkt.correlation_data
            out_pkt = m.PublishPacket(topic=reply_topic, payload=json.dumps({"id": body.get("id"), "ok": "acted"}).encode(),
                                      qos=m.QoS.AT_MOST_ONCE, correlation_data=cd.encode() if isinstance(cd, str) else cd)
            link.client.publish(out_pkt)

        link.on(serve)
        codes, _ = link.subscribe(f"strands/{tenant}/{dev['id']}/@rpc/+", qos=1)
        assert codes[0] < 0x80, codes
        op = Mqtt5(endpoint, name("e4", run, "operator"))
        reply_topic = f"strands/{tenant}/@reply/{op.client_id}"
        pending: dict[bytes, float] = {}
        done: dict[bytes, float] = {}
        ev = threading.Event()

        def on_reply(pkt) -> None:
            if __import__("os").environ.get("EXP_DEBUG"):
                print("OP got", str(pkt.topic), pkt.correlation_data, file=sys.stderr)
            cd = pkt.correlation_data  # awscrt hands it back as str when it is valid UTF-8, else bytes
            cd = cd.encode() if isinstance(cd, str) else bytes(cd or b"")
            cd = cd or json.loads(bytes(pkt.payload or b"{}")).get("id", "").encode()
            if cd in pending:
                done[cd] = time.perf_counter()
                ev.set()

        op.on(on_reply)
        codes, _ = op.subscribe(reply_topic, qos=1)
        assert codes[0] < 0x80, codes
        rpc = f"strands/{tenant}/{dev['id']}/@rpc/act"

        def rr(qos: int, i: int) -> float:
            cd = f"{run}-{qos}-{i}".encode()
            pending[cd] = time.perf_counter()
            ev.clear()
            pkt = m.PublishPacket(topic=rpc, payload=json.dumps({"id": cd.decode(), "action": {"shoulder_pan.pos": 0.001 * i}}).encode(),
                                  qos=m.QoS(qos), response_topic=reply_topic, correlation_data=cd)
            fut = op.client.publish(pkt)
            if qos:
                fut.result(10.0)
            while cd not in done and time.perf_counter() - pending[cd] < 2.0:
                ev.wait(0.5)
            return (done[cd] - pending.pop(cd)) * 1e3 if cd in done else float("nan")

        rr(0, -1); rr(1, -1)  # warm
        for qos in (0, 1):
            xs = [rr(qos, i) for i in range(a.n)]
            out[f"mqtt5_request_response_qos{qos}_ms"] = {**pct([x for x in xs if x == x]), "lost": sum(1 for x in xs if x != x)}
        out["request_payload_bytes"] = len(json.dumps({"id": f"{run}-0-0", "action": {"shoulder_pan.pos": 0.0}}))

        # (b) Direct Message on the v1 link's path (same boto3 call as transports/iot_link.AwsLink.send_direct)
        cfg = Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 1})
        data = boto3.client("iot-data", region_name=REGION, endpoint_url=f"https://{endpoint}", config=cfg)

        def dm(i: int, confirm: bool) -> tuple[float, float]:
            cd = f"dm-{run}-{int(confirm)}-{i}".encode()
            pending[cd] = time.perf_counter()
            ev.clear()
            body = json.dumps({"id": cd.decode(), "reply": reply_topic, "action": {"shoulder_pan.pos": 0.001 * i}}).encode()
            data.send_direct_message(clientId=p.thing, topic=rpc, payload=body, confirmation=confirm, timeout=10)
            call = (time.perf_counter() - pending[cd]) * 1e3
            while cd not in done and time.perf_counter() - pending[cd] < 2.0:
                ev.wait(0.5)
            return (done[cd] - pending.pop(cd)) * 1e3 if cd in done else float("nan"), call

        dm(-1, True)
        for confirm in (True, False):
            rows = [dm(i, confirm) for i in range(a.n)]
            out[f"direct_message_confirm_{str(confirm).lower()}"] = {
                "rtt_ms": {**pct([r for r, _ in rows if r == r]), "lost": sum(1 for r, _ in rows if r != r)},
                "https_call_ms": pct([c for _, c in rows])}
        out["device_served_requests"] = served["n"]

        # (c) offline: the device vanishes without a goodbye
        link.client.stop()
        link.down.wait(5.0)
        t_off = time.perf_counter()
        offline = []
        statuses = set()
        for i in range(20):
            t = time.perf_counter()
            try:
                data.send_direct_message(clientId=p.thing, topic=rpc, payload=b'{"id":"x","reply":"%s"}' % reply_topic.encode(), confirmation=True, timeout=5)
                statuses.add(200)
            except ClientError as exc:
                statuses.add(int(exc.response["ResponseMetadata"]["HTTPStatusCode"]))
            offline.append((time.perf_counter() - t) * 1e3)
        out["direct_message_to_offline_device"] = {"http_statuses": sorted(statuses), "call_ms": pct(offline),
                                                   "first_404_after_disconnect_s": round(offline[0] / 1e3, 2)}
        cd = b"offline-rr"
        pending[cd] = time.perf_counter(); ev.clear()
        op.client.publish(m.PublishPacket(topic=rpc, payload=b'{"id":"offline-rr"}', qos=m.QoS.AT_LEAST_ONCE, response_topic=reply_topic, correlation_data=cd)).result(10.0)
        ev.wait(3.0)
        out["mqtt5_request_to_offline_device"] = "PUBACK ok, no reply within 3 s (the broker has nobody to deliver to; nothing tells the caller)" if cd not in done else "reply?!"
        op.close()

        # (d) the reference's own end-to-end round trip over its IotTransport (Mesh.discover + invoke)
        if not a.no_ref:
            cmd = [sys.executable, str(ROOT / "tests/mesh/aws_iot_round_trip.py"), "--region", REGION, "--n", str(a.n)]
            t = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
            txt = r.stdout + r.stderr
            keep = [ln for ln in txt.splitlines() if re.search(r"discover over IoT|invoke\(act\)|estop to the killed|broadcast estop", ln)]
            out["ref_round_trip"] = {"rc": r.returncode, "wall_s": round(time.perf_counter() - t, 1), "lines": keep,
                                     "created_and_deleted": sorted({ln.split()[1] for ln in txt.splitlines() if ln.startswith(("created", "deleted"))})}
    finally:
        out["deleted"] = p.delete()
        out["swept_thing_types"] = sweep_deprecated_thing_types(iot)
        lo = leftovers(iot)
        lo["ref_things_left"] = [t["thingName"] for t in iot.list_things()["things"] if t["thingName"].startswith("strands-v1-local-mesh")]
        out["leftovers"] = lo
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
