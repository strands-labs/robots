"""Experiment 5: lifecycle (presence) events vs the reference's heartbeat TTL.

    python scripts/iot/exp5_lifecycle.py [--keepalive 30] [--n 5]

The operator subscribes to `$aws/events/presence/connected/<thing>` and `…/disconnected/<thing>` (lifecycle events need no account
configuration). Measured: connect-success → connected event (×n), clean DISCONNECT → event (×n), and a device process killed with
SIGKILL (no DISCONNECT on the wire) → disconnected event, which is the broker's keep-alive detection time. Compared with the reference's
presence TTL (3 s heartbeats every 1 s).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _common import REGION, ROOT, Mqtt5, Provisioned, account_and_endpoint, leftovers, name, pct, run_id, so101_manifest, sweep_deprecated_thing_types

DEVICE = """
import sys, time
sys.path.insert(0, sys.argv[5])
from _common import Mqtt5
l = Mqtt5(sys.argv[1], sys.argv[2], cert=sys.argv[3], key=sys.argv[4], timeout=15.0, keep_alive=int(sys.argv[6]))
print("up", flush=True)
while True:
    time.sleep(1)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    import boto3

    iot = boto3.client("iot", region_name=REGION)
    run, dev, tenant = run_id(), so101_manifest(), "lab"
    account, endpoint = account_and_endpoint(iot)
    work = Path(tempfile.mkdtemp(prefix="strands-iot-exp5-"))
    p = Provisioned(iot, exp="e5", run=run, tenant=tenant, device_type=dev["device_type"], device_id=dev["id"], workdir=work)
    out: dict = {"experiment": 5, "run": run, "keep_alive_secs": 30}
    try:
        p.create(account, [dev["id"]], dev)
        time.sleep(1.0)
        op = Mqtt5(endpoint, name("e5", run, "operator"))
        events: list[tuple[float, str, dict]] = []
        op.on(lambda pkt: events.append((time.perf_counter(), str(pkt.topic).rsplit("/", 2)[-2], json.loads(bytes(pkt.payload or b"{}")))))
        for kind in ("connected", "disconnected"):
            codes, _ = op.subscribe(f"$aws/events/presence/{kind}/{p.thing}", qos=1)
            assert codes[0] < 0x80, (kind, codes)
        conn, disc = [], []
        for i in range(a.n):
            events.clear()
            link = Mqtt5(endpoint, p.thing, cert=p.cert_path, key=p.key_path, timeout=10.0)
            t_up = time.perf_counter()
            while not any(k == "connected" for _, k, _ in events) and time.perf_counter() - t_up < 10:
                time.sleep(0.001)
            e = [t for t, k, _ in events if k == "connected"]
            conn.append((e[0] - t_up) * 1e3 if e else float("nan"))
            time.sleep(0.5)
            events.clear()
            t_dn = time.perf_counter()
            link.client.stop()  # clean MQTT DISCONNECT
            while not any(k == "disconnected" for _, k, _ in events) and time.perf_counter() - t_dn < 10:
                time.sleep(0.001)
            e = [(t, b) for t, k, b in events if k == "disconnected"]
            disc.append((e[0][0] - t_dn) * 1e3 if e else float("nan"))
            if i == 0 and e:
                out["disconnect_event_example"] = {k: v for k, v in e[0][1].items() if k in ("eventType", "disconnectReason", "clientInitiatedDisconnect", "sessionIdentifier")}
            link.down.wait(5.0)
            time.sleep(0.5)
        out["connect_success_to_connected_event_ms"] = {**pct([x for x in conn if not math.isnan(x)]), "lost": sum(1 for x in conn if math.isnan(x))}
        out["clean_disconnect_to_disconnected_event_ms"] = {**pct([x for x in disc if not math.isnan(x)]), "lost": sum(1 for x in disc if math.isnan(x))}

        # ungraceful: SIGKILL a device process; the broker only notices at keep-alive
        events.clear()
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        proc = subprocess.Popen([sys.executable, "-c", DEVICE, endpoint, p.thing, str(p.cert_path), str(p.key_path), str(Path(__file__).parent), "30"],
                                stdout=subprocess.PIPE, text=True, env=env)
        assert proc.stdout and proc.stdout.readline().strip() == "up"
        time.sleep(1.0)
        events.clear()
        t_kill = time.perf_counter()
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        while not any(k == "disconnected" for _, k, _ in events) and time.perf_counter() - t_kill < 120:
            time.sleep(0.01)
        e = [(t, b) for t, k, b in events if k == "disconnected"]
        out["sigkill_to_disconnected_event_s"] = round((e[0][0] - t_kill), 2) if e else "none within 120 s"
        out["sigkill_disconnect_reason"] = e[0][1].get("disconnectReason") if e else None
        # a real partition: SIGSTOP freezes the process, the TCP socket stays open, no PINGREQ → the broker's keep-alive timer decides
        for ka in (5, 30):
            events.clear()
            proc = subprocess.Popen([sys.executable, "-c", DEVICE, endpoint, p.thing, str(p.cert_path), str(p.key_path), str(Path(__file__).parent), str(ka)],
                                    stdout=subprocess.PIPE, text=True, env=env)
            assert proc.stdout and proc.stdout.readline().strip() == "up"
            time.sleep(1.0)
            events.clear()
            t_stop = time.perf_counter()
            proc.send_signal(signal.SIGSTOP)
            while not any(k == "disconnected" for _, k, _ in events) and time.perf_counter() - t_stop < 120:
                time.sleep(0.05)
            e = [(t, b) for t, k, b in events if k == "disconnected"]
            out[f"sigstop_partition_keepalive{ka}_to_disconnected_event_s"] = round((e[0][0] - t_stop), 1) if e else "none within 120 s"
            out[f"sigstop_partition_keepalive{ka}_reason"] = e[0][1].get("disconnectReason") if e else None
            proc.send_signal(signal.SIGKILL)
            proc.wait()
            time.sleep(1.0)
        out["reference_heartbeat"] = {"presence_ttl_s": 3.0, "beat_every_s": 1.0, "offline_detected_within_s": "3.0–3.1 (ttl + expire tick 0.1)"}
        op.close()
    finally:
        out["deleted"] = p.delete()
        out["swept_thing_types"] = sweep_deprecated_thing_types(iot)
        out["leftovers"] = leftovers(iot)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
