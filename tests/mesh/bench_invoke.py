"""invoke() round trip between two processes on this machine: ``python tests/mesh/bench_invoke.py``."""

import os
import socket
import subprocess
import sys
import time
import uuid

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(n: int = 1000) -> None:
    sys.path.insert(0, ROOT)
    from strands_robots.mesh import Mesh

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tenant = "bench-" + uuid.uuid4().hex[:6]
    peer = os.path.join(ROOT, "tests", "mesh", "peer_robot.py")
    url = f"zenoh://?listen=tcp/127.0.0.1:{port}"
    p = subprocess.Popen(
        [sys.executable, peer, url, tenant, "arm"],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": ROOT},
    )
    assert p.stdout is not None and p.stdout.readline().strip() == "ready"
    try:
        with Mesh(f"zenoh://?connect=tcp/127.0.0.1:{port}", tenant=tenant) as m:
            m.discover("arm", min_count=1, timeout=5)
            for rpc, kw in (("describe", {}), ("observe", {}), ("act", {"action": {"shoulder_pan": 0.1}})):
                for _ in range(50):
                    m.invoke("arm", rpc, **kw)
                ts = []
                for _ in range(n):
                    t = time.perf_counter()
                    m.invoke("arm", rpc, **kw)
                    ts.append((time.perf_counter() - t) * 1e3)
                a = np.array(ts)
                p50, p99 = np.percentile(a, 50), np.percentile(a, 99)
                print(f"{rpc:8s} n={n} p50={p50:.3f} ms p99={p99:.3f} ms max={a.max():.3f} ms")
    finally:
        p.kill()
        p.wait()


if __name__ == "__main__":
    main()
