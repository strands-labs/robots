"""broadcast("estop") to two device processes, call → all acked: ``python tests/mesh/bench_estop.py``."""

import os
import socket
import subprocess
import sys
import time
import uuid

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(rounds: int = 50) -> None:
    sys.path.insert(0, ROOT)
    from strands_robots.mesh import Mesh

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tenant = "bench-" + uuid.uuid4().hex[:6]
    peer = os.path.join(ROOT, "tests", "mesh", "peer_robot.py")
    env = {**os.environ, "PYTHONPATH": ROOT}
    urls = [f"zenoh://?listen=tcp/127.0.0.1:{port}", f"zenoh://?connect=tcp/127.0.0.1:{port}"]
    procs = [
        subprocess.Popen([sys.executable, peer, u, tenant, f"arm{i}"], stdout=subprocess.PIPE, text=True, env=env)
        for i, u in enumerate(urls)
    ]
    for p in procs:
        assert p.stdout is not None and p.stdout.readline().strip() == "ready"
    try:
        with Mesh(f"zenoh://?connect=tcp/127.0.0.1:{port}", tenant=tenant) as m:
            m.discover("device(*)", min_count=2, timeout=5)
            ts = []
            for _ in range(rounds):
                t = time.perf_counter()
                acked = m.broadcast("estop")
                ts.append((time.perf_counter() - t) * 1e3)
                assert acked == ["arm0", "arm1"], acked
                assert m.broadcast("resume", approved_by=m.operator) == ["arm0", "arm1"]
            a = np.array(ts)
            p50, p99 = np.percentile(a, 50), np.percentile(a, 99)
            print(f"broadcast estop → 2 acks, n={rounds}: p50={p50:.2f} ms p99={p99:.2f} ms")
    finally:
        for p in procs:
            p.kill()
            p.wait()


if __name__ == "__main__":
    main()
