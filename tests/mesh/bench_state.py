"""subscribe(state) rate and drops between two processes on this machine: ``python tests/mesh/bench_state.py [hz]``."""

import os
import socket
import subprocess
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(hz: float, seconds: float = 5.0) -> None:
    sys.path.insert(0, ROOT)
    from strands_robots.mesh import Mesh

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tenant = "bench-" + uuid.uuid4().hex[:6]
    peer = os.path.join(ROOT, "tests", "mesh", "peer_robot.py")
    url = f"zenoh://?listen=tcp/127.0.0.1:{port}"
    env = {**os.environ, "PYTHONPATH": ROOT}
    p = subprocess.Popen(
        [sys.executable, peer, url, tenant, "arm", str(hz)], stdout=subprocess.PIPE, text=True, env=env
    )
    assert p.stdout is not None and p.stdout.readline().strip() == "ready"
    try:
        with Mesh(f"zenoh://?connect=tcp/127.0.0.1:{port}", tenant=tenant) as m:
            m.discover("arm", min_count=1, timeout=5)
            got: list[int] = []
            unsub = m.subscribe("arm", "state", lambda payload: got.append(payload["seq"]))
            time.sleep(1.0)
            start = len(got)
            time.sleep(seconds)
            window = got[start:]
            unsub()
            gaps = sum(b - a - 1 for a, b in zip(window, window[1:], strict=False) if b > a + 1)
            print(
                f"state_hz={hz:g} received {len(window)} in {seconds:g} s"
                f" = {len(window) / seconds:.1f} Hz, drops={gaps}"
            )
    finally:
        p.kill()
        p.wait()


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 100.0)
