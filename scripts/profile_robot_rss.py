"""Resident memory as robots join one world: ``python scripts/profile_robot_rss.py [N] [model]``.

Prints RSS, the compiled model size and the Python heap (tracemalloc) after each ``Robot``. The Python heap stays
flat: what grows is MuJoCo's C memory, so RSS and ``mj_sizeModel`` are the numbers to read.
"""

import os
import sys
import tracemalloc

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from strands_robots import Robot, Simulation  # noqa: E402


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


def main(n: int, model: str) -> None:
    tracemalloc.start()
    first = None
    print("robots  rss_MB  per_added_MB  model_MB  python_heap_MB")
    for i in range(n):
        Robot(model, position=[0.6 * i, 0.0, 0.0])
        rss = rss_mb()
        first = rss if first is None else first
        per = (rss - first) / i if i else 0.0
        size = mujoco.mj_sizeModel(Simulation.default().backend.model) / 1e6
        heap = tracemalloc.get_traced_memory()[0] / 1e6
        print(f"{i + 1:6d}  {rss:6.0f}  {per:12.0f}  {size:8.1f}  {heap:14.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8, sys.argv[2] if len(sys.argv) > 2 else "so101")
