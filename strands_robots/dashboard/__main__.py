"""``python -m strands_robots.dashboard``: a world with robots in it, served on loopback until Ctrl-C.

    python -m strands_robots.dashboard                      # the browser preset's robot on http://127.0.0.1:8090
    python -m strands_robots.dashboard so101 g1 --port 8091 # an arm and a humanoid, side by side

Robots are registry names, placed one metre apart along x; with none, the one preset that declares a ``browser``
block (the robot the docs page shows) is used. The page's chat needs a model: ``--model`` takes a
Bedrock model id (it also needs the ``[tools]`` extra and AWS credentials); without one the chat says so.
``--voice`` turns on speech to speech (``openai`` needs OPENAI_API_KEY, ``nova`` AWS credentials; ``auto`` picks).
``--fetch`` downloads the page's two browser engines (20 MB, not in the package) into the user cache and exits.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m strands_robots.dashboard", description=__doc__.split("\n\n")[0])
    p.add_argument("robots", nargs="*", default=[], help="registry names (default: the browser preset's robot)")
    p.add_argument("--port", type=int, default=8090, help="loopback port (default 8090)")
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 · localhost · ::1 (loopback only)")
    p.add_argument("--model", default=None, help="a Bedrock model id for the page's chat (optional)")
    p.add_argument("--voice", default=None, choices=("openai", "nova", "auto"), help="speech to speech (optional)")
    p.add_argument("--fetch", action="store_true", help="download the page's browser engines now, then exit")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse(argv)
    if args.fetch:  # for a robot LAN with no internet: fetch here, or copy the cache dir from a machine that did
        from .vendor import fetch_all

        for path in fetch_all():
            print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        return
    if not args.robots:
        from .standalone import browser_preset

        args.robots = [browser_preset()[1]["browser"]["robot"]]
    from ..robot import Robot
    from ..simulation import Simulation
    from .dashboard import Dashboard

    sim = Simulation.default()
    for i, name in enumerate(args.robots):
        Robot(name, sim=sim, position=[float(i), 0.0, 0.0])
    url = f"http://{args.host}:{args.port}/"
    print(f"strands-robots dashboard: {url}  ({', '.join(args.robots)}; Ctrl-C stops)", flush=True)
    Dashboard(sim, model=args.model, voice=args.voice).serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
