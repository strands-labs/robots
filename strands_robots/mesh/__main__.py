"""``python -m strands_robots.mesh``: make a mesh CA, issue each machine its certificate, write zenoh configs.

    python -m strands_robots.mesh ca                                  # the lab CA in ~/.strands/mesh/ca (once)
    python -m strands_robots.mesh cert lab-pc --serves 'so101,arm-*'  # a robot machine: the ids it answers for
    python -m strands_robots.mesh cert alice --out ./alice            # an operator: copy ./alice to their machine
    python -m strands_robots.mesh config --mode router --listen tls/0.0.0.0:7447 -o zenohd.json5

``cert`` writes ``~/.strands/mesh/identity`` by default, which every ``zenoh://`` mesh on that machine then uses: it
joins over mutual TLS only and signs what it sends. ``config`` writes the same identity as a zenoh config file, for a
``zenohd`` router or any other zenoh client (``STRANDS_ZENOH_CONFIG``, LeRobot's ``zenoh_config=``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import pki
from .transports import zenoh_tls


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m strands_robots.mesh", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    ca = sub.add_parser("ca", help="make the mesh CA (keep ca.key on one admin machine)")
    ca.add_argument("--ca", default=str(pki.DEFAULT_CA), help=f"where (default {pki.DEFAULT_CA})")
    cert = sub.add_parser("cert", help="issue one machine its certificate and key")
    cert.add_argument("name", help="the peer's name: letters, digits, _ or - (peer ids become <name>.<random>)")
    cert.add_argument("--serves", default="", help="comma-separated device ids (* and ? wildcards) it answers for")
    cert.add_argument("--ca", default=str(pki.DEFAULT_CA), help=f"the CA (default {pki.DEFAULT_CA})")
    cert.add_argument("--out", default=str(pki.DEFAULT_IDENTITY), help=f"identity dir (default {pki.DEFAULT_IDENTITY})")
    cert.add_argument("--days", type=int, default=825, help="validity in days (default 825)")
    cfg = sub.add_parser("config", help="write a zenoh config file for an identity (mutual TLS, tls/ links only)")
    cfg.add_argument("--identity", default=str(pki.DEFAULT_IDENTITY), help="identity dir")
    cfg.add_argument("--mode", default="peer", choices=("peer", "client", "router"))
    cfg.add_argument("--listen", action="append", default=[], help="tls/<host>:<port>, repeatable")
    cfg.add_argument("--connect", action="append", default=[], help="tls/<host>:<port>, repeatable")
    cfg.add_argument("-o", "--output", default="-", help="file to write (default: stdout)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse(argv)
    try:
        if args.command == "ca":
            print(f"mesh CA in {pki.make_ca(args.ca)}; issue certificates with `cert NAME`")
        elif args.command == "cert":
            serves = [s for s in args.serves.split(",") if s.strip()]
            out = pki.issue(args.ca, args.name, args.out, serves=serves, days=args.days)
            what = f"answers for {', '.join(serves)}" if serves else "an operator: answers for no device"
            print(f"identity {args.name!r} in {out} ({what})")
        else:
            cfg = zenoh_tls.tls_config(args.identity, mode=args.mode, listen=args.listen, connect=args.connect)
            text = json.dumps(cfg, indent=2) + "\n"
            if args.output == "-":
                sys.stdout.write(text)
            else:
                with open(args.output, "w") as f:
                    f.write(text)
                print(f"zenoh config ({args.mode}, mutual TLS) in {args.output}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
