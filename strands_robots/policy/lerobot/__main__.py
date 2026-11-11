"""``python -m strands_robots.policy.lerobot zenoh-config --deployment D --out DIR``: LeRobot's role-based setup.

Writes a CA, the router, ``policy-server`` and ``robot-client`` certificates and the three zenoh configs
(``roles.make``), then prints how to run each end.
"""

from __future__ import annotations

import argparse
import sys

from .roles import make


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and write the configs; 0 on success."""
    parser = argparse.ArgumentParser(prog="python -m strands_robots.policy.lerobot")
    sub = parser.add_subparsers(dest="command", required=True)
    cfg = sub.add_parser("zenoh-config", help="certificates + router/server/robot zenoh configs for one deployment")
    cfg.add_argument("--deployment", required=True, help="the LeRobot deployment name the router's ACL covers")
    cfg.add_argument("--out", required=True, help="directory for certs/ and {router,server,robot}.json5")
    cfg.add_argument("--host", action="append", help="name or address the router is dialled by (repeatable)")
    cfg.add_argument("--port", type=int, default=7447)
    args = parser.parse_args(argv)
    roles = make(args.deployment, args.out, tuple(args.host or ["localhost"]), args.port)
    d, ep = args.deployment, roles.endpoint
    print(f"router: zenohd -c {roles.router}")
    print(f"server: server_config(desc, ckpt, deployment={d!r}, endpoint={ep!r}, zenoh_config={str(roles.server)!r})")
    print(f"robot:  Policy('zenoh://{ep}/{d}', zenoh_config={str(roles.robot)!r})")
    print(f"certs:  {roles.router.parent / 'certs'} (a robot needs ca.pem and robot-client*.pem; ca-key.pem stays)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
