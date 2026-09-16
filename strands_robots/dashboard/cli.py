"""``python -m strands_robots dashboard`` - serve the operator dashboard.

Loopback by default. Binding any other address is refused unless something
guards the API - an enrolled passkey (``auth.auth_enabled()``) or a static
``security.auth_token`` - because the open posture in :mod:`.access` exists for
the machine the dashboard runs on and for nothing else.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from collections.abc import Sequence

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, separate so tests can read the defaults."""
    p = argparse.ArgumentParser(
        prog="python -m strands_robots dashboard",
        description="Serve the strands-robots operator dashboard.",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8090, help="TCP port (default 8090)")
    p.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    p.add_argument("--log-level", default="info", choices=("critical", "error", "warning", "info", "debug"))
    return p


def bind_verdict(host: str, *, guarded: bool) -> str | None:
    """Why this bind must be refused, or None when it may proceed."""
    if host in _LOOPBACK_HOSTS:
        return None
    if guarded:
        return None
    return (
        f"refusing to bind {host}: nothing guards the API yet. Enrol a passkey on "
        "http://127.0.0.1 first (or set DASHBOARD_AUTH_TOKEN), then bind a LAN address."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, refuse an unguarded LAN bind, then serve. Returns the exit code."""
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        print(f"--port {args.port} is outside 1..65535", file=sys.stderr)
        return 2

    from strands_robots.dashboard import auth, settings

    guarded = auth.auth_enabled() or bool(settings.get("security", "auth_token"))
    refusal = bind_verdict(args.host, guarded=guarded)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    import uvicorn

    from strands_robots.dashboard.server import create_app

    url = f"http://{'localhost' if args.host in _LOOPBACK_HOSTS else args.host}:{args.port}/"
    print(f"strands-robots dashboard on {url}", file=sys.stderr)
    if args.open:
        webbrowser.open(url)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
