"""Entry point for ``python -m strands_robots <command>``."""

from __future__ import annotations

import sys

_COMMANDS = ("doctor", "verify-dataset", "dashboard")


def main() -> None:
    """Dispatch ``python -m strands_robots <command>`` to its subcommand.

    Routes the first argv token to the ``doctor``, ``verify-dataset`` or
    ``dashboard`` entry point (stripping it so the subcommand parses clean args) and exits non-zero
    on a missing or unknown command.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m strands_robots <command>")
        print(f"Commands: {', '.join(_COMMANDS)}")
        sys.exit(1)

    cmd = sys.argv[1]
    # The two flags every console script is tried with first. Answering them
    # with "Unknown command" and exit 1 makes a fresh install look broken.
    if cmd in ("-h", "--help"):
        print("Usage: strands-robots <command> [options]")
        print(f"Commands: {', '.join(_COMMANDS)}")
        return
    if cmd in ("-V", "--version"):
        from importlib.metadata import version

        print(f"strands-robots {version('strands-robots')}")
        return
    # Remove the command from argv so sub-parsers see clean args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if cmd == "doctor":
        from strands_robots.doctor import main as doctor_main

        doctor_main()
    elif cmd == "verify-dataset":
        from strands_robots.verify_dataset import main as verify_main

        sys.exit(verify_main())
    elif cmd == "dashboard":
        from strands_robots.dashboard.cli import main as dashboard_main

        sys.exit(dashboard_main())
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available commands: {', '.join(_COMMANDS)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
