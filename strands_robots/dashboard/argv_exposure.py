"""Was this dashboard's bearer token handed to it on the COMMAND LINE?"""

from __future__ import annotations

FLAG = "--auth-token"


def token_flag_in_argv(argv: list[str] | tuple[str, ...] | None) -> str | None:
    """The exposing argument as it appears in argv, or None."""
    items = list(argv or ())
    for i, arg in enumerate(items):
        if arg == FLAG:
            nxt = items[i + 1] if i + 1 < len(items) else ""
            if nxt and not nxt.startswith("-"):
                return FLAG
        elif arg.startswith(f"{FLAG}="):
            if arg[len(FLAG) + 1 :]:
                return FLAG
    return None


def argv_token_notice(argv: list[str] | tuple[str, ...] | None) -> dict[str, str] | None:
    """The sentence for the settings screen, or None when there is nothing to say."""
    if not token_flag_in_argv(argv):
        return None
    return {
        "kind": "token_in_argv",
        "severity": "warn",
        "text": (
            "This dashboard was started with --auth-token on the command line, so its bearer token "
            "is readable by every local user via `ps` - and that token is what stops a stranger on "
            "this machine from driving the arms."
        ),
        "remedy": (
            "Next restart, pass --auth-token-file ~/.strands_dashboard/local_api_token.txt instead: "
            "same token, same auth, out of argv and out of shell history. Rotate the token if this "
            "machine is shared."
        ),
    }
