"""``Policy(spec, **kw)``: pick a provider from the FORM of ``spec``, never from a policy family name."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_PRESET = re.compile(r"^[a-z][a-z0-9_]*(\?[^\s/]*)?$")
FORMS = (
    "<preset>[?k=v&k2=v2]            a named ONNX preset, run in-process",
    "zenoh://host:port/<deployment>   a lerobot-policy-server (tcp); zenoh://tls/host:port/<deployment> any zenoh link",
    "<owner>/<repo>                   a LeRobot checkpoint on the Hugging Face Hub, run in-process",
    "<dir with config.json>           a local LeRobot checkpoint, run in-process",
    "<dir with meta/info.json>[?episode=n]   a recorded LeRobot dataset, one episode replayed",
    "<callable>                       f(obs) -> action",
)
HOLD_HINT = "use robot.hold() to stand still, robot.act(..., seconds=) to reach a pose"
LINKS = ("tcp", "tls", "quic", "ws", "wss", "udp")  # zenoh locator protocols a lerobot-policy-server may listen on
ZENOH_FORM = "zenoh://[<link>/]host:port/<deployment>, link one of " + ", ".join(LINKS) + " (default tcp)"


def zenoh_locator(spec: str) -> tuple[str, str]:
    """``zenoh://host:7447/d`` -> (``tcp/host:7447``, ``d``); ``zenoh://tls/host:7447/d`` -> (``tls/host:7447``, ``d``).

    The endpoint is LeRobot's own zenoh locator (``<link>/host:port``), so a lease that says ``tls/…`` (a TLS
    router on the internet) is written as it was given; ``zenoh+tls://host:7447/d`` spells the same thing.
    """
    scheme, _, rest = spec.partition("://")
    link = scheme.removeprefix("zenoh").removeprefix("+") or "tcp"
    parts = rest.split("/")
    if parts and parts[0] in LINKS and ":" not in parts[0]:  # zenoh://tls/host:port/<deployment>
        link, parts = parts[0], parts[1:]
    if link not in LINKS or len(parts) != 2 or not all(parts):
        raise ValueError(f"{spec!r}: expected {ZENOH_FORM}")
    hostport, deployment = parts
    host, _, port = hostport.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"{spec!r}: expected {ZENOH_FORM}")
    return f"{link}/{hostport}", deployment


def make(spec: Any, **kw: Any) -> Any:
    if not isinstance(spec, (str, Path)):
        if callable(spec) and not kw:
            from .callable import CallablePolicy

            return CallablePolicy(spec)
        raise TypeError(f"Policy({type(spec).__name__}) is not a policy spec; forms:\n  " + "\n  ".join(FORMS))
    s = str(spec)
    if s.startswith("zenoh://") or s.startswith("zenoh+"):
        from .lerobot import ZenohPolicy

        endpoint, deployment = zenoh_locator(s)
        return ZenohPolicy(endpoint=endpoint, deployment=deployment, **kw)
    if s == "hold":  # a run of it moved nothing, yet an agent read the word as "reach a pose": no such spec
        raise ValueError(f"Policy({s!r}): not a policy; {HOLD_HINT}")
    path, _, query = s.partition("?")
    if _PRESET.match(s):  # a known preset wins: the spec is a name, never a path in the CWD
        from .onnx import preset, preset_names

        if path in preset_names():
            return preset(s, **kw)
    local = Path(path).expanduser()
    if (local / "meta" / "info.json").is_file():
        from .dataset import ReplayPolicy

        try:  # the query is a grammar, not a hint: one key, ``episode``; anything else is refused by name
            params = dict(parse_qsl(query, keep_blank_values=True, strict_parsing=bool(query)))
        except ValueError as exc:
            raise ValueError(f"Policy({s!r}): malformed query {query!r}; the form is <dir>?episode=n ({exc})") from exc
        if unknown := sorted(set(params) - {"episode"}):
            raise ValueError(f"Policy({s!r}): unknown parameter {', '.join(map(repr, unknown))}; known: episode")
        episode = params.get("episode", "0")
        if not episode.isdigit():
            raise ValueError(f"Policy({s!r}): episode must be an integer >= 0, got {episode!r}")
        return ReplayPolicy(local, int(episode), **kw)
    if (local / "config.json").is_file() or _REPO.match(s):
        from .chunked import ChunkedPolicy
        from .lerobot import LeRobotPolicy

        replan = kw.pop("replan_every", None)
        return ChunkedPolicy(LeRobotPolicy(s, **kw), replan_every=replan)
    if local.is_dir():
        raise ValueError(
            f"Policy({s!r}): a directory, but neither a checkpoint nor a dataset; forms:\n  " + "\n  ".join(FORMS)
        )
    if _PRESET.match(s):
        from .onnx import preset_names

        known = f"; known presets: {', '.join(preset_names())}"
        raise ValueError(f"Policy({s!r}): unknown form{known}; forms:\n  " + "\n  ".join(FORMS))
    raise ValueError(f"Policy({s!r}): unknown form; forms:\n  " + "\n  ".join(FORMS))


def from_spec(spec: str) -> Any:
    """``from_spec("microduck?vx=0.3")`` is ``Policy("microduck?vx=0.3")``: a spec string alone, no keywords."""
    if not isinstance(spec, str):
        raise TypeError(f"from_spec takes a spec string, got {type(spec).__name__}")
    return make(spec)


def spec_help() -> str:
    """What a model is told a policy spec may be: every shipped preset and every form that crosses a wire."""
    from .onnx import preset_names

    forms = "; ".join(" ".join(f.split()) for f in FORMS if not f.startswith("<callable>"))
    return f"a policy spec. Shipped presets: {', '.join(map(repr, preset_names()))}. Forms: {forms}"
