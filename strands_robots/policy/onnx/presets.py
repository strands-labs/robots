"""Named ONNX presets: their facts live in ``presets.json``; ``preset("microduck?vx=0.3")`` builds one.

A query value is converted to the type of the parameter's default (``vx=0.3`` → float, ``algorithm=ppo`` → str).
An unknown preset or an unknown key is refused, naming what is known.
"""

from __future__ import annotations

import importlib
import inspect
import json
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

_DATA = Path(__file__).with_name("presets.json")


@cache
def _table() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = json.loads(_DATA.read_text())
    return data


def preset_names(robot: str | None = None) -> tuple[str, ...]:
    """Every shipped preset, or only those that drive the registry robot ``robot``."""
    return tuple(sorted(n for n, facts in _table().items() if robot is None or robot in facts["robots"]))


def preset_robots(name: str) -> tuple[str, ...]:
    """The registry robots preset ``name`` drives."""
    return tuple(_table()[name]["robots"])


def preset_data(cls: type) -> dict[str, Any]:
    """The facts the preset implemented by ``cls`` reads (repo, files, hashes, layout)."""
    for name, facts in _table().items():
        if facts["class"].rpartition(":")[2] == cls.__name__:
            return {"name": name, **facts}
    raise KeyError(f"{cls.__name__} is not a preset in {_DATA.name}; presets: {', '.join(preset_names())}")


def _class(name: str) -> type:
    module, _, attr = _table()[name]["class"].partition(":")
    found: type = getattr(importlib.import_module(f"{__package__}.{module}"), attr)
    return found


def parameters(name: str) -> dict[str, inspect.Parameter]:
    """The keyword parameters a preset accepts (what a query string may set)."""
    cls = _class(name)
    sig = inspect.signature(cls.__dict__["__init__"])
    return {k: p for k, p in sig.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY and k != "providers"}


def _convert(name: str, key: str, text: str, default: Any) -> Any:
    if isinstance(default, bool):
        if text.lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"{name}?{key}={text}: expected true or false")
        return text.lower() in ("true", "1")
    if isinstance(default, (int, float)):
        try:
            return type(default)(float(text)) if isinstance(default, float) else int(text)
        except ValueError as exc:
            raise ValueError(f"{name}?{key}={text}: expected a number") from exc
    return text


def split(spec: str) -> tuple[str, dict[str, Any]]:
    """``"microduck?vx=0.3"`` → ``("microduck", {"vx": 0.3})``, refused naming the known presets or keys."""
    name, _, query = spec.partition("?")
    known = preset_names()
    if name not in known:
        raise ValueError(f"unknown ONNX preset {name!r}; known presets: {', '.join(known)}")
    params = parameters(name)
    out: dict[str, Any] = {}
    for key, text in parse_qsl(query, keep_blank_values=True, strict_parsing=bool(query)):
        if key not in params:
            raise ValueError(f"{name}: unknown parameter {key!r}; known: {', '.join(sorted(params))}")
        out[key] = _convert(name, key, text, params[key].default)
    return name, out


def preset(spec: str, **kw: Any) -> Any:
    """Build a preset from ``"name?k=v&k2=v2"``; keywords override the query and are checked the same way."""
    name, args = split(spec)
    params = parameters(name)
    unknown = sorted(set(kw) - set(params) - {"providers"})
    if unknown:
        raise TypeError(f"{name}: unexpected keyword {unknown[0]!r}; known: {', '.join(sorted(params))}")
    return _class(name)(**{**args, **kw})
