"""JSON schema for a verb, generated from its Python signature (no spec is ever written by hand).

An annotation is read as an expression tree (``ast``), so nested generics and unions are exact without importing
the names a module only imports for type checking. ``int``/``float``/``str``/``bool`` map to JSON types, ``X | None``
becomes nullable, ``tuple``/``list``/``Sequence`` become arrays, ``Mapping``/``dict`` objects, ``Any`` any JSON, an
``Action`` one number per joint and a policy a spec string (an ``Any`` whose default is a tuple of numbers is
that many numbers). Parameters that cannot cross a wire (callables,
events, live objects) are left out; a required one is an error, so a model is never offered a verb it cannot call.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable, Sequence
from typing import Any

_SCALARS = {"int": "integer", "float": "number", "str": "string", "bool": "boolean"}
_ARRAYS = {"tuple", "list", "Sequence", "Tuple", "List"}
_OBJECTS = {"dict", "Mapping", "Dict", "MutableMapping"}
_WIRE_DEFAULTS = (int, float, str, bool)


class NotOnTheWire(TypeError):
    """A required parameter whose type a model cannot send (a callable, an event, ...)."""


def _name(node: ast.expr) -> str:
    return node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")


def _union(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union(node.left) + _union(node.right)
    return [node]


def _args(node: ast.Subscript) -> list[ast.expr]:
    return list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]


def _schema(node: ast.expr, joints: Sequence[str]) -> dict[str, Any] | None:
    """The JSON schema of one annotation node, or None when it cannot cross a wire."""
    alts = _union(node)
    nullable = any(isinstance(a, ast.Constant) and a.value is None for a in alts)
    kinds = [s for s in (_one(a, joints) for a in alts if not (isinstance(a, ast.Constant) and a.value is None)) if s]
    if not kinds:
        return None
    out = dict(kinds[0])
    if nullable and isinstance(out.get("type"), str):
        out["type"] = [out["type"], "null"]
    return out


def _one(node: ast.expr, joints: Sequence[str]) -> dict[str, Any] | None:
    if isinstance(node, ast.Subscript):
        base, args = _name(node.value), _args(node)
        if base in _ARRAYS:
            items = [s for s in (_schema(a, joints) for a in args if not isinstance(a, ast.Constant)) if s]
            out: dict[str, Any] = {"type": "array"}
            if items:
                out["items"] = items[0]
            if base in ("tuple", "Tuple") and not any(isinstance(a, ast.Constant) for a in args):
                out["minItems"] = out["maxItems"] = len(args)
            return out
        if base in _OBJECTS and len(args) == 2 and _name(args[0]) == "str":  # JSON keys are strings
            value = _schema(args[1], joints)
            return {"type": "object", "additionalProperties": value} if value else None
        return None
    name = _name(node)
    if name in _SCALARS:
        return {"type": _SCALARS[name]}
    if name == "Any":
        return {"description": "any JSON value"}
    if name == "Action":
        return {
            "type": "object",
            "properties": {k: {"type": "number"} for k in joints},
            "additionalProperties": False,
            "description": f"joint -> target, any subset of {list(joints)}",
        }
    if name == "PolicyLike":
        return {"type": "string", "description": "a policy spec, e.g. a LeRobot checkpoint id"}
    return None


def property_for(param: inspect.Parameter, joints: Sequence[str]) -> dict[str, Any] | None:
    """The JSON schema of one parameter, or None when it cannot cross a wire."""
    ann = param.annotation
    if ann is inspect.Parameter.empty:
        if not isinstance(param.default, _WIRE_DEFAULTS):
            return None
        ann = type(param.default).__name__
    text = ann if isinstance(ann, str) else inspect.formatannotation(ann)
    try:
        prop = _schema(ast.parse(text, mode="eval").body, joints)
    except SyntaxError:
        return None
    d = param.default
    if prop is not None and "type" not in prop and isinstance(d, tuple) and all(isinstance(v, int | float) for v in d):
        prop = {
            "type": "array",
            "items": {"type": "number"},
            "minItems": len(d),
            "maxItems": len(d),
            "default": list(d),
        }
    if prop is not None and d is not inspect.Parameter.empty and isinstance(d, _WIRE_DEFAULTS):
        prop["default"] = param.default
    return prop


def parameters(
    fn: Callable[..., Any], *, joints: Sequence[str], exclude: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """``{"type": "object", "properties": ..., "required": [...]}`` for ``fn``'s wire-able parameters."""
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if name in exclude or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        prop = property_for(param, joints)
        if prop is None:
            if param.default is inspect.Parameter.empty:
                raise NotOnTheWire(f"{getattr(fn, '__qualname__', fn)}: parameter {name!r} cannot cross a wire")
            continue
        props[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}
