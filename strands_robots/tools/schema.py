"""JSON schema for a verb, generated from its Python signature (no spec is ever written by hand).

An annotation is read as an expression tree (``ast``), so nested generics and unions are exact without importing
the names a module only imports for type checking. ``int``/``float``/``str``/``bool`` map to JSON types, ``X | None``
becomes nullable, ``X | Y`` an ``anyOf``, ``Literal[...]`` an enum, ``tuple``/``list``/``Sequence`` arrays,
``Mapping``/``dict`` objects (a key union that admits ``str`` has string keys), ``Any`` any JSON, an
``Action`` one number per joint and a policy a spec string naming the shipped presets and forms (an ``Any``
whose default is a tuple of numbers is that many numbers). Parameters that cannot cross a wire (callables,
events, live objects) are left out; a required one is an error, so a model is never offered a verb it cannot call.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

_SCALARS = {"int": "integer", "float": "number", "str": "string", "bool": "boolean"}
_ARRAYS = {"tuple", "list", "Sequence", "Tuple", "List"}
_OBJECTS = {"dict", "Mapping", "Dict", "MutableMapping"}
_WIRE_DEFAULTS = (int, float, str, bool)
# The joints an Action names: a sequence of names, or name -> what a model is told of it (unit, minimum, maximum).
Joints = Sequence[str] | Mapping[str, Mapping[str, Any]]


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


def _schema(node: ast.expr, joints: Joints) -> dict[str, Any] | None:
    """The JSON schema of one annotation node, or None when it cannot cross a wire."""
    alts = _union(node)
    nullable = any(isinstance(a, ast.Constant) and a.value is None for a in alts)
    kinds = [s for s in (_one(a, joints) for a in alts if not (isinstance(a, ast.Constant) and a.value is None)) if s]
    if not kinds:
        return None
    if len(kinds) > 1:  # a factor or a (lo, hi) range: the model is told it may send either
        return {"anyOf": kinds + [{"type": "null"}] * nullable}
    out = dict(kinds[0])
    if nullable and isinstance(out.get("type"), str):
        out["type"] = [out["type"], "null"]
    if nullable and "enum" in out:
        out["enum"] = [*out["enum"], None]
    return out


def _one(node: ast.expr, joints: Joints) -> dict[str, Any] | None:
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
        if base == "Literal" and all(isinstance(a, ast.Constant) for a in args):
            return {"type": "string", "enum": [a.value for a in args]}  # type: ignore[attr-defined]
        if base in _OBJECTS and len(args) == 2 and "str" in map(_name, _union(args[0])):  # JSON keys are strings
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
            "properties": {k: {"type": "number", **(joints[k] if isinstance(joints, Mapping) else {})} for k in joints},
            "additionalProperties": False,
            "description": f"joint -> target, any subset of {list(joints)}",
        }
    if name == "PolicyLike":
        from ..policy.factory import spec_help

        return {"type": "string", "description": spec_help()}
    return None


def property_for(param: inspect.Parameter, joints: Joints) -> dict[str, Any] | None:
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
    if (
        prop is not None
        and not {"type", "anyOf"} & set(prop)
        and isinstance(d, tuple)
        and all(isinstance(v, int | float) for v in d)
    ):
        prop = {
            "type": "array",
            "items": {"type": "number"},
            "minItems": len(d),
            "maxItems": len(d),
            "default": list(d),
        }
    if prop is not None and d is not inspect.Parameter.empty and isinstance(d, _WIRE_DEFAULTS):
        prop["default"] = param.default
    if prop is not None and isinstance(d, tuple) and "default" not in prop:
        prop["default"] = list(d)
    return prop


def parameters(fn: Callable[..., Any], *, joints: Joints, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
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
