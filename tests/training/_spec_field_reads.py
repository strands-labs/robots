"""Discovery rule shared by the field-scoped shared-domain guards.

Each shared numeric domain on :class:`~strands_robots.training.base.Trainer`
(:meth:`~strands_robots.training.base.Trainer._seed_problems` and its siblings)
documents a biconditional: a backend that *reads* the field MUST route it
through the gate, and one that ignores the field MUST NOT report on it. The
guards that pin the first half derive their scope from the tree rather than
from a list, so that a new backend which starts reading the field fails them
until it routes through the gate.

That derivation is only as wide as its notion of "reads the field". A backend
may read a spec field two ways:

* by name, ``spec.seed`` - an :class:`ast.Attribute` on ``spec``; or
* through a forwarding table, ``getattr(spec, field)`` over a tuple of field
  names - which is how a transport-only provider serializes every field it
  passes on without naming any of them in an attribute access.

A rule that recognizes only the first form reports a complete sweep while a
table-driven reader sits outside it, so the guards' promise ("a new backend
that starts reading the field fails this test") does not hold for that backend
and the biconditional goes unenforced for it. :func:`reads_spec_field`
recognizes both, and is the one place the two forms are defined so the
field-scoped guards cannot drift apart on what counts as a read.
"""

from __future__ import annotations

import ast
from collections.abc import Collection


def _reads_a_field_by_name(tree: ast.AST, fields: Collection[str]) -> bool:
    """Does *tree* contain ``spec.<field>`` for one of *fields*?"""
    return any(
        isinstance(node, ast.Attribute) and node.attr in fields and getattr(node.value, "id", None) == "spec"
        for node in ast.walk(tree)
    )


def _forwards_spec_fields_by_name(tree: ast.AST) -> bool:
    """Does *tree* read spec fields through ``getattr(spec, ...)``?

    The marker of a forwarding table: the field being read is a value rather
    than an attribute name, so no attribute access mentions it.
    """
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "spec"
        for node in ast.walk(tree)
    )


def _names_a_field_as_a_string(tree: ast.AST, fields: Collection[str]) -> bool:
    """Does *tree* carry one of *fields* as a string constant (a table entry)?"""
    return any(
        isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in fields
        for node in ast.walk(tree)
    )


def reads_spec_field(source: str, fields: Collection[str]) -> bool:
    """Does *source* read any of *fields* off a ``spec``, by either form?

    Args:
        source: Python source of one backend module.
        fields: The :class:`~strands_robots.training.base.TrainSpec` field names
            one shared domain owns.

    Returns:
        True when the module reads such a field by name (``spec.seed``) or
        forwards it through a table (``getattr(spec, field)`` over a tuple that
        names the field). The table form requires both halves, so a module that
        merely mentions the field name in a message - or one that uses
        ``getattr(spec, ...)`` for unrelated fields - is not a reader.
    """
    tree = ast.parse(source)
    if _reads_a_field_by_name(tree, fields):
        return True
    return _forwards_spec_fields_by_name(tree) and _names_a_field_as_a_string(tree, fields)
