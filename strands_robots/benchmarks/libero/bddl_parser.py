"""BDDL parser - LIBERO task files → named-predicate AST.

LIBERO ships its tasks as ``.bddl`` files written in a PDDL-derived
s-expression syntax:

.. code-block:: lisp

    (define (problem libero_pick_cube)
      (:domain kitchen)
      (:language "pick up the red cube and place it on the plate")
      (:objects cube_1 plate_1 - object)
      (:init
        (on cube_1 table_1))
      (:goal
        (and
          (on cube_1 plate_1)
          (not (grasped cube_1)))))

This module parses that into a :class:`BDDLProblem` whose ``:goal`` compiles
to a single ``(SimEngine) -> bool`` callable via
:mod:`strands_robots.simulation.predicates`. Crucially it **never** evaluates
user code - the BDDL grammar is a closed set of tokens plus a whitelisted
predicate vocabulary; anything outside that set raises
:class:`BDDLParseError`.

Unknown predicates are rejected rather than silently evaluated to ``False``:
a BDDL parse failure is always preferable to a misleading success rate.

Scope of this parser (matches what LIBERO actually uses):

* Top-level form: ``(define (problem <name>) ...)``
* Section markers: ``:domain``, ``:objects``, ``:init``, ``:goal``, ``:language``
* Boolean combinators: ``and``, ``or``, ``not``
* Predicate vocabulary: ``on``, ``near``, ``inside``, ``open``, ``closed``,
  ``grasped``, ``upright``

Everything else is either dropped silently (typed-object annotations like
``obj1 - object`` are flattened to just the symbols) or raises
:class:`BDDLParseError` depending on whether leniency would mask real bugs.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.predicates import make_predicate

if TYPE_CHECKING:
    from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)


class BDDLParseError(ValueError):
    """Raised when a BDDL file fails to tokenize, parse, or compile."""


# AST nodes


@dataclass(frozen=True)
class Pred:
    """Leaf predicate: ``(on cube_1 plate_1)`` → ``Pred("on", ["cube_1", "plate_1"])``."""

    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class And:
    clauses: tuple[Node, ...]


@dataclass(frozen=True)
class Or:
    clauses: tuple[Node, ...]


@dataclass(frozen=True)
class Not:
    clause: Node


Node = Pred | And | Or | Not


@dataclass
class BDDLProblem:
    """Parsed representation of a BDDL file."""

    name: str
    domain: str | None = None
    language: str | None = None
    objects: list[str] = field(default_factory=list)
    init: list[Node] = field(default_factory=list)
    goal: Node | None = None


# Tokenizer + s-expression parser


_COMMENT_RE = re.compile(r";[^\n]*")
_PAREN_RE = re.compile(r"([()])")


def _tokenize(text: str) -> list[str]:
    """Split BDDL text into paren / atom tokens.

    LIBERO BDDL allows ``;`` line comments and double-quoted ``:language``
    strings. We strip comments first, then walk the text pairing up quoted
    regions so whitespace inside them isn't split.
    """
    # Strip ``;`` line comments (but not inside quotes).
    out_tokens: list[str] = []
    i = 0
    # Use a simple hand-rolled scanner so quoted strings stay intact.
    s = text
    n = len(s)
    while i < n:
        c = s[i]
        if c == ";":
            # Skip until newline.
            nl = s.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue
        if c.isspace():
            i += 1
            continue
        if c in "()":
            out_tokens.append(c)
            i += 1
            continue
        if c == '"':
            # Quoted string - find matching quote. No escape sequences in LIBERO
            # so this is a plain scan.
            end = s.find('"', i + 1)
            if end == -1:
                raise BDDLParseError(f"unterminated quoted string at offset {i}")
            out_tokens.append(s[i : end + 1])
            i = end + 1
            continue
        # Atom - grab until whitespace or paren.
        j = i
        while j < n and not s[j].isspace() and s[j] not in "()":
            j += 1
        out_tokens.append(s[i:j])
        i = j
    return out_tokens


def _parse_sexp(tokens: list[str]) -> Any:
    """Consume tokens (in place, reversed-stack style) and return a nested list.

    Atoms stay as ``str``. Lists nest normally. Caller is responsible for
    raising on leftover tokens.
    """
    if not tokens:
        raise BDDLParseError("unexpected end of input")
    token = tokens.pop(0)
    if token == "(":
        out: list[Any] = []
        while tokens and tokens[0] != ")":
            out.append(_parse_sexp(tokens))
        if not tokens:
            raise BDDLParseError("missing closing ')'")
        tokens.pop(0)  # consume ")"
        return out
    if token == ")":
        raise BDDLParseError("unexpected ')'")
    return token


# Predicate vocabulary

# BDDL predicate name → (predicate-registry name, args → kwargs adapter).
# The adapter is how LIBERO's positional BDDL args map to our kwarg-style
# predicate library. ``*args`` is the list from ``expr[1:]`` at compile time.


def _on_kwargs(args: list[str]) -> dict[str, Any]:
    if len(args) != 2:
        raise BDDLParseError(f"(on ...) expects 2 args, got {len(args)}: {args}")
    # #170: tighten thresholds to match upstream LIBERO's
    # ``ObjectState.check_ontop`` semantics. The default ``_body_on``
    # uses ``z_offset=0.02 m`` and ``xy_tol=0.15 m`` (loose, intended for
    # coarse "is A on B" checks); upstream LIBERO requires
    # ``a.z >= b.z`` (no offset) AND ``|a.xy - b.xy| < 0.03 m`` AND
    # contact between A and B. The empirical at-success state on
    # ``libero-10/SCENE5`` has mug.z 4 mm above plate.z and mug.xy 1.2 cm
    # off plate.xy — the loose defaults rejected this as ``False`` while
    # ``env.check_success()`` returned ``True`` (#170 diagnostic).
    #
    # #171 sub-task 3e: also require physics contact via
    # ``sim.get_contacts``. Without this, transient
    # "mug-suspended-above-plate" placement states produce BDDL false
    # positives (mug 5 cm above plate, xy-aligned, no contact yet).
    # Graceful degradation: engines without ``get_contacts`` skip the
    # check (preserves pre-#171 behaviour).
    return {
        "body_a": args[0],
        "body_b": args[1],
        "z_offset": 0.0,
        "xy_tol": 0.03,
        "require_contact": True,
    }


def _near_kwargs(args: list[str]) -> dict[str, Any]:
    if len(args) != 2:
        raise BDDLParseError(f"(near ...) expects 2 args, got {len(args)}: {args}")
    return {"body_a": args[0], "body_b": args[1], "threshold": 0.1}


def _inside_kwargs(args: list[str]) -> dict[str, Any]:
    if len(args) != 2:
        raise BDDLParseError(f"(inside ...) expects 2 args, got {len(args)}: {args}")
    return {"body": args[0], "container": args[1]}


def _open_kwargs(args: list[str]) -> dict[str, Any]:
    if len(args) != 1:
        raise BDDLParseError(f"(open ...) expects 1 arg, got {len(args)}: {args}")
    return {"joint": args[0], "value": 0.1}


def _closed_kwargs(args: list[str]) -> dict[str, Any]:
    if len(args) != 1:
        raise BDDLParseError(f"(closed ...) expects 1 arg, got {len(args)}: {args}")
    return {"joint": args[0], "value": 0.01}


def _grasped_kwargs(args: list[str]) -> dict[str, Any]:
    if len(args) != 1:
        raise BDDLParseError(f"(grasped ...) expects 1 arg, got {len(args)}: {args}")
    # LIBERO scenes use robot0_gripper_* for Panda gripper geoms. Adapters that
    # need a different prefix should subclass LiberoAdapter and override
    # ``GRIPPER_PREFIX``.
    return {"body": args[0], "gripper_prefix": "robot0_gripper"}


def _upright_kwargs(args: list[str]) -> dict[str, Any]:
    if len(args) != 1:
        raise BDDLParseError(f"(upright ...) expects 1 arg, got {len(args)}: {args}")
    return {"body": args[0], "tol": 0.15}


#: Whitelist of BDDL predicates → predicate-registry entries. Compiled once
#: at import; mutating this dict in-place is the extension point for
#: adapters that need to add benchmark-specific predicates. Keep in sync
#: with the docstring at the top of this module.
PREDICATE_VOCABULARY: dict[str, tuple[str, Callable[[list[str]], dict[str, Any]]]] = {
    "on": ("body_on", _on_kwargs),
    "near": ("distance_less_than", _near_kwargs),
    "inside": ("body_inside", _inside_kwargs),
    "open": ("joint_above", _open_kwargs),
    "closed": ("joint_below", _closed_kwargs),
    "grasped": ("grasped", _grasped_kwargs),
    "upright": ("body_upright", _upright_kwargs),
}


# Top-level parser


def parse_bddl(text: str) -> BDDLProblem:
    """Parse a BDDL string into a :class:`BDDLProblem`.

    Args:
        text: Contents of a ``.bddl`` file.

    Raises:
        BDDLParseError: On tokenizer, parser, or vocabulary failures. The
            message always names the offending construct.
    """
    tokens = _tokenize(text)
    if not tokens:
        raise BDDLParseError("empty BDDL input")
    sexp = _parse_sexp(tokens)
    if tokens:
        raise BDDLParseError(f"trailing tokens after top-level form: {tokens[:5]!r}")

    if not isinstance(sexp, list) or not sexp or sexp[0] != "define":
        raise BDDLParseError("expected top-level (define ...) form")

    problem_name = ""
    domain: str | None = None
    language: str | None = None
    objects: list[str] = []
    init_nodes: list[Node] = []
    goal_node: Node | None = None

    for child in sexp[1:]:
        if not isinstance(child, list) or not child:
            continue
        head = child[0]
        if head == "problem":
            if len(child) >= 2 and isinstance(child[1], str):
                problem_name = child[1]
        elif head == ":domain":
            if len(child) >= 2 and isinstance(child[1], str):
                domain = child[1]
        elif head == ":language":
            # Language strings are quoted ("pick up the cube"). Strip quotes.
            pieces: list[str] = []
            for c in child[1:]:
                if isinstance(c, str):
                    if c.startswith('"') and c.endswith('"'):
                        pieces.append(c[1:-1])
                    else:
                        pieces.append(c)
            language = " ".join(pieces) if pieces else None
        elif head == ":objects":
            # LIBERO uses PDDL typed syntax: ``cube_1 plate_1 - object``. We
            # flatten to just the symbols; the ``-`` and type annotations
            # don't affect predicate evaluation.
            for c in child[1:]:
                if isinstance(c, str) and c != "-":
                    objects.append(c)
        elif head == ":init":
            for c in child[1:]:
                if not isinstance(c, list):
                    continue
                # Init entries use the same predicate grammar - compile each.
                try:
                    init_nodes.append(_compile_ast(c))
                except BDDLParseError as e:
                    # Init failures are not fatal - they're just "declared
                    # initial state", which the adapter may or may not
                    # enforce. Log and skip; the goal is the authoritative
                    # success criterion.
                    logger.debug("skipping unsupported (:init ...) clause: %s", e)
        elif head == ":goal":
            if len(child) < 2:
                raise BDDLParseError("(:goal ...) is empty")
            goal_node = _compile_ast(child[1])
        # Other markers (:requirements, :constants, etc.) are silently ignored.

    return BDDLProblem(
        name=problem_name or "unnamed",
        domain=domain,
        language=language,
        objects=objects,
        init=init_nodes,
        goal=goal_node,
    )


def _compile_ast(expr: Any) -> Node:
    """Compile a raw s-expression list into the typed :data:`Node` AST."""
    if not isinstance(expr, list) or not expr:
        raise BDDLParseError(f"expected predicate s-expression, got {expr!r}")
    head = expr[0]
    if not isinstance(head, str):
        raise BDDLParseError(f"expected symbol head, got {head!r}")
    # PDDL grammar is case-insensitive for predicates and connectives. Real
    # LIBERO BDDL files mix cases: every spatial / object / goal task uses
    # ``(And (On ...))``  with capital initials in the goal, even though
    # ``PREDICATE_VOCABULARY`` and the connective branches below were
    # written in lowercase. Normalise once at the head; keep the original
    # ``head`` for error messages so debugging shows the source casing.
    head_norm = head.lower()
    if head_norm == "and":
        if len(expr) == 1:
            raise BDDLParseError("(and ...) requires at least one clause")
        return And(tuple(_compile_ast(c) for c in expr[1:]))
    if head_norm == "or":
        if len(expr) == 1:
            raise BDDLParseError("(or ...) requires at least one clause")
        return Or(tuple(_compile_ast(c) for c in expr[1:]))
    if head_norm == "not":
        if len(expr) != 2:
            raise BDDLParseError(f"(not ...) expects 1 clause, got {len(expr) - 1}")
        return Not(_compile_ast(expr[1]))
    if head_norm not in PREDICATE_VOCABULARY:
        valid = sorted(PREDICATE_VOCABULARY)
        raise BDDLParseError(f"unknown predicate {head!r}. Supported: {valid}")
    # Leaf predicate - args are the remainder, must all be strings.
    args = []
    for a in expr[1:]:
        if not isinstance(a, str):
            raise BDDLParseError(f"predicate {head!r}: expected string args, got {a!r}")
        args.append(a)
    # Validate arity by attempting the kwargs conversion now (fail-fast).
    _, adapter = PREDICATE_VOCABULARY[head_norm]
    adapter(args)  # raises BDDLParseError on bad arity
    # Store the normalised name so ``compile_goal`` can look it up in
    # ``PREDICATE_VOCABULARY`` (whose keys are all lowercase).
    return Pred(name=head_norm, args=tuple(args))


# Compile AST → callable


def compile_goal(node: Node) -> Callable[[SimEngine], bool]:
    """Compile a :data:`Node` AST into a single ``(sim) -> bool`` callable.

    The compiled callable is a pure function of ``sim`` state: no hidden RNG,
    no per-call allocation past what the leaf predicate closures capture.
    Boolean combinators are evaluated with short-circuit semantics.
    """
    if isinstance(node, Pred):
        registry_name, adapter = PREDICATE_VOCABULARY[node.name]
        kwargs = adapter(list(node.args))
        return make_predicate(registry_name, **kwargs)
    if isinstance(node, And):
        compiled = [compile_goal(c) for c in node.clauses]
        return lambda sim: all(p(sim) for p in compiled)
    if isinstance(node, Or):
        compiled = [compile_goal(c) for c in node.clauses]
        return lambda sim: any(p(sim) for p in compiled)
    if isinstance(node, Not):
        inner = compile_goal(node.clause)
        return lambda sim: not inner(sim)
    raise BDDLParseError(f"unsupported AST node: {type(node).__name__}")


def parse_bddl_file(path: str | Path) -> BDDLProblem:
    """Convenience loader - reads ``path`` and runs :func:`parse_bddl`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BDDL file not found: {path}")
    if not p.is_file():
        raise ValueError(f"BDDL path is not a file: {path}")
    return parse_bddl(p.read_text())


__all__ = [
    "And",
    "BDDLParseError",
    "BDDLProblem",
    "Node",
    "Not",
    "Or",
    "PREDICATE_VOCABULARY",
    "Pred",
    "compile_goal",
    "parse_bddl",
    "parse_bddl_file",
]
