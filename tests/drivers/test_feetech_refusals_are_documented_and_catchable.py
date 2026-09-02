# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A refusal this codec makes must be one its caller can name in an ``except``.

:mod:`~strands_robots.drivers.feetech.protocol` is a codec whose only caller is
a serial bus module, so its ``Raises:`` blocks are the entire interface for
error handling: a bus that reads back a frame decides between "retry the read"
and "this is my bug" purely from the class it catches.

Two properties are graded here, and the tree-wide
``tests/test_raises_docstring_completeness.py`` can see neither of them.

That guard reads the *abstract syntax tree* and compares only the classes a
function raises **in its own scope**, and it grades a function only if the
docstring already has a ``Raises:`` section - "A docstring with no ``Raises:``
section at all is out of scope" is its documented policy, shared with its
``Args:`` sibling. Both exemptions are load-bearing tree-wide (318 functions in
the package raise with no block), so this file narrows to one module rather
than proposing to change that policy.

Inside this module the two exemptions matter concretely:

* **What escapes is not what the body raises.** ``build_packet`` raises
  ``ValueError`` in its own scope, and a caller who passes ``motor_id="3"``
  receives a ``TypeError`` from the ``_validate_id`` helper one frame down. The
  AST guard cannot see it; a caller absolutely can. So the cells below *call*
  each function and grade the class that actually comes out.
* **A block that does not exist cannot be wrong.** Deleting a ``Raises:``
  section removes a function from the tree-wide guard's population entirely,
  silently. :class:`TestEveryPublicRefusalIsDocumented` requires the block for
  every name the package exports, so leaving is a failure rather than an exit.

The refusing inputs are stated as data rather than derived, because "an input
this function refuses" is not recoverable from a signature.
:meth:`TestTheRuleIsNotVacuous.test_every_public_callable_is_exercised` fails
if a public callable is added without one, so the table cannot fall behind the
surface it grades.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import pytest

import strands_robots.drivers.feetech as package
from strands_robots.drivers.feetech import protocol

# A frame the parser accepts, so a case that should reach a later guard is not
# stopped by the header scan: ID 1, LEN 2, error 0, additive checksum.
_WELL_FORMED = b"\xff\xff\x01\x02\x00\xfc"

# (public name, args, the class a caller receives). Every entry is measured, and
# the ones that come out of a helper rather than the function's own body are the
# reason this file calls instead of reading the syntax tree.
_REFUSALS: tuple[tuple[str, tuple[Any, ...], type[BaseException]], ...] = (
    ("build_packet", ("3", protocol.Instruction.PING), TypeError),
    ("build_packet", (True, protocol.Instruction.PING), TypeError),
    ("build_packet", (0xFE, protocol.Instruction.PING), ValueError),
    ("build_packet", (1, 0x1FF), ValueError),
    ("ping_packet", ("3",), TypeError),
    ("ping_packet", (0xFE,), ValueError),
    ("read_packet", ("3", 0x38, 2), TypeError),
    ("read_packet", (1, 0x1FF, 2), ValueError),
    ("read_packet", (1, 0x38, 0), ValueError),
    ("write_packet", (True, 0x2A, b"\x01"), TypeError),
    ("write_packet", (1, 0x2A, b""), ValueError),
    ("sync_write_packet", (0x2A, 2, [("1", b"\x01\x02")]), TypeError),
    ("sync_write_packet", (0x2A, 2, []), ValueError),
    ("sync_write_packet", (0x2A, 2, [(1, b"\x01")]), ValueError),
    ("parse_status_packet", ("notbytes", 1, 0), TypeError),
    ("parse_status_packet", (_WELL_FORMED, 0x1FF, 0), ValueError),
    ("parse_status_packet", (b"\x00\x00", 1, 0), protocol.ProtocolError),
    ("encode_word", (True,), TypeError),
    ("encode_word", (0x10000,), ValueError),
    ("decode_word", ([0xFF, 0x03],), TypeError),
    ("decode_word", (b"\x01",), ValueError),
)


def _public_callables() -> dict[str, Any]:
    """Every callable the package exports, keyed by its exported name.

    Derived from ``__all__`` rather than listed, so a builder added to the
    codec is held to the same contract without editing this file.
    """
    found = {}
    for name in package.__all__:
        value = getattr(package, name)
        if inspect.isfunction(value):
            found[name] = value
    return found


def _documented_classes(func: Any) -> set[str]:
    """The exception names a function's own ``Raises:`` block mentions.

    Read permissively - any capitalised ``Error``-shaped token inside the block
    counts - so this can only ever under-report a documented class, never
    invent one.
    """
    doc = inspect.getdoc(func) or ""
    match = re.search(r"^Raises:\s*$(.*?)(?=^\S|\Z)", doc, re.M | re.S)
    if match is None:
        return set()
    return set(re.findall(r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception))\b", match.group(1)))


def _has_raises_block(func: Any) -> bool:
    return re.search(r"^Raises:\s*$", inspect.getdoc(func) or "", re.M) is not None


class TestEveryPublicRefusalIsDocumented:
    """A class a caller can receive must be a class the docstring names."""

    @pytest.mark.parametrize(
        ("name", "args", "expected"),
        _REFUSALS,
        ids=[f"{n}-{e.__name__}-{i}" for i, (n, _, e) in enumerate(_REFUSALS)],
    )
    def test_the_escaping_class_is_named_in_the_functions_own_block(
        self, name: str, args: tuple[Any, ...], expected: type[BaseException]
    ) -> None:
        func = _public_callables()[name]
        with pytest.raises(expected) as caught:
            func(*args)
        # Grade the class the caller actually receives, which for the builders
        # is raised by _validate_id rather than by the function's own body.
        raised = type(caught.value).__name__
        documented = _documented_classes(func)
        assert raised in documented, (
            f"{name}{args!r} raises {raised}, which its Raises: block does not name "
            f"(it names {sorted(documented) or 'nothing'})"
        )

    def test_every_public_callable_carries_a_raises_block(self) -> None:
        """A block that does not exist leaves the tree-wide guard's population."""
        missing = sorted(name for name, func in _public_callables().items() if not _has_raises_block(func))
        assert not missing, f"public callables with no Raises: block, so ungraded tree-wide: {missing}"


class TestTheDocumentedHandlerIsWritable:
    """The class every parser docstring names must be one the package hands out."""

    def test_the_refusal_class_is_exported(self) -> None:
        assert "ProtocolError" in package.__all__
        assert getattr(package, "ProtocolError") is protocol.ProtocolError

    def test_a_wire_fault_is_catchable_as_documented(self) -> None:
        with pytest.raises(package.ProtocolError):
            package.parse_status_packet(b"\x00\x00", 1, 0)

    def test_a_caller_bug_is_not_reported_as_a_wire_fault(self) -> None:
        """The reason ProtocolError subclasses ValueError rather than replacing it."""
        with pytest.raises(ValueError) as caught:
            package.parse_status_packet(_WELL_FORMED, 0x1FF, 0)
        assert not isinstance(caught.value, protocol.ProtocolError)


class TestTheRuleIsNotVacuous:
    """The grading must fail on a block that omits a class it can receive."""

    def test_every_public_callable_is_exercised(self) -> None:
        exercised = {name for name, _, _ in _REFUSALS}
        assert exercised == set(_public_callables()), (
            "the refusal table and the exported surface have drifted; "
            f"table={sorted(exercised)} surface={sorted(_public_callables())}"
        )

    def test_a_block_that_omits_a_raised_class_is_reported(self) -> None:
        def omits() -> None:
            """Refuse something without saying so.

            Raises:
                ProtocolError: Never, actually.
            """
            raise TypeError("a class the block does not name")

        assert "TypeError" not in _documented_classes(omits)

    def test_a_block_that_names_the_raised_class_is_accepted(self) -> None:
        def names_it() -> None:
            """Refuse something and say so.

            Raises:
                TypeError: When the caller passes the wrong type.
            """
            raise TypeError("a class the block names")

        assert "TypeError" in _documented_classes(names_it)

    def test_a_class_named_outside_the_raises_block_is_not_credited(self) -> None:
        """Reading the whole docstring would credit a class the block never promised."""

        def elsewhere() -> None:
            """Name a class in prose without documenting it as a refusal.

            Args:
                value: Passing a :class:`str` here is a TypeError.

            Raises:
                ProtocolError: Never, actually.
            """
            raise TypeError("named in Args, not in Raises")

        assert "TypeError" not in _documented_classes(elsewhere)
        assert "ProtocolError" in _documented_classes(elsewhere)

    def test_a_missing_block_is_distinguishable_from_an_empty_one(self) -> None:
        def undocumented() -> None:
            """No Raises: section at all."""

        assert not _has_raises_block(undocumented)
        assert _documented_classes(undocumented) == set()


class TestWhatIsUnchanged:
    """Documenting a refusal must not move a byte the codec puts on the wire."""

    def test_the_accepted_frames_are_untouched(self) -> None:
        assert protocol.ping_packet(1) == b"\xff\xff\x01\x02\x01\xfb"
        assert protocol.build_packet(1, protocol.Instruction.PING) == b"\xff\xff\x01\x02\x01\xfb"
        assert protocol.read_packet(1, 0x38, 2)[:5] == b"\xff\xff\x01\x04\x02"

    def test_a_well_formed_status_packet_still_parses(self) -> None:
        assert protocol.parse_status_packet(_WELL_FORMED, 1, 0) == (0, b"")
