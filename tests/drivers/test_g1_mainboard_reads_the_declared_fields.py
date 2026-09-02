"""``G1Driver._on_mainboard`` writes the field names ``MainBoardState_`` declares.

``_on_mainboard`` reaches into the IDL message with
``getattr(msg, name, default)``.  That call cannot fail: a name the message
type does not declare yields the default, and the default is a well-formed
value that lands in the published record looking exactly like a reading.  So
a decoder that reads a name the IDL never had publishes a constant, and the
fleet card shows a plausible number forever.  This is the same failure mode
:mod:`tests.drivers.test_g1_lidar_state_reads_the_declared_fields` pins for
``LidarState_``; the mainboard record has to be held to the same rule.

The pattern here has three layers, matching that sibling file:

* A *faithful double* carrying exactly the declared field names and nothing
  else.  Reading an undeclared name off it produces the default, which is the
  defect, so these cells grade the decoder on any install.
* :data:`_DECLARED_MAINBOARD_FIELDS`, a frozen copy of the declaration, and a
  cell that checks it against the real ``MainBoardState_`` when the SDK *is*
  importable.  That is what keeps the double faithful as the IDL moves.
* A derivation over the decoder's own source, so a name added later is held
  to the same rule without anyone remembering to add a case.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
import types
from typing import Any

import pytest

from strands_robots.drivers.g1 import (
    _TOPIC_MAINBOARD,
    G1Driver,
)

#: Every field ``unitree_hg.msg.dds_.MainBoardState_`` declares, all four of
#: them six-element vectors: ``fan_state`` (``uint16``), ``temperature``
#: (``int16``), ``value`` (``float32``) and ``state`` (``uint32``).  Frozen
#: here because that SDK is installed from a git clone rather than PyPI, so it
#: cannot be a test dependency; :func:`test_the_frozen_declaration_matches_the_sdk`
#: proves this copy is still true wherever the SDK *is* importable.  If a
#: firmware update lands a field rename, the fidelity cell fails there first
#: and the frozen copy is rewritten to match.
_DECLARED_MAINBOARD_FIELDS: frozenset[str] = frozenset(
    {
        "fan_state",
        "temperature",
        "value",
        "state",
    }
)


def _mainboard_message(**overrides: Any) -> types.SimpleNamespace:
    """Return a stand-in carrying exactly the declared MainBoardState fields.

    Faithful in the one way that matters here: it declares the names the
    real message declares and no others, so a decoder reaching for a name
    the IDL does not have gets ``getattr``'s default from this object
    exactly as it would from the real one.  The defaults are six-element
    vectors like the real declaration, so a decoder that reads them and
    coerces will succeed end-to-end without needing a matching firmware
    on the box.
    """
    fields: dict[str, Any] = {
        "fan_state": [0] * 6,
        "temperature": [0.0] * 6,
        "value": [0.0] * 6,
        "state": [0] * 6,
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _getattr_names(func: Any) -> set[str]:
    """Return every literal attribute name ``func`` reads with ``getattr``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            continue
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            value = node.args[1].value
            if isinstance(value, str):
                names.add(value)
    return names


# =========================================================================
# Premises - the absence this suite is about is real.                     #
# =========================================================================


class TestThePremisesHold:
    """What makes an undeclared read silent, stated as checkable facts."""

    def test_the_double_declares_the_fields_and_no_others(self) -> None:
        """The stand-in is faithful in shape, which is what makes it useful."""
        msg = _mainboard_message()
        assert set(vars(msg)) == set(_DECLARED_MAINBOARD_FIELDS)

    @pytest.mark.parametrize(
        "undeclared",
        ["fan_speed", "cpu_temperature", "bms_state", "sys_bat_state", "sys_state", "tick"],
    )
    def test_the_declaration_carries_no_such_field(self, undeclared: str) -> None:
        """The neon-side reads that are not on this firmware's layout.

        The neon-the-g1 ``g1_mainboard`` port read each of these off the IDL
        behind a ``hasattr`` gate, so a name the message did not declare was
        simply absent from its output.  ``sys_state`` and ``tick`` are the
        two that reached this decoder without that gate, where an
        undeclared read lands the ``getattr`` default in the record instead
        (``tick`` is declared on ``LowState_``, not here).  If a future
        firmware adds one back, this cell fires so the declaration and the
        decoder can be updated together.
        """
        assert undeclared not in _DECLARED_MAINBOARD_FIELDS
        assert not hasattr(_mainboard_message(), undeclared)

    def test_an_undeclared_read_yields_the_default_rather_than_raising(self) -> None:
        """This is the whole mechanism: the miss is silent, not an error."""
        msg = _mainboard_message(state=[1] * 6)
        assert getattr(msg, "fan_speed", 0) == 0
        assert getattr(msg, "bms_state", 0) == 0
        assert getattr(msg, "sys_state", 0) == 0


# =========================================================================
# The regression - a reading the message carries reaches the cache.       #
# =========================================================================


class TestTheDecoderReadsTheDeclaredNames:
    """A reading the message carries has to survive into the record."""

    def test_a_healthy_fan_reaches_the_record(self) -> None:
        """``fan_state`` is a vector of fan flags, one per fan on the board."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_mainboard(_mainboard_message(fan_state=[1, 1, 1, 0]))
        assert driver._mainboard is not None
        assert driver._mainboard["fan_state"] == [1, 1, 1, 0]

    def test_a_thermistor_reading_reaches_the_record(self) -> None:
        """``temperature`` is a per-thermistor vector on the mainboard."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_mainboard(_mainboard_message(temperature=[42.5, 44.0, 41.75]))
        assert driver._mainboard is not None
        assert driver._mainboard["temperature"] == pytest.approx([42.5, 44.0, 41.75])

    def test_the_state_vector_reaches_the_record(self) -> None:
        """``state`` is a ``uint32`` vector; the IDL gives it no semantics.

        Published under the vendor's own name rather than interpreted, so
        a caller reads what the board sent.  The decoder carrying it at
        all is the point: a decoder that reads a name the IDL does not
        declare drops this reading and reports a constant instead.
        """
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_mainboard(_mainboard_message(state=[2, 0, 0, 0, 0, 1]))
        assert driver._mainboard is not None
        assert driver._mainboard["state"] == [2, 0, 0, 0, 0, 1]

    def test_the_value_vector_reaches_the_record(self) -> None:
        """``value`` is a ``float32`` vector, likewise undocumented."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_mainboard(_mainboard_message(value=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]))
        assert driver._mainboard is not None
        assert driver._mainboard["value"] == pytest.approx([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

    def test_the_wall_time_is_stamped(self) -> None:
        """``t`` is the wall time of decode, not a field on the message.

        The IDL declares four vectors and no clock of any kind, so the
        decoder stamps ``time.time()`` on every message.  This is how the mesh's health
        chip knows whether the reading is stale.
        """
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_mainboard(_mainboard_message())
        assert driver._mainboard is not None
        assert driver._mainboard["t"] is not None
        assert driver._mainboard["t"] > 0.0

    @pytest.mark.parametrize("renamed", sorted(_DECLARED_MAINBOARD_FIELDS))
    def test_a_missing_field_lands_none_not_a_plausible_reading(self, renamed: str) -> None:
        """A firmware that renames a field yields ``None`` for it here.

        ``_on_mainboard`` reads through ``getattr(msg, name, None)`` at
        every field, so a rename does not write a typed default into the
        record.  That matters because every plausible default is also a
        real reading: ``0`` looks like a healthy system, ``[]`` looks like
        a unit reporting no fans.  ``None`` is the one value that cannot
        be mistaken for a measurement, and the ``g1_mainboard`` verb
        passes it through.  A stand-in that omits one attribute models
        exactly that miss, one field at a time.
        """
        fields = {name: [1] * 6 for name in _DECLARED_MAINBOARD_FIELDS if name != renamed}
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_mainboard(types.SimpleNamespace(**fields))
        assert driver._mainboard is not None
        assert driver._mainboard[renamed] is None
        for present in fields:
            assert driver._mainboard[present] is not None


# =========================================================================
# The subscription plan - the topic and IDL are wired.                    #
# =========================================================================


class TestTheDriverSubscribesTheTopic:
    """Every cached record starts with a subscription: this one is wired."""

    def test_the_topic_constant_names_the_mainboard_topic(self) -> None:
        """``_TOPIC_MAINBOARD`` is the wire the firmware publishes on."""
        assert _TOPIC_MAINBOARD == "rt/mainboardstate"

    def test_the_subscription_plan_includes_mainboard(self) -> None:
        """``G1Driver._subscription_plan`` names the mainboard topic and IDL.

        The DDS engine walks the plan at connect time to open one
        subscriber per entry; a plan that does not name the topic means
        the driver connects and no ``MainBoardState_`` messages ever
        arrive, so ``_on_mainboard`` never fires and ``_mainboard``
        stays ``None`` on the mesh forever.
        """
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        plan = driver._subscription_plan()
        topics = [row[0] for row in plan]
        assert _TOPIC_MAINBOARD in topics

        # The IDL class name and decoder must match the handler.
        for topic, (idl_module, idl_class), decoder in plan:
            if topic == _TOPIC_MAINBOARD:
                assert idl_module == "unitree_sdk2py.idl.unitree_hg.msg.dds_"
                assert idl_class == "MainBoardState_"
                assert decoder == driver._on_mainboard
                break


# =========================================================================
# The derivation - a name added later is held to the same rule.           #
# =========================================================================


class TestEveryNameReadIsDeclared:
    """Derived from the decoder's source, so a new read is graded on arrival."""

    def test_the_decoder_reads_only_declared_fields(self) -> None:
        """Any name here that the IDL does not declare is a silent constant."""
        read = _getattr_names(G1Driver._on_mainboard)
        assert read, "the derivation found no getattr reads to grade"
        undeclared = read - _DECLARED_MAINBOARD_FIELDS
        assert not undeclared, (
            f"_on_mainboard reads {sorted(undeclared)} which MainBoardState_ does "
            "not declare on the current firmware layout. Either add the field to "
            "_DECLARED_MAINBOARD_FIELDS (and to the double) or drop the read. "
            "Refs strands-labs/robots#358."
        )

    def test_at_least_one_declared_field_is_read(self) -> None:
        """Non-vacuity: the rule above is not satisfied by reading nothing."""
        read = _getattr_names(G1Driver._on_mainboard)
        assert read & _DECLARED_MAINBOARD_FIELDS, (
            f"_on_mainboard reads none of the declared MainBoardState fields "
            f"({sorted(_DECLARED_MAINBOARD_FIELDS)}). This would land a record "
            "of all-default values on the mesh forever."
        )


# =========================================================================
# Controls - a malformed message does not crash the DDS thread.           #
# =========================================================================


class TestTheDecoderIsResilient:
    """The DDS thread has to survive a message it cannot read."""

    def test_a_malformed_message_is_swallowed(self) -> None:
        """The decoder catches, logs at debug, and moves on."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_mainboard("not a message")
        driver._on_mainboard(None)
        driver._on_mainboard(object())
        # ``_mainboard`` may be None (no successful decode) or hold whatever
        # the last coercion-tolerant read produced -- but the driver still
        # runs.


# =========================================================================
# Fidelity - the frozen declaration is checked where the SDK exists.      #
# =========================================================================


class TestTheFrozenDeclarationIsTrue:
    """Without this the double could drift into agreeing with a bug."""

    def test_the_frozen_declaration_matches_the_sdk(self) -> None:
        """Compare the frozen copy against the real IDL when it is importable.

        ``unitree_sdk2py`` is installed from a git clone rather than PyPI,
        so it is absent on an ordinary contributor machine and in CI;
        skipping there is the point of freezing the declaration in the
        first place.  On a host with the SDK present (an office bring-up
        box), this cell holds the frozen copy honest against the real
        message the firmware ships.
        """
        dds = pytest.importorskip(
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            reason="unitree_sdk2py is installed from a git clone, not PyPI",
        )
        declared = {field.name for field in dataclasses.fields(dds.MainBoardState_)}
        # Equality, not containment.  A subset check sees only the half of
        # the drift that removes a field; the other half is a firmware that
        # declares one this side never reads, which is a reading dropped on
        # the floor rather than a constant published - both need a decision
        # before landing, so both fail here.
        assert declared == set(_DECLARED_MAINBOARD_FIELDS), (
            f"MainBoardState_ declares {sorted(declared)}; the frozen copy says "
            f"{sorted(_DECLARED_MAINBOARD_FIELDS)}. Fields gone: "
            f"{sorted(set(_DECLARED_MAINBOARD_FIELDS) - declared)}; fields the decoder "
            f"does not read: {sorted(declared - set(_DECLARED_MAINBOARD_FIELDS))}. "
            "Update _DECLARED_MAINBOARD_FIELDS and _on_mainboard together."
        )
