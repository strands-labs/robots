"""The pressure decoder reads the field names ``PressSensorState_`` declares.

``_on_pressure`` reaches into the IDL message with
``getattr(msg, name, default)``.  That call cannot fail: a name the message
type does not declare yields the default, and the default is a well-formed
value that lands in the published record looking exactly like a reading.
So a decoder that reads a name the IDL never had publishes a constant, and
the fleet card shows a plausible number forever.

The suite has three layers because the SDK that owns the IDL is not on
PyPI and so cannot be a test dependency:

* A *faithful double* carrying exactly the declared field names and
  nothing else.  Reading an undeclared name off it produces the default,
  which is the defect, so these cells grade the decoder on any install.
* :data:`_DECLARED_PRESSURE_FIELDS`, a frozen copy of the declaration,
  and a cell that checks it against the real ``PressSensorState_`` when
  the SDK *is* importable.  That is what keeps the double faithful as
  the IDL moves.
* A derivation over the decoder's own source, so a name added later is
  held to the same rule without anyone remembering to add a case.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
import types
from typing import Any

import pytest

from strands_robots.drivers.g1 import _TOPIC_PRESSURE, G1Driver

#: Every field ``unitree_hg.msg.dds_.PressSensorState_`` declares, as
#: shipped by ``unitree_sdk2py`` 1.0.1.  Frozen here because that SDK is
#: installed from a git clone rather than PyPI, so it cannot be a test
#: dependency; :func:`test_the_frozen_declaration_matches_the_sdk` proves
#: this copy is still true wherever the SDK *is* importable.
_DECLARED_PRESSURE_FIELDS: frozenset[str] = frozenset(
    {
        "pressure",
        "temperature",
        "lost",
        "reserve",
    }
)


def _pressure_message(**overrides: Any) -> types.SimpleNamespace:
    """Return a stand-in carrying exactly the declared PressSensorState fields.

    Faithful in the one way that matters here: it declares the names the
    real message declares and no others, so a decoder reaching for a
    name the IDL does not have gets ``getattr``'s default from this
    object exactly as it would from the real one.
    """
    fields: dict[str, Any] = {
        "pressure": [0.0] * 12,
        "temperature": [0.0] * 12,
        "lost": 0,
        "reserve": 0,
    }
    fields.update(overrides)
    # Sanity: reject overrides that widen the field set the double declares.
    assert set(fields) == set(_DECLARED_PRESSURE_FIELDS), (
        f"stand-in got fields the IDL does not declare: {set(fields) - _DECLARED_PRESSURE_FIELDS}"
    )
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
        msg = _pressure_message()
        assert set(vars(msg)) == set(_DECLARED_PRESSURE_FIELDS)

    @pytest.mark.parametrize("undeclared", ["voltage", "timestamp", "tick"])
    def test_the_declaration_carries_no_such_field(self, undeclared: str) -> None:
        """The neon-side reads that are not on the current IDL layout.

        The neon-the-g1 ``g1_pressure`` port additionally read these
        fields directly off ``PressSensorState_``.  On the layout the
        current firmware ships with (``float32[12]`` vectors for
        ``pressure`` / ``temperature`` and ``uint32`` scalars for
        ``lost`` / ``reserve``, and nothing else) they are not declared,
        so a decoder reading them would land the ``getattr`` default in
        the record forever.  If a future firmware adds one back, this
        cell fires so the declaration and the decoder can be updated
        together.
        """
        assert undeclared not in _DECLARED_PRESSURE_FIELDS
        assert not hasattr(_pressure_message(), undeclared)

    def test_an_undeclared_read_yields_the_default_rather_than_raising(self) -> None:
        """This is the whole mechanism: the miss is silent, not an error."""
        msg = _pressure_message(lost=1)
        assert getattr(msg, "voltage", 0) == 0
        assert getattr(msg, "tick", 0) == 0


# =========================================================================
# The regression - a reading the message carries reaches the cache.       #
# =========================================================================


class TestTheDecoderReadsTheDeclaredNames:
    """A reading the message carries has to survive into the record."""

    def test_a_healthy_pressure_vector_reaches_the_record(self) -> None:
        """``pressure`` is a per-sensor vector of raw pressure readings."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        reading = [1.5, 2.1, 0.8, 0.9, 1.2, 1.4, 1.6, 2.3, 0.7, 1.0, 1.1, 1.3]
        driver._on_pressure(_pressure_message(pressure=reading))
        assert driver._pressure is not None
        assert driver._pressure["pressure"] == pytest.approx(reading)

    def test_a_thermistor_reading_reaches_the_record(self) -> None:
        """``temperature`` is a per-sensor vector on the pressure board."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        reading = [
            25.5,
            26.0,
            25.8,
            25.9,
            26.1,
            25.7,
            25.6,
            26.2,
            25.5,
            25.9,
            26.0,
            25.8,
        ]
        driver._on_pressure(_pressure_message(temperature=reading))
        assert driver._pressure is not None
        assert driver._pressure["temperature"] == pytest.approx(reading)

    def test_the_lost_counter_reaches_the_record(self) -> None:
        """``lost`` is the packet-loss counter; a caller reads it as ``int``."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_pressure(_pressure_message(lost=99))
        assert driver._pressure is not None
        assert driver._pressure["lost"] == 99

    def test_the_reserve_scalar_reaches_the_record(self) -> None:
        """``reserve`` is a ``uint32`` scalar the IDL declares next to ``lost``."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_pressure(_pressure_message(reserve=7))
        assert driver._pressure is not None
        assert driver._pressure["reserve"] == 7

    def test_the_wall_time_is_stamped(self) -> None:
        """``t`` is the wall time of decode, not a field on the message.

        The IDL does not carry a wall-time field, so the decoder stamps
        ``time.time()`` on every message.  This is how the mesh's
        health chip knows whether the reading is stale.
        """
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_pressure(_pressure_message())
        assert driver._pressure is not None
        assert driver._pressure["t"] is not None
        assert driver._pressure["t"] > 0.0

    def test_a_missing_scalar_field_lands_none_not_zero(self) -> None:
        """A firmware that renames ``lost`` yields ``None`` on this side.

        ``_on_pressure`` reads through ``getattr(msg, name, None)`` at
        every field so a firmware rename does not silently write ``0``
        into the record (which would look like a wire in perfect health
        on the current layout).  A message stand-in that omits the
        attribute entirely models exactly that miss.
        """
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        msg = types.SimpleNamespace(
            pressure=[1.0] * 12,
            temperature=[25.0] * 12,
            # lost / reserve deliberately absent
        )
        driver._on_pressure(msg)
        assert driver._pressure is not None
        assert driver._pressure["lost"] is None
        assert driver._pressure["reserve"] is None

    def test_a_missing_vector_field_lands_none_not_empty(self) -> None:
        """An undeclared ``pressure`` lands ``None`` in the record.

        An empty list would be a plausible reading (a build with no
        sensors declared), but is not the same as a firmware rename
        where the field is absent entirely.  ``None`` is the decidable
        value for the missing case; the ``g1_pressure`` verb passes it
        through.
        """
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        msg = types.SimpleNamespace(
            temperature=[25.0] * 12,
            lost=0,
            reserve=0,
            # pressure deliberately absent
        )
        driver._on_pressure(msg)
        assert driver._pressure is not None
        assert driver._pressure["pressure"] is None


# =========================================================================
# The subscription plan - the topic and IDL are wired.                    #
# =========================================================================


class TestTheDriverSubscribesTheTopic:
    """Every cached record starts with a subscription: this one is wired."""

    def test_the_topic_constant_names_the_pressure_topic(self) -> None:
        """``_TOPIC_PRESSURE`` is the wire the firmware publishes on."""
        assert _TOPIC_PRESSURE == "rt/pressuresensorstate"

    def test_the_subscription_plan_includes_pressure(self) -> None:
        """``G1Driver._subscription_plan`` names the pressure topic and IDL.

        The DDS engine walks the plan at connect time to open one
        subscriber per entry; a plan that does not name the topic means
        the driver connects and no ``PressSensorState_`` messages ever
        arrive, so ``_on_pressure`` never fires and ``_pressure`` stays
        ``None`` on the mesh forever.
        """
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        plan = driver._subscription_plan()
        topics = [row[0] for row in plan]
        assert _TOPIC_PRESSURE in topics

        # The IDL class name and decoder must match the handler.
        for topic, (idl_module, idl_class), decoder in plan:
            if topic == _TOPIC_PRESSURE:
                assert idl_module == "unitree_sdk2py.idl.unitree_hg.msg.dds_"
                assert idl_class == "PressSensorState_"
                assert decoder == driver._on_pressure
                break


# =========================================================================
# The derivation - a name added later is held to the same rule.           #
# =========================================================================


class TestEveryNameReadIsDeclared:
    """Derived from the decoder's source, so a new read is graded on arrival."""

    def test_the_decoder_reads_only_declared_fields(self) -> None:
        """Any name here that the IDL does not declare is a silent constant."""
        read = _getattr_names(G1Driver._on_pressure)
        assert read, "the derivation found no getattr reads to grade"
        undeclared = read - _DECLARED_PRESSURE_FIELDS
        assert not undeclared, (
            f"_on_pressure reads {sorted(undeclared)} which PressSensorState_ does "
            "not declare on the current firmware layout. Either add the field to "
            "_DECLARED_PRESSURE_FIELDS (and to the double) or drop the read. "
            "Refs strands-labs/robots#358."
        )

    def test_at_least_one_declared_field_is_read(self) -> None:
        """Non-vacuity: the rule above is not satisfied by reading nothing."""
        read = _getattr_names(G1Driver._on_pressure)
        assert read & _DECLARED_PRESSURE_FIELDS, (
            f"_on_pressure reads none of the declared PressSensorState fields "
            f"({sorted(_DECLARED_PRESSURE_FIELDS)}). This would land a record "
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
        driver._on_pressure("not a message")
        driver._on_pressure(None)
        driver._on_pressure(object())
        # ``_pressure`` may be None (no successful decode) or hold whatever
        # the last coercion-tolerant read produced -- but the driver still
        # runs.


# =========================================================================
# Fidelity - the frozen declaration is checked where the SDK exists.      #
# =========================================================================


class TestTheFrozenDeclarationIsTrue:
    """Without this the double could drift into agreeing with a bug."""

    def test_the_frozen_declaration_matches_the_sdk(self) -> None:
        """Compare the frozen copy against the real IDL when it is importable.

        ``unitree_sdk2py`` is installed from a git clone rather than
        PyPI, so it is absent on an ordinary contributor machine and in
        CI; skipping there is the point of freezing the declaration in
        the first place.  On a host with the SDK present (an office
        bring-up box), this cell holds the frozen copy honest against
        the real message the firmware ships.
        """
        dds = pytest.importorskip(
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            reason="unitree_sdk2py is installed from a git clone, not PyPI",
        )
        declared = {field.name for field in dataclasses.fields(dds.PressSensorState_)}
        # The frozen set names the fields the driver reads; on a firmware
        # that declares a superset this cell fails, so a rewriter has to
        # decide which fields to lift into the cache before landing.
        assert declared == set(_DECLARED_PRESSURE_FIELDS), (
            f"PressSensorState_ dropped fields: "
            f"{sorted(set(_DECLARED_PRESSURE_FIELDS) - declared)}. "
            "Update _DECLARED_PRESSURE_FIELDS to the real declaration."
        )
