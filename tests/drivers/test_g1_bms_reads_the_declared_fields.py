"""The BMS decoder reads the field names ``BmsState_`` declares.

``_on_bms`` reaches into the IDL message with ``getattr(msg, name, default)``.
That call cannot fail: a name the message type does not declare yields the
default, and a *typed* default is a well-formed value that lands in the
published record looking exactly like a reading.  So a decoder that reads a
name the IDL never had publishes a constant, and the fleet card shows a
plausible reading forever.

That is not hypothetical here.  ``_on_bms`` read ``getattr(msg, "charge", 0)``
into a ``charging`` flag, and ``BmsState_`` declares no charge field of any
spelling, so every G1 reported ``charging=False`` - indistinguishable from a
pack measured to be discharging, and published on the mesh health wire by
the health reader in :mod:`strands_robots.mesh.sensors`.  Three of the driver's
six DDS decoders had a declared-fields grader like this one; this decoder was
one of the two without, which is why the read went unnoticed.

The suite has three layers because the SDK that owns the IDL is not on PyPI
and so cannot be a test dependency:

* A *faithful double* carrying exactly the declared field names and nothing
  else.  Reading an undeclared name off it produces the default, which is
  the defect, so these cells grade the decoder on any install.
* :data:`_DECLARED_BMS_FIELDS`, a frozen copy of the declaration, and a cell
  that checks it against the real ``BmsState_`` when the SDK *is*
  importable.  That is what keeps the double faithful as the IDL moves.
* A derivation over the decoder's own source, so a name added later is held
  to the same rule without anyone remembering to add a case.

Sibling files: :mod:`tests.drivers.test_g1_mainboard_reads_the_declared_fields`,
:mod:`tests.drivers.test_g1_pressure_reads_the_declared_fields` and
:mod:`tests.drivers.test_g1_lidar_state_reads_the_declared_fields`.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
import types
from typing import Any

import pytest

from strands_robots.drivers.g1 import _TOPIC_BMS, G1Driver

#: Every field ``unitree_hg.msg.dds_.BmsState_`` declares.  Frozen here
#: because that SDK is installed from a git clone rather than PyPI, so it
#: cannot be a test dependency; :func:`test_the_frozen_declaration_matches_the_sdk`
#: proves this copy is still true wherever the SDK *is* importable.  The
#: decoder reads three of these (``soc``, ``current``, ``cycle``); the rest
#: are declared but uncached, which the ``g1_battery`` module docstring
#: records as a decision that belongs to the decoder rather than the verb.
_DECLARED_BMS_FIELDS: frozenset[str] = frozenset(
    {
        "version_high",
        "version_low",
        "fn",
        "cell_vol",
        "bmsvoltage",
        "current",
        "soc",
        "soh",
        "temperature",
        "cycle",
        "manufacturer_date",
        "bmsstate",
        "reserve",
    }
)


def _bms_message(**overrides: Any) -> types.SimpleNamespace:
    """Return a stand-in carrying exactly the declared BmsState fields.

    Faithful in the one way that matters here: it declares the names the real
    message declares and no others, so a decoder reaching for a name the IDL
    does not have gets ``getattr``'s default from this object exactly as it
    would from the real one.
    """
    fields: dict[str, Any] = {
        "version_high": 1,
        "version_low": 0,
        "fn": 0,
        "cell_vol": [4000] * 40,
        "bmsvoltage": [50000, 0, 0],
        "current": 0,
        "soc": 100,
        "soh": 100,
        "temperature": [25] * 12,
        "cycle": 0,
        "manufacturer_date": 0,
        "bmsstate": [0] * 5,
        "reserve": [0] * 3,
    }
    fields.update(overrides)
    # Sanity: reject overrides that widen the field set the double declares.
    assert set(fields) == set(_DECLARED_BMS_FIELDS), (
        f"stand-in got fields the IDL does not declare: {set(fields) - _DECLARED_BMS_FIELDS}"
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


def _decode(**overrides: Any) -> dict[str, Any]:
    """Return the ``_battery`` record one decode of the double produces."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._on_bms(_bms_message(**overrides))
    assert driver._battery is not None, "the decoder wrote no record at all"
    return driver._battery


# =========================================================================
# Premises - the absence this suite is about is real.                     #
# =========================================================================


class TestThePremisesHold:
    """What makes an undeclared read silent, stated as checkable facts."""

    def test_the_double_declares_the_fields_and_no_others(self) -> None:
        """The stand-in is faithful in shape, which is what makes it useful."""
        assert set(vars(_bms_message())) == set(_DECLARED_BMS_FIELDS)

    @pytest.mark.parametrize("undeclared", ["charge", "charging", "status", "charge_state"])
    def test_the_declaration_carries_no_charge_field(self, undeclared: str) -> None:
        """No spelling of a charge flag is declared on this message.

        ``charge`` is the name the decoder used to read.  ``status`` is the
        near miss: it *is* declared on ``unitree_go.msg.dds_.BmsState_``, the
        quadruped's BMS message, and this driver subscribes the humanoid's
        ``unitree_hg`` one - so a reader who checked the wrong IDL would
        conclude a status word was available here.
        """
        assert undeclared not in _DECLARED_BMS_FIELDS
        assert not hasattr(_bms_message(), undeclared)

    def test_an_undeclared_read_yields_a_plausible_false_not_an_error(self) -> None:
        """This is the whole mechanism: the miss is silent, and it lies.

        ``bool(getattr(msg, "charge", 0))`` on a message with no charge
        field is ``False``, which is a perfectly well-formed answer to "is
        this pack charging" - so nothing downstream can tell it apart from
        a measurement.  That is why the defect survived: the read could not
        raise and the value could not look wrong.
        """
        msg = _bms_message(soc=42)
        assert bool(getattr(msg, "charge", 0)) is False
        assert getattr(msg, "charge", 0.0) == 0.0


# =========================================================================
# The regression - the record carries readings and nothing else.          #
# =========================================================================


class TestTheRecordCarriesOnlyWhatTheMessageSaid:
    """A key in the record has to trace back to a field on the wire."""

    def test_the_record_carries_no_charge_flag(self) -> None:
        """The defect, pinned: no charge key on a decode of a real layout.

        Reporting one would mean answering a question the message does not
        answer.  A firmware that does declare a charge field is one read in
        ``_on_bms`` plus one key on the ``g1_battery`` envelope, and this
        cell is where that decision gets recorded.
        """
        record = _decode()
        assert "charging" not in record, (
            f"_on_bms published {sorted(record)}; BmsState_ declares no charge field, so a "
            "charging key can only be the getattr default dressed as a reading."
        )

    def test_the_record_names_exactly_the_readings_the_decoder_lifts(self) -> None:
        """Three declared fields plus the decode timestamp, nothing more."""
        assert set(_decode()) == {"pct", "current", "cycle", "t"}

    def test_the_state_of_charge_reaches_the_record(self) -> None:
        """``soc`` is the percentage the motion gates read as ``pct``."""
        assert _decode(soc=77)["pct"] == pytest.approx(77.0)

    def test_the_pack_current_reaches_the_record(self) -> None:
        """``current`` is signed; the sign is the vendor's, not converted."""
        assert _decode(current=-3210)["current"] == pytest.approx(-3210.0)

    def test_the_cycle_count_reaches_the_record(self) -> None:
        """``cycle`` is the pack's charge-cycle count as an integer."""
        assert _decode(cycle=142)["cycle"] == 142

    def test_the_wall_time_is_stamped(self) -> None:
        """``t`` is the host clock at decode, not a field on the message.

        ``BmsState_`` declares ``manufacturer_date`` and no clock, so the
        decoder stamps ``time.time()``.  This is how the mesh's health chip
        knows whether the reading is stale.
        """
        assert _decode()["t"] > 0.0

    @pytest.mark.parametrize("renamed", ["soc", "current", "cycle"])
    def test_a_missing_field_lands_none_not_a_plausible_zero(self, renamed: str) -> None:
        """A firmware that renames a field yields ``None``, never ``0``.

        ``0.0`` amps is a real reading (an idle pack) and ``0`` cycles is a
        real reading (a new pack), so a typed default here is exactly as
        undetectable as the charge flag was.  ``None`` is the one value that
        cannot be mistaken for a measurement.
        """
        fields = {name: 1 for name in _DECLARED_BMS_FIELDS if name != renamed}
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_bms(types.SimpleNamespace(**fields))
        assert driver._battery is not None
        key = {"soc": "pct", "current": "current", "cycle": "cycle"}[renamed]
        assert driver._battery[key] is None

    def test_a_non_numeric_field_lands_none_rather_than_raising(self) -> None:
        """A value that will not coerce is a miss, not a dead DDS thread."""
        record = _decode(soc="n/a")
        assert record["pct"] is None


# =========================================================================
# The derivation - a name added later is held to the same rule.           #
# =========================================================================


class TestEveryNameReadIsDeclared:
    """Derived from the decoder's source, so a new read is graded on arrival."""

    def test_the_decoder_reads_only_declared_fields(self) -> None:
        """Any name here that the IDL does not declare is a silent constant."""
        read = _getattr_names(G1Driver._on_bms)
        assert read, "the derivation found no getattr reads to grade"
        undeclared = read - _DECLARED_BMS_FIELDS
        assert not undeclared, (
            f"_on_bms reads {sorted(undeclared)} which BmsState_ does not declare. "
            "Either add the field to _DECLARED_BMS_FIELDS (and to the double) or "
            "drop the read; a read with a typed default publishes a constant."
        )

    def test_at_least_one_declared_field_is_read(self) -> None:
        """Non-vacuity: the rule above is not satisfied by reading nothing."""
        read = _getattr_names(G1Driver._on_bms)
        assert read & _DECLARED_BMS_FIELDS, (
            f"_on_bms reads none of the declared BmsState fields "
            f"({sorted(_DECLARED_BMS_FIELDS)}). This would land a record of "
            "all-None values on the mesh forever."
        )


# =========================================================================
# Controls - a malformed message does not crash the DDS thread.           #
# =========================================================================


class TestTheDecoderIsResilient:
    """The DDS thread has to survive a message it cannot read."""

    def test_a_malformed_message_is_swallowed(self) -> None:
        """The decoder catches, logs at debug, and moves on."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_bms("not a message")
        driver._on_bms(None)
        driver._on_bms(object())
        # ``_battery`` may be None (no successful decode) or hold whatever the
        # last coercion-tolerant read produced -- but the driver still runs.


# =========================================================================
# The subscription plan - the topic and IDL are wired.                    #
# =========================================================================


class TestTheDriverSubscribesTheTopic:
    """Every cached record starts with a subscription: this one is wired."""

    def test_the_topic_constant_names_the_bms_topic(self) -> None:
        """``_TOPIC_BMS`` is the wire the firmware publishes the pack on."""
        assert _TOPIC_BMS == "rt/lf/bmsstate"

    def test_the_subscription_plan_wires_the_humanoid_idl(self) -> None:
        """The plan names ``unitree_hg``, which is what fixes the field set.

        ``unitree_go.msg.dds_.BmsState_`` is a different declaration under
        the same class name, so the module the plan names is what decides
        which fields exist.  A plan that pointed at the quadruped's IDL
        would make the frozen declaration in this file wrong.
        """
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        for topic, (idl_module, idl_class), decoder in driver._subscription_plan():
            if topic == _TOPIC_BMS:
                assert idl_module == "unitree_sdk2py.idl.unitree_hg.msg.dds_"
                assert idl_class == "BmsState_"
                assert decoder == driver._on_bms
                break
        else:  # pragma: no cover - the assertion below names the failure
            raise AssertionError(f"{_TOPIC_BMS} is not in the subscription plan")


# =========================================================================
# Fidelity - the frozen declaration is checked where the SDK exists.      #
# =========================================================================


class TestTheFrozenDeclarationIsTrue:
    """Without this the double could drift into agreeing with a bug."""

    def test_the_frozen_declaration_matches_the_sdk(self) -> None:
        """Compare the frozen copy against the real IDL when it is importable.

        ``unitree_sdk2py`` is installed from a git clone rather than PyPI, so
        it is absent on an ordinary contributor machine and in CI; skipping
        there is the point of freezing the declaration in the first place.
        On a host with the SDK present this cell holds the frozen copy honest
        against the real message the firmware ships.

        Equality, not containment: a subset check sees only the drift that
        removes a field, and the other direction - a field declared that this
        side never reads - is a reading dropped on the floor.  Both need a
        decision before landing.
        """
        dds = pytest.importorskip(
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            reason="unitree_sdk2py is installed from a git clone, not PyPI",
        )
        declared = {field.name for field in dataclasses.fields(dds.BmsState_)}
        assert declared == set(_DECLARED_BMS_FIELDS), (
            f"BmsState_ declares {sorted(declared)}; the frozen copy says "
            f"{sorted(_DECLARED_BMS_FIELDS)}. Fields gone: "
            f"{sorted(set(_DECLARED_BMS_FIELDS) - declared)}; fields added: "
            f"{sorted(declared - set(_DECLARED_BMS_FIELDS))}. Update "
            "_DECLARED_BMS_FIELDS, the double and _on_bms together."
        )
