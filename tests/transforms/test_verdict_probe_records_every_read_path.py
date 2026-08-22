"""The verdict probe records every way a mapping hands out a stored value.

:class:`~strands_robots.transforms.base.DatasetTransform.transform` decides
whether a re-validation run may be reported as gated by wrapping the source
episode in a probe and asking, afterwards, whether the verdict consulted an
``observation.images.*`` column. That accusation is only honest if the probe
observed everything the verdict observed: a value-reaching read the probe
cannot see makes "consulted no image column" mean "read nothing" and "read
everything through a path I do not intercept" indistinguishably, and only the
first is true of a vacuous gate.

So the graded surface here is ``dict`` itself, partitioned exhaustively: every
member that can hand a stored value to a caller must be intercepted and must
record, and every member that cannot must not falsely record (a false record
claims a gate the verdict never applied, which is the unsafe direction). The
partition is asserted against ``vars(dict)`` rather than hand-listed alone, so
a member a future interpreter adds is unclassified and fails here instead of
becoming a silent hole.
"""

from collections.abc import Hashable
from typing import Any

import pytest

from strands_robots.transforms.base import _KeyRecordingEpisode

IMG = "observation.images.cam"
EPISODE: dict[str, Any] = {IMG: "PIXELS", "observation.state": [1.0], "action": [2.0], "timestamp": [0.0]}

# ``dict`` members that can hand a value stored in the mapping to a caller.
# Each must be overridden by the probe and must record.
VALUE_BEARING = frozenset(
    {
        "__getitem__",
        "get",
        "pop",
        "popitem",
        "setdefault",
        "items",
        "values",
        "__eq__",
        "__ne__",
        "copy",
        "__or__",
        "__ror__",
    }
)

# ``dict`` members that provably cannot hand a stored value out, with the
# reason - so a reader can check the classification rather than trust it.
CANNOT_HAND_OUT_A_VALUE: dict[str, str] = {
    "__contains__": "answers with a bool",
    "__len__": "answers with a count",
    "__sizeof__": "answers with a byte count",
    "__iter__": "yields keys",
    "__reversed__": "yields keys",
    "keys": "yields keys",
    "__setitem__": "stores, returns None",
    "__delitem__": "removes, returns None",
    "clear": "empties, returns None",
    "update": "stores, returns None",
    "__ior__": "stores in place and returns the probe itself, so any later read of a value goes through one of the recorded paths",
    "fromkeys": "classmethod building a new mapping from keys, never from this one's values",
    "__init__": "construction",
    "__new__": "construction",
    "__class_getitem__": "typing subscription of the class",
    "__doc__": "the class docstring",
    "__getattribute__": "attribute access, not item access",
    "__hash__": "None on dict",
    "__lt__": "dict is unordered - raises TypeError",
    "__le__": "dict is unordered - raises TypeError",
    "__gt__": "dict is unordered - raises TypeError",
    "__ge__": "dict is unordered - raises TypeError",
    "__repr__": (
        "a formatted rendering rather than a value handed to the caller, and recording on it would let an "
        "unrelated debug log of the episode claim the gate - the over-claiming direction"
    ),
}


def _probe() -> _KeyRecordingEpisode:
    return _KeyRecordingEpisode(dict(EPISODE))


ALL_KEYS = frozenset(EPISODE)


def _read_popitem(d: _KeyRecordingEpisode) -> set[str]:
    key, _value = d.popitem()
    return {key}


# One driver per value-bearing member, keyed by the member it exercises so the
# table and the partition above cannot drift apart. Each performs the read and
# returns the keys whose values it received, which is the property under test:
# a read is recorded when every key it handed over is in ``consulted``.
VALUE_BEARING_READS: dict[str, Any] = {
    "__getitem__": lambda d: {IMG} if d[IMG] else set(),
    "get": lambda d: {IMG} if d.get(IMG) else set(),
    "pop": lambda d: {IMG} if d.pop(IMG) else set(),
    # ``popitem`` hands over whichever key it removed, not one the caller names.
    "popitem": _read_popitem,
    "setdefault": lambda d: {IMG} if d.setdefault(IMG) else set(),
    "items": lambda d: {k for k, _ in d.items()},
    "values": lambda d: ALL_KEYS if list(d.values()) else set(),
    "__eq__": lambda d: ALL_KEYS if (d == EPISODE) else set(),
    "__ne__": lambda d: ALL_KEYS if not (d != EPISODE) else set(),
    "copy": lambda d: set(d.copy()),
    "__or__": lambda d: set(d | {}),
    "__ror__": lambda d: set({} | d),
}

# Bulk-copy spellings: syntax rather than members, and the ordinary defensive
# copy a verdict makes before touching the caller's mapping.
BULK_COPY_SPELLINGS: dict[str, Any] = {
    "dict(episode)": lambda d: dict(d)[IMG],
    "{**episode}": lambda d: {**d}[IMG],
    "episode.copy()": lambda d: d.copy()[IMG],
    "episode | {}": lambda d: (d | {})[IMG],
    "{} | episode": lambda d: ({} | d)[IMG],
}

KEYS_ONLY_READS: dict[str, Any] = {
    "list(episode.keys())": lambda d: list(d.keys()),
    "for key in episode": lambda d: [k for k in d],
    "key in episode": lambda d: IMG in d,
    "len(episode)": len,
    "sorted(episode)": sorted,
    "list(reversed(episode))": lambda d: list(reversed(d)),
}


class TestTheGradedSurfaceIsEveryDictMember:
    """The classification is exhaustive over ``dict``, not a hand-picked subset."""

    def test_every_dict_member_is_classified(self):
        """A member a future interpreter adds fails here rather than escaping."""
        unclassified = set(vars(dict)) - VALUE_BEARING - set(CANNOT_HAND_OUT_A_VALUE)
        assert not unclassified, (
            f"unclassified dict member(s) {sorted(unclassified)}: decide whether each can hand a stored "
            "value to a caller (then the probe must record it) or cannot (then say why here)"
        )

    def test_no_member_is_classified_both_ways(self):
        assert not VALUE_BEARING & set(CANNOT_HAND_OUT_A_VALUE)

    def test_both_classifications_name_only_real_dict_members(self):
        members = set(vars(dict))
        assert VALUE_BEARING <= members
        assert set(CANNOT_HAND_OUT_A_VALUE) <= members

    def test_the_driver_table_covers_exactly_the_value_bearing_members(self):
        """Premise: the behavioural sweep below is not silently narrower."""
        assert set(VALUE_BEARING_READS) == set(VALUE_BEARING)


class TestEveryValueBearingReadIsRecorded:
    """The probe intercepts each one, and each one records."""

    @pytest.mark.parametrize("member", sorted(VALUE_BEARING))
    def test_the_probe_overrides_it(self, member):
        assert member in vars(_KeyRecordingEpisode), (
            f"dict.{member} can hand a stored value to a verdict and the probe does not intercept it, "
            "so a verdict reading pixels that way is recorded as having read none"
        )

    @pytest.mark.parametrize("member", sorted(VALUE_BEARING_READS))
    def test_it_records_every_key_whose_value_it_handed_over(self, member):
        probe = _probe()
        received = VALUE_BEARING_READS[member](probe)
        assert received, f"premise: the dict.{member} driver received no value to record"
        assert probe.consulted >= received, (
            f"a verdict reading the episode through dict.{member} received the values of "
            f"{sorted(received)}, and the probe recorded only {sorted(probe.consulted)}"
        )

    @pytest.mark.parametrize("member", sorted(m for m, read in VALUE_BEARING_READS.items() if IMG in read(_probe())))
    def test_reading_the_image_column_through_it_claims_the_gate(self, member):
        """The consequence: the run's gated label turns on exactly this answer."""
        probe = _probe()
        VALUE_BEARING_READS[member](probe)
        assert probe.consulted_an_image_column(), (
            f"a verdict reading the episode through dict.{member} received the pixel values, "
            "yet the probe reports it consulted no image column"
        )


class TestABulkCopyIsRecordedTheSameWay:
    """A defensive copy hands over every value, including every pixel column.

    ``dict(episode)`` and ``{**episode}`` are what make this a pin on the
    interpreter rather than on this module alone: CPython performs them without
    calling ``__getitem__`` while its dict-merge fast path applies, and the
    probe's ``__iter__`` override is what takes that fast path away. Should a
    future interpreter merge by some other route, these fail loudly instead of
    the probe quietly returning to under-reporting.
    """

    @pytest.mark.parametrize("spelling", sorted(BULK_COPY_SPELLINGS))
    def test_the_copy_records_every_key(self, spelling):
        probe = _probe()
        copied = BULK_COPY_SPELLINGS[spelling](probe)
        assert "PIXELS" in repr(copied), "premise: the copy carries the pixels"
        assert probe.consulted, f"{spelling} recorded nothing at all"
        assert probe.consulted >= set(EPISODE), f"{spelling} is a bulk read and must record every key"

    def test_the_probe_overrides_iter_so_the_bulk_copies_are_visible(self):
        """The mechanism, pinned: a revert to inherited iteration is a regression."""
        assert "__iter__" in vars(_KeyRecordingEpisode)


class TestAKeysOnlyReadIsNotRecorded:
    """No false record: claiming a gate the verdict never applied is the unsafe direction."""

    @pytest.mark.parametrize("spelling", sorted(KEYS_ONLY_READS))
    def test_it_reaches_no_value_and_records_nothing(self, spelling):
        probe = _probe()
        KEYS_ONLY_READS[spelling](probe)
        assert not probe.consulted_an_image_column(), (
            f"{spelling} hands over no value, so recording an image column there would claim a gate "
            "the verdict never applied"
        )


class TestTheProbeStaysATransparentStandIn:
    """It is handed to caller code typed for ``dict`` - it must behave like one."""

    def test_it_is_a_dict(self):
        assert isinstance(_probe(), dict)

    def test_it_compares_equal_to_the_episode_it_wraps(self):
        assert _probe() == EPISODE

    def test_it_is_unhashable_exactly_as_a_dict_is(self):
        """``consulted`` is instrumentation, not identity, so hashing stays refused.

        Stated through ``Hashable`` rather than by calling ``hash()``: parity
        with the wrapped ``dict`` is the contract, and asserting it this way
        says so directly - including that the mapping it stands in for is
        unhashable too - instead of pinning one raise.
        """
        assert not isinstance(_probe(), Hashable)
        assert not isinstance(EPISODE, Hashable), "premise: the wrapped dict is unhashable too"
        assert _KeyRecordingEpisode.__hash__ is None
        assert dict.__hash__ is None

    @pytest.mark.parametrize(
        ("spelling", "call"),
        [
            ("copy()", lambda d: d.copy()),
            ("| other", lambda d: d | {"extra": 1}),
            ("other |", lambda d: {"extra": 1} | d),
        ],
    )
    def test_a_copy_is_a_plain_dict_as_it_is_for_any_dict_subclass(self, spelling, call):
        assert type(call(_probe())) is dict, spelling

    def test_a_copy_carries_the_episode_contents_unchanged(self):
        assert _probe().copy() == EPISODE
        assert (_probe() | {}) == EPISODE
        assert dict(_probe()) == EPISODE

    def test_iterating_still_yields_the_keys_in_insertion_order(self):
        assert list(_probe()) == list(EPISODE)
