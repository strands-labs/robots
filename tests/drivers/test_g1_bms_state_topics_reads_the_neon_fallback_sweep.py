"""The BMS DDS topic fallback envelope tools name exactly what the neon sweep tries.

The neon bundle's ``g1_battery.py`` module
(``cagataycali/neon-the-g1/tools/g1_battery.py``) carries a
three-topic ``_BMS_TOPICS`` tuple that its ``_ensure_subscriber``
helper walks in order when opening the battery-management subscribe:
the canonical ``rt/lf/bmsstate`` first, then two historical spellings
(``rt/bmsstate`` and ``rt/bms_state``). The
:mod:`strands_robots.tools.g1.g1_bms_state_topics` module snapshots
that tuple into module-level constants and exposes two agent-facing
verbs - :func:`g1_list_bms_state_topics` (list the whole envelope) and
:func:`g1_bms_state_topic_admits` (decide one membership query) - so
a caller planning a firmware-portable subscribe can name the
candidate set decidably before a future driver-side wrapper for the
neon sweep fires. The tests here fix that contract without pulling
the SDK: the module is loadable on a host without ``unitree_sdk2py``
(the same SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs strands-labs/robots#358),
and every membership answer is read off the module's own snapshot
rather than restated in the tests, so a widen or narrow to the
constants surfaces here as a shape change rather than as a diverging
table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The neon bundle's own answer at wire time. The verbs answer
  against the module-level snapshot, not against a live import of
  the neon bundle's ``_BMS_TOPICS`` tuple (the whole point of the
  port is that the snapshot lets a headless host answer without
  pulling the neon module or the CycloneDDS bindings). A future
  driver-side wrapper for the neon sweep will re-validate against
  its own live tuple at wire time; testing the snapshot vs the live
  tuple is a driver-side test, not a lookup-side one.
* Which topic the driver or the neon wrapper resolves to at any
  particular moment. Live topic selection is a firmware-runtime
  answer (the neon sweep walks in priority order and stops at the
  first topic that yields a :class:`BmsState_` message); the verbs
  answer only the narrower static question of which topics the
  sweep would consider. A future test that pinned the driver's live
  choice would cross a scope line this file keeps.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_bms_state_topics import (
    _BMS_STATE_IDL_TYPE,
    _BMS_STATE_TOPIC_DESCRIPTIONS,
    _BMS_STATE_TOPIC_PRIORITY,
    _BMS_STATE_TOPICS,
    g1_bms_state_topic_admits,
    g1_list_bms_state_topics,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim. This helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import
    time would break every headless CI runner and Thor before an
    office bring-up. The driver enforces the same rule against
    itself (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is
    the only path that loads the SDK); this cell holds the
    bms-state-topics envelope verbs to it too (refs
    strands-labs/robots#358).
    """
    sys.modules.pop("strands_robots.tools.g1.g1_bms_state_topics", None)
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_bms_state_topics")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bms_state_topics imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the "
        "SDK loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_neon_observed_fallback_sweep() -> None:
    """The snapshot names every topic the neon ``_BMS_TOPICS`` sweep tries today.

    The neon bundle's ``_BMS_TOPICS`` tuple has 3 entries observed
    across G1 firmware releases (``rt/lf/bmsstate``, ``rt/bmsstate``,
    ``rt/bms_state``). A drift on either side surfaces here: a
    driver-side sweep wrapper (when it lands) will validate the same
    set at wire time and its topic-resolution loop will name the
    same membership test. The count is pinned rather than listed
    value-by-value so a caller widening the sweep on the neon side
    updates one number here rather than 3 assertions.
    """
    assert len(_BMS_STATE_TOPICS) == 3, (
        f"expected 3 topics in the neon _BMS_TOPICS snapshot, got "
        f"{len(_BMS_STATE_TOPICS)}: {sorted(_BMS_STATE_TOPICS)}. A "
        "neon-side widen or narrow would update this count; refs "
        "strands-labs/robots#358."
    )


def test_the_canonical_current_firmware_topic_is_a_member() -> None:
    """``rt/lf/bmsstate`` is a member of the fallback set.

    The current-firmware canonical spelling is the topic the driver
    subscribes at connect time (see
    :mod:`~strands_robots.tools.g1.g1_dds_topics`), and it must
    remain the rank-0 candidate of the neon sweep. If a future
    firmware release renames the canonical topic and the neon
    module updates without also updating this snapshot, the test
    surfaces the drift before wire.
    """
    assert "rt/lf/bmsstate" in _BMS_STATE_TOPICS, (
        "rt/lf/bmsstate is the current-firmware canonical BMS topic and "
        "the driver's own subscribe target; it must be a member of the "
        "neon fallback set. Refs strands-labs/robots#358."
    )
    assert _BMS_STATE_TOPIC_PRIORITY["rt/lf/bmsstate"] == 0, (
        f"rt/lf/bmsstate priority "
        f"{_BMS_STATE_TOPIC_PRIORITY['rt/lf/bmsstate']} disagrees with "
        "the neon sweep's rank-0 preference; refs strands-labs/robots#358."
    )


def test_every_snapshot_topic_carries_a_priority_and_description() -> None:
    """Every topic has a matching priority rank and description entry.

    :data:`_BMS_STATE_TOPIC_PRIORITY` and
    :data:`_BMS_STATE_TOPIC_DESCRIPTIONS` are the tables both verbs
    read when they build the returned descriptor. A drift between
    the membership set and either table would surface here: a topic
    in :data:`_BMS_STATE_TOPICS` without a matching key in either
    table would raise ``KeyError`` on the ``_describe`` call at
    :func:`g1_list_bms_state_topics` time, and a key not in the
    membership set would be silently unreachable. Pinning both
    directions here catches the drift before wire.
    """
    assert set(_BMS_STATE_TOPIC_PRIORITY) == set(_BMS_STATE_TOPICS), (
        f"priority-table keys {sorted(_BMS_STATE_TOPIC_PRIORITY)} do "
        f"not match the membership set {sorted(_BMS_STATE_TOPICS)}. A "
        "neon-side widen must update both together; refs "
        "strands-labs/robots#358."
    )
    assert set(_BMS_STATE_TOPIC_DESCRIPTIONS) == set(_BMS_STATE_TOPICS), (
        f"description-table keys "
        f"{sorted(_BMS_STATE_TOPIC_DESCRIPTIONS)} do not match the "
        f"membership set {sorted(_BMS_STATE_TOPICS)}. A neon-side "
        "widen must update both together; refs strands-labs/robots#358."
    )
    for topic, rank in _BMS_STATE_TOPIC_PRIORITY.items():
        assert isinstance(rank, int) and rank >= 0, (
            f"priority rank for topic {topic!r} must be a non-negative int; got {rank!r}. Refs strands-labs/robots#358."
        )
    for topic, description in _BMS_STATE_TOPIC_DESCRIPTIONS.items():
        assert isinstance(description, str) and description != "", (
            f"description for topic {topic!r} must be a non-empty str; "
            f"got {description!r}. Refs strands-labs/robots#358."
        )


def test_the_priority_ranks_are_a_contiguous_zero_based_range() -> None:
    """Priority ranks form a contiguous ``[0, N)`` range with no ties.

    The neon sweep walks the tuple by index, so the ranks must form a
    zero-based contiguous range with no gaps or ties. A widen that
    left a hole in the rank space or two topics at the same rank
    would break the neon walk order silently at wire time; pinning
    the range here catches the drift before wire.
    """
    ranks = sorted(_BMS_STATE_TOPIC_PRIORITY.values())
    expected = list(range(len(_BMS_STATE_TOPICS)))
    assert ranks == expected, (
        f"priority ranks {ranks} are not a contiguous [0, "
        f"{len(_BMS_STATE_TOPICS)}) range; the neon sweep walks by "
        f"index and every rank slot must be filled exactly once. "
        f"Refs strands-labs/robots#358."
    )


def test_the_idl_type_names_the_neon_bmsstate_message() -> None:
    """The IDL type is the ``BmsState_`` message class the neon sweep decodes.

    The neon ``_ensure_subscriber`` helper passes
    ``unitree_sdk2py.idl.unitree_hg.msg.dds_.BmsState_`` explicitly
    to :class:`~unitree_sdk2py.core.channel.ChannelSubscriber`; the
    IDL type is what makes the decode wire-compatible across the
    three fallback topics. Named here as a plain string constant
    rather than an import so the module stays SDK-load-hygiene
    clean.
    """
    assert _BMS_STATE_IDL_TYPE == ("unitree_sdk2py.idl.unitree_hg.msg.dds_.BmsState_"), (
        f"BMS IDL type {_BMS_STATE_IDL_TYPE!r} disagrees with the neon "
        "wrapper's own type argument; refs strands-labs/robots#358."
    )


def test_list_returns_every_topic_in_neon_priority_order() -> None:
    """The list verb returns descriptors sorted by neon sweep-order rank.

    The order is fixed to mirror the neon tuple's own index order so a
    caller comparing two returned envelopes (before and after a
    widen, for instance) sees a stable diff rather than a
    permutation. Sorting is done at the verb boundary rather than in
    the snapshot so the underlying membership constant stays a
    ``frozenset`` (order-free) and the verb-time order tracks the
    priority table.
    """
    payload = _call(g1_list_bms_state_topics)
    assert payload["status"] == "success", payload
    topic_names = payload["topic_names"]
    expected_order = sorted(
        _BMS_STATE_TOPICS,
        key=lambda t: _BMS_STATE_TOPIC_PRIORITY[t],
    )
    assert topic_names == expected_order, (
        f"list verb returned topic_names {topic_names!r} out of "
        f"neon-priority order; expected {expected_order!r}. Refs "
        "strands-labs/robots#358."
    )
    descriptor_topics = [row["topic"] for row in payload["topics"]]
    assert descriptor_topics == topic_names, (
        f"descriptor list ordering {descriptor_topics!r} disagrees "
        f"with topic_names field ordering {topic_names!r}; both must "
        "present in the same neon-priority order. Refs "
        "strands-labs/robots#358."
    )


def test_list_names_every_snapshot_topic_and_no_others() -> None:
    """The list verb surfaces exactly the membership set.

    A drift here means the verb's returned envelope disagrees with
    the module-level constant it is supposed to snapshot; the whole
    point of the port is that the two agree by construction.
    """
    payload = _call(g1_list_bms_state_topics)
    assert payload["count"] == len(_BMS_STATE_TOPICS), (
        f"list verb count {payload['count']} disagrees with snapshot "
        f"size {len(_BMS_STATE_TOPICS)}. Refs strands-labs/robots#358."
    )
    surfaced = set(payload["topic_names"])
    assert surfaced == set(_BMS_STATE_TOPICS), (
        f"list verb topic_names {sorted(surfaced)} disagree with the "
        f"snapshot {sorted(_BMS_STATE_TOPICS)}. A widen must update "
        "both together; refs strands-labs/robots#358."
    )


def test_list_names_the_primary_topic_as_the_rank_zero_candidate() -> None:
    """The ``primary_topic`` field is the rank-0 topic the driver subscribes.

    The neon sweep starts from rank 0 and settles on the first topic
    that yields a message; the primary topic named on the returned
    envelope is a display of that rank-0 preference so a caller
    reading the envelope does not have to sort the priority table
    themselves.
    """
    payload = _call(g1_list_bms_state_topics)
    expected_primary = min(
        _BMS_STATE_TOPICS,
        key=lambda t: _BMS_STATE_TOPIC_PRIORITY[t],
    )
    assert payload["primary_topic"] == expected_primary, (
        f"list verb primary_topic {payload['primary_topic']!r} "
        f"disagrees with the rank-0 candidate {expected_primary!r}. "
        "Refs strands-labs/robots#358."
    )


def test_list_surfaces_every_descriptor_with_the_snapshot_priority_and_description() -> None:
    """Every descriptor carries the topic's snapshot priority and description.

    The ``priority`` and ``description`` fields on each returned
    descriptor must match the module's
    :data:`_BMS_STATE_TOPIC_PRIORITY` and
    :data:`_BMS_STATE_TOPIC_DESCRIPTIONS` entries for that topic
    byte-for-byte; a re-wording or re-ranking that only landed on
    one side of the verb boundary would surface here.
    """
    payload = _call(g1_list_bms_state_topics)
    for row in payload["topics"]:
        topic = row["topic"]
        assert row["priority"] == _BMS_STATE_TOPIC_PRIORITY[topic], (
            f"descriptor row for topic {topic!r} carries priority="
            f"{row['priority']}; snapshot says "
            f"{_BMS_STATE_TOPIC_PRIORITY[topic]}. Refs "
            "strands-labs/robots#358."
        )
        assert row["description"] == _BMS_STATE_TOPIC_DESCRIPTIONS[topic], (
            f"descriptor row for topic {topic!r} carries description "
            f"{row['description']!r}; snapshot says "
            f"{_BMS_STATE_TOPIC_DESCRIPTIONS[topic]!r}. Refs "
            "strands-labs/robots#358."
        )
        assert row["idl_type"] == _BMS_STATE_IDL_TYPE, (
            f"descriptor row for topic {topic!r} carries idl_type "
            f"{row['idl_type']!r}; snapshot says {_BMS_STATE_IDL_TYPE!r}. "
            "Refs strands-labs/robots#358."
        )


def test_admits_returns_true_on_every_snapshot_topic() -> None:
    """Every topic in the snapshot admits.

    Pins the round-trip: whatever the list verb returns as a valid
    topic, the admits verb agrees is a member of the fallback set. A
    drift between the two would leave a caller unable to trust the
    two verbs' answers together.
    """
    for topic in sorted(_BMS_STATE_TOPICS):
        payload = _call(g1_bms_state_topic_admits, topic=topic)
        assert payload["status"] == "success", payload
        assert payload["admitted"] is True, (
            f"admits verb refused {topic!r} but it is a member of the "
            f"BMS fallback snapshot {sorted(_BMS_STATE_TOPICS)}. Refs "
            "strands-labs/robots#358."
        )
        assert payload["target"]["topic"] == topic, (
            f"admits verb target topic {payload['target']['topic']!r} "
            f"disagrees with the queried topic {topic!r}. Refs "
            "strands-labs/robots#358."
        )
        assert payload["target"]["priority"] == _BMS_STATE_TOPIC_PRIORITY[topic], (
            f"admits verb target for {topic!r} carries priority="
            f"{payload['target']['priority']}; snapshot says "
            f"{_BMS_STATE_TOPIC_PRIORITY[topic]}. Refs "
            "strands-labs/robots#358."
        )
        assert payload["target"]["idl_type"] == _BMS_STATE_IDL_TYPE, (
            f"admits verb target for {topic!r} carries idl_type "
            f"{payload['target']['idl_type']!r}; snapshot says "
            f"{_BMS_STATE_IDL_TYPE!r}. Refs strands-labs/robots#358."
        )


def test_admits_refuses_an_off_sweep_topic_with_the_valid_candidates_listed() -> None:
    """A topic outside the sweep returns ``admitted=False`` with the valid set listed.

    The neon ``_ensure_subscriber`` helper's behaviour on an
    unrecognised topic is silent-skip (the topic is simply not
    walked, and the wrapper returns no message); the verb's
    off-sweep path surfaces the caller-side note so a caller reading
    the refusal does not silently pass an unknown topic through. The
    refusal also lists the valid candidates so the caller can
    resolve the drift without a follow-up call.
    """
    payload = _call(g1_bms_state_topic_admits, topic="rt/battery")
    assert payload["status"] == "success", payload
    assert payload["admitted"] is False, (
        "admits verb admitted 'rt/battery' but it is not on the BMS "
        f"fallback snapshot {sorted(_BMS_STATE_TOPICS)}. Refs "
        "strands-labs/robots#358."
    )
    assert "'rt/battery'" in payload["refusal_advice"], (
        f"admits verb refusal_advice {payload['refusal_advice']!r} "
        "must name the queried topic verbatim. Refs "
        "strands-labs/robots#358."
    )
    for known in _BMS_STATE_TOPICS:
        assert repr(known) in payload["refusal_advice"], (
            f"admits verb refusal_advice {payload['refusal_advice']!r} "
            f"must list every known candidate; {known!r} is missing. "
            "Refs strands-labs/robots#358."
        )


def test_admits_refuses_a_bool_argument_as_a_shape_error() -> None:
    """``bool`` is refused decidably, not resolved through ``str()`` coercion.

    ``True``/``False`` are not valid DDS topic strings under any
    convention; a caller that passed a boolean is making a caller
    mistake, and the verb surfaces it as an error rather than
    resolving through Python's coercions.
    """
    for value in (True, False):
        payload = _call(g1_bms_state_topic_admits, topic=value)
        assert payload["status"] == "error", (
            f"admits verb accepted bool topic {value!r}; the argument must be str. Refs strands-labs/robots#358."
        )
        assert "bool" in payload["message"], (
            f"admits verb error message for bool topic {value!r} must "
            f"name the refused type; got {payload['message']!r}. Refs "
            "strands-labs/robots#358."
        )


def test_admits_refuses_a_non_str_argument_as_a_shape_error() -> None:
    """Non-str inputs are refused decidably.

    ``int``, ``float``, ``None``, ``list``, ``tuple`` are all not
    valid topic strings and the verb surfaces each as a shape error
    rather than resolving through Python's coercions.
    """
    for value in (1, 1.5, None, ["rt/lf/bmsstate"], ("rt/lf/bmsstate",)):
        payload = _call(g1_bms_state_topic_admits, topic=value)
        assert payload["status"] == "error", (
            f"admits verb accepted non-str topic {value!r}; the argument must be str. Refs strands-labs/robots#358."
        )
        assert type(value).__name__ in payload["message"], (
            f"admits verb error message for non-str topic {value!r} "
            f"must name the refused type; got {payload['message']!r}. "
            "Refs strands-labs/robots#358."
        )


def test_admits_refuses_the_empty_string_as_a_shape_error() -> None:
    """The empty string is refused decidably.

    An empty topic name has no membership answer to compute; the verb
    refuses it as a shape error rather than returning
    ``admitted=False`` (which would let a caller silently pass the
    empty string through as an off-sweep topic).
    """
    payload = _call(g1_bms_state_topic_admits, topic="")
    assert payload["status"] == "error", (
        "admits verb accepted an empty topic string; the argument must be non-empty. Refs strands-labs/robots#358."
    )
    assert "non-empty" in payload["message"], (
        f"admits verb error message for the empty string must name "
        f"the non-empty contract; got {payload['message']!r}. Refs "
        "strands-labs/robots#358."
    )


def test_every_refusal_string_cites_the_open_issue() -> None:
    """Every refuse path cites ``strands-labs/robots#358`` as its issue anchor.

    The repo's refusal-strings rule (#2872) is that a refusal message
    cites a resolvable issue reference; every refuse path in this
    module cites #358 (the g1 tools port tracker). A refuse path
    that dropped the citation would fail the rule at CI; pinning
    the citation here surfaces the drift before wire.
    """
    refuse_calls: list[dict[str, Any]] = [
        {"topic": True},
        {"topic": False},
        {"topic": 1},
        {"topic": 1.5},
        {"topic": None},
        {"topic": ["rt/lf/bmsstate"]},
        {"topic": ("rt/lf/bmsstate",)},
        {"topic": ""},
    ]
    for kwargs in refuse_calls:
        payload = _call(g1_bms_state_topic_admits, **kwargs)
        assert payload["status"] == "error", (
            f"admits verb accepted refuse-path input {kwargs!r}; expected error. Refs strands-labs/robots#358."
        )
        assert "strands-labs/robots#358" in payload["message"], (
            f"admits verb refusal for {kwargs!r} does not cite "
            f"strands-labs/robots#358; got {payload['message']!r}. "
            "The refusal-strings rule requires a resolvable issue "
            "reference on every refuse path."
        )
    off_sweep = _call(g1_bms_state_topic_admits, topic="rt/battery")
    assert off_sweep["admitted"] is False, off_sweep
    assert "strands-labs/robots#358" in off_sweep["refusal_advice"], (
        f"off-sweep refusal_advice {off_sweep['refusal_advice']!r} "
        "does not cite strands-labs/robots#358; the refusal-strings "
        "rule requires a resolvable issue reference on every refuse "
        "path."
    )
