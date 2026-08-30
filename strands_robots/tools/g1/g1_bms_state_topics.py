"""Agent-facing lookup for the BMS DDS topic fallback set the neon battery wrapper sweeps.

The Unitree G1 firmware publishes the battery-management snapshot on
``rt/lf/bmsstate`` in the release
:class:`~strands_robots.drivers.g1.G1Driver` targets today, but the
neon bundle's ``g1_battery`` tool wrapper carries a three-topic
fallback list in a ``_BMS_TOPICS`` module constant
(``cagataycali/neon-the-g1/tools/g1_battery.py``) that the wrapper
tries in order when the primary topic returns no message: the
canonical ``rt/lf/bmsstate`` first, then two historical spellings
(``rt/bmsstate`` and ``rt/bms_state``) observed across earlier G1
firmware releases. The neon wrapper's ``_ensure_subscriber`` helper
walks the list and settles on the first topic that returns a
:class:`unitree_sdk2py.idl.unitree_hg.msg.dds_.BmsState_` frame; that
choice is what makes the neon wrapper portable across firmware
revisions where only the current spelling would otherwise resolve.

This module snapshots that three-topic fallback set as a
module-level ``frozenset`` and surfaces it as two agent-facing verbs -
:func:`g1_list_bms_state_topics` (list the whole envelope) and
:func:`g1_bms_state_topic_admits` (decide one membership query) -
so a caller planning a battery-state read can decide the
firmware-portable topic set decidably before a future driver-side
wrapper for the neon fallback sweep is dispatched. Refs
strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon wrapper's ``_ensure_subscriber`` helper
  actually opens a :class:`unitree_sdk2py.core.channel.ChannelSubscriber`
  against each candidate topic under the same
  :func:`~strands_robots.tools.g1._g1_common.ensure_dds` singleton the
  driver holds, and settles on the first that resolves; that
  subscribe path is out of scope for this lookup. Today's
  :class:`~strands_robots.drivers.g1.G1Driver` subscribes only
  ``rt/lf/bmsstate`` (the current-firmware spelling) at connect
  time; a future driver-side wrapper for the neon fallback sweep
  will cross-reference the membership answer this lookup returns.
  This module ports the read-only membership half without also
  introducing a second BMS subscriber path the driver does not yet
  own.
* An SDK or CycloneDDS re-import. The topic names live here as
  string constants snapshotted from the neon bundle's
  ``_BMS_TOPICS`` tuple; the snapshot lives here rather than being
  re-read from the neon module so
  ``import strands_robots.tools.g1.g1_bms_state_topics`` pulls
  no ``unitree_sdk2py`` submodule - the import-hygiene contract
  every other file in this package carries, refs
  strands-labs/robots#358. A firmware release that widens or
  narrows the fallback set is a caller-side update; when a
  driver-side sweep lands, its topic-resolution loop will name the
  same membership test this lookup answers.

What this module does not decide.

* Which topic the driver or the neon wrapper actually settled on at
  any particular moment. Live topic selection is a driver-side
  runtime answer that depends on the firmware revision the robot is
  running; the neon wrapper resolves it by walking the tuple in
  order and stopping at the first topic that yields a message. This
  lookup answers the narrower static question: which topic *names*
  the neon fallback set carries, so a caller planning a subscribe
  can list the candidates before a future driver-side wrapper for
  the sweep fires.
* Whether a topic outside the set is a valid BMS wire path. A
  topic outside the fallback set is *not* automatically wrong: a
  hypothetical future firmware could publish the BMS snapshot under
  a new topic that the neon table has not yet observed. This verb
  answers only the narrower membership question the current neon
  snapshot answers; a widen to the neon tuple lands both there and
  here in the same PR, or the two silently drift.
* The IDL type each topic carries. Every entry resolves to
  ``unitree_sdk2py.idl.unitree_hg.msg.dds_.BmsState_`` under the
  neon wrapper (the wrapper passes the type explicitly to
  :class:`~unitree_sdk2py.core.channel.ChannelSubscriber`), but
  the type contract is a driver-side subscribe answer, not a
  lookup-side one; :mod:`~strands_robots.tools.g1.g1_dds_topic_idl_types`
  is the sibling module that ports the IDL side of the driver's own
  seven-topic subscription set. A future widen to that module can
  cross-reference the BMS type when a driver-side sweep for the
  fallback set lands; this module carries only the topic-name
  snapshot.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: Snapshot of the three DDS topic names the neon bundle's
#: ``_BMS_TOPICS`` tuple sweeps in order when subscribing the
#: battery-management state. The neon
#: (``cagataycali/neon-the-g1/tools/g1_battery.py``) wrapper's
#: ``_ensure_subscriber`` helper walks this exact tuple and settles
#: on the first topic that yields a
#: :class:`~unitree_sdk2py.idl.unitree_hg.msg.dds_.BmsState_` frame;
#: the ordered walk is what makes the neon wrapper portable across
#: firmware revisions where only one spelling resolves at a time.
#: Named here as a module-level ``frozenset`` rather than being
#: re-imported from the neon module so
#: ``import strands_robots.tools.g1.g1_bms_state_topics`` pulls
#: zero ``unitree_sdk2py`` submodules - the import-hygiene contract
#: every other file in this package carries. A neon-side widen or
#: narrow updates this snapshot and the neon module in the same PR
#: or the two silently drift.
#:
#: The set is a ``frozenset`` because the lookup answers a
#: membership question (`does topic X appear in the neon fallback
#: sweep?`); the ordered-preference side of the neon tuple is
#: captured separately by :data:`_BMS_STATE_TOPIC_PRIORITY` for the
#: descriptor payload so a caller reading a single topic gets both
#: the membership answer and the neon-order rank for that topic.
_BMS_STATE_TOPICS: frozenset[str] = frozenset(
    {
        "rt/lf/bmsstate",
        "rt/bmsstate",
        "rt/bms_state",
    }
)

#: Per-topic priority rank the neon wrapper's ``_ensure_subscriber``
#: helper walks in order. The primary topic ``rt/lf/bmsstate`` is
#: rank ``0`` (the first candidate the neon sweep tries), the two
#: historical spellings follow at ranks ``1`` and ``2``. Named here
#: rather than derived from the ``frozenset`` above because a
#: ``frozenset`` carries no order; the rank is what a caller
#: planning a firmware-specific subscribe would consult to decide
#: which topic to open first before falling back. Kept in sync with
#: the neon tuple's own index order; a re-order in the neon module
#: lands both there and here in the same PR, or a caller reading the
#: rank surfaces a different sweep order than the neon wrapper runs.
_BMS_STATE_TOPIC_PRIORITY: dict[str, int] = {
    "rt/lf/bmsstate": 0,
    "rt/bmsstate": 1,
    "rt/bms_state": 2,
}

#: Per-topic description surfaced on the returned descriptor. The
#: descriptions name the wire lineage of each topic name (the
#: current-firmware canonical spelling, and the two historical
#: spellings the neon sweep carries as fallbacks). Named here rather
#: than inlined so a widen to the tuple surfaces the description in
#: one place; kept plain (no emoji) so the source-string emoji guard
#: the repo runs at CI does not fire on the description table.
_BMS_STATE_TOPIC_DESCRIPTIONS: dict[str, str] = {
    "rt/lf/bmsstate": (
        "Canonical BMS state topic on current G1 firmware (the topic "
        "the driver subscribes at connect time and the neon sweep's "
        "first candidate)."
    ),
    "rt/bmsstate": (
        "Historical BMS state topic name observed on earlier G1 firmware releases; the neon sweep's second fallback."
    ),
    "rt/bms_state": (
        "Historical BMS state topic name observed on earlier G1 "
        "firmware releases (underscore variant); the neon sweep's "
        "third fallback."
    ),
}

#: The BMS payload IDL type every topic in the fallback set carries
#: under the neon wrapper's ``_ensure_subscriber`` helper. Named as
#: a plain string constant (not an import) so
#: ``import strands_robots.tools.g1.g1_bms_state_topics`` pulls no
#: ``unitree_sdk2py`` submodules; the IDL type is a caller-side
#: cross-reference for a future driver-side wrapper for the sweep,
#: not a subscribe path this lookup itself opens.
_BMS_STATE_IDL_TYPE: str = "unitree_sdk2py.idl.unitree_hg.msg.dds_.BmsState_"


def _describe(topic: str) -> dict[str, Any]:
    """Build the per-topic descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_bms_state_topics` so
    :func:`g1_bms_state_topic_admits`'s admitted-path payload names
    the same fields, and so a widen to the descriptor lands in one
    place. Every field is a snapshot read; no bus is touched. The
    ``priority`` field carries the neon sweep-order rank
    (:data:`_BMS_STATE_TOPIC_PRIORITY`); the ``idl_type`` field
    carries the BMS payload type string so a caller planning a
    subscribe can name the IDL up front without re-reading the
    driver-side type table.
    """
    return {
        "topic": topic,
        "priority": _BMS_STATE_TOPIC_PRIORITY[topic],
        "description": _BMS_STATE_TOPIC_DESCRIPTIONS[topic],
        "idl_type": _BMS_STATE_IDL_TYPE,
    }


@tool
def g1_list_bms_state_topics() -> dict[str, Any]:
    """Return the BMS DDS topic fallback set the neon battery wrapper sweeps.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side
    wrapper for the neon ``_BMS_TOPICS`` sweep is dispatched, so a
    caller planning a firmware-portable BMS subscribe can list the
    candidate topic names and their neon sweep-order rank without
    reaching the bus.

    The envelope names three topics: ``rt/lf/bmsstate`` (rank 0, the
    current-firmware canonical spelling and the driver's own
    subscribe topic), ``rt/bmsstate`` (rank 1, the neon sweep's
    second fallback), and ``rt/bms_state`` (rank 2, the third
    fallback). Each descriptor carries the topic string, the neon
    priority rank, a plain-text description of the topic's wire
    lineage, and the BMS payload IDL type string
    (``unitree_sdk2py.idl.unitree_hg.msg.dds_.BmsState_``) every
    entry resolves to under the neon wrapper.

    Returns:
        A dict with ``status``; a ``count`` naming the number of
        candidate topics in the sweep; a ``topics`` list of
        descriptors (one per topic, sorted by neon priority rank so
        the returned list mirrors the neon sweep order) carrying
        ``topic``, ``priority``, ``description``, and ``idl_type``;
        a ``topic_names`` field listing the topic strings in the
        same neon-priority order; a ``primary_topic`` field naming
        the rank-0 topic the driver subscribes today; and an
        ``idl_type`` field naming the shared payload type. Every
        field is a snapshot of a neon-observed constant; no dynamic
        decode runs here.
    """
    topics_in_priority_order = sorted(
        _BMS_STATE_TOPICS,
        key=lambda t: _BMS_STATE_TOPIC_PRIORITY[t],
    )
    return {
        "status": "success",
        "count": len(_BMS_STATE_TOPICS),
        "topics": [_describe(topic) for topic in topics_in_priority_order],
        "topic_names": topics_in_priority_order,
        "primary_topic": topics_in_priority_order[0],
        "idl_type": _BMS_STATE_IDL_TYPE,
    }


@tool
def g1_bms_state_topic_admits(topic: str = "") -> dict[str, Any]:
    """Decide whether a topic is inside the neon BMS fallback sweep set.

    Read-only. Reads the module's snapshot of the neon bundle's
    ``_BMS_TOPICS`` tuple and returns the same membership answer the
    neon wrapper's ``_ensure_subscriber`` helper would recognise
    when walking its candidate list. A caller with a topic string
    resolves it against the fallback set before a future driver-side
    sweep dispatches, rather than triggering the wrapper's
    silent-skip on an unrecognised topic at wire time.

    A topic inside :data:`_BMS_STATE_TOPICS` is a topic the neon
    wrapper would try to subscribe against; a topic outside the set
    is *not* automatically wrong (a hypothetical future firmware
    could publish the BMS snapshot under a new topic that the neon
    table has not yet observed) but is outside the set the current
    neon sweep would try. This verb answers the narrower
    membership question the neon fallback set answers.

    Args:
        topic: The topic string to test. Must be a non-empty
            ``str``; ``bool`` is refused (``True``/``False`` are
            not valid DDS topic strings under any convention) and
            the empty string is refused as a shape error (no topic
            name means no membership query to answer). Non-str
            inputs are refused decidably rather than resolved
            through Python's ``str()`` coercion.

    Returns:
        A dict with ``status``; a ``query`` sub-dict carrying the
        supplied ``topic``; an ``admitted`` boolean naming whether
        the topic is a member of the three-entry fallback set; and
        (when ``admitted`` is ``True``) a ``target`` sub-dict
        carrying the same descriptor
        :func:`g1_list_bms_state_topics` returns for the topic
        (``topic``, ``priority``, ``description``, ``idl_type``).
        On a not-admitted query the dict carries a
        ``refusal_advice`` field naming that the topic is outside
        the neon fallback set and listing the three known candidates
        (in neon-priority order) so the caller can resolve the drift
        without a follow-up call. On a shape error (``bool``,
        non-str, empty string) the dict carries ``status="error"``
        with a message naming the type refused.
    """
    if isinstance(topic, bool):
        return {
            "status": "error",
            "message": (f"topic must be str, got bool ({topic!r}). Refs strands-labs/robots#358."),
        }
    if not isinstance(topic, str):
        return {
            "status": "error",
            "message": (f"topic must be str, got {type(topic).__name__} ({topic!r}). Refs strands-labs/robots#358."),
        }
    if topic == "":
        return {
            "status": "error",
            "message": (
                "topic must be a non-empty str; an empty topic name "
                "has no membership answer to compute. Refs "
                "strands-labs/robots#358."
            ),
        }

    admitted = topic in _BMS_STATE_TOPICS
    if not admitted:
        known_in_priority = sorted(
            _BMS_STATE_TOPICS,
            key=lambda t: _BMS_STATE_TOPIC_PRIORITY[t],
        )
        return {
            "status": "success",
            "admitted": False,
            "query": {"topic": topic},
            "refusal_advice": (
                f"topic {topic!r} is not on the neon BMS fallback "
                "sweep; the neon _ensure_subscriber helper would "
                "silently skip an unrecognised topic and return "
                "empty. Known candidates in neon-priority order: "
                f"{known_in_priority!r}. Refs "
                "strands-labs/robots#358."
            ),
        }

    return {
        "status": "success",
        "admitted": True,
        "query": {"topic": topic},
        "target": _describe(topic),
    }
