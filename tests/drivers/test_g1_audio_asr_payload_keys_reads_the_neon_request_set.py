"""The audio-ASR payload-key lookup tools name the neon-observed request-payload set.

The Unitree G1 audio SDK
(:class:`unitree_sdk2py.g1.audio.g1_audio_client.AudioClient`) registers
the on-robot ASR (speech-to-text) dispatch as raw
``_Call(1002, payload_json)`` in ``AudioClient.Init()`` but exposes no
Python helper for it and does not publish the payload's key shape. The
neon bundle's ``cagataycali/neon-the-g1/tools/g1_audio.py`` builds the
payload dict by JSON-encoding two keys - ``duration`` (a positive-finite
float; always set) and ``pcm_file`` (an optional string path; set only
when the caller passes it) - and the neon wrapper's own construction is
the only observed source of the shape today. The
:mod:`strands_robots.tools.g1.g1_audio_asr_payload_keys` module snapshots
that catalogue into a module-level constant and exposes two
agent-facing verbs -
:func:`g1_list_audio_asr_payload_keys` (list the whole set) and
:func:`g1_audio_asr_payload_key_admits` (decide one query) - so a caller
can decide the refusal decidably before a future call path is
attempted. The tests here fix that contract without pulling the SDK:
the module is loadable on a host without ``unitree_sdk2py`` (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off the
module's own snapshot rather than restated in the tests, so a widen or
narrow to the constant surfaces here as a shape change rather than as
a diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The SDK's own answer at wire time. The verbs answer against the
  module-level snapshot, not against a live import of the SDK's
  ``_Call`` handler table (the whole point of the port is that the
  snapshot lets a headless host answer). A driver-side wrapper for
  the id that lands later will re-validate against the SDK's live
  handler at wire time; testing the snapshot vs the live handler is
  a driver-side test, not a lookup-side one.
* Whether the firmware on this robot honours the key at wire time.
  A firmware that registers ``1002`` in ``AudioClient.Init()`` but
  has the underlying ASR service disabled returns a non-zero rc
  regardless of the payload shape; neither this lookup nor the
  SDK's own admission set can decide it ahead of wire time, and the
  ``description`` prose on the module docstring says so verbatim.
  The membership tests here grade the snapshot only.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_audio_asr_payload_keys import (
    _ASR_API_ID,
    _AUDIO_ASR_PAYLOAD_KEYS,
    _INVALID_KEY_CODE,
    _RPC_TIMEOUT_CODE,
    g1_audio_asr_payload_key_admits,
    g1_list_audio_asr_payload_keys,
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

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent; a module that pulled a
    submodule at import time would break every headless CI runner
    and Thor before an office bring-up (refs
    strands-labs/robots#358).
    """
    sys.modules.pop("strands_robots.tools.g1.g1_audio_asr_payload_keys", None)
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_audio_asr_payload_keys")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_audio_asr_payload_keys imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the "
        "SDK loads only inside function bodies (refs "
        "strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_neon_observed_keys() -> None:
    """The snapshot names every ASR payload key the neon bundle constructs.

    The neon bundle's ``g1_asr`` builds the payload dict from exactly
    two keys: ``duration`` (always) and ``pcm_file`` (optional). The
    keys are pinned so a caller widening the map on the neon side
    (e.g. a future firmware that adds a language-selector key)
    updates this cell and the constant together.
    """
    assert len(_AUDIO_ASR_PAYLOAD_KEYS) == 2
    assert set(_AUDIO_ASR_PAYLOAD_KEYS) == {"duration", "pcm_file"}


def test_the_snapshot_pins_the_asr_api_id() -> None:
    """The payload keys front API id ``1002``.

    The id is pinned as a module-level constant so this file and the
    twin :mod:`~strands_robots.tools.g1.g1_audio_call_api_ids` (which
    catalogues the id itself) name the same number rather than let
    the two sides drift.
    """
    assert _ASR_API_ID == 1002


def test_the_duration_key_is_required_and_float_typed() -> None:
    """The ``duration`` key is required and coerces to ``float``.

    The neon wrapper's ``payload = {"duration": float(duration_s)}``
    always sets the key and always coerces the value to ``float``.
    Pinned here so a widen that flips the required flag or narrows
    the type surfaces as a shape change.
    """
    entry = _AUDIO_ASR_PAYLOAD_KEYS["duration"]
    assert entry["type"] == "float"
    assert entry["required"] is True


def test_the_pcm_file_key_is_optional_and_str_typed() -> None:
    """The ``pcm_file`` key is optional and carries a ``str`` path.

    The neon wrapper's ``if pcm_file: payload["pcm_file"] = pcm_file``
    gates the key on the caller's argument and treats the value as a
    string path (the wrapper does not upload the file - it names an
    existing path on the robot). Pinned so a widen that flips the
    required flag or changes the type surfaces as a shape change.
    """
    entry = _AUDIO_ASR_PAYLOAD_KEYS["pcm_file"]
    assert entry["type"] == "str"
    assert entry["required"] is False


def test_the_refusal_codes_decode_against_the_shared_table() -> None:
    """The refusal codes carry the same names the shared table carries.

    ``3103`` decodes to ``RPC_CLIENT_API_NOT_REG`` (mis-typed key or
    unknown API id); ``3104`` decodes to ``RPC_CLIENT_API_TIMEOUT``
    (RPC future in flight). Pinned so a re-word of either code lands
    in :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` rather
    than drifting between the SDK-side log and this lookup.
    """
    assert ERR_CODES[_INVALID_KEY_CODE] == "RPC_CLIENT_API_NOT_REG"
    assert ERR_CODES[_RPC_TIMEOUT_CODE] == "RPC_CLIENT_API_TIMEOUT"


def test_list_names_the_whole_envelope() -> None:
    """:func:`g1_list_audio_asr_payload_keys` returns a full snapshot.

    ``status`` is ``success``; ``api_id`` is ``1002``; ``count`` is
    the length of the constant; ``payload_keys`` names one descriptor
    per admitted key in sorted order; ``keys`` names just the key
    names in sorted order; ``required`` and ``optional`` partition
    the keys; ``refusals`` carries both refusal codes and their
    decoded text.
    """
    envelope = _call(g1_list_audio_asr_payload_keys)
    assert envelope["status"] == "success"
    assert envelope["api_id"] == _ASR_API_ID
    assert envelope["count"] == len(_AUDIO_ASR_PAYLOAD_KEYS)
    assert envelope["keys"] == sorted(_AUDIO_ASR_PAYLOAD_KEYS)
    assert envelope["required"] == sorted(k for k, v in _AUDIO_ASR_PAYLOAD_KEYS.items() if v["required"])
    assert envelope["optional"] == sorted(k for k, v in _AUDIO_ASR_PAYLOAD_KEYS.items() if not v["required"])
    assert [d["key"] for d in envelope["payload_keys"]] == sorted(_AUDIO_ASR_PAYLOAD_KEYS)
    assert envelope["refusals"] == [
        {"code": _INVALID_KEY_CODE, "text": ERR_CODES[_INVALID_KEY_CODE]},
        {"code": _RPC_TIMEOUT_CODE, "text": ERR_CODES[_RPC_TIMEOUT_CODE]},
    ]


def test_list_descriptors_carry_the_snapshot_fields() -> None:
    """Every descriptor names ``key`` / ``role`` / ``type`` / ``required``.

    The descriptor shape is fixed here so a widen (a new field on the
    descriptor) surfaces in one place; a caller filtering the
    envelope reads the same field names both verbs return.
    """
    envelope = _call(g1_list_audio_asr_payload_keys)
    for descriptor in envelope["payload_keys"]:
        assert set(descriptor) == {"key", "role", "type", "required"}
        entry = _AUDIO_ASR_PAYLOAD_KEYS[descriptor["key"]]
        assert descriptor["role"] == entry["role"]
        assert descriptor["type"] == entry["type"]
        assert descriptor["required"] is entry["required"]


def test_admits_on_a_known_key_returns_the_descriptor() -> None:
    """:func:`g1_audio_asr_payload_key_admits` admits every documented key.

    Every key in :data:`_AUDIO_ASR_PAYLOAD_KEYS` returns
    ``status=success`` with a ``payload_key`` descriptor carrying
    the same fields the list verb returns.
    """
    for key in _AUDIO_ASR_PAYLOAD_KEYS:
        result = _call(g1_audio_asr_payload_key_admits, key=key)
        assert result["status"] == "success"
        assert result["payload_key"]["key"] == key
        assert result["payload_key"]["role"] == _AUDIO_ASR_PAYLOAD_KEYS[key]["role"]
        assert result["payload_key"]["type"] == _AUDIO_ASR_PAYLOAD_KEYS[key]["type"]
        assert result["payload_key"]["required"] is _AUDIO_ASR_PAYLOAD_KEYS[key]["required"]


def test_admits_refuses_a_missing_argument() -> None:
    """A missing argument is refused decidably.

    ``key=None`` is not a valid dispatch query; the verb refuses
    with the ``3103`` code rather than picking a default and reads
    the refusal text off the shared table.
    """
    result = _call(g1_audio_asr_payload_key_admits, key=None)
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_KEY_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_KEY_CODE]
    assert "key is required" in result["reason"]
    assert "strands-labs/robots#358" in result["reason"]


def test_admits_refuses_a_bool_argument() -> None:
    """A ``bool`` argument is refused before it aliases a str.

    ``bool`` is not a subclass of ``str`` but a passed-through boolean
    is a caller mistake, and the neon wrapper's own construction only
    ever names string keys. The verb refuses with the ``3103`` code
    and reads the refusal text off the shared table.
    """
    result = _call(g1_audio_asr_payload_key_admits, key=True)
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_KEY_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_KEY_CODE]
    assert "is a bool" in result["reason"]
    assert "strands-labs/robots#358" in result["reason"]


def test_admits_refuses_a_non_str_argument() -> None:
    """A non-``str`` argument is refused decidably.

    Ints, floats, and other non-string types are not valid keys; the
    verb refuses with the ``3103`` code rather than coercing.
    """
    for bad in (1002, 3.14, ["duration"], {"duration": 1.0}):
        result = _call(g1_audio_asr_payload_key_admits, key=bad)  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert result["refusal_code"] == _INVALID_KEY_CODE
        assert result["refusal_text"] == ERR_CODES[_INVALID_KEY_CODE]
        assert "is not a str" in result["reason"]
        assert "strands-labs/robots#358" in result["reason"]


def test_admits_refuses_an_unknown_key() -> None:
    """A string key that is not in the admitted set is refused decidably.

    An unknown key is one plan away from also being an unknown API id;
    the verb refuses with the same ``3103`` code the twin uses for
    unknown ids so both refusals decode against the same table.
    """
    result = _call(g1_audio_asr_payload_key_admits, key="language")
    assert result["status"] == "error"
    assert result["refusal_code"] == _INVALID_KEY_CODE
    assert result["refusal_text"] == ERR_CODES[_INVALID_KEY_CODE]
    assert "'language'" in result["reason"]
    assert "not in the admitted set" in result["reason"]
    assert "strands-labs/robots#358" in result["reason"]


def test_every_refusal_reason_cites_the_open_issue() -> None:
    """Every refusal reason cites strands-labs/robots#358.

    The rule (refs strands-labs/robots#2872) is that a refusal
    string names a resolvable issue reference so a caller reaching
    the refusal has one hop to the open decision. The four refusal
    paths are exercised together so a re-word that drops the ref
    surfaces at once rather than one test at a time.
    """
    reasons = [
        _call(g1_audio_asr_payload_key_admits, key=None)["reason"],
        _call(g1_audio_asr_payload_key_admits, key=True)["reason"],
        _call(g1_audio_asr_payload_key_admits, key=42)["reason"],  # type: ignore[arg-type]
        _call(g1_audio_asr_payload_key_admits, key="language")["reason"],
    ]
    for reason in reasons:
        assert "strands-labs/robots#358" in reason


def test_list_partitions_agree_with_the_snapshot() -> None:
    """``required`` and ``optional`` sum to ``keys`` with no overlap.

    A caller consuming the list envelope reads three keys
    (``required``, ``optional``, ``keys``); those three fields must
    partition the same set (union covers, intersection empty). Pinned
    so a widen that adds a third partition (say, ``conditional``)
    surfaces as a shape change here rather than a silent drift.
    """
    envelope = _call(g1_list_audio_asr_payload_keys)
    required = set(envelope["required"])
    optional = set(envelope["optional"])
    keys = set(envelope["keys"])
    assert required | optional == keys
    assert required & optional == set()


def test_the_two_verbs_return_the_same_descriptor_shape() -> None:
    """A key admitted by both verbs returns the same descriptor.

    ``g1_list_audio_asr_payload_keys`` names one descriptor per
    admitted key; ``g1_audio_asr_payload_key_admits`` names one
    descriptor per admitted call. Both descriptors carry the same
    field names for the same key so a caller mixing the two verbs
    reads consistent shapes.
    """
    envelope = _call(g1_list_audio_asr_payload_keys)
    for descriptor in envelope["payload_keys"]:
        one = _call(g1_audio_asr_payload_key_admits, key=descriptor["key"])
        assert one["status"] == "success"
        assert one["payload_key"] == descriptor
