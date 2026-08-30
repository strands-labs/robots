"""Agent-facing lookup for the ``AudioClient._Call(1002, ...)`` payload keys the neon bundle admits.

The Unitree G1 audio SDK
(:class:`unitree_sdk2py.g1.audio.g1_audio_client.AudioClient`) registers
the on-robot ASR (speech-to-text) dispatch as raw
``_Call(1002, payload_json)`` in ``AudioClient.Init()`` but exposes no
Python helper for it. The neon bundle's
``cagataycali/neon-the-g1/tools/g1_audio.py`` reaches the id by
JSON-encoding a payload dict whose keys the SDK does not publicly
document; the neon wrapper's own construction of the dict is the only
observed source of the shape today:

.. code-block:: python

    payload = {"duration": float(duration_s)}
    if pcm_file:
        payload["pcm_file"] = pcm_file
    code, data = audio._Call(1002, json.dumps(payload))

Two keys, two roles: ``duration`` (a positive-finite float naming the
number of seconds the robot listens on its own mic when no ``pcm_file``
is passed) and ``pcm_file`` (an optional string path on the robot to a
raw PCM file to transcribe instead of listening live). The twin lookup
:mod:`~strands_robots.tools.g1.g1_audio_call_api_ids` catalogues the
API-id (``1002``) itself; this module snapshots the payload-key half
so a caller planning a future driver-side wrapper for the id decides
the key set decidably before a mis-typed key triggers whatever the
firmware surfaces at wire time. Refs strands-labs/robots#358.

The two agent-facing verbs mirror the twin's shape:
:func:`g1_list_audio_asr_payload_keys` names the whole envelope, and
:func:`g1_audio_asr_payload_key_admits` decides one query.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_asr`` verb wrapped
  ``AudioClient._Call(1002, ...)`` and returned whatever payload the
  firmware reported; the underlying SDK future is a single in-flight
  slot per ``AudioClient`` instance, so concurrent ``_Call`` from
  different threads returns ``rc=3104`` (``RPC_CLIENT_API_TIMEOUT``).
  That call is the same audio RPC channel today's
  :class:`~strands_robots.drivers.g1.G1Driver` does not front (the
  driver's :meth:`~strands_robots.drivers.g1.G1Driver.stream` spec
  declares only ``sensors`` / ``status`` / ``stop`` verbs), and a
  future audio-side driver method that fronts the read will land
  alongside the twin at
  :mod:`~strands_robots.tools.g1.g1_audio_call_api_ids`. This module
  ports the read-only enumeration half of the payload shape without
  also introducing a second audio writer path the driver does not yet
  own.
* An SDK re-import. The key table is a module-level constant snapshot
  of what the neon bundle observed against the real robot; the
  constant lives here rather than being re-imported from the SDK so
  ``import strands_robots.tools.g1.g1_audio_asr_payload_keys`` pulls no
  ``unitree_sdk2py`` submodule - the import-hygiene contract every
  other file in this package carries, refs
  strands-labs/robots#358. An SDK release that widens the ASR payload
  vocabulary (a new firmware knob) is a driver-side update; when the
  driver's audio read method lands, its refusal for a mis-typed key
  will quote the same ``rc=3103`` ("RPC_CLIENT_API_NOT_REG") /
  ``rc=3104`` ("RPC_CLIENT_API_TIMEOUT") entries the
  :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` carries.

What this module does not decide.

* Whether the firmware on this robot honours the key at wire time.
  The neon bundle's ``g1_asr`` docstring notes that the id itself
  "may not be enabled on this firmware" - a firmware that registers
  the id but has the underlying ASR service disabled returns a
  non-zero rc regardless of the payload shape. A caller reads the
  ``description`` field this verb returns and knows the key is
  firmware-gated in ways this snapshot does not surface.
* Whether the payload as a whole is well-formed. The lookup grades
  one key at a time; a caller planning the full JSON encode reads
  the returned ``required`` / ``optional`` sets and constructs the
  dict itself. A future driver-side wrapper for the id will validate
  the whole payload at wire time before serialising to JSON.
* The response schema. The neon bundle's ``g1_asr`` returns the raw
  response for the caller to inspect ("the exact response schema
  isn't publicly documented - this returns the raw response so you
  can discover it"). This module names only the *request* payload's
  keys.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES

#: Snapshot of the ``AudioClient._Call(1002, ...)`` request-payload keys
#: the neon bundle
#: (``cagataycali/neon-the-g1/tools/g1_audio.py::g1_asr``) admits today.
#: Each descriptor names:
#:
#: * ``role`` - a short prose description of what the key controls;
#: * ``type`` - the Python type the neon wrapper coerces the key to
#:   before ``json.dumps`` (``"float"`` for ``duration``, ``"str"``
#:   for ``pcm_file``);
#: * ``required`` - whether the neon wrapper always sets the key
#:   (``duration``) or gates it on the caller's argument
#:   (``pcm_file``).
#:
#: The label table lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the ASR (API-id ``1002``) side of the audio
#: ``_Call`` conversation; a caller that needs the API-id enumeration
#: reaches :mod:`~strands_robots.tools.g1.g1_audio_call_api_ids`
#: directly. Colocating the key table with the enumeration verb
#: mirrors ``_AUDIO_CALL_API_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_audio_call_api_ids`,
#: ``_LOCO_CALL_API_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_loco_call_api_ids`, and
#: ``_ARM_ACTION_MAP`` in
#: :mod:`~strands_robots.tools.g1.g1_arm_actions`: one snapshot per
#: SDK-facing table, one verb pair per snapshot.
_AUDIO_ASR_PAYLOAD_KEYS: dict[str, dict[str, Any]] = {
    "duration": {
        "role": (
            "Seconds the robot listens on its own mic before returning "
            "the transcript. The neon wrapper always sets this key; when "
            "``pcm_file`` is also passed the neon wrapper still sends "
            "``duration`` but the on-robot ASR service is expected to "
            "prefer the PCM file (the exact firmware behaviour is not "
            "publicly documented)."
        ),
        "type": "float",
        "required": True,
    },
    "pcm_file": {
        "role": (
            "Path on the robot's filesystem to a raw PCM file to "
            "transcribe instead of listening on the mic. Absent when the "
            "neon wrapper is called with an empty ``pcm_file`` argument. "
            "The wire path is a string, not a bytes blob: the neon "
            "wrapper does not upload the file, it names an existing "
            "path on the robot (a firmware that has not staged the file "
            "at that path will surface whatever rc the on-robot ASR "
            "service returns)."
        ),
        "type": "str",
        "required": False,
    },
}

#: The API id the payload keys in :data:`_AUDIO_ASR_PAYLOAD_KEYS`
#: front. Pinned as a module-level constant so a widen (a firmware
#: that adds a second audio-side ``_Call`` id with a different key
#: shape) lands as a shape change on both this module and its twin
#: :mod:`~strands_robots.tools.g1.g1_audio_call_api_ids` rather than
#: as a magic number floating loose.
_ASR_API_ID: int = 1002

#: The error-table entry the SDK's ``_Call`` returns when the
#: AudioClient's RPC future is already in flight
#: (``RPC_CLIENT_API_TIMEOUT``). Named here because a caller reading
#: this envelope planning a full-payload construction still has to
#: cross this refusal at wire time; surfaced alongside the mis-typed
#: key refusal on the returned envelope so both refusals appear at
#: the same level of the caller's plan. Mirrors
#: :data:`~strands_robots.tools.g1.g1_audio_call_api_ids._RPC_TIMEOUT_CODE`.
_RPC_TIMEOUT_CODE: int = 3104

#: The error-table entry the SDK returns for an API id the handler
#: table does not admit (``RPC_CLIENT_API_NOT_REG``). Surfaced on the
#: refuse path because a caller who names a key the payload snapshot
#: does not admit is one plan away from also naming an API id the
#: SDK does not admit; both refusals decode against the same table,
#: and naming the code twice would let the two entries drift.
_INVALID_KEY_CODE: int = 3103


def _describe(key: str) -> dict[str, Any]:
    """Build the per-key descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_audio_asr_payload_keys` so
    :func:`g1_audio_asr_payload_key_admits`'s admitted-path payload
    names the same fields, and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched.
    """
    entry = _AUDIO_ASR_PAYLOAD_KEYS[key]
    return {
        "key": key,
        "role": entry["role"],
        "type": entry["type"],
        "required": entry["required"],
    }


@tool
def g1_list_audio_asr_payload_keys() -> dict[str, Any]:
    """Return the ``AudioClient._Call(1002, ...)`` payload keys the neon bundle admits.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for the ASR id is called, so a caller can compare an intended
    payload key against the set the neon bundle observed against the
    real robot, and decide alongside that whether the key is required
    or optional in the neon wrapper's own construction.

    Returns:
        A dict with ``status``; an ``api_id`` naming the audio API id
        the keys front (``1002``, the on-robot ASR); a ``count``
        naming the number of admitted keys; a ``payload_keys`` list
        of descriptors (one per admitted key, sorted alphabetically)
        carrying ``key`` (the JSON key name), ``role`` (the
        neon-observed purpose), ``type`` (the Python type the neon
        wrapper coerces the value to), and ``required`` (whether the
        neon wrapper always sets the key or gates it); a ``keys``
        list of just the key names in sorted order; a ``required``
        list naming the keys the neon wrapper always sets; an
        ``optional`` list naming the keys the neon wrapper gates on
        a caller argument; and a ``refusals`` list carrying the two
        refusal codes (``3103`` unknown key, ``3104`` RPC future in
        flight) and their decoded text that a future call verb would
        surface. Every field is a snapshot of a neon constant; no
        dynamic decode runs here.
    """
    keys = sorted(_AUDIO_ASR_PAYLOAD_KEYS)
    required = sorted(k for k, v in _AUDIO_ASR_PAYLOAD_KEYS.items() if v["required"])
    optional = sorted(k for k, v in _AUDIO_ASR_PAYLOAD_KEYS.items() if not v["required"])
    return {
        "status": "success",
        "api_id": _ASR_API_ID,
        "count": len(_AUDIO_ASR_PAYLOAD_KEYS),
        "payload_keys": [_describe(key) for key in keys],
        "keys": keys,
        "required": required,
        "optional": optional,
        "refusals": [
            {"code": _INVALID_KEY_CODE, "text": ERR_CODES[_INVALID_KEY_CODE]},
            {"code": _RPC_TIMEOUT_CODE, "text": ERR_CODES[_RPC_TIMEOUT_CODE]},
        ],
    }


@tool
def g1_audio_asr_payload_key_admits(key: str | None = None) -> dict[str, Any]:
    """Decide whether ``key`` is inside the neon-observed ASR payload set.

    Read-only. Compares one argument against the neon-observed
    :data:`_AUDIO_ASR_PAYLOAD_KEYS` and reports the admitted
    descriptor on match, or the ``3103`` refusal code a future
    driver-side wrapper would quote on miss. No driver instance, no
    DDS, no SDK: the decision reads only module-level constants and
    the argument itself.

    A key inside the admitted set is *not* the same as an admitted
    call: any well-formed payload is still refused with ``rc=3104``
    while the singleton ``AudioClient``'s RPC future is in flight,
    and a firmware that admits the id but has the underlying ASR
    service disabled returns a non-zero rc at wire time regardless
    of the payload shape. Neither is a snapshot answer; both are
    live-driver reads a caller reaches after this verb admits the
    key. The returned payload's ``required`` field names whether the
    neon wrapper always sets the key so a caller planning the full
    payload construction sees at once which side of the gate the key
    lands on.

    Args:
        key: The payload key to check. Must be a ``str``; ``bool``
            is refused with the ``3103`` code because a passed-through
            boolean is a caller mistake, not a valid key query.
            A missing argument (``None``) is refused decidably rather
            than treated as a default.

    Returns:
        A dict with ``status``; on admit, a ``payload_key``
        descriptor with ``key``, ``role``, ``type``, and ``required``
        (the same shape :func:`g1_list_audio_asr_payload_keys`
        returns). On refuse, ``refusal_code`` and ``refusal_text``
        name the ``3103`` code and its decoded text, plus a
        ``reason`` string that names why the argument was refused
        (missing argument, bool argument, non-str argument, or
        unknown key).
    """
    if key is None:
        return {
            "status": "error",
            "refusal_code": _INVALID_KEY_CODE,
            "refusal_text": ERR_CODES[_INVALID_KEY_CODE],
            "reason": (
                f"key is required; pass one of {sorted(_AUDIO_ASR_PAYLOAD_KEYS)} "
                "so the lookup is decidable. Refs strands-labs/robots#358."
            ),
        }
    if isinstance(key, bool):
        return {
            "status": "error",
            "refusal_code": _INVALID_KEY_CODE,
            "refusal_text": ERR_CODES[_INVALID_KEY_CODE],
            "reason": (
                f"key={key!r} is a bool; pass one of "
                f"{sorted(_AUDIO_ASR_PAYLOAD_KEYS)} as a str. "
                "Refs strands-labs/robots#358."
            ),
        }
    if not isinstance(key, str):
        return {
            "status": "error",
            "refusal_code": _INVALID_KEY_CODE,
            "refusal_text": ERR_CODES[_INVALID_KEY_CODE],
            "reason": (
                f"key={key!r} is not a str; pass one of "
                f"{sorted(_AUDIO_ASR_PAYLOAD_KEYS)} as a str. "
                "Refs strands-labs/robots#358."
            ),
        }
    if key not in _AUDIO_ASR_PAYLOAD_KEYS:
        return {
            "status": "error",
            "refusal_code": _INVALID_KEY_CODE,
            "refusal_text": ERR_CODES[_INVALID_KEY_CODE],
            "reason": (
                f"key={key!r} is not in the admitted set "
                f"{sorted(_AUDIO_ASR_PAYLOAD_KEYS)}. "
                "Refs strands-labs/robots#358."
            ),
        }
    return {
        "status": "success",
        "payload_key": _describe(key),
    }
