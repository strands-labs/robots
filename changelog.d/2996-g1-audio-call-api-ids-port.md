### `feat(tools/g1)`: port the `AudioClient._Call` API-id enumeration verbs from the neon bundle

Two agent-facing verbs surface the one `_Call` API id the neon bundle
(`cagataycali/neon-the-g1/tools/g1_audio.py`) fronts through
`AudioClient._Call(1002, payload_json)`:

- `g1_list_audio_call_api_ids()` names the whole envelope: the one
  observed id (`1002` = on-robot ASR), the descriptor's `role` /
  `kind` / `payload` / `description` / `admits_audio_writes` flag,
  the write-id subset (empty today), and the two refusal codes
  a future driver-side wrapper would quote (`3103` invalid API id,
  `3104` RPC future in flight).
- `g1_audio_call_api_id_admits(api_id: int)` decides one query. On
  admission returns the same descriptor `g1_list_audio_call_api_ids`
  returns for that id; on refusal names the `3103` code and its
  decoded text, plus a `reason` string that names why the argument
  was refused (missing, bool, non-int, or unknown id). `bool`,
  non-int, and `None` inputs are refused decidably rather than
  resolved through Python's coercions.

Twin of `g1_list_loco_call_api_ids` / `g1_loco_call_api_id_admits`
(refs strands-labs/robots#2992): same envelope shape, same refusal-code
set, same import-hygiene contract. The two modules stay separate
because the loco and audio SDKs are two different singleton clients
(`LocoClient` vs `AudioClient`) with disjoint `_Call` admission
tables. Testing that surface: an id admitted on the loco side
(`7001`) is refused on the audio side, and vice versa - captured in
`test_g1_audio_call_api_id_admits_refuses_an_unknown_id`.

The module ports the read-only enumeration half of the neon
bundle's audio `_Call` catalogue - the actual call path is deferred
to a future driver-side wrapper that will front the RPC. The ASR
id is firmware-gated in ways the SDK's own admission set cannot
decide ahead of wire time (a build that registers `1002` in
`AudioClient.Init()` but has the ASR service disabled returns a
non-zero rc); the `description` field says so verbatim.

Import hygiene: no `unitree_sdk2py` submodule pulls at import time
(the SDK-load-hygiene contract every other file in this package
carries, refs strands-labs/robots#358). Zero-argument verb path,
snapshot-only reads.
