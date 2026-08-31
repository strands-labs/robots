### Added: `g1_list_audio_asr_payload_keys` and `g1_audio_asr_payload_key_admits` verb pair

Ports `g1_list_audio_asr_payload_keys` and `g1_audio_asr_payload_key_admits`
from `cagataycali/neon-the-g1/tools/g1_audio.py::g1_asr` into
`strands_robots.tools.g1`. The lookup names the two request-payload keys
(``duration``, ``pcm_file``) the neon wrapper JSON-encodes into
``AudioClient._Call(1002, ...)``, so a caller planning a future driver-side
wrapper for the id decides an intended key decidably before triggering the
SDK's ``rc=3103`` (``RPC_CLIENT_API_NOT_REG``) or ``rc=3104``
(``RPC_CLIENT_API_TIMEOUT``) refusal at wire time. Twin of the already-shipped
:mod:`~strands_robots.tools.g1.g1_audio_call_api_ids` - one snapshot per
SDK-facing table, one verb pair per snapshot. The import pulls no
``unitree_sdk2py`` submodule (SDK-load-hygiene rule, refs
strands-labs/robots#358). Refs strands-labs/robots#358.
