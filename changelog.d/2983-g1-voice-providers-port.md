### Feature

- Added `strands_robots.tools.g1.g1_list_voice_providers` and
  `strands_robots.tools.g1.g1_voice_provider_admits`, two read-only ``@tool``
  verbs that snapshot the four voice-provider names the neon bundle's
  ``g1_speak`` verb (``cagataycali/neon-the-g1/tools/g1_speak.py``) admits:
  ``openai`` / ``openai_realtime`` (both routed to the OpenAI Realtime bidi
  factory, both guarded on ``OPENAI_API_KEY`` by the bundle's own
  ``prov in ("openai", "openai_realtime")`` check), ``nova_sonic`` (Amazon
  Nova Sonic, reached through AWS credentials), and ``gemini`` (Google
  Gemini Live, reached through ``GOOGLE_API_KEY``). Each admitted provider
  carries a ``credential_env`` field naming the env-var its factory reaches
  for so a caller comparing an intended provider against both conditions
  (membership + credential) has the env-var name on hand; ``openai`` and
  ``openai_realtime`` name the same ``OPENAI_API_KEY`` so a caller's
  credential check admits both aliases identically. The two verbs let a
  caller decide the ``rc=7404`` gate-refused refusal decidably before a
  future driver-side ``g1_speak`` wrapper fires. No ``unitree_sdk2py``
  submodule and no bidi audio-stack submodule (``pywebrtc_audio``,
  ``pyaudio``, ``strands.experimental.bidi``) load on import - the
  provider table is a module-level snapshot, matching the SDK-load
  hygiene the rest of this package carries and letting a headless CI
  runner read the admitted set without the optional audio deps present.
  Non-string, bool, empty-string, and missing arguments to
  ``g1_voice_provider_admits`` refuse decidably with reason strings that
  name the shape error rather than falling through to a confusing
  "unknown provider" refusal. Refs strands-labs/robots#358.
