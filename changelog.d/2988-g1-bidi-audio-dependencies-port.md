### Feature

- Added `strands_robots.tools.g1.g1_list_bidi_audio_dependencies` and
  `strands_robots.tools.g1.g1_bidi_audio_dependency_admits`, two read-only
  ``@tool`` verbs that snapshot the three module-import names the neon
  bundle's ``g1_speak`` verb
  (``cagataycali/neon-the-g1/tools/g1_speak.py``) reads inside its
  ``_probe_bidi`` guard: ``pywebrtc_audio`` (the WebRTC-lineage
  acoustic-echo-cancellation front-end), ``pyaudio`` (the PortAudio mic
  capture the ``autopick_mic`` helper selects a device on), and
  ``strands.experimental.bidi`` (the strands experimental bidi agent
  factory ``BidiAgent`` is imported from). Each admitted dependency
  carries a ``role`` label classifying its contribution to the bidi audio
  path (``aec_frontend``, ``mic_capture``, ``bidi_agent``) and a
  ``pip_hint`` naming a suggested distribution name a caller would
  ``pip install`` to satisfy the probe on a host missing the module.
  The two verbs let a caller decide the ``rc=7404`` gate-refused refusal
  decidably before a future driver-side ``g1_speak`` wrapper fires
  ``_probe_bidi`` against the host, so a headless CI runner reading the
  dependency set can enumerate the three module names without pulling
  the audio stack itself. No ``unitree_sdk2py`` submodule and no bidi
  audio-stack submodule (``pywebrtc_audio``, ``pyaudio``,
  ``strands.experimental.bidi``) load on import - the dependency table
  is a module-level string snapshot, matching the SDK-load hygiene the
  rest of this package carries and letting a machine without the optional
  audio deps read the admitted set. Non-string, bool, empty-string, and
  missing arguments to ``g1_bidi_audio_dependency_admits`` refuse
  decidably with reason strings that name the shape error rather than
  falling through to a confusing "unknown dependency" refusal. Refs
  strands-labs/robots#358.
