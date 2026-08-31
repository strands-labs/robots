### Feature

- Added `strands_robots.tools.g1.g1_list_capture_rate_candidates` and
  `strands_robots.tools.g1.g1_capture_rate_candidate_admits`, two
  read-only ``@tool`` verbs that snapshot the four PCM sample rates the
  neon bundle's ``g1_bidi_audio`` module
  (``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``) iterates against
  ``PyAudio.is_format_supported`` inside its ``pick_capture_rate``
  helper before opening the USB mic: ``16000`` (Brio-family
  16 kHz-native devices, also the internal AEC and bidi-pipeline rate),
  ``48000`` (DJI Mic Mini and other 48 kHz-only consumer USB mics, also
  the fallback rate the neon helper returns on a whole-list miss),
  ``44100`` (CD-lineage consumer USB rate), and ``32000`` (an unusual
  low-rate widening safety net). The tuple order matches the neon
  source byte-for-byte: the neon helper stops at the first accepted
  rate, so preserving the ordering pins which rate a given USB mic
  family lands on. Each admitted rate carries a ``role`` label
  classifying its mic family, a ``description`` naming the observed
  device set the rate is known-good for, an ``is_pipeline_native``
  flag naming the 16 kHz entry the neon path uses without a downsample,
  and an ``is_fallback`` flag naming the 48 kHz rate the neon
  ``pick_capture_rate`` returns when every candidate is refused; the
  envelope surfaces the fallback rate separately in
  ``fallback_rate_hz`` so a caller does not have to model the fallback
  branch as a candidate itself. The two verbs let a caller decide the
  ``rc=7404`` gate-refused refusal decidably before a future
  driver-side ``g1_speak`` wrapper runs the mic-open probe against the
  host, so a headless CI runner reading the candidate set can
  enumerate the four rates without pulling the audio stack itself. No
  ``unitree_sdk2py`` submodule and no bidi audio-stack submodule
  (``pywebrtc_audio``, ``pyaudio``, ``strands.experimental.bidi``)
  load on import - the candidate list is an immutable integer tuple,
  matching the SDK-load hygiene the rest of this package carries and
  letting a machine without ``pyaudio`` installed read the admitted
  set. Non-integer (including ``float``), bool, non-positive, and
  missing arguments to ``g1_capture_rate_candidate_admits`` refuse
  decidably with reason strings that name the shape error rather than
  falling through to a confusing "unknown rate" refusal. Refs
  strands-labs/robots#358.
