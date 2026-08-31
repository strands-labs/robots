### Added

- `strands_robots.tools.g1.g1_bidi_audio_mic_keywords` ports the read-only
  half of neon-the-g1's `autopick_mic` helper into two agent-facing lookups:
  `g1_list_bidi_audio_mic_keywords` (name the priority-ordered substring
  set the helper walks when `VOICE_MIC_NAME` is unset - ``DJI``, ``Logi``,
  ``Brio``, ``USB``, ``Mic``, in that order) and
  `g1_bidi_audio_mic_keyword_admits` (decide one query, refusing bool,
  non-string, empty-string, mis-cased, and off-set arguments with a
  message that names the admitted casing so a caller planning a
  `VOICE_MIC_NAME` override reads the neon-observed set without a
  second lookup). Every descriptor names `match_case_insensitive=True`
  because the bundle's helper lowercases both operands before its `in`
  check; every descriptor also names `override_env="VOICE_MIC_NAME"` so
  the driver-side write path and this lookup share a single string.
  Read-only, no driver instance, no DDS, no SDK, no PyAudio session:
  `import strands_robots.tools.g1.g1_bidi_audio_mic_keywords` pulls no
  `unitree_sdk2py` submodule and no optional audio-stack submodule
  (`pyaudio`, `pywebrtc_audio`, `strands.experimental.bidi`).
  Refs #358.
