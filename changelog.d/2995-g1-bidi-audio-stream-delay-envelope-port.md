### Added

- `strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope` ports the
  read-only envelope half of neon-the-g1's `g1_bidi_audio.py` AEC
  `stream_delay_ms` argument (the speaker->mic loopback delay hint the WebRTC
  `AudioProcessor` compensates for on the G1's DDS audio path) into two
  agent-facing lookups: `g1_list_bidi_audio_stream_delay_envelope` (name the
  observed `[0, 500]` ms clamp and the neon-tuned default of `120` ms named
  in `g1_bidi_audio.py` as `DEFAULT_STREAM_DELAY_MS`) and
  `g1_stream_delay_ms_admits` (decide one query, refusing negative values,
  above-ceiling values past which WebRTC silently truncates to its internal
  delay-buffer bound, bool-masquerading-as-int, and non-int-non-bool values
  with a module-local refusal text that names the stream-delay envelope). The
  refusal descriptor omits any `code` field because the audio-processing
  pipeline ships no rc for a bounds-violated stream-delay argument and the
  nearest neighbour on the driver's side (`7404`, motion-FSM) would hand a
  planner a locomotion FSM remedy for an argument that never touches
  `rt/lowcmd`. Read-only, no driver instance, no DDS, no SDK, no
  `pywebrtc_audio` import: `import
  strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope` pulls no
  `unitree_sdk2py` and no `pywebrtc` submodule. Refs #358.
