### Added

- `strands_robots.tools.g1.g1_bidi_audio_sample_rates` ports the read-only
  sample-rate half of neon-the-g1's `g1_bidi_audio.py` audio-config block
  (the three integer rates the `G1BidiAudioIO` pins for its three
  audio surfaces: `MIC_RATE = 16000` for the laptop USB mic capture the
  WebRTC AEC runs on, `G1_RATE = 16000` for the G1 DDS chest-speaker feed
  on `rt/audio_stream` and the WebRTC AEC's far-buffer reference queue,
  and `OPENAI_RATE = 24000` for the OpenAI Realtime endpoint's own sample
  rate the `BidiAgent` upsamples to before publishing and downsamples
  from before consuming) into two agent-facing lookups:
  `g1_list_bidi_audio_sample_rates` (name the whole role -> rate mapping,
  with a `description` per row disambiguating the two numerically-equal
  16 kHz roles by signal direction) and `g1_bidi_audio_sample_rate_admits`
  (decide one query against the admitted role set, refusing mis-cased
  names, bool-masquerading-as-str, non-str non-bool values, the empty
  string, and missing arguments with the `7404` gate-refusal code a
  future driver-side wrapper would quote at the same boundary; the WebRTC
  / OpenAI Realtime / DDS-speaker factories ship no numbered SDK rc for
  a bad sample-rate argument, so the lookup shares the neighbouring
  provider-lookup's refusal shape). Read-only, no driver instance, no
  DDS, no SDK, no `pywebrtc_audio` import, no `pyaudio` import: `import
  strands_robots.tools.g1.g1_bidi_audio_sample_rates` pulls no
  `unitree_sdk2py` submodule and no optional audio-stack submodule. Refs
  #358.
