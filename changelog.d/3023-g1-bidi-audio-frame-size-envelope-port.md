### Feature

- Added `strands_robots.tools.g1.g1_list_bidi_audio_frame_size_envelope` and
  `strands_robots.tools.g1.g1_frame_size_admits`, two read-only ``@tool``
  verbs that snapshot the WebRTC AEC frame-size pin the neon bundle
  (`cagataycali/neon-the-g1/tools/g1_bidi_audio.py`) uses on both the
  AEC-processing thread (`FRAME_SIZE = 160`) and the mic-capture thread
  (`PYAUDIO_FRAMES = 160`). Both constants are the same 10 ms envelope at
  16 kHz (`160 / 16000 == 0.010`) viewed from two threads and share the
  single `_FRAME_SIZE_SAMPLES = 160` pin; a caller planning a future
  bidi-audio write against the driver reads the shared pin off this
  lookup rather than re-deriving it from a sample-rate / duration
  product where a wrong-size frame would surface at wire time only as
  a WebRTC `RTC_DCHECK` at the C++ layer with no refusal visible to the
  Python caller. Refuses at the value boundary (`value != pin` reads
  as `frame_size out of envelope - need frame_size == 160 (10 ms at
  16 kHz, the WebRTC AudioProcessor pin)`), the type boundary
  (`bool`, `float`, `Decimal`, `str`, `None`, list, tuple each read as
  `non-int`), and every refusal cites `strands-labs/robots#358` on the
  audio-processing surface rather than re-borrowing the motion-FSM
  `7404` refusal from `ERR_CODES`. Sibling of
  `g1_bidi_audio_sample_rates` (the three-role rate lookup),
  `g1_bidi_audio_stream_delay_envelope` (the WebRTC AEC delay clamp),
  and `g1_bidi_audio_dependencies` (the runtime probe set); the four
  together are the read-only lookup half of the bidi-audio surface
  every future driver-side wrapper will front. Import pulls zero
  `unitree_sdk2py`, `pywebrtc_audio`, or `pyaudio` submodules (the
  SDK-load-hygiene contract every other file in this package carries,
  refs `strands-labs/robots#358`). Refs `strands-labs/robots#358`,
  `strands-labs/robots#2916`.
