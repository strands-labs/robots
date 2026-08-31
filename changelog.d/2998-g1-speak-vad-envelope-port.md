### Added

- `strands_robots.tools.g1.g1_speak_vad_envelope` ports the read-only half of
  neon-the-g1's `g1_speak(action="start")` turn-detector envelope
  (`cagataycali/neon-the-g1/tools/g1_speak.py`): two agent-facing verbs
  (`g1_list_speak_vad_envelope`, `g1_speak_vad_admits`) surface the
  `BidiAgent` turn-detector's ``vad_threshold`` clamp (``[0.0, 1.0]``) and the
  neon-observed ``silence_duration_ms`` positive-integer domain (``>= 1``),
  plus the neon-tuned defaults (``0.7`` "stops echo triggers" and ``700`` ms
  "relaxed"). Module-local refusal texts stay on-surface — the bidi voice
  pipeline ships no distinct rc, so no motion-FSM ``7404`` code is re-borrowed
  for a turn-detector bounds violation. Twin of
  `g1_bidi_audio_stream_delay_envelope` (the AEC half of the same argument
  tuple; the two libraries have disjoint refusal shapes so the modules stay
  separate). Read-only. No driver instance, no DDS, no SDK, no optional
  audio-stack import (refs #358).
