"""Agent-facing lookup for the 10 ms frame-size pin the neon bidi bundle uses.

The neon bundle's bidirectional audio IO
(``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``) pins the WebRTC
:class:`AudioProcessor`'s frame size to a single integer value:
``FRAME_SIZE = 160`` (int16 samples) is the 10 ms frame at 16 kHz the
AEC contract requires (``160 / 16000 == 0.010``), and
``PYAUDIO_FRAMES = 160`` is the ``frames_per_buffer`` argument the
neon bundle hands the ``pyaudio`` mic-input callback at the same
rate; the neon mic thread rescales that second constant by the
observed capture rate (``fpb = int(PYAUDIO_FRAMES * capture_rate /
MIC_RATE)``) so a 48 kHz USB mic yields ``480`` frames per callback
and the downsampler still surfaces a ``FRAME_SIZE``-sized frame to
the AEC. Both numbers are the same 10 ms envelope viewed from the
two threads (``FRAME_SIZE`` on the AEC-processing thread,
``PYAUDIO_FRAMES`` on the mic-capture thread) and neither is a
caller-tunable value: WebRTC's ``AudioProcessor`` admits only 10 ms
frames at 16 kHz (its ``ProcessStream`` contract refuses any other
length at the C++ layer without surfacing the refusal to the Python
caller), and the mic callback needs the same rate as the AEC frame
so ``_frame_residual``'s remainder-carry logic does not desync.

This module snapshots that shared ``160``-sample pin into
module-level constants and exposes two agent-facing verbs so a
caller planning a bidi start can name the same integer the AEC
pipeline requires without re-deriving it from a sample-rate/duration
product, rather than pinning the value inside the write path where
the refusal is invisible to the planner. The verb pair mirrors
:mod:`~strands_robots.tools.g1.g1_bidi_audio_sample_rates` (the
three-role sample-rate lookup): one snapshot lookup naming the two
roles at their shared pinned value, one membership decision on one
query.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_speak(action="start")``
  verb spins up a ``G1BidiAudioIO`` under a single-thread lock; that
  construction is the same audio-processing path the driver's
  future bidi wrapper would front. A future driver method that
  fronts the bidi IO will land the write verb; refs
  strands-labs/robots#358 for the SDK-facing seam that write
  belongs on. This module ports the read-only lookup half without
  also introducing a second audio-processing writer path the
  driver does not yet own.
* An SDK re-import. The frame-size pin is captured here as a
  ``tuple`` of two ``(role, samples)`` descriptors and a shared
  integer bound; the constant lives here rather than being
  re-imported from ``pywebrtc_audio`` or ``pyaudio`` so ``import
  strands_robots.tools.g1.g1_bidi_audio_frame_size_envelope`` pulls
  no ``unitree_sdk2py`` submodule *and* pulls no optional audio
  submodule (the module-load hygiene contract every other file in
  this package carries, refs strands-labs/robots#358). A revision
  of the observed pin is a driver-side update; when the driver's
  bidi audio method lands, its refusal will surface the same
  module-local :data:`_REFUSAL_TEXT` this module names for an
  off-pin frame size.

Why this module quotes a module-local refusal text rather than an
``ERR_CODES`` entry.

The G1 SDK ships no distinct rc for a mis-sized audio frame (the
frame size is a caller-side integer the WebRTC ``AudioProcessor``
consumes; neither the SDK nor ``pywebrtc_audio`` ships a numbered
rc for a bad frame-length argument, and WebRTC's own refusal
happens inside a C++ ``RTC_DCHECK`` that neither returns to nor
surfaces on the Python side). Naming a module-local refusal text
keeps the refusal remedy on the audio-processing surface a caller
reading it can act on, rather than re-borrowing the motion-FSM
``7404`` entry (its nearest neighbour in :data:`ERR_CODES`) whose
text reads ``"Invalid FSM id - need FSM in {500, 501, 801}"`` - a
remedy that would point a planner at locomotion FSM transitions to
fix an audio-processing frame size. The sibling
:mod:`~strands_robots.tools.g1.g1_bidi_audio_stream_delay_envelope`
carries the same convention for the same reason.

What this module does not decide.

* Whether the current host has the audio dependencies installed.
  The neon bundle's ``_probe_bidi`` probe (``pywebrtc_audio`` +
  ``pyaudio`` + ``strands.experimental.bidi.BidiAgent``) is a live
  runtime check answered on the write path; a caller planning a
  bidi start compares an intended frame size against the pin this
  verb surfaces first, and only then reaches the runtime probe for
  the missing-dep refusal.
* The relationship between ``FRAME_SIZE`` and the AEC sample rate.
  The neon bundle pins ``FRAME_SIZE = 160`` at 16 kHz so the frame
  is 10 ms long; a caller reading the sample rate reaches
  :mod:`~strands_robots.tools.g1.g1_bidi_audio_sample_rates` for
  the three-role rate lookup, and colocating the rate here would
  restate a fact the sibling module already carries verbatim.
* The mic capture rate the ``pyaudio`` callback runs at. The neon
  bundle probes an ordered candidate list
  (:mod:`~strands_robots.tools.g1.g1_capture_rate_candidates`
  ports the sweep set) and rescales ``PYAUDIO_FRAMES`` by the
  observed rate; the rescale product is a runtime derivation this
  lookup does not carry. This lookup answers a pure membership
  question on the pinned ``160``-sample value the AEC-processing
  thread requires; the mic-thread's per-callback frame count is a
  derived quantity on top of that pin.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: The single admitted frame size (int16 samples) the neon bundle
#: pins for the WebRTC AEC path. Fixed at ``160`` because WebRTC's
#: ``AudioProcessor::ProcessStream`` admits only 10 ms frames at
#: the sample rate the processor was constructed with
#: (``160 / 16000 == 0.010``), and the neon bundle constructs the
#: processor at 16 kHz (the :mod:`~strands_robots.tools.g1.g1_bidi_audio_sample_rates`
#: ``mic_rate`` role). Named as a plain int rather than as
#: ``(min, max)`` clamp pair because the envelope is a single pin,
#: not a range: WebRTC does not admit a 9 ms or 11 ms frame
#: alternately, and the neon mic thread's remainder-carry logic in
#: ``_frame_residual`` depends on every enqueued frame being
#: exactly this size. A widen to a range would silently break
#: either the AEC (WebRTC refuses at the C++ layer without
#: surfacing) or the mic thread (a wrong-sized carry desyncs the
#: near/far alignment).
_FRAME_SIZE_SAMPLES: int = 160

#: Snapshot of the ``(role -> integer sample count)`` mapping the
#: neon bundle pins today. The two roles are the two threads that
#: consume the same pinned value:
#:
#: * ``aec_frame_size`` - the WebRTC :class:`AudioProcessor` frame
#:   size on the AEC-processing thread. Named on the neon side as
#:   ``FRAME_SIZE = 160``. The AEC contract requires 10 ms frames
#:   at 16 kHz; this is that requirement in int16-sample units.
#: * ``pyaudio_frames_per_buffer`` - the ``frames_per_buffer``
#:   argument the neon mic-capture thread hands the ``pyaudio``
#:   input stream on open. Named on the neon side as
#:   ``PYAUDIO_FRAMES = 160``. Equal to ``aec_frame_size`` at
#:   16 kHz capture; at a higher capture rate the neon thread
#:   rescales it by the observed rate ratio
#:   (``fpb = int(PYAUDIO_FRAMES * capture_rate / MIC_RATE)``) so
#:   the callback still yields a ``FRAME_SIZE``-sized frame after
#:   downsampling.
#:
#: The two entries share the same ``160``-sample pin because both
#: are the same 10 ms envelope viewed from two threads. A caller
#: reading either entry sees the same integer; a caller reading
#: both from the same call sees the pair's identity, which is
#: what makes the neon carry-remainder logic and the AEC frame
#: alignment work as one contract rather than two loosely-related
#: pins.
_ROLE_PINS: tuple[tuple[str, int], ...] = (
    ("aec_frame_size", _FRAME_SIZE_SAMPLES),
    ("pyaudio_frames_per_buffer", _FRAME_SIZE_SAMPLES),
)

#: The module-local refusal text every ``g1_frame_size_admits``
#: refusal quotes when the caller's argument does not match the
#: pinned value. Named here rather than borrowed from
#: :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` because
#: the audio-processing pipeline ships no distinct rc for a
#: mis-sized frame argument and the motion-FSM ``7404`` entry (its
#: nearest neighbour) reads ``"Invalid FSM id - need FSM in {500,
#: 501, 801}"`` - a remedy that points a planner at locomotion FSM
#: transitions to fix an audio pre-processing argument. Surfacing
#: the module-local text keeps the refusal payload's remedy on the
#: same surface the write belongs on, and a future driver-side
#: bidi audio wrapper will surface this same text rather than
#: re-borrowing a motion code.
_REFUSAL_TEXT: str = (
    f"frame_size out of envelope - need frame_size == {_FRAME_SIZE_SAMPLES} "
    f"(10 ms at 16 kHz, the WebRTC AudioProcessor pin)"
)


def _envelope() -> dict[str, Any]:
    """Build the envelope descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_bidi_audio_frame_size_envelope` so
    :func:`g1_frame_size_admits` names the same fields on its
    admitted-path payload and so a widen to the descriptor lands
    in one place. Every field is a snapshot read; no bus is
    touched.
    """
    return {
        "frame_size_samples": _FRAME_SIZE_SAMPLES,
        "roles": [{"role": role, "samples": samples} for role, samples in _ROLE_PINS],
    }


@tool
def g1_list_bidi_audio_frame_size_envelope() -> dict[str, Any]:
    """Return the shared frame-size pin the neon bundle observed as required.

    Read-only.  No driver instance, no DDS, no SDK, no
    ``pywebrtc_audio`` or ``pyaudio`` import: every field is a
    module-level constant. Useful before a future driver-side
    wrapper for ``G1BidiAudioIO`` is called, so a caller can
    compare an intended frame size against the pin the neon bundle
    observed as required by the WebRTC AEC contract and can carry
    the module-local refusal text a driver-side wrapper would
    surface on a mis-sized frame. The two roles ``aec_frame_size``
    (the WebRTC processor's per-frame size) and
    ``pyaudio_frames_per_buffer`` (the mic-input ``frames_per_buffer``
    at 16 kHz capture) share the same ``160``-sample pin, so a
    caller reading either entry sees the same integer.

    Returns:
        A dict with ``status``; an ``envelope`` sub-dict carrying
        ``frame_size_samples`` (the single admitted integer, in
        int16 samples) and ``roles`` (a list of ``(role, samples)``
        descriptors naming the two threads that consume the pin);
        and a ``refusals`` list carrying a single descriptor with
        the module-local :data:`_REFUSAL_TEXT` a future write verb
        would surface on a mis-sized frame. Every field is a
        snapshot of an observed pin or a module-local text; no
        dynamic decode runs here.
    """
    return {
        "status": "success",
        "envelope": _envelope(),
        "refusals": [
            {"text": _REFUSAL_TEXT},
        ],
    }


@tool
def g1_frame_size_admits(frame_size: int = 160) -> dict[str, Any]:
    """Decide whether a ``frame_size`` argument matches the pinned envelope.

    Read-only.  Compares the argument against the single pin
    :func:`g1_list_bidi_audio_frame_size_envelope` returns and
    reports the refusal shape if the value does not match. No
    driver instance, no DDS, no SDK, no ``pywebrtc_audio`` or
    ``pyaudio`` import: the decision reads only module-level
    constants and the argument itself.

    A ``frame_size`` matching the pin is *not* the same as an
    admitted write: the driver's audio singleton may refuse on
    liveness grounds (an in-flight bidi run, a not-yet-constructed
    ``AudioProcessor``, a stalled far-buffer feed), which this
    verb does not read (that is a live driver-instance query
    answered by a future bidi state verb). The returned envelope
    names only the numeric pin decision.

    Args:
        frame_size: integer int16 samples per frame; must be
            exactly ``160`` (the WebRTC 10 ms frame at 16 kHz the
            neon bundle pins on both the ``aec_frame_size`` and
            ``pyaudio_frames_per_buffer`` roles). The default
            ``160`` matches the pin so a caller who does not pass
            an explicit argument lands on the admitted value.
            Every other integer is refused because WebRTC's
            ``AudioProcessor::ProcessStream`` admits only 10 ms
            frames at the constructor's sample rate and the neon
            mic thread's ``_frame_residual`` carry logic depends
            on every enqueued frame being exactly this size; a
            widen would silently break either the AEC (WebRTC
            refuses at the C++ layer without surfacing to Python)
            or the mic thread (a wrong-sized carry desyncs the
            near/far alignment). Boolean values are refused
            explicitly at the boundary because Python's ``bool``
            is a subclass of ``int``, so ``True`` would otherwise
            silently look up ``1`` and hide the type mistake;
            naming the refusal at the boundary surfaces the
            mistake instead. Non-integer numeric values
            (``float``, ``Decimal``) are refused with the same
            shape so a caller passing ``frame_size=160.0`` sees
            an actionable refusal rather than a silent truncation
            the ``AudioProcessor`` constructor would perform.

    Returns:
        A dict with ``status``; an ``admits`` bool naming whether
        the value matches the pin; a ``refusals`` list of refusal
        descriptors, each carrying the dimension name, the
        offending value, the pin it violated, the comparison shape
        (``value != pin`` or ``non-int``), and the module-local
        :data:`_REFUSAL_TEXT` a driver-side wrapper would surface
        if the write were attempted with the mismatched value;
        the same ``envelope`` sub-dict
        :func:`g1_list_bidi_audio_frame_size_envelope` returns.
        On an admitted value the ``refusals`` list is empty; on a
        rejected value the single mismatch is named.
    """
    envelope = _envelope()
    refusals: list[dict[str, Any]] = []

    def _reject(value: Any, cmp: str) -> None:
        refusals.append(
            {
                "dimension": "frame_size",
                "value": value,
                "pin_key": "frame_size_samples",
                "pin": _FRAME_SIZE_SAMPLES,
                "comparison": cmp,
                "text": _REFUSAL_TEXT,
            }
        )

    # bool subclasses int; refuse first so True/False do not silently
    # look up 1/0 and hide a type mistake at the boundary.
    if isinstance(frame_size, bool):
        _reject(frame_size, "non-int")
    elif not isinstance(frame_size, int):
        _reject(frame_size, "non-int")
    else:
        v = int(frame_size)
        if v != _FRAME_SIZE_SAMPLES:
            _reject(frame_size, "value != pin")

    return {
        "status": "success",
        "admits": not refusals,
        "refusals": refusals,
        "envelope": envelope,
    }
