"""The frame-size-envelope lookup tools name the pin the neon bidi bundle uses.

The neon bundle's bidirectional audio IO
(``cagataycali/neon-the-g1/tools/g1_bidi_audio.py``) pins the WebRTC
:class:`AudioProcessor` frame size at a single integer value:
``FRAME_SIZE = 160`` on the AEC-processing thread and
``PYAUDIO_FRAMES = 160`` on the mic-capture thread (which the neon
mic callback rescales by the observed capture rate so the
downsampled frame still lands at ``FRAME_SIZE`` samples). Both
constants are the same 10 ms envelope at 16 kHz
(``160 / 16000 == 0.010``) viewed from two threads; the
:mod:`strands_robots.tools.g1.g1_bidi_audio_frame_size_envelope`
module snapshots that shared pin into module-level constants and
exposes two agent-facing verbs -
:func:`g1_list_bidi_audio_frame_size_envelope` (name the whole
envelope) and :func:`g1_frame_size_admits` (decide one query) - so a
caller can decide the refusal decidably before a future
audio-processing write path is attempted. The tests here fix that
contract without pulling the SDK, ``pywebrtc_audio``, or ``pyaudio``:
the module is loadable on a host without ``unitree_sdk2py`` and
without the WebRTC or PortAudio Python bindings (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off
the module's own snapshot rather than restated in the tests, so a
widen or narrow of the observed pin surfaces here as a shape change
rather than as a diverging table this file would need to manually
update.

Two things this file's cells deliberately do not pin:

* The WebRTC library's own answer at wire time. The envelope is
  the neon bundle's observed pin, not WebRTC's
  ``AudioProcessor::ProcessStream`` refusal at the C++ layer (which
  ``RTC_DCHECK``s a mis-sized frame without surfacing to the
  Python caller). A driver-side wrapper for the bidi IO that
  lands later will re-check the pin at wire time and its refusal
  string will surface the same module-local :data:`_REFUSAL_TEXT`
  the admits-verb quotes today.
* The live bidi state. Whether the ``G1BidiAudioIO`` singleton is
  currently constructed, whether the mic autopick has resolved a
  device, whether the far-buffer queue is draining: those are
  live driver-instance reads and belong on a future bidi state
  verb; the envelope surfaces only the numeric pin decision.

One property this file explicitly refuses to pin: the ``7404``
motion-FSM refusal code from
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES`. That code
is the driver's :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
refusal on ``rt/lowcmd`` writes and its decoded text reads
``"Invalid FSM id - need FSM in {500, 501, 801}"`` - a remedy that
points at locomotion FSM transitions, not the audio-processing
surface a mis-sized frame belongs on. The audio pipeline ships no
distinct rc for a mis-sized frame argument, and the refusal text
this module surfaces is module-local so a planner reading a
frame-size refusal sees a remedy that matches the surface, not a
re-borrowed motion FSM code. Cells below pin only the module-local
text; a re-borrowing of ``7404`` would fail
``test_the_refusal_text_names_the_frame_size_envelope_not_the_motion_fsm``.
"""

from __future__ import annotations

import importlib
import sys
from decimal import Decimal
from typing import Any

import pytest

from strands_robots.tools.g1._g1_common import ERR_CODES
from strands_robots.tools.g1.g1_bidi_audio_frame_size_envelope import (
    _FRAME_SIZE_SAMPLES,
    _REFUSAL_TEXT,
    _ROLE_PINS,
    g1_frame_size_admits,
    g1_list_bidi_audio_frame_size_envelope,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process; this helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be
    importable with the SDK absent (refs strands-labs/robots#358);
    a module that pulled a submodule at import time would break
    every headless CI runner and Thor before an office bring-up.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_bidi_audio_frame_size_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bidi_audio_frame_size_envelope "
        f"imports pulled SDK submodules: {leaked}. The rule for this "
        f"package is that the SDK loads only inside function bodies "
        f"(refs strands-labs/robots#358)."
    )


def test_the_import_pulls_no_pywebrtc_audio_module() -> None:
    """The tool module is loadable on a host without ``pywebrtc_audio``.

    The neon bundle's ``g1_bidi_audio`` imports ``pywebrtc_audio``
    at module load to construct the WebRTC :class:`AudioProcessor`;
    the envelope port must not close that dependency on this
    module so a headless CI runner can decide the numeric refusal
    without the audio stack present. Pinned here so a future edit
    that reaches into ``pywebrtc_audio`` at import time (for a
    compile-time frame-size bound, say) fails this cell first, not
    as a dependency surprise on a mesh peer that never installs
    the audio stack.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_bidi_audio_frame_size_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if "pywebrtc" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bidi_audio_frame_size_envelope "
        f"imports pulled pywebrtc_audio submodules: {leaked}. The "
        f"envelope port is numeric-only; the WebRTC library "
        f"belongs inside the driver-side wrapper that lands "
        f"later, refs strands-labs/robots#358."
    )


def test_the_import_pulls_no_pyaudio_module() -> None:
    """The tool module is loadable on a host without ``pyaudio``.

    The neon bundle's ``g1_bidi_audio`` also imports ``pyaudio``
    for the mic-capture callback; the envelope port must not
    close that dependency on this module either. Pinned as a
    sibling of the ``pywebrtc_audio`` scan so a future edit that
    reaches into ``pyaudio`` at import time fails here rather
    than as a PortAudio-linker surprise on a CI runner that never
    installs the sound backend.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_bidi_audio_frame_size_envelope")
    after = set(sys.modules)
    leaked = {name for name in after - before if name.lower() == "pyaudio" or name.lower().startswith("pyaudio.")}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_bidi_audio_frame_size_envelope "
        f"imports pulled pyaudio submodules: {leaked}. The envelope "
        f"port is numeric-only; the PortAudio binding belongs "
        f"inside the driver-side wrapper that lands later, refs "
        f"strands-labs/robots#358."
    )


def test_the_pin_is_a_positive_integer() -> None:
    """The pinned frame size is a positive plain integer.

    A non-int pin would let a caller passing ``160.0`` slip through
    the type-refusal path; a zero or negative pin would name an
    audio frame no thread can carry. Pins the invariant so a
    typo-driven widen (e.g. ``_FRAME_SIZE_SAMPLES = 160.0``)
    surfaces here rather than as a silent Boolean-passes-int
    admission at runtime.
    """
    assert isinstance(_FRAME_SIZE_SAMPLES, int) and not isinstance(_FRAME_SIZE_SAMPLES, bool), (
        f"_FRAME_SIZE_SAMPLES is not a plain int: {_FRAME_SIZE_SAMPLES!r}"
    )
    assert _FRAME_SIZE_SAMPLES > 0, (
        f"_FRAME_SIZE_SAMPLES {_FRAME_SIZE_SAMPLES} is not positive; a "
        f"non-positive frame size would name an audio frame no "
        f"thread can carry."
    )


def test_the_pin_matches_the_neon_observed_value() -> None:
    """The pinned frame size matches the neon-bundle-observed ``160`` samples.

    The neon bundle names ``FRAME_SIZE = 160`` on the
    AEC-processing thread and ``PYAUDIO_FRAMES = 160`` on the
    mic-capture thread; both are the same 10 ms envelope at 16 kHz
    (``160 / 16000 == 0.010``). Pinning the number here surfaces
    a drift in either direction: a widen to ``320`` (a 20 ms
    frame, which WebRTC refuses at the C++ layer) or a narrow to
    ``80`` (a 5 ms frame, which the mic-thread carry logic
    desyncs on) would fail this cell first.
    """
    assert _FRAME_SIZE_SAMPLES == 160


def test_the_pin_encodes_a_10_ms_frame_at_16_khz() -> None:
    """The pin evaluates to exactly 10 ms at the ``mic_rate`` sample rate.

    The neon pin is the WebRTC AEC contract's frame-length
    requirement expressed in samples at the constructor's
    16 kHz sample rate. Encoding the mathematical relationship as
    a cell surfaces a future edit that changes only one of the
    two numbers (frame size or sample rate) without also updating
    the sibling; pins the relationship at
    ``160 / 16000 == 0.010`` (10 ms).
    """
    mic_rate_hz = 16000
    frame_duration_seconds = _FRAME_SIZE_SAMPLES / mic_rate_hz
    assert frame_duration_seconds == pytest.approx(0.010), (
        f"pin {_FRAME_SIZE_SAMPLES} samples at {mic_rate_hz} Hz "
        f"encodes {frame_duration_seconds * 1000:.3f} ms, not 10 ms; "
        f"the WebRTC AudioProcessor contract admits only 10 ms "
        f"frames at the constructor's sample rate."
    )


def test_the_role_pins_are_two_entries_at_the_shared_value() -> None:
    """The ``_ROLE_PINS`` snapshot names two entries at the same pin.

    The neon bundle uses two constants (``FRAME_SIZE`` on the
    AEC-processing thread, ``PYAUDIO_FRAMES`` on the mic-capture
    thread) that share the same ``160``-sample value because both
    are the same 10 ms envelope viewed from two threads. Pinning
    the count and the shared value here surfaces a future edit
    that adds a third role, drops one of the two, or splits the
    shared pin into two different integers.
    """
    assert len(_ROLE_PINS) == 2, (
        f"_ROLE_PINS has {len(_ROLE_PINS)} entries, expected 2 (``aec_frame_size`` and ``pyaudio_frames_per_buffer``)."
    )
    role_names = [role for role, _ in _ROLE_PINS]
    assert set(role_names) == {"aec_frame_size", "pyaudio_frames_per_buffer"}, (
        f"_ROLE_PINS role names are {role_names!r}, expected the two neon-bundle roles."
    )
    for role, samples in _ROLE_PINS:
        assert samples == _FRAME_SIZE_SAMPLES, (
            f"role {role!r} carries pin {samples}, expected the "
            f"shared _FRAME_SIZE_SAMPLES {_FRAME_SIZE_SAMPLES}. "
            f"Both roles must share the pin because both are the "
            f"same 10 ms envelope viewed from two threads."
        )


def test_the_role_pins_have_no_duplicate_role_names() -> None:
    """Each role in ``_ROLE_PINS`` names a distinct thread.

    A duplicate role would let a widen accidentally list the same
    role twice with two different values; the admits verb would
    then read a stable pin off ``_FRAME_SIZE_SAMPLES`` while the
    list verb surfaced two inconsistent samples for the same
    role. Pin the uniqueness so the drift surfaces here.
    """
    role_names = [role for role, _ in _ROLE_PINS]
    assert len(role_names) == len(set(role_names)), f"_ROLE_PINS carries duplicate role names: {role_names!r}"


def test_the_refusal_text_names_the_frame_size_envelope_not_the_motion_fsm() -> None:
    """The refusal text is module-local, not a re-borrowed motion FSM code.

    The G1 driver's :meth:`_check_motion_gates` refuses locomotion
    writes with rc=``7404`` whose text reads ``"Invalid FSM id -
    need FSM in {500, 501, 801}"``. The ``AudioProcessor`` runs
    on the AEC-processing thread in the Python process itself and
    never touches ``rt/lowcmd``; the audio-processing pipeline
    ships no distinct rc for a mis-sized frame argument. The
    refusal shape this module surfaces is module-local text that
    names the frame-size envelope (not the motion FSM) so an
    agent planner reading a frame-size refusal sees a remedy on
    the same surface the write belongs on. Pinned here so a
    re-borrowing of ``7404`` (or any other motion-FSM entry from
    ``ERR_CODES``) fails this cell first, not as a wrong-remedy
    surprise in production.
    """
    assert isinstance(_REFUSAL_TEXT, str) and _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT is not a non-empty string: {_REFUSAL_TEXT!r}"
    )
    assert "frame_size" in _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT does not name the frame_size dimension: "
        f"{_REFUSAL_TEXT!r}. A caller reading the refusal must see "
        f"a remedy on the audio-processing surface."
    )
    fsm_text = ERR_CODES[7404]
    assert _REFUSAL_TEXT != fsm_text, (
        f"_REFUSAL_TEXT re-borrows the motion-FSM ``7404`` text "
        f"{fsm_text!r}. The AudioProcessor runs on the mic "
        f"pre-processing thread in-process and never touches "
        f"rt/lowcmd; the refusal shape must be module-local so a "
        f"planner does not read a motion FSM remedy for a "
        f"frame-size error."
    )
    assert "FSM" not in _REFUSAL_TEXT, (
        f"_REFUSAL_TEXT names the motion FSM: {_REFUSAL_TEXT!r}. "
        f"The AudioProcessor runs on the mic pre-processing thread "
        f"in-process; the refusal remedy belongs on the "
        f"audio-processing surface, not the locomotion FSM."
    )


def test_g1_list_bidi_audio_frame_size_envelope_returns_the_full_envelope() -> None:
    """The verb's payload names the pin, every role, and the refusal.

    ``envelope`` carries the pin integer and the two role
    descriptors; ``refusals`` names the module-local
    :data:`_REFUSAL_TEXT` a future driver-side bidi audio wrapper
    would surface on a mis-sized frame.
    """
    result = _call(g1_list_bidi_audio_frame_size_envelope)
    assert result["status"] == "success"
    env = result["envelope"]
    assert env["frame_size_samples"] == _FRAME_SIZE_SAMPLES
    assert len(env["roles"]) == 2
    role_map = {row["role"]: row["samples"] for row in env["roles"]}
    assert role_map == {
        "aec_frame_size": _FRAME_SIZE_SAMPLES,
        "pyaudio_frames_per_buffer": _FRAME_SIZE_SAMPLES,
    }
    assert result["refusals"] == [{"text": _REFUSAL_TEXT}]


def test_g1_list_bidi_audio_frame_size_envelope_refusal_omits_a_borrowed_code() -> None:
    """The list-envelope refusal descriptor names no ``code`` field.

    A ``code`` field on this refusal would only be honest if the
    audio-processing pipeline shipped a distinct rc for a
    mis-sized frame argument, and it does not; borrowing a
    motion-FSM code from
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` would
    hand a planner a wrong-surface remedy. Pins the omission so a
    future re-introduction of ``code`` fails this cell first.
    """
    result = _call(g1_list_bidi_audio_frame_size_envelope)
    for refusal in result["refusals"]:
        assert "code" not in refusal, (
            f"refusal descriptor carries a ``code`` field: {refusal!r}. "
            f"The audio-processing pipeline ships no rc for a "
            f"mis-sized frame argument; borrowing one from "
            f"ERR_CODES puts a wrong-surface remedy on an "
            f"audio-processing refusal."
        )


def test_g1_list_bidi_audio_frame_size_envelope_returns_fresh_containers() -> None:
    """Successive calls do not share the envelope dict, roles list, or refusals list.

    A caller mutating one call's ``envelope`` (or ``roles``,
    ``refusals``) must not affect the next call's payload. Pins
    the isolation so a mutation-at-callsite bug lands here rather
    than as a ghost-state-across-calls regression in production.
    """
    first = _call(g1_list_bidi_audio_frame_size_envelope)
    second = _call(g1_list_bidi_audio_frame_size_envelope)
    assert first is not second
    assert first["envelope"] is not second["envelope"]
    assert first["envelope"]["roles"] is not second["envelope"]["roles"]
    assert first["refusals"] is not second["refusals"]


def test_g1_frame_size_admits_the_pin() -> None:
    """The pinned value is admitted with an empty refusals list.

    ``frame_size=160`` matches ``FRAME_SIZE`` in the neon bundle -
    a caller who reads the envelope's ``frame_size_samples`` and
    passes it to the admits verb must see an admit, not a refuse.
    Pins the round-trip so a narrow of the pin that drops the
    neon value fails this cell first.
    """
    result = _call(g1_frame_size_admits, frame_size=160)
    assert result["status"] == "success"
    assert result["admits"] is True
    assert result["refusals"] == []


def test_g1_frame_size_admits_uses_the_pin_as_its_default() -> None:
    """The verb's default argument matches the pin so no-arg calls admit.

    A caller who invokes ``g1_frame_size_admits()`` without an
    explicit argument must land on an admitted result; a drift
    between the default and the pin would silently refuse the
    zero-argument call. Pins the default so a future edit that
    changes the pin without also updating the default fails here.
    """
    result = _call(g1_frame_size_admits)
    assert result["admits"] is True
    assert result["refusals"] == []


@pytest.mark.parametrize("frame_size", [0, 1, 80, 159, 161, 320, 480, 1024, -1])
def test_g1_frame_size_admits_refuses_a_value_off_the_pin(frame_size: int) -> None:
    """A ``frame_size`` not matching the pin reads as one refusal.

    Every integer other than ``160`` is refused because WebRTC
    admits only the pinned 10 ms frame length. Common off-pin
    guesses (``0`` the null frame, ``80`` a 5 ms frame, ``161``
    an off-by-one, ``320`` a 20 ms frame, ``480`` the raw 48 kHz
    mic count before downsampling, ``1024`` the pyaudio default,
    ``-1`` the "any" sentinel) each read as a single refusal
    naming the pin and the module-local remedy.
    """
    result = _call(g1_frame_size_admits, frame_size=frame_size)
    assert result["admits"] is False, (
        f"frame_size={frame_size} was admitted but does not match the pin {_FRAME_SIZE_SAMPLES}"
    )
    assert len(result["refusals"]) == 1
    refusal = result["refusals"][0]
    assert refusal["dimension"] == "frame_size"
    assert refusal["value"] == frame_size
    assert refusal["pin_key"] == "frame_size_samples"
    assert refusal["pin"] == _FRAME_SIZE_SAMPLES
    assert refusal["comparison"] == "value != pin"
    assert refusal["text"] == _REFUSAL_TEXT
    assert "code" not in refusal, (
        f"off-pin refusal carries a ``code`` field: {refusal!r}. "
        f"The audio-processing pipeline ships no rc for a "
        f"mis-sized frame argument."
    )


def test_g1_frame_size_admits_refuses_a_bool_frame_size() -> None:
    """A ``bool`` ``frame_size`` reads as refused with a ``non-int`` comparison.

    Python's ``bool`` is a subclass of ``int``, so ``True`` would
    otherwise silently look up ``1`` (a legitimate off-pin
    integer, refused by the value branch anyway) and hide the
    type mistake behind a refuse-with-wrong-comparison result.
    Naming the refusal at the boundary surfaces the mistake with
    the right shape.
    """
    for value in (True, False):
        result = _call(g1_frame_size_admits, frame_size=value)
        assert result["admits"] is False, f"bool {value!r} was admitted as a frame_size"
        refusal = result["refusals"][0]
        assert refusal["dimension"] == "frame_size"
        assert refusal["comparison"] == "non-int"
        assert refusal["text"] == _REFUSAL_TEXT


@pytest.mark.parametrize(
    "frame_size",
    [160.0, 160.5, Decimal("160"), "160", None, [], (160,)],
)
def test_g1_frame_size_admits_refuses_a_non_int_frame_size(frame_size: Any) -> None:
    """A non-int-non-bool ``frame_size`` reads as refused with a ``non-int`` comparison.

    ``float`` values (even integer-valued like ``160.0``) are
    refused because the WebRTC ``AudioProcessor`` and ``pyaudio``
    ``frames_per_buffer`` expect an integer; a caller passing
    ``160.0`` learns the shape mistake here rather than at wire
    time via a silent truncation. ``Decimal`` and ``str`` follow
    the same rule (a str is refused rather than parsed - the
    verb does not fabricate a value the caller did not supply).
    """
    result = _call(g1_frame_size_admits, frame_size=frame_size)
    assert result["admits"] is False, f"{frame_size!r} was admitted as a frame_size"
    refusal = result["refusals"][0]
    assert refusal["dimension"] == "frame_size"
    assert refusal["comparison"] == "non-int"
    assert refusal["text"] == _REFUSAL_TEXT


def test_g1_frame_size_admits_carries_the_envelope_on_admit_and_refuse() -> None:
    """The verb returns the same envelope shape on admitted and refused paths.

    A caller reading ``envelope`` from a refused result must see
    the same shape as one reading it from an admitted result, so
    the payload does not switch between two schemas depending on
    the verdict.
    """
    admitted = _call(g1_frame_size_admits, frame_size=160)
    refused = _call(g1_frame_size_admits, frame_size=161)
    assert admitted["envelope"].keys() == refused["envelope"].keys()
    assert admitted["envelope"]["frame_size_samples"] == refused["envelope"]["frame_size_samples"]
    assert admitted["envelope"]["roles"] == refused["envelope"]["roles"]


def test_g1_frame_size_admits_declares_a_non_handle_first_parameter_type() -> None:
    """The ``frame_size`` parameter is annotated ``int``, not ``Any``.

    ``Any`` is the annotation the derived
    ``TestEveryLiveHandleVerbRefusesAWrongHandle`` scanner in
    ``tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py``
    keys on to grade a verb as a live-handle verb, and a
    live-handle verb owes an ``{"status": "error"}`` envelope on a
    wrong handle - a shape ``g1_frame_size_admits`` does not owe
    because its first parameter is a numeric pin, not a live
    driver instance.  The same guard is pinned on the sibling
    envelope-verb tests; this cell keeps this port in the same
    shape.

    This guard reads the annotation off the wrapped function so a
    future widen back to ``Any`` fails this cell first, rather
    than re-entering the live-handle population and tripping the
    scanner a second time.
    """
    import inspect

    undecorated = getattr(g1_frame_size_admits, "__wrapped__", g1_frame_size_admits)
    signature = inspect.signature(undecorated)
    parameter = signature.parameters["frame_size"]
    assert parameter.annotation in ("int", int), (
        f"g1_frame_size_admits.frame_size annotation is "
        f"{parameter.annotation!r}; Any would re-enter the "
        f"live-handle-verb population and trip the wrong-handle "
        f"scanner in tests/tools/g1/."
    )
