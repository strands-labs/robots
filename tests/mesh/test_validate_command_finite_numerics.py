"""Pin: ``validate_command`` rejects NaN / inf / non-coercible numerics.

Two defences in the numeric coercion helpers in
``strands_robots/mesh/security.py``:

* ``_coerce_float`` rejects NaN and +/-inf via ``math.isfinite``.
  Without the check, ``nan < lo`` and ``nan > hi`` are both False
  (IEEE-754), so a payload like ``{"duration": NaN}`` would pass the
  bounds clamp, reach the robot adapter, and turn ``time.sleep(nan)``
  / ``time.monotonic() + nan`` into a never-terminating deadline.
* ``_coerce_int`` wraps ``int(...)`` so NaN/inf and overflowing values
  raise :class:`ValidationError` instead of bare ``ValueError`` /
  ``OverflowError``. ``_exec_cmd`` only catches ``ValidationError``,
  so a bare exception would bypass the structured ``command_rejected``
  audit + wire response and surface as a generic "dispatch error".

These tests pin the rejection class + path on every numeric field
that flows through ``validate_command`` so a future refactor that
drops a guard is caught here.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.mesh.security import ValidationError, validate_command


def _execute_cmd(**overrides):
    """Build a minimal valid ``execute`` cmd; overrides one field."""
    base = {
        "action": "execute",
        "instruction": "go",
        "policy_provider": "mock",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_execute_duration_rejects_non_finite(bad):
    """``duration`` must be finite -- pre-fix this silently passed for NaN."""
    with pytest.raises(ValidationError, match="finite"):
        validate_command(_execute_cmd(duration=bad))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_step_count_rejects_non_finite(bad):
    """``step.steps`` is coerced via ``_coerce_int``; NaN/inf must raise ValidationError, not ValueError."""
    cmd = {"action": "step", "steps": bad}
    with pytest.raises(ValidationError):
        validate_command(cmd)


def test_step_count_rejects_nan_via_validation_error_not_value_error():
    """Pin H2: the rejection class must be ValidationError so ``_exec_cmd`` audits it.

    Pre-fix code raised a bare ``ValueError("cannot convert float NaN to integer")``
    from ``int(nan)``; ``_exec_cmd`` only catches ``ValidationError`` so the
    bare ValueError bypassed the structured ``command_rejected`` audit.
    """
    cmd = {"action": "step", "steps": float("nan")}
    raised = None
    try:
        validate_command(cmd)
    except Exception as e:  # noqa: BLE001 -- intentionally broad to detect non-ValidationError regression
        raised = e
    assert raised is not None, "NaN must raise; got silent passthrough"
    assert isinstance(raised, ValidationError), (
        f"NaN must raise ValidationError, not bare {type(raised).__name__}: {raised}"
    )


def test_execute_duration_in_bounds_still_accepted():
    """Sanity: legitimate finite ``duration`` values still pass through."""
    out = validate_command(_execute_cmd(duration=42.0))
    assert out["duration"] == 42.0


def test_step_count_finite_int_still_accepted():
    """Sanity: legitimate finite ``steps`` values still pass through."""
    out = validate_command({"action": "step", "steps": 5})
    assert out["steps"] == 5


def test_execute_duration_overflow_raises_validation_error():
    """Pin H2 partner: a Python int that overflows IEEE-754 still raises ValidationError."""
    # Float coercion of huge int would overflow; ensure we surface
    # ValidationError from the wrapped try/except.
    huge = 10**400  # bigger than IEEE-754 max
    with pytest.raises(ValidationError):
        validate_command(_execute_cmd(duration=huge))


def test_no_silent_nan_passthrough():
    """End-to-end: build the command, validate, assert no ``nan`` sneaks into ``out``."""
    cmd = _execute_cmd(duration=float("nan"))
    with pytest.raises(ValidationError):
        validate_command(cmd)
    # If somehow it went through (regression), this would catch it:
    try:
        out = validate_command(cmd)
    except ValidationError:
        return  # expected
    pytest.fail(f"NaN must never appear in out; got {out!r}")


def test_isnan_check_independent_from_bounds():
    """Verify the isfinite check fires regardless of the bounds setup."""
    # With duration=nan, even though nan < 0 and nan > 3600 are both False
    # (so a bounds-only check would pass), isfinite catches it.
    assert not math.isfinite(float("nan")), "sanity"
    with pytest.raises(ValidationError, match="finite"):
        validate_command(_execute_cmd(duration=float("nan")))
