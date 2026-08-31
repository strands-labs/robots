"""Pin: ``validate_command`` numeric coercion handles ``None`` and non-numbers.

Companion to ``test_validate_command_finite_numerics`` (which pins the NaN/inf
defences). This pins the two remaining edges of the shared ``_coerce_float`` /
``_coerce_int`` helpers in ``strands_robots.mesh.security``:

* **None handling.** A numeric field present with value ``None`` is NOT a free
  pass. When the field carries a default (``start.duration`` -> 30.0,
  ``step.steps`` -> 1) the helper substitutes that default; when it has no
  default (``control_frequency``, ``n_steps``) the helper raises
  :class:`ValidationError` with ``"<field> is required"`` rather than silently
  forwarding ``None`` to the robot adapter. Silent defaults are not honoured on
  the security boundary.
* **Type rejection.** A non-number (``str``/``list``) and -- critically --
  ``bool`` (a subclass of ``int`` that must not be accepted as a count or rate)
  raise :class:`ValidationError`, so a payload like ``{"control_frequency":
  true}`` cannot smuggle ``1`` past the validator.

All assertions go through the public ``validate_command`` surface so they pin
observable behavior, not the private helpers.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh.security import ValidationError, validate_command


def _start_cmd(**overrides):
    """Minimal valid ``start`` command; overrides one policy field."""
    base = {"action": "start", "instruction": "pick up the cube", "policy_provider": "mock"}
    base.update(overrides)
    return base


# --- None substitutes the default when the field has one -------------------


def test_start_duration_none_falls_back_to_default():
    """``duration`` has a 30.0 default, so an explicit None resolves to it."""
    out = validate_command(_start_cmd(duration=None))
    assert out["duration"] == 30.0


def test_step_steps_none_falls_back_to_default():
    """``step.steps`` defaults to 1, so an explicit None resolves to 1."""
    out = validate_command({"action": "step", "steps": None})
    assert out["steps"] == 1


# --- None is rejected when the field has no default ------------------------


def test_start_control_frequency_none_is_required():
    """``control_frequency`` has no default; None must raise, not pass through."""
    with pytest.raises(ValidationError, match="control_frequency is required"):
        validate_command(_start_cmd(control_frequency=None))


def test_start_n_steps_none_is_required():
    """``n_steps`` has no default; None must raise, not pass through."""
    with pytest.raises(ValidationError, match="n_steps is required"):
        validate_command(_start_cmd(n_steps=None))


# --- Non-numeric types are rejected (bool included) ------------------------


@pytest.mark.parametrize("bad", ["fast", [50.0], {"hz": 50}])
def test_start_control_frequency_rejects_non_number(bad):
    """A float field rejects str / list / dict with a 'must be a number' error."""
    with pytest.raises(ValidationError, match="control_frequency must be a number"):
        validate_command(_start_cmd(control_frequency=bad))


def test_start_control_frequency_rejects_bool():
    """``bool`` is an ``int`` subclass but must NOT be accepted as a rate."""
    with pytest.raises(ValidationError, match="control_frequency must be a number, got bool"):
        validate_command(_start_cmd(control_frequency=True))


@pytest.mark.parametrize("bad", ["3", [3], {"n": 3}])
def test_step_steps_rejects_non_integer(bad):
    """``step.steps`` (an int field) rejects str / list / dict.

    A float is accepted only when it is integral (``3.0``); a fractional one is
    refused under its own name rather than truncated, which
    ``test_validate_command_counts_are_whole_numbers`` pins.
    """
    with pytest.raises(ValidationError, match="steps must be an integer"):
        validate_command({"action": "step", "steps": bad})


def test_step_steps_rejects_bool():
    """``bool`` must not be accepted as an integer count."""
    with pytest.raises(ValidationError, match="steps must be an integer, got bool"):
        validate_command({"action": "step", "steps": True})
