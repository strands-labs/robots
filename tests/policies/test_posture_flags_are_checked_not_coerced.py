"""A posture a policy constructor stores is checked, not coerced with ``bool()``.

``pad_short_actions`` (``lerobot_local``, ``lerobot_async``) and ``walk``
(``wbc``) each select one of two postures rather than scaling a quantity, and
each constructor stored the caller's value through ``bool(...)``;
``lerobot_local``'s ``strict_keys`` and ``cache_model`` were stored as given and
read by truthiness, which is the same inversion without the call. Every
non-empty string is truthy, so ``bool("false")`` is ``True``: the spellings a
caller reaches for to opt OUT selected the posture the word asks to skip, and
``None`` / ``0`` took the other branch without ever being a declared spelling of
it. Nothing raised and nothing logged on either half.

Both postures move a joint. ``pad_short_actions=True`` commands the unmatched
actuators ``0.0``, which is an absolute target on a LeRobot ``<motor>.pos``
follower or a MuJoCo position actuator, so they TRAVEL there
(:func:`~strands_robots.policies.base.align_action_values` says so); measured on
a 6-actuator SO-101 driven by a 4-value chunk, ``pad_short_actions="false"``
swung joints 5 and 6 by -1.06 rad and -1.02 rad where ``False`` held both at
0.0000 rad. ``walk="false"`` loads and prefers the locomotion policy instead of
running the balance policy alone.

So each is now checked with :func:`~strands_robots.utils.boolean_flag_error` -
the domain the package already applies to the postures in ``mesh.iot`` and to
``fast_mode`` via ``SimEngine._validate_posture_flags`` - and the value is
stored as given rather than converted.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.base import align_action_values
from strands_robots.policies.lerobot_async import LerobotAsyncPolicy
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.wbc.policy import WBCPolicy
from strands_robots.utils import boolean_flag_error


def _async(value: Any) -> LerobotAsyncPolicy:
    return LerobotAsyncPolicy(
        server_address="h:1",
        policy_type="act",
        pretrained_name_or_path="x/y",
        pad_short_actions=value,
    )


def _local(value: Any) -> LerobotLocalPolicy:
    return LerobotLocalPolicy(pad_short_actions=value)


def _local_strict(value: Any) -> LerobotLocalPolicy:
    return LerobotLocalPolicy(strict_keys=value)


def _local_cache(value: Any) -> LerobotLocalPolicy:
    return LerobotLocalPolicy(cache_model=value)


def _wbc(value: Any) -> WBCPolicy:
    return WBCPolicy(walk=value, allow_missing_models=True)


#: ``(context label, parameter, factory, attribute)`` for every posture a policy
#: constructor stores. Derived from the ``bool()``-coercion sweep below, which
#: keeps this table and the tree in step.
SITES: list[tuple[str, str, Callable[[Any], Any], str]] = [
    ("lerobot_async", "pad_short_actions", _async, "pad_short_actions"),
    ("lerobot_local", "pad_short_actions", _local, "pad_short_actions"),
    ("lerobot_local", "strict_keys", _local_strict, "strict_keys"),
    ("lerobot_local", "cache_model", _local_cache, "cache_model"),
    ("WBCPolicy", "walk", _wbc, "_walk"),
]

#: Spellings of *off* that ``bool()`` read as *on*. Each selected the posture it
#: asks to skip.
TRUTHY_OFF = ["false", "False", "no", "off", "0", "none"]

#: Values that are not a posture at all: the first three were truthy (*on*), the
#: rest took the other branch while spelling neither.
OTHER_NON_BOOLEAN = ["true", 1.5, [1], None, 0, 0.0, []]

POLICIES_DIR = pathlib.Path(__file__).resolve().parents[2] / "strands_robots" / "policies"


@pytest.mark.parametrize(("context", "param", "factory", "_attr"), SITES, ids=[f"{s[0]}-{s[1]}" for s in SITES])
@pytest.mark.parametrize("spelling", TRUTHY_OFF)
def test_a_truthy_spelling_of_off_is_refused(
    context: str, param: str, factory: Callable[[Any], Any], _attr: str, spelling: str
) -> None:
    """The spellings ``bool()`` inverted are refused, naming the parameter."""
    with pytest.raises(ValueError, match=param):
        factory(spelling)


@pytest.mark.parametrize(("context", "param", "factory", "_attr"), SITES, ids=[f"{s[0]}-{s[1]}" for s in SITES])
@pytest.mark.parametrize("value", OTHER_NON_BOOLEAN, ids=[repr(v) for v in OTHER_NON_BOOLEAN])
def test_a_value_that_is_not_a_posture_is_refused(
    context: str, param: str, factory: Callable[[Any], Any], _attr: str, value: Any
) -> None:
    """Neither branch is taken for a value that spells neither posture."""
    with pytest.raises(ValueError, match=param):
        factory(value)


@pytest.mark.parametrize(("context", "param", "factory", "_attr"), SITES, ids=[f"{s[0]}-{s[1]}" for s in SITES])
def test_the_refusal_is_the_shared_domain_verbatim(
    context: str, param: str, factory: Callable[[Any], Any], _attr: str
) -> None:
    """One domain owns the wording, so the three cannot drift to three answers."""
    expected = boolean_flag_error("false", param, context)
    assert expected is not None
    with pytest.raises(ValueError) as excinfo:
        factory("false")
    assert str(excinfo.value) == expected


@pytest.mark.parametrize(("context", "param", "factory", "attr"), SITES, ids=[f"{s[0]}-{s[1]}" for s in SITES])
@pytest.mark.parametrize("posture", [True, False])
def test_both_postures_are_still_accepted_and_stored_as_given(
    context: str, param: str, factory: Callable[[Any], Any], attr: str, posture: bool
) -> None:
    """The over-reach control: a usable value is honoured, not converted."""
    assert getattr(factory(posture), attr) is posture


@pytest.mark.parametrize(("context", "param", "factory", "attr"), SITES, ids=[f"{s[0]}-{s[1]}" for s in SITES])
def test_a_numpy_boolean_is_a_boolean(context: str, param: str, factory: Callable[[Any], Any], attr: str) -> None:
    """A provider handed a NumPy boolean is not refused - it IS a boolean."""
    assert bool(getattr(factory(np.bool_(True)), attr)) is True


def test_the_two_postures_really_move_different_actuators() -> None:
    """Premise: the flag is load-bearing, so reading it wrong is not cosmetic."""
    keys = ["1", "2", "3", "4", "5", "6"]
    chunk = [0.9, -0.9, 0.9, -0.6]
    padded_values, padded_keys = align_action_values(chunk, keys, pad_short=True)
    omitted_values, omitted_keys = align_action_values(chunk, keys, pad_short=False)
    assert padded_keys == keys and padded_values[4:] == [0.0, 0.0]
    assert omitted_keys == keys[:4] and len(omitted_values) == 4


def _coerced_postures(source: str) -> list[str]:
    """Report ``bool``-annotated parameters a function assigns through ``bool()``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.unparse(node)
        for arg in [*node.args.args, *node.args.kwonlyargs]:
            annotation = ast.unparse(arg.annotation) if arg.annotation else ""
            if annotation.strip() == "bool" and f"bool({arg.arg})" in body:
                found.append(f"{node.name}({arg.arg})")
    return found


def test_no_policy_constructor_coerces_a_posture_with_bool() -> None:
    """The whole package, so the next provider cannot reintroduce the coercion."""
    offenders = {
        f"{path.relative_to(POLICIES_DIR)}::{site}"
        for path in sorted(POLICIES_DIR.rglob("*.py"))
        for site in _coerced_postures(path.read_text(encoding="utf-8"))
    }
    assert offenders == set()


def test_the_coercion_sweep_reports_a_planted_offender() -> None:
    """Non-vacuity: the sweep above passes because the tree is clean."""
    planted = "def __init__(self, walk: bool = True) -> None:\n    self._walk = bool(walk)\n"
    assert _coerced_postures(planted) == ["__init__(walk)"]
