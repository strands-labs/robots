"""Every consolidated g1 execution verb writes the driver envelope through.

One table-driven suite for the thirteen verbs in
:mod:`strands_robots.tools.g1.g1_actions`, replacing the file-per-verb suites
the ports carried. The facts pinned per verb are the same four the originals
pinned: the import pulls no SDK module, an unusable ``driver`` handle is
refused with an envelope naming the verb and the parameter, the data
parameters are refused by name before the driver is touched, and a usable
driver is called exactly once with the envelope returned verbatim (success or
refusal alike - the verbs do not reshape).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from strands_robots.tools.g1.g1_actions import (
    _ACTIONS,
    g1_arm_action,
    g1_balance_stand,
    g1_move_velocity,
    g1_release_arm,
    g1_safe_lie_to_stand,
    g1_safe_squat_to_stand,
    g1_safe_stand_to_squat,
    g1_set_fsm,
    g1_set_stand_height,
    g1_set_swing_height,
    g1_shake_hand_loco,
    g1_stop_move,
    g1_wave_hand_loco,
)

# verb function, accessor, happy-path kwargs, positional args the driver
# method must receive, kwargs the driver method must receive.
_VERBS: list[tuple[Any, str, dict[str, Any], tuple[Any, ...], dict[str, Any]]] = [
    (g1_arm_action, "arm_action", {"action": "clap"}, ("clap", None), {}),
    (g1_balance_stand, "balance_stand", {"balance_mode": 0}, (0,), {}),
    (
        g1_move_velocity,
        "move_velocity",
        {"vx": 0.2, "vy": 0.0, "vyaw": 0.1, "duration": 2.0},
        (0.2, 0.0, 0.1, 2.0),
        {},
    ),
    (g1_release_arm, "release_arm", {}, (), {}),
    (g1_safe_lie_to_stand, "safe_lie_to_stand", {"preamble_s": 0.5}, (0.5,), {}),
    (g1_safe_squat_to_stand, "safe_squat_to_stand", {"preamble_s": 0.5}, (0.5,), {}),
    (g1_safe_stand_to_squat, "safe_stand_to_squat", {"preamble_s": 0.5}, (0.5,), {}),
    (g1_set_fsm, "set_fsm", {"fsm_id": 500, "wait": 1.0}, (500,), {"wait": 1.0}),
    (g1_set_stand_height, "set_stand_height", {"height": 0.5}, (0.5,), {}),
    (g1_set_swing_height, "set_swing_height", {"height": 0.1}, (0.1,), {}),
    (g1_shake_hand_loco, "shake_hand_loco", {"stage": 0}, (0,), {}),
    (g1_stop_move, "stop_move", {}, (), {}),
    (g1_wave_hand_loco, "wave_hand_loco", {"turn_flag": False}, (False,), {}),
]

_IDS = [row[1] for row in _VERBS]

# verb function, kwargs that must be refused, substrings the refusal names.
_PARAM_REFUSALS: list[tuple[Any, dict[str, Any], tuple[str, ...]]] = [
    (g1_arm_action, {}, ("g1_arm_action", "`action`", "`action_id`")),
    (g1_balance_stand, {}, ("g1_balance_stand", "`balance_mode`", "required")),
    (g1_balance_stand, {"balance_mode": True}, ("g1_balance_stand", "`balance_mode`", "bool")),
    (g1_balance_stand, {"balance_mode": "static"}, ("g1_balance_stand", "`balance_mode`", "'str'")),
    (g1_move_velocity, {"vy": 0.0, "vyaw": 0.0, "duration": 1.0}, ("g1_move_velocity", "vx")),
    (g1_move_velocity, {"vx": 0.0, "vyaw": 0.0, "duration": 1.0}, ("g1_move_velocity", "vy")),
    (g1_move_velocity, {"vx": 0.0, "vy": 0.0, "duration": 1.0}, ("g1_move_velocity", "vyaw")),
    (g1_move_velocity, {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}, ("g1_move_velocity", "duration")),
    (
        g1_move_velocity,
        {"vx": float("nan"), "vy": 0.0, "vyaw": 0.0, "duration": 1.0},
        ("g1_move_velocity", "vx"),
    ),
    (
        g1_move_velocity,
        {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "duration": -1.0},
        ("g1_move_velocity", "duration"),
    ),
    (g1_safe_lie_to_stand, {"preamble_s": 0.0}, ("g1_safe_lie_to_stand", "preamble_s")),
    (g1_safe_squat_to_stand, {"preamble_s": float("inf")}, ("g1_safe_squat_to_stand", "preamble_s")),
    (g1_safe_stand_to_squat, {"preamble_s": -0.5}, ("g1_safe_stand_to_squat", "preamble_s")),
    (g1_set_fsm, {}, ("g1_set_fsm", "`fsm_id`", "required")),
    (g1_set_fsm, {"fsm_id": True}, ("g1_set_fsm", "`fsm_id`", "'bool'")),
    (g1_set_fsm, {"fsm_id": "damp"}, ("g1_set_fsm", "`fsm_id`", "'str'")),
    (g1_set_fsm, {"fsm_id": 500, "wait": 0.0}, ("g1_set_fsm", "wait")),
    (g1_set_stand_height, {}, ("g1_set_stand_height", "`height`", "required")),
    (g1_set_stand_height, {"height": float("nan")}, ("g1_set_stand_height", "height")),
    (g1_set_swing_height, {}, ("g1_set_swing_height", "`height`", "required")),
    (g1_set_swing_height, {"height": "high"}, ("g1_set_swing_height", "height")),
    (g1_shake_hand_loco, {}, ("g1_shake_hand_loco", "`stage`", "required")),
    (g1_shake_hand_loco, {"stage": False}, ("g1_shake_hand_loco", "`stage`", "bool")),
    (g1_shake_hand_loco, {"stage": 1.5}, ("g1_shake_hand_loco", "`stage`", "'float'")),
    (g1_wave_hand_loco, {}, ("g1_wave_hand_loco", "`turn_flag`", "required")),
    (g1_wave_hand_loco, {"turn_flag": 1}, ("g1_wave_hand_loco", "`turn_flag`", "'int'")),
]


class _Recorder:
    """A driver stub exposing one accessor that records its call."""

    def __init__(self, accessor: str, envelope: dict[str, Any]) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._envelope = envelope

        def _method(*args: Any, **kwargs: Any) -> dict[str, Any]:
            self.calls.append((args, kwargs))
            return self._envelope

        setattr(self, accessor, _method)


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in envelope["content"])


def test_the_import_pulls_no_sdk_module() -> None:
    """SDK-load hygiene: importing the verbs must not import unitree_sdk2py."""
    importlib.import_module("strands_robots.tools.g1.g1_actions")
    pulled = [name for name in sys.modules if name.startswith("unitree_sdk2py")]
    assert not pulled, f"strands_robots.tools.g1.g1_actions imports pulled SDK submodules: {pulled}"


def test_every_verb_has_an_actions_table_row() -> None:
    """The suite's table and the module's table cover the same verbs."""
    assert sorted(_IDS) == sorted(accessor for accessor, _, _ in _ACTIONS.values())
    assert sorted(row[0].__name__ for row in _VERBS) == sorted(_ACTIONS)


@pytest.mark.parametrize(("verb", "accessor", "kwargs", "args", "call_kwargs"), _VERBS, ids=_IDS)
class TestEveryVerbHoldsTheHandleContract:
    def test_a_missing_driver_is_refused_naming_the_parameter(
        self, verb: Any, accessor: str, kwargs: dict[str, Any], args: Any, call_kwargs: Any
    ) -> None:
        out = verb(driver=None, **kwargs)
        assert out["status"] == "error"
        assert verb.__name__ in _text(out)
        assert "`driver`" in _text(out)

    def test_a_wrong_shape_driver_is_refused_naming_the_type(
        self, verb: Any, accessor: str, kwargs: dict[str, Any], args: Any, call_kwargs: Any
    ) -> None:
        out = verb(driver="g1-lab", **kwargs)
        assert out["status"] == "error"
        assert "'str'" in _text(out)

    def test_a_usable_driver_is_called_once_and_the_envelope_round_trips(
        self, verb: Any, accessor: str, kwargs: dict[str, Any], args: tuple[Any, ...], call_kwargs: dict[str, Any]
    ) -> None:
        envelope = {"status": "success", "content": [{"json": {"rc": 0}}]}
        driver = _Recorder(accessor, envelope)
        out = verb(driver=driver, **kwargs)
        assert out is envelope, "the verb must return the driver's envelope verbatim"
        assert driver.calls == [(args, call_kwargs)]

    def test_a_driver_side_refusal_surfaces_verbatim(
        self, verb: Any, accessor: str, kwargs: dict[str, Any], args: Any, call_kwargs: Any
    ) -> None:
        refusal = {"status": "error", "content": [{"text": "rc=3104 rpc timeout"}]}
        out = verb(driver=_Recorder(accessor, refusal), **kwargs)
        assert out is refusal


@pytest.mark.parametrize(
    ("verb", "kwargs", "substrings"),
    _PARAM_REFUSALS,
    ids=[f"{row[0].__name__}-{'-'.join(sorted(row[1])) or 'missing'}" for row in _PARAM_REFUSALS],
)
def test_a_bad_data_parameter_is_refused_by_name_before_the_driver_is_touched(
    verb: Any, kwargs: dict[str, Any], substrings: tuple[str, ...]
) -> None:
    accessor = _ACTIONS[verb.__name__][0]
    driver = _Recorder(accessor, {"status": "success", "content": []})
    out = verb(driver=driver, **kwargs)
    assert out["status"] == "error"
    for expected in substrings:
        assert expected in _text(out), f"refusal must name {expected!r}: {_text(out)}"
    assert driver.calls == [], "a refused parameter must not reach the driver"
