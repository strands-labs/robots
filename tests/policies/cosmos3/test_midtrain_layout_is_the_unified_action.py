"""A ``midtrain`` action layout must name the unified action it is served from.

A Cosmos 3 embodiment declares two things about the same model output: its width
(``raw_action_dim``) and the column names for the model's own unified action
(``raw_action_layout`` - a 9D effector pose ``tx,ty,tz,r0..r5`` plus a 1D
``grasp``). ``action_layouts`` then names the columns per served action space.
``joint_pos`` legitimately differs: it is the one space the RoboLab server
post-processes, converting the effector pose into joint targets (DROID: 7 joints
+ gripper = 8). ``midtrain`` is the unified action served through un-converted,
so its column names are the unified action's.

Nothing enforced that. ``_action_column_names`` takes ``layout[:width]`` and
pads any shortfall with synthesized ``action_<i>`` names, so a ``midtrain``
layout narrower than the row it names does not shorten the row - it slides every
name it does carry onto the wrong column and leaves the surplus unnamed. On a
10-wide unified action named by an 8-entry layout, column 7 (a rotation
component) is emitted under the layout's 8th name while the grasp leaves as
``action_9``, a key no actuator answers to. ``action_mapping`` validation then
compounds it: it accepts the misplaced name (so a caller can map a rotation
component onto a gripper actuator and be told nothing) and refuses ``grasp``
(so the one column a gripper needs cannot be named at all).

These tests pin the invariant on every embodiment that declares a ``midtrain``
layout, and the two consequences that follow from it - every column of a
unified-action row is named, and a mapped ``grasp`` reaches the actuator
carrying the row's grasp value. The controls hold the boundary: ``joint_pos``
stays the server's converted layout (narrower than ``raw_action_dim`` on
purpose), an embodiment with no gripper declares no ``grasp`` column, and an
``action_mapping`` key that names no column is still refused rather than
aliased.

They are transport-free - the injected client owns its own address and is never
dialled - so no server, checkpoint, or GL context is needed.
"""

from typing import Any, cast

import numpy as np
import pytest

from strands_robots.policies.cosmos3.embodiments import EMBODIMENTS, get_embodiment
from strands_robots.policies.cosmos3.policy import Cosmos3Policy

_MIDTRAIN = "midtrain"

#: Embodiments that serve the unified action under ``midtrain``.
_MIDTRAIN_EMBODIMENTS = sorted(name for name, e in EMBODIMENTS.items() if _MIDTRAIN in e.action_layouts)

#: Of those, the ones whose unified action carries a gripper column.
_GRASP_EMBODIMENTS = sorted(name for name in _MIDTRAIN_EMBODIMENTS if "grasp" in get_embodiment(name).raw_action_layout)


class _AddressedClient:
    """A service client that owns its endpoint, so the constructor never dials."""

    host = "localhost"
    port = 8000

    def __init__(self, action: np.ndarray | None = None) -> None:
        self._action = action

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {"action": self._action}

    def reset(self) -> None:
        return None

    def get_server_metadata(self) -> dict[str, Any]:
        return {}


def _midtrain_policy(embodiment: str, action: np.ndarray | None = None, **kwargs: Any) -> Cosmos3Policy:
    return Cosmos3Policy(
        embodiment=embodiment,
        action_space=_MIDTRAIN,
        backend="service",
        # A stand-in for the websocket client, which owns its own endpoint.
        client=cast(Any, _AddressedClient(action)),
        **kwargs,
    )


def _unified_action_row(width: int) -> np.ndarray:
    """One timestep of a unified action, distinct per column so a shift shows."""
    return np.asarray([[round(0.1 * (i + 1), 4) for i in range(width)]], dtype=np.float32)


def test_some_embodiment_serves_the_unified_action_under_midtrain() -> None:
    """Premise: a clean sweep below would prove nothing if nothing declared midtrain."""
    assert _MIDTRAIN_EMBODIMENTS, "no embodiment declares a 'midtrain' action layout"
    assert _GRASP_EMBODIMENTS, "no midtrain embodiment carries a 'grasp' column"


@pytest.mark.parametrize("embodiment", _MIDTRAIN_EMBODIMENTS)
def test_a_midtrain_layout_names_the_unified_action_columns(embodiment: str) -> None:
    """``midtrain`` is the unified action, so it names the unified action's columns."""
    e = get_embodiment(embodiment)
    assert e.action_layouts[_MIDTRAIN] == e.raw_action_layout, (
        f"{embodiment!r} serves the unified action under 'midtrain' but names its columns "
        f"{e.action_layouts[_MIDTRAIN]} instead of the unified action's {e.raw_action_layout}"
    )


@pytest.mark.parametrize("embodiment", _MIDTRAIN_EMBODIMENTS)
def test_every_column_of_a_unified_action_row_is_named(embodiment: str) -> None:
    """No column of the served row falls through to a synthesized ``action_<i>`` name."""
    e = get_embodiment(embodiment)
    policy = _midtrain_policy(embodiment)
    step = policy._unpack_actions(_unified_action_row(e.raw_action_dim))[0]

    synthesized = sorted(k for k in step if k.startswith("action_") and k[len("action_") :].isdigit())
    assert not synthesized, (
        f"{embodiment!r} emits {len(step)} columns for its {e.raw_action_dim}-wide unified action, "
        f"{len(synthesized)} of them unnamed ({synthesized}): the 'midtrain' layout "
        f"{e.action_layouts[_MIDTRAIN]} is {len(e.action_layouts[_MIDTRAIN])} entries wide"
    )
    assert len(step) == e.raw_action_dim


@pytest.mark.parametrize("embodiment", _GRASP_EMBODIMENTS)
def test_a_mapped_grasp_column_carries_the_rows_grasp_value(embodiment: str) -> None:
    """Mapping ``grasp`` onto an actuator delivers the grasp, not a rotation component."""
    e = get_embodiment(embodiment)
    row = _unified_action_row(e.raw_action_dim)
    grasp = float(row[0, -1])

    policy = _midtrain_policy(embodiment, action_mapping={"grasp": "finger_joint1"})
    step = policy._unpack_actions(row)[0]

    assert "finger_joint1" in step, f"{embodiment!r} emitted {sorted(step)} with no mapped gripper actuator"
    assert step["finger_joint1"] == pytest.approx(grasp), (
        f"{embodiment!r} drove finger_joint1 with {step['finger_joint1']} instead of the row's grasp {grasp}"
    )


def test_droid_joint_pos_stays_the_servers_converted_layout() -> None:
    """``joint_pos`` is the converted space, so it is narrower than the unified action.

    A guard that simply forced every layout to ``raw_action_dim`` would break
    this: the server really does serve 7 joint targets plus a gripper for a
    10-wide unified action.
    """
    e = get_embodiment("droid")
    joint_pos = e.action_layouts["joint_pos"]
    assert joint_pos == [f"joint_{i}" for i in range(7)] + ["gripper"]
    assert len(joint_pos) < e.raw_action_dim


def test_an_embodiment_with_no_gripper_declares_no_grasp_column() -> None:
    """The invariant follows the unified action, which for ``av`` carries no grasp."""
    e = get_embodiment("av")
    assert "grasp" not in e.action_layouts[_MIDTRAIN]
    assert len(e.action_layouts[_MIDTRAIN]) == e.raw_action_dim == 9


@pytest.mark.parametrize("embodiment", _MIDTRAIN_EMBODIMENTS)
def test_an_action_mapping_key_that_names_no_column_is_still_refused(embodiment: str) -> None:
    """A key naming no column is refused, not aliased onto a nearby one."""
    with pytest.raises(ValueError, match="are not in the"):
        _midtrain_policy(embodiment, action_mapping={"not_a_column": "finger_joint1"})


def _served_observation(embodiment: str) -> dict[str, Any]:
    """An observation carrying every camera the embodiment declares."""
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    return {key: frame for key in get_embodiment(embodiment).camera_keys}


@pytest.mark.parametrize("embodiment", _GRASP_EMBODIMENTS)
def test_the_served_chunk_reaches_the_gripper_through_get_actions(embodiment: str) -> None:
    """The public inference path delivers the row's grasp to the mapped actuator.

    ``_unpack_actions`` is exercised directly above; this drives the documented
    ``get_actions_sync`` path over a served chunk so the contract is pinned where
    a caller actually meets it.
    """
    e = get_embodiment(embodiment)
    chunk = np.concatenate([_unified_action_row(e.raw_action_dim)] * 2, axis=0)
    grasp = float(chunk[0, -1])

    policy = _midtrain_policy(embodiment, action=chunk, action_mapping={"grasp": "finger_joint1"})
    steps = policy.get_actions_sync(_served_observation(embodiment), "close the gripper")

    assert len(steps) == 2
    assert steps[0].get("finger_joint1") == pytest.approx(grasp), (
        f"{embodiment!r} served {e.raw_action_dim} columns and delivered "
        f"finger_joint1={steps[0].get('finger_joint1')} instead of the grasp {grasp}; "
        f"emitted keys {sorted(steps[0])}"
    )
