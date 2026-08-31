"""The joint-reference tools name exactly what ``G1Driver.send_action`` accepts.

``send_action`` refuses any action-dict key that is not in
:data:`~strands_robots.drivers.g1._G1_JOINT_INDEX`. Three agent-facing tools -
:func:`g1_joint_reference`, :func:`g1_joint_name`, :func:`g1_joint_index` -
exist to surface that same map before the write is attempted, so a caller can
decide the refusal decidably rather than triggering it from the driver.

The rules the tools carry are read here off the driver's constants rather than
being restated in the tests, so a driver-side rename of a joint (which is what
``refs strands-labs/robots#2765`` might land for the ankle-pitch / ankle-roll
pair) does not require also editing this file. What the tests do restate is
the *shape* of each returned record and the SDK-load-hygiene contract every
file under :mod:`strands_robots.tools.g1` carries: importing the module must
not pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.drivers.g1 import (
    _G1_JOINT_INDEX,
    _G1_NAMED_JOINTS,
    _SDK_KD,
    _SDK_KP,
)
from strands_robots.tools.g1.g1_joints import (
    _GROUP_SLOTS,
    g1_joint_index,
    g1_joint_name,
    g1_joint_reference,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function directly
    when called in-process, but a caller cannot rely on that: the wrapper's
    contract is that it returns the wrapped function's return value verbatim.
    This helper is where a shape drift would surface once, rather than at
    every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable with
    the SDK absent; a module that pulled a submodule at import time would
    break every headless CI runner and Thor before an office bring-up. The
    driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only path
    that loads the SDK); this cell holds the joint-reference verb to it too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_joints")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_joints imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#2765)."
    )


def test_reference_returns_every_slot_the_driver_names() -> None:
    """No argument = every slot in the driver's map, once each."""
    result = _call(g1_joint_reference)
    assert result["status"] == "success"
    assert result["count"] == _G1_NAMED_JOINTS
    names = [row["name"] for row in result["joints"]]
    # Every driver-named joint is present exactly once
    assert sorted(names) == sorted(_G1_JOINT_INDEX)
    # The slot list is dense over [0, N)
    indices = [row["index"] for row in result["joints"]]
    assert indices == sorted(indices)
    assert indices[0] == 0 and indices[-1] == _G1_NAMED_JOINTS - 1


def test_every_row_carries_the_gain_pair_the_driver_would_use() -> None:
    """Kp and Kd on each row are the driver's own SDK-derived tables."""
    result = _call(g1_joint_reference)
    for row in result["joints"]:
        slot = row["index"]
        assert row["kp"] == _SDK_KP[slot], (
            f"kp for slot {slot} ({row['name']}) drifted from the driver's "
            "table - the reference tool must not restate gains."
        )
        assert row["kd"] == _SDK_KD[slot]


def test_group_filter_returns_that_group_verbatim() -> None:
    """A group name selects the slot tuple the module publishes for it."""
    for group, slots in _GROUP_SLOTS.items():
        result = _call(g1_joint_reference, group=group)
        assert result["status"] == "success"
        assert result["count"] == len(slots)
        assert [row["index"] for row in result["joints"]] == list(slots)
        assert result["group"] == group


def test_group_filter_refuses_an_unknown_group_by_name() -> None:
    """An unknown group returns an error whose message names the domain."""
    result = _call(g1_joint_reference, group="left_tentacle")
    assert result["status"] == "error"
    assert "left_tentacle" in result["message"]
    assert "left_arm" in result["message"]  # domain is named
    assert "strands-labs/robots#2765" in result["message"]


def test_joint_name_returns_the_driver_key_for_a_slot() -> None:
    """``g1_joint_name(slot)`` returns the same name the driver's map holds."""
    for name, slot in _G1_JOINT_INDEX.items():
        result = _call(g1_joint_name, index=slot)
        assert result["status"] == "success"
        assert result["name"] == name
        assert result["index"] == slot
        assert result["kp"] == _SDK_KP[slot]
        assert result["kd"] == _SDK_KD[slot]


def test_joint_name_refuses_a_slot_out_of_range_and_names_the_bound() -> None:
    """Only slots ``[0, 28]`` are named; the refusal states this."""
    for bad in (-1, _G1_NAMED_JOINTS, _G1_NAMED_JOINTS + 6):
        result = _call(g1_joint_name, index=bad)
        assert result["status"] == "error"
        assert f"[0, {_G1_NAMED_JOINTS - 1}]" in result["message"]
        assert "strands-labs/robots#2765" in result["message"]


def test_joint_name_refuses_bool_as_index_despite_being_int_subclass() -> None:
    """``True`` is an ``int(1)``; the tool must not silently accept it as slot 1.

    A dict-key typo of ``True`` for a numeric index is exactly the class of
    caller mistake this domain refusal exists to name - a live G1 write built
    on a silently-accepted ``True`` would land at the left hip.
    """
    result = _call(g1_joint_name, index=True)  # type: ignore[arg-type]
    assert result["status"] == "error"
    assert "bool" in result["message"]


def test_joint_index_accepts_the_driver_key_verbatim() -> None:
    """The snake_case key round-trips through the lookup."""
    for name, slot in _G1_JOINT_INDEX.items():
        result = _call(g1_joint_index, name=name)
        assert result["status"] == "success", f"snake_case {name} refused: {result}"
        assert result["index"] == slot
        assert result["name"] == name  # the returned name is the canonical key


def test_joint_index_accepts_pascal_case_and_camel_case_alias() -> None:
    """``LeftKnee`` and ``leftKnee`` both normalise to ``left_knee``."""
    for alias in ("LeftKnee", "leftKnee", "  LeftKnee  "):
        result = _call(g1_joint_index, name=alias)
        assert result["status"] == "success", f"alias {alias!r} refused: {result}"
        assert result["name"] == "left_knee"
        assert result["index"] == _G1_JOINT_INDEX["left_knee"]


def test_joint_index_refuses_an_unknown_name_and_names_the_domain() -> None:
    """An unknown joint name renders the driver's map, not a guess."""
    result = _call(g1_joint_index, name="ThirdEye")
    assert result["status"] == "error"
    assert "ThirdEye" in result["message"]
    # The domain (the driver's map) is named; a caller sees what is accepted.
    assert "left_knee" in result["message"]
    assert "strands-labs/robots#2765" in result["message"]


def test_joint_index_refuses_non_string_name() -> None:
    """A non-string ``name`` is a type error, refused with the type named."""
    for bad in (5, None, ["left_knee"]):
        result = _call(g1_joint_index, name=bad)  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert type(bad).__name__ in result["message"]


def test_group_slots_cover_every_joint_exactly_once() -> None:
    """The group partition is a partition, not a cover with overlaps.

    A slot that landed in two groups would produce two rows in the unfiltered
    ``g1_joint_reference()`` response for that joint, and the ``group`` field
    on a lookup would depend on which group the code found first. Neither is
    the shipped behaviour, and this test is where a future edit that broke it
    would surface.
    """
    covered: list[int] = []
    for slots in _GROUP_SLOTS.values():
        covered.extend(slots)
    assert sorted(covered) == list(range(_G1_NAMED_JOINTS))
    assert len(covered) == len(set(covered))  # no slot repeats
