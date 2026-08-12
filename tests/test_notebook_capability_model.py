"""Notebook 6's capability model may not advertise a capability the robot lacks.

``examples/notebooks/06_fleet_orchestration.ipynb`` allocates fleet work by
matching a sub-task's required capability against what each robot advertises, and
it derives what a robot advertises from the registry. Two properties of that
derivation are load-bearing and neither is visible in a run that succeeds:

* ``grasp`` must come from the registry's ``gripper`` block, never from
  ``category``. The block names the gripper's actuators and which end of their
  range is closed, so it is the only field that says the mechanism exists; the
  library resolves a gripper the same way, preferring it over a name heuristic
  (``MuJoCoMotionPrimitives._registry_gripper_metadata``, GH #1658). Measured on
  the shipped registry, 23 robots have category ``arm`` and 3 declare a gripper,
  so a category-derived ``grasp`` over-advertises 20 arms. That is the one
  allocation error an allocator cannot detect afterwards: the robot accepts the
  task, the plan reads complete, and the failure happens in the world.

* An unknown robot name must raise the notebook's own ``ValueError``.
  ``get_robot`` returns ``None`` for a name it does not know, so
  ``get_robot(name).get("category")`` dies with ``AttributeError`` before
  reaching a carefully worded message written for exactly that input.

The notebook is the artifact under test, so the test executes the notebook's own
capability cell rather than a copy of it - a copy would keep passing after the
notebook regressed. See #2173.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.registry import get_robot, list_robots

_NOTEBOOK = Path(__file__).resolve().parent.parent / "examples" / "notebooks" / "06_fleet_orchestration.ipynb"


def _capability_cell_source() -> str:
    """Source of the notebook cell defining the capability model.

    Selected by content rather than index so inserting a cell above it does not
    silently point this test at the wrong code.
    """
    nb = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    matches = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code" and "def capabilities_of" in "".join(cell.get("source", []))
    ]
    assert len(matches) == 1, (
        f"expected exactly one notebook cell defining capabilities_of, found {len(matches)}. "
        "The capability model moved or was duplicated; update this test to match."
    )
    return matches[0]


@pytest.fixture(scope="module")
def capabilities_of() -> Any:
    """``capabilities_of`` as the notebook defines it, executed from the notebook.

    Executed statement by statement, stopping at the first statement that fails
    once ``capabilities_of`` exists. What this test needs is the definition; a
    trailing statement that only demonstrates it may depend on the fleet, the
    world or the GL backend, and letting that decide the verdict would report a
    ``NameError`` about ``Simulation`` in place of the capability defect the
    assertions below are written to name.
    """
    import ast

    source = _capability_cell_source()
    module = ast.parse(source, filename=str(_NOTEBOOK))
    namespace: dict[str, Any] = {}
    for node in module.body:
        chunk = ast.Module(body=[node], type_ignores=[])
        try:
            exec(compile(chunk, str(_NOTEBOOK), "exec"), namespace)  # noqa: S102
        except Exception:
            if callable(namespace.get("capabilities_of")):
                break  # demo code past the definition; not this test's subject
            raise
    fn = namespace.get("capabilities_of")
    assert callable(fn), "notebook cell defined no callable capabilities_of"
    return fn


def _arms_by_gripper() -> tuple[list[str], list[str]]:
    """(arms declaring a gripper, arms not declaring one) from the live registry."""
    arms = [r["name"] for r in list_robots() if r.get("category") == "arm"]
    with_gripper = [n for n in arms if (get_robot(n) or {}).get("gripper")]
    without = [n for n in arms if n not in with_gripper]
    return with_gripper, without


def test_grasp_is_claimed_only_where_a_gripper_is_declared(capabilities_of: Any) -> None:
    """``grasp`` tracks the ``gripper`` block exactly, over the whole registry."""
    for entry in list_robots():
        name = entry["name"]
        try:
            caps = capabilities_of(name)
        except ValueError:
            continue  # a category the notebook declines to map: covered below
        declares = bool((get_robot(name) or {}).get("gripper"))
        assert ("grasp" in caps) is declares, (
            f"{name!r} (category {entry.get('category')!r}) advertises "
            f"grasp={'grasp' in caps} but declares gripper={declares}. grasp must come "
            "from the registry gripper block, never from the category."
        )


def test_the_gripper_split_is_real_so_the_check_is_not_vacuous() -> None:
    """Both sides of the split must be populated, or the assertion above proves nothing.

    If every arm declared a gripper (or none did), a category-derived ``grasp``
    would satisfy the test and the regression would be invisible.
    """
    with_gripper, without = _arms_by_gripper()
    assert with_gripper, "no arm declares a gripper; the grasp check would be vacuous"
    assert without, "every arm declares a gripper; the grasp check would be vacuous"


def test_a_category_arm_without_a_gripper_does_not_advertise_grasp(capabilities_of: Any) -> None:
    """The specific pre-fix defect, pinned on a named robot.

    An arm with no declared gripper is exactly the robot a category-derived model
    over-advertises, and it is the one a planner is most likely to hand
    pick-and-place work to.
    """
    _, without = _arms_by_gripper()
    caps = capabilities_of(without[0])
    assert "grasp" not in caps, f"{without[0]!r} declares no gripper but advertises {sorted(caps)}"
    assert "manipulate" in caps, f"{without[0]!r} is an arm and should still advertise manipulate: {sorted(caps)}"


def test_unknown_robot_raises_value_error_not_attribute_error(capabilities_of: Any) -> None:
    """An unknown name must reach the notebook's message, not die on ``None.get``."""
    with pytest.raises(ValueError, match="registry"):
        capabilities_of("definitely_not_a_registered_robot")


def test_every_registry_category_is_mapped(capabilities_of: Any) -> None:
    """No registry category may fall through to the unmapped-category refusal.

    ``aerial`` and ``expressive`` were both absent when this was written, so a
    drone and ``reachy_mini`` were refused as gaps rather than described. A
    category may legitimately map to an EMPTY capability set - that is the honest
    answer for an expressive robot - but it may not be missing.
    """
    unmapped: dict[str, str] = {}
    for entry in list_robots():
        try:
            capabilities_of(entry["name"])
        except ValueError as exc:
            unmapped[str(entry.get("category"))] = f"{entry['name']}: {exc}"
    assert not unmapped, (
        "registry categories the notebook maps to no capabilities: "
        f"{sorted(unmapped)}. Add a row to CATEGORY_CAPABILITIES (an empty set is "
        f"a valid row). Details: {unmapped}"
    )
