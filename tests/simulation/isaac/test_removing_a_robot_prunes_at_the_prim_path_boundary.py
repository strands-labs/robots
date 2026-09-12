# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: removing one robot leaves a prefix-sharing sibling's prim tracked.

``IsaacSimulation.remove_robot`` documents that it "prunes any prims rooted at
the robot's prim path from ``self._prim_registry``". A prim path is interpolated
from the robot's name (``{stage_path}/Robots/{name}``), and the prune tested that
relation with a bare ``p.startswith(prim_path)`` -- which is a test on the
*string*, not on the USD path. So every robot whose NAME extends the removed
one's was treated as a prim rooted under it:

* two robots ``arm`` and ``arm_left``, ``remove_robot("arm")`` ->
  ``/World/Robots/arm_left`` dropped from the teardown registry while ``arm_left``
  stayed registered in ``_robots``. The registry is what :meth:`destroy` releases
  and counts (``num_prims_released = len(self._prim_registry)``), so a live robot
  was left with zero tracked prims and destroy under-reported by one.
* ``arm``/``arm2`` and ``panda``/``panda_gripper`` behave identically -- the
  arrangement needs no unusual name, only a name that is a prefix of another.

This is the same corruption the empty name used to cause, which
``tests/simulation/test_entity_name_domain_at_creation.py`` closed at the
creation site: ``add_robot("")`` reserved the *container* path
``/World/Robots/``, so ``remove_robot("")`` pruned every robot. Refusing the
empty name stopped that one arrangement from arising; it left the prune rule
itself unchanged, and the rule over-matches for any two ordinary names in that
relation.

The two sibling removal verbs never had this: :meth:`remove_object` and
:meth:`remove_camera` ask for their exact path (``if prim_path in
self._prim_registry``), which cannot over-match. ``remove_robot`` is the one that
prunes a subtree, so the fix bounds the subtree at ``/`` -- USD's path separator,
and therefore what separates a descendant prim from a sibling that merely shares
a prefix -- rather than narrowing it to an exact match, which would drop the
subtree prune the docstring promises.

None of this needs Isaac Sim or a GPU: the procedural ``add_robot`` branch and
all three removal verbs touch no stage, so the unbound methods run against the
small ``types.SimpleNamespace`` stand-in for ``self`` that the neighbouring Isaac
name-domain tests already use.
"""

from __future__ import annotations

import threading
import types

import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import (
    IsaacSimulation,
    _CameraState,
    _ObjectState,
    _RobotState,
)

#: ``(removed, survivor)`` name pairs. The first row is the arrangement where the
#: string test and the path test AGREE -- neither name is a prefix of the other --
#: so it is the control that held before the fix and must still hold. The rest are
#: the relation the string test cannot see: ``removed`` is a prefix of
#: ``survivor``, one character short of the ``/`` that would make it a parent.
_PAIRS = (
    ("arm", "helper"),
    ("arm", "arm_left"),
    ("arm", "arm2"),
    ("panda", "panda_gripper"),
    ("so101", "so101_leader"),
)


def _stub() -> types.SimpleNamespace:
    """A stand-in for ``self`` carrying only what add/remove read."""
    return types.SimpleNamespace(
        _lock=threading.RLock(),
        _world_created=True,
        _world=None,
        _config=IsaacConfig(),
        _robots={},
        _objects={},
        _cameras={},
        _action_controllers={},
        _replicated=False,
        _prim_registry=[],
    )


def _with_robots(*names: str) -> types.SimpleNamespace:
    """Register ``names`` in the two registries a prune reads, and nothing else.

    Written directly rather than through ``add_robot``. These cells are about how
    ``remove_robot`` scopes its ``_prim_registry`` prune, and the only inputs to
    that are the ``_robots`` keys and the prim paths - so going through
    ``add_robot`` only coupled them to whatever that method needs to *load* a
    robot. It did once: the call was ``add_robot(stub, name, data_config="panda")``
    against a stub whose ``_world`` is ``None``, which succeeded only because the
    "procedural" branch it reached created nothing at all. Now that the branch
    resolves and imports a real description, the same call needs an asset, a
    converter and a live stage, none of which a prune has any opinion about.

    Registering the two facts the prune reads keeps the fixture honest about its
    own subject, and asserts the arrangement it depends on rather than a status.
    """
    stub = _stub()
    for name in names:
        prim_path = f"/World/Robots/{name}"
        stub._robots[name] = _RobotState(name=name, prim_path=prim_path, joint_names=[])
        stub._prim_registry.append(prim_path)
    assert stub._prim_registry == [f"/World/Robots/{n}" for n in names]
    assert sorted(stub._robots) == sorted(names)
    return stub


class TestAPruneIsBoundedAtThePathSeparator:
    """A removal drops the removed robot's prims and only those."""

    @pytest.mark.parametrize(("removed", "survivor"), _PAIRS)
    def test_the_surviving_robot_keeps_its_prim(self, removed, survivor):
        stub = _with_robots(removed, survivor)

        assert IsaacSimulation.remove_robot(stub, removed)["status"] == "success"  # type: ignore[arg-type]

        assert f"/World/Robots/{survivor}" in stub._prim_registry
        assert f"/World/Robots/{removed}" not in stub._prim_registry

    @pytest.mark.parametrize(("removed", "survivor"), _PAIRS)
    def test_every_robot_still_registered_has_a_tracked_prim(self, removed, survivor):
        """The teardown invariant, stated as the count :meth:`destroy` reports.

        ``destroy`` releases ``_prim_registry`` and reports its length as
        ``prims_released``, so a robot present in ``_robots`` with no entry there
        is a prim nothing will release and nothing will count.
        """
        stub = _with_robots(removed, survivor)

        IsaacSimulation.remove_robot(stub, removed)  # type: ignore[arg-type]

        assert sorted(stub._robots) == [survivor]
        assert len(stub._prim_registry) == len(stub._robots)

    def test_a_descendant_prim_is_still_pruned(self):
        """The prune is bounded, not narrowed to an exact match.

        ``remove_robot`` is documented as pruning "any prims rooted at the
        robot's prim path", so a child prim registered beneath it goes with the
        robot. An exact-membership prune -- what the two sibling removal verbs
        use -- would leave it behind.
        """
        stub = _with_robots("arm", "arm_left")
        stub._prim_registry.append("/World/Robots/arm/gripper")

        IsaacSimulation.remove_robot(stub, "arm")  # type: ignore[arg-type]

        assert stub._prim_registry == ["/World/Robots/arm_left"]

    def test_a_refused_removal_prunes_nothing(self):
        """An unknown name is an error, and an error changes no bookkeeping."""
        stub = _with_robots("arm", "arm_left")

        assert IsaacSimulation.remove_robot(stub, "arm_")["status"] == "error"  # type: ignore[arg-type]

        assert stub._prim_registry == ["/World/Robots/arm", "/World/Robots/arm_left"]
        assert sorted(stub._robots) == ["arm", "arm_left"]


class TestEveryRemovalVerbScopesItsPrune:
    """All three removal verbs leave a prefix-sharing sibling's prim tracked.

    ``remove_object`` and ``remove_camera`` already did, by asking for their
    exact path. Graded together so the three cannot drift back apart: the prim
    path of every entity is interpolated from its name, so the relation exists on
    all three surfaces.
    """

    def test_remove_object(self):
        stub = _stub()
        for name in ("crate", "crate_lid"):
            path = f"/World/Objects/{name}"
            stub._objects[name] = _ObjectState(name=name, prim_path=path, shape="box", is_static=True)
            stub._prim_registry.append(path)

        assert IsaacSimulation.remove_object(stub, "crate")["status"] == "success"  # type: ignore[arg-type]

        assert stub._prim_registry == ["/World/Objects/crate_lid"]

    def test_remove_camera(self):
        stub = _stub()
        for name in ("wrist", "wrist_left"):
            path = f"/World/Cameras/{name}"
            stub._cameras[name] = _CameraState(name=name, prim_path=path, width=64, height=64)
            stub._prim_registry.append(path)

        assert IsaacSimulation.remove_camera(stub, "wrist")["status"] == "success"  # type: ignore[arg-type]

        assert stub._prim_registry == ["/World/Cameras/wrist_left"]

    def test_remove_robot(self):
        stub = _with_robots("arm", "arm_left")

        assert IsaacSimulation.remove_robot(stub, "arm")["status"] == "success"  # type: ignore[arg-type]

        assert stub._prim_registry == ["/World/Robots/arm_left"]
