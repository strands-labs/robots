"""A USD/PhysX articulation reaches every body through exactly one joint.

``_validate_kinematic_tree`` is the fail-fast topology guard every URDF / MJCF /
USD loader runs before handing a ``ProceduralRobot`` back. Its stated invariant
is that each non-root link has exactly one inbound joint, and three shapes
violate it:

  * two joints sharing one ``(parent_body, child_body)`` edge -- an MJCF body
    carrying two ``<joint>`` children, i.e. a 2-DOF compound joint;
  * two joints naming the same ``child_body`` with *different* parents, i.e. a
    link with two parents;
  * a joint whose ``parent_body`` is its own ``child_body``.

Only the first was checked. The second is what URDF makes easy: a URDF is a
flat list of joints each naming ``<parent>`` and ``<child>`` explicitly, so a
generated or hand-edited file expresses a two-parent link as readily as a
well-formed chain -- and ``load_urdf`` already refuses a duplicate ``<link>``,
an unknown parent or child link, a missing ``<parent>`` / ``<child>`` and an
unknown joint type, so the topology guard is the one that should catch it.

What is *not* an error here: several roots. A body with no inbound joint is a
root, and an MJCF whose ``<worldbody>`` declares several top-level bodies has
several of them. Connectivity and general acyclicity stay out of scope -- the
guard answers how many joints reach a body, not whether every body is reached.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.isaac.loaders import load_mjcf, load_urdf
from strands_robots.simulation.isaac.procedural import BodyDef, JointDef, ProceduralRobot

# A joint's remedy depends on which shape it is: a compound joint is split
# with an intermediate massless link, a two-parent link has one joint too
# many. This phrase belongs to the first and must not be offered for the
# second.
_COMPOUND_JOINT_REMEDY = "intermediate massless link"


def _urdf(*joints: str, links: tuple[str, ...] = ("base", "shoulder", "forearm")) -> str:
    """Render a minimal URDF: named links plus the given <joint> elements."""
    link_xml = "".join(f'<link name="{name}"/>' for name in links)
    return f'<?xml version="1.0"?><robot name="probe">{link_xml}{"".join(joints)}</robot>'


def _joint(name: str, parent: str, child: str) -> str:
    return (
        f'<joint name="{name}" type="revolute">'
        f'<parent link="{parent}"/><child link="{child}"/>'
        f'<axis xyz="0 0 1"/><limit lower="-1" upper="1"/>'
        f"</joint>"
    )


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "probe.urdf"
    path.write_text(text)
    return str(path)


def _robot(*edges: tuple[str, int, int], bodies: int = 4) -> ProceduralRobot:
    """Build a ProceduralRobot from ``(joint_name, parent_index, child_index)``."""
    return ProceduralRobot(
        name="probe",
        bodies=[BodyDef(name=f"b{i}") for i in range(bodies)],
        joints=[JointDef(name=n, parent_body=p, child_body=c) for n, p, c in edges],
    )


def _load_or_refusal(tmp_path, urdf: str) -> str:
    """Return the refusal text, or fail naming the topology that was accepted.

    A bare "DID NOT RAISE" says nothing about what the loader handed back, so
    the diagnosis reports how many joints reached each body instead.
    """
    try:
        robot = load_urdf(_write(tmp_path, urdf))
    except ValueError as refused:
        return str(refused)
    inbound: dict[int, list[str]] = {}
    for joint in robot.joints:
        inbound.setdefault(joint.child_body, []).append(joint.name)
    multi = {robot.bodies[c].name: n for c, n in inbound.items() if len(n) > 1}
    raise AssertionError(
        f"load_urdf accepted a robot no articulation can instantiate: bodies reached by "
        f"more than one joint {multi}, and it reports num_joints={robot.num_joints} "
        f"with joint_names={robot.joint_names}."
    )


class TestABodyReachedTwiceIsRefused:
    """The invariant the guard states, not only its duplicate-edge case."""

    def test_two_parents_on_one_link_is_refused(self, tmp_path):
        # forearm is named as the child of two joints with *different* parents,
        # so the two edges are distinct and a per-edge count sees nothing --
        # while forearm still has two inbound joints, which is the invariant.
        message = _load_or_refusal(
            tmp_path,
            _urdf(
                _joint("shoulder_pan", "base", "shoulder"),
                _joint("elbow_from_shoulder", "shoulder", "forearm"),
                _joint("elbow_from_base", "base", "forearm"),
            ),
        )
        # The offending body is named, not just indexed: a URDF author wrote
        # link names, so the name is what makes it locatable.
        assert "forearm" in message
        assert "elbow_from_shoulder" in message
        assert "elbow_from_base" in message
        # The link that is reached once is not accused.
        assert "shoulder_pan" not in message

    def test_the_two_parent_refusal_does_not_prescribe_an_intermediate_link(self, tmp_path):
        # Splitting with an intermediate link is the remedy for a compound
        # joint. A link with two parents carries one joint too many, so that
        # advice would send the author to add a body rather than drop a joint.
        message = _load_or_refusal(
            tmp_path,
            _urdf(
                _joint("shoulder_pan", "base", "shoulder"),
                _joint("elbow_from_shoulder", "shoulder", "forearm"),
                _joint("elbow_from_base", "base", "forearm"),
            ),
        )
        assert _COMPOUND_JOINT_REMEDY not in message
        assert "redundant" in message

    def test_a_cycle_that_gives_a_body_two_parents_is_refused(self, tmp_path):
        # b is the child of both a->b and c->b, so the chain closes on itself.
        message = _load_or_refusal(
            tmp_path,
            _urdf(
                _joint("j1", "base", "shoulder"),
                _joint("j2", "shoulder", "forearm"),
                _joint("j3", "forearm", "shoulder"),
            ),
        )
        assert "shoulder" in message
        assert "j1" in message
        assert "j3" in message

    def test_a_self_parented_joint_is_refused(self, tmp_path):
        # A joint whose parent is its own child is one inbound joint by count,
        # so it needs naming on its own terms.
        message = _load_or_refusal(
            tmp_path,
            _urdf(
                _joint("shoulder_pan", "base", "shoulder"),
                _joint("loop", "shoulder", "shoulder"),
                links=("base", "shoulder"),
            ),
        )
        assert "loop" in message
        assert "two distinct bodies" in message
        assert "shoulder_pan" not in message

    def test_the_owner_refuses_a_two_parent_robot_directly(self):
        from strands_robots.simulation.isaac.procedural import _validate_kinematic_tree

        # Pinned at the guard, not only through one loader: every builder and
        # every other loader calls this same function. ``ProceduralRobot``
        # itself does not validate, so the guard is driven explicitly.
        with pytest.raises(ValueError, match="more than one joint") as exc:
            _validate_kinematic_tree(_robot(("j1", 0, 2), ("j2", 1, 2)))
        assert "b2" in str(exc.value)

    def test_an_out_of_range_child_index_is_reported_bare(self):
        from strands_robots.simulation.isaac.procedural import _validate_kinematic_tree

        # Body-index validity is a different question, so the label falls back
        # to the bare index rather than guessing at a name.
        with pytest.raises(ValueError, match="more than one joint") as exc:
            _validate_kinematic_tree(_robot(("j1", 0, 99), ("j2", 1, 99), bodies=2))
        assert "99" in str(exc.value)


class TestTheTreeContractIsNotWidened:
    """Shapes that must keep loading, and the boundary of the invariant."""

    def test_a_proper_chain_is_accepted(self, tmp_path):
        robot = load_urdf(
            _write(
                tmp_path,
                _urdf(
                    _joint("shoulder_pan", "base", "shoulder"),
                    _joint("elbow", "shoulder", "forearm"),
                ),
            )
        )
        assert robot.joint_names == ["shoulder_pan", "elbow"]

    def test_several_roots_are_accepted(self, tmp_path):
        # Two disjoint sub-chains: bodies 0 and 2 have no inbound joint. This
        # is a shipped shape, so connectivity must not be enforced here.
        robot = load_urdf(
            _write(
                tmp_path,
                _urdf(
                    _joint("j1", "base", "shoulder"),
                    _joint("j2", "forearm", "wrist"),
                    links=("base", "shoulder", "forearm", "wrist"),
                ),
            )
        )
        assert robot.num_joints == 2

    def test_a_multi_root_description_has_two_roots_and_still_loads(self, tmp_path):
        """Several roots is not an error, and a real description is the witness.

        This used to assert the same thing about the shipped ``unitree_g1``
        procedural builder, which had two roots. That builder is gone - it
        described no real robot and ``add_robot`` never turned it into one - but
        the conclusion it supported still holds for the reason this module's
        docstring already gives: an MJCF whose ``<worldbody>`` declares several
        top-level bodies produces exactly this topology, and it is a shape
        MuJoCo itself accepts. So the premise moves from a fabricated robot to a
        description file, which is the stronger witness anyway.
        """
        mjcf = (
            '<mujoco model="two_roots"><worldbody>'
            '<body name="arm_base"><joint name="j1" axis="0 0 1"/><geom type="box" size="0.1 0.1 0.1"/>'
            '<body name="arm_link"><joint name="j2" axis="0 1 0"/><geom type="box" size="0.05 0.05 0.2"/>'
            "</body></body>"
            '<body name="fixture"><geom type="box" size="0.2 0.2 0.02"/></body>'
            "</worldbody></mujoco>"
        )
        path = tmp_path / "two_roots.xml"
        path.write_text(mjcf)
        robot = load_mjcf(str(path))

        reached = {joint.child_body for joint in robot.joints}
        roots = [i for i in range(len(robot.bodies)) if i not in reached]
        # The premise: enforcing a single root would refuse this description.
        assert len(roots) > 1, f"expected several roots, got {roots} for bodies {[b.name for b in robot.bodies]}"

    def test_a_well_formed_chain_still_loads(self, tmp_path):
        """The guard refuses the three bad shapes and nothing else.

        Replaces a sweep over the three deleted procedural builders. Their only
        contribution was "a real robot is not refused", which any well-formed
        chain witnesses - and unlike those tables, this one is a description the
        loader actually parses.
        """
        robot = load_urdf(
            _write(
                tmp_path,
                _urdf(
                    _joint("j1", "base", "shoulder"),
                    _joint("j2", "shoulder", "forearm"),
                ),
            )
        )
        assert robot.num_joints == 2

    def test_general_acyclicity_stays_out_of_scope(self):
        from strands_robots.simulation.isaac.procedural import _validate_kinematic_tree

        # b1 -> b2 -> b1: each body is reached by exactly one joint, so the
        # invariant this guard states holds even though the pair has no root.
        # Rootedness is a broader question and is deliberately not asked here.
        _validate_kinematic_tree(_robot(("a", 1, 2), ("b", 2, 1)))

    def test_the_duplicate_edge_refusal_is_unchanged(self, tmp_path):
        # A compound joint keeps its own wording and its own remedy, so the
        # two shapes stay distinguishable to the author who hit one of them.
        message = _load_or_refusal(
            tmp_path,
            _urdf(
                _joint("hip_roll", "base", "shoulder"),
                _joint("hip_pitch", "base", "shoulder"),
            ),
        )
        assert "duplicate parent->child body edges" in message
        assert _COMPOUND_JOINT_REMEDY in message
        assert "hip_roll" in message
        assert "hip_pitch" in message
