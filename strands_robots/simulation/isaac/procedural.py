"""The description dataclasses every Isaac robot loader returns.

:class:`ProceduralRobot` and its :class:`BodyDef` / :class:`JointDef` parts are
the shape ``load_urdf`` / ``load_mjcf`` / ``load_usd`` in
:mod:`strands_robots.simulation.isaac.loaders` parse a description file into, and
:func:`_validate_kinematic_tree` is the guard all three apply to the result.

This module used to also ship three hardcoded robots - ``so100``, ``panda`` and
``unitree_g1`` - built from literal body and joint tables, plus a
``get_procedural_robot`` lookup that ``IsaacSimulation.add_robot`` consulted when
the caller named no asset. That layer is gone, because it did not describe any of
those robots and nothing ever turned it into a robot:

* ``add_robot`` read only ``joint_names`` off it and made no USD call at all, so
  the branch reported success having created zero prims, left
  ``_RobotState.articulation`` as ``None``, and returned an empty
  ``get_observation()`` for the whole lifecycle - measured on Isaac Sim 6.0.1,
  before and after ``world.reset()``.
* the joint names it reported disagreed with the MuJoCo backend's for the same
  robot name: ``so100`` as ``shoulder_pan``/``shoulder_lift``/... against
  MuJoCo's ``Rotation``/``Pitch``/..., and ``panda`` as 7 joints against
  MuJoCo's 9. So it broke the joint-name parity ``docs/simulation/isaac.md``
  promises even as pure metadata.
* the data could not have been authored into a working articulation anyway.
  ``JointDef`` carries no joint anchor frame, which is what a USD revolute joint
  is defined by; ``BodyDef.position`` meant parent-relative from the loaders but
  cumulative world-frame in those tables; ``JointDef.axis`` is a free vector
  where ``UsdPhysics`` takes an X/Y/Z token, and ``panda_joint4``'s ``(0,-1,0)``
  is unrepresentable as one; ``stiffness`` was ``0.0`` on every joint, so nothing
  would have held a pose; and six ``unitree_g1`` bodies declared ``mass=0.0``,
  which is not a valid PhysX rigid body.

``add_robot`` now resolves the same description file the MuJoCo backend loads and
imports it, so those three names spawn a real articulation whose joints match
MuJoCo's exactly. See :mod:`strands_robots.simulation.isaac.mjcf_assets`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JointDef:
    """Definition of a single joint in a procedural robot."""

    name: str
    joint_type: str = "revolute"  # revolute, prismatic, fixed
    parent_body: int = 0
    child_body: int = 1
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    limit_lower: float = -3.14159
    limit_upper: float = 3.14159
    damping: float = 0.1
    stiffness: float = 0.0
    armature: float = 0.01


@dataclass
class BodyDef:
    """Definition of a single body/link in a procedural robot."""

    name: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float = 1.0
    shape: str = "box"  # box, sphere, capsule, cylinder
    shape_size: tuple[float, ...] = (0.05, 0.05, 0.05)


@dataclass
class ProceduralRobot:
    """Complete procedural robot definition."""

    name: str
    bodies: list[BodyDef] = field(default_factory=list)
    joints: list[JointDef] = field(default_factory=list)
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def num_joints(self) -> int:
        """Number of actuated (non-fixed) joints."""
        return sum(1 for j in self.joints if j.joint_type != "fixed")

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of actuated joint names."""
        return [j.name for j in self.joints if j.joint_type != "fixed"]


def _body_label(robot: ProceduralRobot, index: int) -> str:
    """Render a body index as ``index (name)`` for a refusal message.

    A caller who wrote a URDF named links, not indices, so the name is what
    makes the offender locatable. An out-of-range index is reported bare
    rather than guessed at -- body-index validity is not this helper's
    question.
    """
    if 0 <= index < len(robot.bodies):
        return f"{index} ({robot.bodies[index].name})"
    return str(index)


def _validate_kinematic_tree(robot: ProceduralRobot) -> None:
    """Fail-fast guard: every body is reached by at most one joint.

    A USD/MuJoCo articulation requires a tree where each non-root link has
    exactly one inbound joint, so a body is never reached twice and never
    parented to itself. Three shapes violate that invariant and would each
    surface two layers down as a cryptic articulation error at instantiation
    time:

      * Two joints sharing one ``(parent_body, child_body)`` edge -- an MJCF
        body carrying two ``<joint>`` children, i.e. a 2-DOF compound joint.
      * Two joints naming the same ``child_body`` with *different* parents,
        i.e. a link with two parents. URDF spells ``<parent>`` and
        ``<child>`` explicitly per joint, so a generated or hand-edited file
        expresses this as easily as a well-formed chain.
      * A joint whose ``parent_body`` is its own ``child_body``.

    Each is reported separately because the remedies differ: a compound
    joint is split with an intermediate massless link, whereas a link with
    two parents carries one joint too many and inserting a link would not
    help.

    Validation runs unconditionally on every procedural builder + every
    URDF / MJCF / USD loader: shipping a robot we know cannot instantiate
    has no good use case in this package, and silently producing one is
    worse than failing fast at builder time. The check raises ``ValueError``
    with body indices + joint names so the offender is obvious from the
    traceback alone.

    For 2-DOF compound joints (e.g. a hip with both roll and pitch axes),
    insert an intermediate massless link body between the two joints so each
    joint has its own ``(parent, child)`` edge -- which is how a humanoid's
    hip / ankle / shoulder axes are expressed as a tree.

    Multiple *roots* are not an error: a body with no inbound joint is a root,
    and an MJCF whose ``<worldbody>`` declares several top-level bodies has
    several of them. Connectivity and general acyclicity are therefore out of
    scope for this guard -- it answers how many joints reach a body, not
    whether every body is reached.
    """
    from collections import Counter

    edges = [(j.parent_body, j.child_body) for j in robot.joints]
    dups = {edge: count for edge, count in Counter(edges).items() if count > 1}
    if dups:
        offenders = {edge: [j.name for j in robot.joints if (j.parent_body, j.child_body) == edge] for edge in dups}
        raise ValueError(
            f"{robot.name}: duplicate parent->child body edges: {offenders}. "
            f"Insert intermediate massless link bodies before instantiating articulation."
        )

    self_parented = [j.name for j in robot.joints if j.parent_body == j.child_body]
    if self_parented:
        raise ValueError(
            f"{robot.name}: joints whose parent_body is their own child_body: {self_parented}. "
            f"A joint connects two distinct bodies; re-point its parent_body or child_body."
        )

    inbound: dict[int, list[str]] = {}
    for joint in robot.joints:
        inbound.setdefault(joint.child_body, []).append(joint.name)
    multi = {child: names for child, names in sorted(inbound.items()) if len(names) > 1}
    if multi:
        reached_twice = {_body_label(robot, child): names for child, names in multi.items()}
        raise ValueError(
            f"{robot.name}: bodies reached by more than one joint: {reached_twice}. "
            f"A tree articulation reaches each body through exactly one joint, so one of "
            f"these joints is redundant -- drop it or re-point its child_body."
        )
