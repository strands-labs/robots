"""ROS 2 sidecar that exposes ``moveit_py`` planning over ZMQ + msgpack.

This is a **reference implementation** matching the wire protocol the
client (:class:`strands_robots.policies.moveit2.MoveIt2Policy`) expects.
It is intentionally minimal - production deployments should fork this
file and harden it for their own collision world / planner pipeline /
auth posture.

Run it with::

    source /opt/ros/jazzy/setup.bash         # or your distro
    pip install pyzmq msgpack                # the only non-ROS deps
    python -m strands_robots.policies.moveit2.server.zmq_node \\
        --port 5556 --planning-group arm

The sidecar is single-threaded REQ/REP - one in-flight plan request at a
time. That matches the ``MoveItPy.plan()`` API which is itself
single-threaded. Because REQ/REP is lockstep, every request that is
received gets exactly one reply: a request that cannot be served is
answered with a failure response, never by dropping the reply and
exiting. A planning failure is reported in the ``plan`` response
(``success=False`` plus a ``status`` naming the stage); anything else is
reported as ``{"error": ...}``, which the client raises as
``RuntimeError``.

Wire protocol::

    request  = {"endpoint": "plan",
                "data": {"joint_state": list[float] | None,
                         "planning_group": str,
                         "target_pose": [x, y, z, qw, qx, qy, qz] | None,
                         "target_joints": dict[str, float] | None,
                         "world_update": dict | None}}
    response = {"trajectory": list[list[float]],
                "success": bool,
                "status": str}

The trajectory rows are ``[time_from_start_seconds, q0, q1, ..., qN]`` -
the time column lets the client / runner schedule waypoints precisely.

Notes for forks:

* The ``world_update`` payload is intentionally schema-free here. A
  production sidecar should validate it against a known schema (e.g.
  expect ``{"depth_topic": str, "stamp": int, "frame_id": str}``) before
  pushing to the planning scene.
* ``api_token`` validation is left as an exercise - the reference
  implementation accepts any client. Enable it by checking
  ``request.get("api_token")`` against an env-var / file secret - by
  then ``request`` is known to be a map.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

# These imports deliberately happen inside ``main`` so this file can be
# imported and statically analysed without ROS 2 sourced. Top-level
# imports of ``rclpy`` / ``moveit_py`` would crash on dev boxes that
# only have the strands-robots client installed.

logger = logging.getLogger("moveit2.zmq_node")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoveIt2 ZMQ sidecar for strands-robots MoveIt2Policy",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address for the ZMQ REP socket. Default 0.0.0.0 inside "
        "the container (the host-level firewall / docker port mapping is "
        "the security boundary).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="Bind port. Default 5556 - matches MoveIt2Policy's default.",
    )
    parser.add_argument(
        "--planning-group",
        default="arm",
        help="Default MoveIt2 planning-group name. Per-request overrides win.",
    )
    parser.add_argument(
        "--robot-description-package",
        default=None,
        help="ROS 2 package providing the URDF/SRDF (``MoveItPyConfigBuilder``).",
    )
    parser.add_argument(
        "--moveit-config-package",
        default=None,
        help="moveit_py config package (e.g. ``moveit_resources_panda_moveit_config``).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for the sidecar. ROS 2 spinner logs are independent.",
    )
    return parser.parse_args(argv)


def _build_moveit_py(args: argparse.Namespace) -> Any:
    """Construct the ``moveit_py`` runtime.

    Kept in its own function so a fork can mock / replace the planner
    initialisation without rewriting the ZMQ loop.
    """
    from moveit.planning import MoveItPy
    from moveit_configs_utils import MoveItConfigsBuilder

    builder = MoveItConfigsBuilder(robot_name="moveit2_sidecar")
    if args.robot_description_package:
        builder = builder.robot_description(package=args.robot_description_package)
    if args.moveit_config_package:
        builder = builder.moveit_cpp(file_path=args.moveit_config_package)

    moveit_config = builder.to_moveit_configs().to_dict()
    moveit_py = MoveItPy(node_name="strands_robots_moveit2_sidecar", config_dict=moveit_config)
    logger.info("MoveItPy initialised; planning groups: %s", moveit_py.get_planning_component_names())
    return moveit_py


def _plan(
    moveit_py: Any,
    *,
    planning_group: str,
    joint_state: list[float] | None,
    target_pose: list[float] | None,
    target_joints: dict[str, float] | None,
    world_update: dict[str, Any] | None,  # noqa: ARG001 - reserved, see schema TODO above
) -> dict[str, Any]:
    """Run a single ``moveit_py`` plan and serialise the result.

    Returns the wire-format response dict. Every ``moveit_py``
    interaction is guarded, so a failing plan comes back as a structured
    ``{"success": False, "status": ...}`` response instead of raising:
    the REP loop owes its peer exactly one reply, and an escaping
    exception would end the sidecar with that reply never sent.

    ``moveit_py`` is a C++ binding whose exception types are not
    enumerable from Python, so each guard catches broadly and reports the
    stage that failed in ``status``:

    * ``unknown_planning_group`` - the group name does not resolve.
    * ``start_state_error`` - the current robot state is not readable
      (no ``/joint_states`` yet, monitor not warmed up).
    * ``missing_goal`` - neither goal field was supplied.
    * ``invalid_goal`` - the goal was rejected: a joint the group does
      not have, an unresolvable pose link, or a ``target_pose`` that is
      not 7 values.
    * ``planner_exception`` / ``planner_returned_empty`` - planning ran
      and failed.
    * ``trajectory_error`` - the result did not serialise.
    """
    from geometry_msgs.msg import PoseStamped

    try:
        component = moveit_py.get_planning_component(planning_group)
    except Exception as e:
        return {"trajectory": [], "success": False, "status": f"unknown_planning_group:{e}"}

    try:
        component.set_start_state_to_current_state()
    except Exception as e:  # noqa: BLE001 - report the failing stage structurally
        logger.exception("Reading the current robot state failed: %s", e)
        return {"trajectory": [], "success": False, "status": f"start_state_error:{e}"}

    if joint_state is not None:
        # Forks that need start-state override should plug their own
        # ``RobotState`` builder here. Reference implementation trusts
        # the planner's current state.
        logger.debug("joint_state hint received but unused in reference impl: %s", joint_state)

    try:
        if target_joints is not None:
            component.set_goal_state(joint_values=target_joints)
        elif target_pose is not None:
            x, y, z, qw, qx, qy, qz = target_pose
            pose = PoseStamped()
            pose.header.frame_id = "base_link"  # Forks: parameterise this.
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = qw
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            component.set_goal_state(pose_stamped_msg=pose, pose_link="end_effector_link")
        else:
            return {
                "trajectory": [],
                "success": False,
                "status": "missing_goal:expected_target_pose_or_target_joints",
            }
    except Exception as e:  # noqa: BLE001 - report the failing stage structurally
        logger.exception("Setting the goal state failed: %s", e)
        return {"trajectory": [], "success": False, "status": f"invalid_goal:{e}"}

    try:
        plan_result = component.plan()
    except Exception as e:  # noqa: BLE001 - report planner failure structurally
        logger.exception("Planning failed: %s", e)
        return {"trajectory": [], "success": False, "status": f"planner_exception:{e}"}

    if not plan_result:
        return {"trajectory": [], "success": False, "status": "planner_returned_empty"}

    # ``trajectory_msg.joint_trajectory.points`` is a list of
    # ``trajectory_msgs/JointTrajectoryPoint``. Each has
    # ``time_from_start`` (Duration) + ``positions`` (list[float]).
    try:
        trajectory_msg = plan_result.trajectory
        rows: list[list[float]] = []
        for point in trajectory_msg.joint_trajectory.points:
            t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            rows.append([float(t)] + [float(q) for q in point.positions])
    except Exception as e:  # noqa: BLE001 - report the failing stage structurally
        logger.exception("Serialising the planned trajectory failed: %s", e)
        return {"trajectory": [], "success": False, "status": f"trajectory_error:{e}"}

    return {"trajectory": rows, "success": True, "status": "ok"}


def main(argv: list[str] | None = None) -> int:
    """ZMQ REP loop entry point. Returns the desired process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Lazy imports - see module docstring for rationale.
    import msgpack
    import rclpy
    import zmq

    rclpy.init()
    try:
        moveit_py = _build_moveit_py(args)
    except Exception as e:
        logger.exception("Failed to construct MoveItPy: %s", e)
        rclpy.shutdown()
        return 1

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")
    logger.info("MoveIt2 ZMQ sidecar listening on tcp://%s:%d", args.host, args.port)

    try:
        while True:
            try:
                raw = socket.recv()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt; shutting down.")
                break

            try:
                request = msgpack.unpackb(raw, raw=False)
            except Exception as e:  # noqa: BLE001
                socket.send(msgpack.packb({"error": f"malformed_request:{e}"}, use_bin_type=True))
                continue

            # ``unpackb`` decodes any valid msgpack value, not just a map: the
            # single byte ``0x2a`` is the integer 42, and a string, list, nil or
            # bool decode just as cleanly. Reading ``endpoint`` off one of those
            # raises before the dispatch guard below exists to answer it, so it
            # is rejected here in the same class as bytes that do not decode at
            # all - either way the peer did not send a request. This is also
            # what makes the ``api_token`` check the fork notes suggest safe to
            # add: nothing reads a key off ``request`` until it is a map.
            if not isinstance(request, dict):
                socket.send(
                    msgpack.packb(
                        {"error": f"malformed_request:expected a msgpack map, got {type(request).__name__}"},
                        use_bin_type=True,
                    )
                )
                continue

            endpoint = request.get("endpoint", "")
            data = request.get("data") or {}

            # REQ/REP is lockstep: this peer is blocked until we reply, so
            # every request must produce exactly one response. Dispatch is
            # guarded because forks are invited to edit the handlers above
            # (start-state override, pose frame, world_update schema) and an
            # exception escaping here would close the socket with the reply
            # never sent - taking the sidecar down for every other client
            # too, not just the one that sent the bad request.
            try:
                if endpoint == "ping":
                    response = {"status": "ok"}
                elif endpoint == "reset":
                    # Reference implementation has nothing to reset; forks
                    # with stateful planners (RRT-Connect cache, etc.)
                    # should plug seed handling here.
                    seed = (data.get("options") or {}).get("seed")
                    logger.info("reset called (seed=%r); reference impl is a no-op", seed)
                    response = {"status": "ok"}
                elif endpoint == "plan":
                    response = _plan(
                        moveit_py,
                        planning_group=data.get("planning_group", args.planning_group),
                        joint_state=data.get("joint_state"),
                        target_pose=data.get("target_pose"),
                        target_joints=data.get("target_joints"),
                        world_update=data.get("world_update"),
                    )
                else:
                    response = {"error": f"unknown_endpoint:{endpoint}"}
                payload = msgpack.packb(response, use_bin_type=True)
            except Exception as e:  # noqa: BLE001 - answer the peer, keep serving
                logger.exception("Handling endpoint %r failed: %s", endpoint, e)
                payload = msgpack.packb({"error": f"internal_error:{type(e).__name__}:{e}"}, use_bin_type=True)

            socket.send(payload)
    finally:
        socket.close()
        context.term()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
