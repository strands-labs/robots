"""Reference ROS 2 / ``moveit_py`` sidecar for :class:`MoveIt2Policy`.

This package is **import-only** Python source. The ROS 2 / ``moveit_py``
deps stay out of ``pyproject.toml`` - the sidecar is meant to run inside a
ROS 2 environment (``source /opt/ros/jazzy/setup.bash`` + ``moveit_py``
installed) and the strands-robots client side talks to it via ZMQ +
msgpack from a plain Python venv.

Two recommended ways to run the sidecar:

1. **Docker compose** - see ``docker-compose.yml`` next to this module.
   ``docker compose up`` brings up a ``moveit_py`` container exposing
   port 5556 and connects the client transparently.

2. **Native ROS 2** - source your ROS 2 distro and ``moveit_py``, then::

       python -m strands_robots.policies.moveit2.server.zmq_node \\
           --port 5556 --planning-group arm

   This is the recommended dev-loop setup; the docker path is for
   pinned reproducible deployments.

The wire protocol is documented in :mod:`strands_robots.policies.moveit2.client`.
"""

# Note: importing the server module here would force-import rclpy /
# moveit_py at package import time, which would defeat the whole
# "import-only Python source" framing. Users who want to run the
# sidecar import ``zmq_node`` explicitly. Documented above.
__all__: list[str] = []
