"""Remote policy inference: client/server split for edge robots + remote GPU.

A resource-constrained robot host (CPU / edge device) often cannot run a large
VLA (pi0 / SmolVLA / MolmoAct2) at control rate. This package splits inference
across two machines over a portable WS-JSON WebSocket protocol:

* :class:`~strands_robots.inference.server.PolicyServer` wraps ANY
  :class:`~strands_robots.policies.base.Policy` and serves it (run it on the
  GPU box).
* :class:`~strands_robots.inference.client.RemotePolicy` is a drop-in
  ``Policy`` that forwards observations to the server and returns the action
  chunk (construct it on the robot host). It is wired into
  :func:`~strands_robots.policies.create_policy` as the ``remote`` provider, so
  ``create_policy("remote", endpoint="ws://gpu-box:8765")`` (or the smart
  string ``create_policy("ws://gpu-box:8765")``) yields one.

Requires the ``inference`` extra::

    pip install 'strands-robots[inference]'

See ``docs/inference/remote.md`` for the two-machine setup.
"""

from strands_robots.inference.client import RemotePolicy
from strands_robots.inference.server import PolicyServer

__all__ = ["PolicyServer", "RemotePolicy"]
