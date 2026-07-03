"""Two Unitree G1 humanoids autonomously make a bed in NVIDIA Isaac Sim,
coordinating as equal peers over Arm Device Connect.

This is a self-contained example (it does not modify the ``strands_robots``
product package or Arm's Device Connect). It runs under the Isaac Sim / Isaac
Lab Python on a DGX Spark and bundles its own copy of the Device Connect swarm
driver (:mod:`swarm_driver`), so it has no dependency on the MuJoCo demo.

Modules:

* :mod:`locomotion`   — the velocity walk-in policy + the whole-body bed-reach RL policy.
* :mod:`scene`        — build the room/bed/sheet/robots in Isaac Sim.
* :mod:`cloth`        — PhysX particle-cloth bedsheet + grasp attachment.
* :mod:`perception`   — robotics-connect-calibrated LiDAR + head-camera sensing (real-to-sim).
* :mod:`coordination` / :mod:`swarm_driver` — in-process Device Connect swarm of two G1 peers.
* :mod:`manipulation` / :mod:`replay` — legacy ``--pink`` / ``--replay`` paths.
* ``rl/``             — the bed-reach RL package (see ``RL_WHOLE_BODY_REACH.md``).
* ``demo.py``         — entrypoint: ``isaaclab.sh -p examples/isaac_bed_making/demo.py``.
"""
