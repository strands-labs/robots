"""Original Unitree G1 MuJoCo demos, organised by iteration.

Three iterations build on each other, each in its own sub-package with its README:

* ``humanoid_simulation/`` — bring a single G1 up in MuJoCo and drive it.
* ``bed_making_demo/``      — two G1s physically make a bed in MuJoCo (single run).
* ``bed_making_swarm/``     — the two G1s coordinate as equal peers over Arm Device
  Connect; its ``--mujoco`` mode drives the ``bed_making_demo`` visualisation.

The later Isaac Sim work lives in ``examples/isaac_bed_making/`` (it carries its own
swarm driver, ``swarm_driver.py``, rather than importing these).
"""
