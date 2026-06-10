"""Cosmos 3 Policy — NVIDIA omnimodal VLA policy for strands-robots.

Wraps the Cosmos 3 Generator *action* surface (e.g.
``nvidia/Cosmos3-Nano-Policy-DROID``) as a robots :class:`Policy`. Service mode
speaks to the Cosmos Framework RoboLab WebSocket policy server using a
self-contained msgpack+NumPy wire client (no ``openpi-client`` dependency
— see ``client.py`` for the rationale).

Quickstart::

    # 1. Start the policy server (holds the GPU), from a Cosmos Framework checkout:
    #    uv sync --all-extras --group=cu130-train --group=policy-server
    #    python -m cosmos_framework.scripts.action_policy_server_robolab \
    #        --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000
    #    # wait for: curl http://localhost:8000/healthz  -> 200
    #
    # 2. Client install (numpy-version agnostic — composes with lerobot):
    #    pip install 'strands-robots[cosmos3-service]'

    from strands_robots.policies import create_policy

    policy = create_policy("cosmos3", embodiment="droid", port=8000)
    chunk = policy.get_actions_sync(observation, "pick up the cube")

In MuJoCo (the ``droid`` embodiment drives a Franka/DROID-class arm — use the
``franka`` or ``panda`` sim asset)::

    from strands_robots import Simulation
    sim = Simulation(tool_name="sim", mesh=False); sim.create_world()
    sim.add_robot(name="arm", data_config="franka")   # DROID == Franka Emika Panda
    ...  # add cube + wrist/front/side cameras
    sim.run_policy(robot_name="arm", policy_provider="cosmos3",
                   policy_config={"embodiment": "droid", "port": 8000,
                                  "observation_mapping": {...}},
                   instruction="pick up the red cube", n_steps=24,
                   control_frequency=15.0)

See ``examples/cosmos3_sim_rollout.py`` for a complete, runnable rollout +
recording. Available embodiments: droid, umi, av, bridge (see ``embodiments.py``).
"""

from .client import Cosmos3WebsocketClient
from .embodiments import (
    EMBODIMENTS,
    ROBOT_ACTION_MAPPINGS,
    Cosmos3Embodiment,
    get_embodiment,
    get_robot_action_mapping,
    list_embodiments,
    list_robot_action_mappings,
)
from .policy import Cosmos3Policy

__all__ = [
    "Cosmos3Policy",
    "Cosmos3WebsocketClient",
    "Cosmos3Embodiment",
    "EMBODIMENTS",
    "ROBOT_ACTION_MAPPINGS",
    "get_embodiment",
    "get_robot_action_mapping",
    "list_embodiments",
    "list_robot_action_mappings",
]
