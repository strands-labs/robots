"""VERA policy provider - video-to-action (two-stage planner + Jacobian IDM).

VERA (Video-to-Embodied Robot Action, MIT/CSAIL - https://github.com/sizhe-li/VERA)
is a two-stage closed-loop video-to-action policy: an embodiment-agnostic video
planner (DFoT / WAN) dreams future frames, and an embodiment-specific Jacobian
IDM translates the dream into robot actions. *One video planner, many IDMs.*

This provider wraps VERA's websocket policy server
(``vera.server.start_vera_server``) as a strands-robots :class:`Policy`, mirroring
the ``cosmos3`` service pattern: a self-contained msgpack+websocket client plus an
optional managed server subprocess.

Quickstart::

    # 1. Download Wave-1 checkpoints (~4 GB; full repo ~42 GB):
    #    hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts
    #    export VERA_CKPT_ROOT=$PWD/vera-ckpts
    #
    # 2. Install VERA + extras (Python 3.11, torch 2.6 / CUDA 12.4):
    #    pip install 'vera @ git+https://github.com/sizhe-li/VERA.git'
    #    pip install websockets msgpack 'numpy>=1.24'   # VERA websocket client deps
    #    (VERA is git-only - not a strands-robots extra; PyPI rejects VCS refs)

    from strands_robots.policies import create_policy

    policy = create_policy("vera", embodiment="pusht")
    chunk = policy.get_actions_sync(observation, "push the T to the goal")

In MuJoCo / sim::

    sim.run_policy(
        robot_name="pusher",
        policy_provider="vera",
        policy_config={"embodiment": "pusht"},
        instruction="push the T to the goal",
        n_steps=200,
    )

Available embodiments: ``pusht``, ``mimicgen`` (Wave 1) · ``allegro``, ``droid``
(Wave 2, code in-tree; checkpoints land later). See ``README.md``.
"""

from .client import VeraWebsocketClient
from .config import VeraConfig
from .provider import VeraPolicy
from .server_runner import DockerServerRunner, VeraServerRunner, make_server_runner

__all__ = [
    "VeraPolicy",
    "VeraConfig",
    "VeraWebsocketClient",
    "VeraServerRunner",
    "DockerServerRunner",
    "make_server_runner",
]
