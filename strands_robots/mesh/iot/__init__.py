"""AWS IoT Core, native: shadows, rules, X.509 provisioning and fleet claims for a mesh that runs over
``iot://``. ``Mesh.mount(robot, iot=IotOptions(...))`` turns it on per device. ``import strands_robots.mesh`` loads
only stdlib modules from here; ``awsiot``, ``boto3``, and ``cryptography`` are imported inside the functions
that need them (``pip install 'strands-robots[iot]'``). Modules: ``shadow``, ``rules``, ``policy``, ``provision``,
``claim``, ``registry``, ``lifecycle``, ``direct``, ``greengrass``, ``kvs``, ``options``."""

from __future__ import annotations

from .options import IotOptions
from .shadow import ShadowMirror, shadow_topic

__all__ = ["IotOptions", "ShadowMirror", "shadow_topic"]
