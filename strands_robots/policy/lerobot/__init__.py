"""LeRobot as a policy provider: any checkpoint in-process, or any ``lerobot-policy-server`` over Zenoh.

LeRobot owns loading, saved processors, admission and the chunk runtime; this package only translates
names, units, cameras and holds at the edge. Nothing here imports lerobot, torch or zenoh at load time.
"""

from .contract import Contract, server_config, write_server_yaml
from .local import LeRobotPolicy
from .remote import ZenohPolicy
from .server import serve

__all__ = ["LeRobotPolicy", "ZenohPolicy", "Contract", "server_config", "write_server_yaml", "serve"]
