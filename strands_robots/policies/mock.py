"""Mock policy for testing — generates smooth sinusoidal trajectories."""

import logging
import math
from typing import Any

from strands_robots.policies.base import Policy

logger = logging.getLogger(__name__)


class MockPolicy(Policy):
    """Mock policy for testing — generates smooth sinusoidal trajectories."""

    def __init__(self, **kwargs: Any) -> None:
        self.robot_state_keys: list[str] = []
        self._step = 0
        logger.info("Mock Policy initialized")

    @property
    def provider_name(self) -> str:
        return "mock"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = robot_state_keys

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Return smooth sinusoidal actions."""
        if not self.robot_state_keys:
            if "observation.state" in observation_dict:
                state = observation_dict["observation.state"]
                dim = len(state) if hasattr(state, "__len__") else 6
            else:
                dim = 6
            self.robot_state_keys = [f"joint_{i}" for i in range(dim)]

        mock_actions = []
        for i in range(8):
            action_dict = {}
            t = (self._step + i) * 0.02
            for j, key in enumerate(self.robot_state_keys):
                freq = 0.3 + j * 0.15
                phase = j * math.pi / 3
                action_dict[key] = 0.5 * math.sin(2 * math.pi * freq * t + phase)
            mock_actions.append(action_dict)

        self._step += len(mock_actions)
        return mock_actions
