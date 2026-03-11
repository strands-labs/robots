"""MujocoBackend — the main simulation class (AgentTool).

Thin orchestrator that delegates to the submodules in this package.
"""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, Optional

from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolSpec, ToolUse

from ._rendering import apply_sim_action, get_sim_observation
from ._tool import build_tool_spec, dispatch_action
from ._tool import stream as _stream
from ._types import SimWorld
from ._viewer import close_viewer_internal

logger = logging.getLogger(__name__)


class MujocoBackend(AgentTool):
    """Programmatic MuJoCo simulation environment as a Strands AgentTool.

    Gives AI agents the ability to create, modify, and control MuJoCo
    simulation environments through natural language → tool actions.
    """

    def __init__(
        self,
        tool_name: str = "sim",
        default_timestep: float = 0.002,
        default_width: int = 640,
        default_height: int = 480,
        mesh: bool = True,
        peer_id: str = None,
        **kwargs,
    ):
        super().__init__()
        self.tool_name_str = tool_name
        self.default_timestep = default_timestep
        self.default_width = default_width
        self.default_height = default_height

        # World state
        self._world: Optional[SimWorld] = None
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix=f"{tool_name}_sim"
        )
        self._policy_threads: Dict[str, Future] = {}
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

        # Viewer handle
        self._viewer_handle = None
        self._viewer_thread = None

        # Cache renderers per (width, height) to avoid GPU context churn
        self._renderers: Dict[tuple, Any] = {}
        self._renderer_model = None  # Track which model the renderers are for

        logger.info("🎮 Simulation tool '%s' initialized", tool_name)

        # Zenoh mesh — every simulation backend is a peer by default
        try:
            from strands_robots.zenoh_mesh import init_mesh

            self.mesh = init_mesh(self, peer_id=peer_id, peer_type="sim", mesh=mesh)
        except Exception as e:
            logger.debug("Mesh init skipped: %s", e)
            self.mesh = None

    # -------------------------------------------------------------------
    # Public Properties — Direct MuJoCo model/data access
    # -------------------------------------------------------------------

    @property
    def mj_model(self):
        """Direct access to the MuJoCo model (mujoco.MjModel)."""
        return self._world._model if self._world else None

    @property
    def mj_data(self):
        """Direct access to the MuJoCo data (mujoco.MjData)."""
        return self._world._data if self._world else None

    # -------------------------------------------------------------------
    # Robot ABC compatible interface
    # -------------------------------------------------------------------

    def get_observation(
        self, robot_name: str = None, camera_name: str = None
    ) -> Dict[str, Any]:
        """Get observation from simulation (Robot ABC compatible)."""
        if self._world is None or self._world._model is None:
            return {}

        if robot_name is None:
            if not self._world.robots:
                return {}
            robot_name = next(iter(self._world.robots))

        if robot_name not in self._world.robots:
            return {}

        return get_sim_observation(self, robot_name, cam_name=camera_name)

    def send_action(
        self, action: Dict[str, Any], robot_name: str = None, n_substeps: int = 1
    ) -> None:
        """Apply action to simulation (Robot ABC compatible)."""
        if self._world is None or self._world._model is None:
            return

        if robot_name is None:
            if not self._world.robots:
                return
            robot_name = next(iter(self._world.robots))

        if robot_name not in self._world.robots:
            return

        apply_sim_action(self, robot_name, action, n_substeps=n_substeps)

    # -------------------------------------------------------------------
    # AgentTool Interface
    # -------------------------------------------------------------------

    @property
    def tool_name(self) -> str:
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        return "simulation"

    @property
    def tool_spec(self) -> ToolSpec:
        return build_tool_spec(self.tool_name_str)

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        async for event in _stream(self, tool_use, invocation_state, **kwargs):
            yield event

    def _dispatch_action(self, action: str, d: Dict[str, Any]) -> Dict[str, Any]:
        """Route action to method via dispatch table."""
        return dispatch_action(self, action, d)

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def cleanup(self):
        if hasattr(self, "mesh") and self.mesh:
            self.mesh.stop()
        if self._world:
            for r in self._world.robots.values():
                r.policy_running = False
            self._world = None
        close_viewer_internal(self)
        self._executor.shutdown(wait=False)
        self._shutdown_event.set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
