"""``use_ros``: one Strands tool for any ROS 2 graph. Reads are free; motion writes need an operator.

    from strands import Agent
    from strands_robots.ros import use_ros
    agent = Agent(tools=[use_ros()])                       # a Strands interrupt asks the human for writes
    agent = Agent(tools=[use_ros(approve=lambda v, t, p: input(f"{v} {t}? ") == "y")])

Approval order for a write on a motion surface: exact ``STRANDS_ROS_ALLOW`` names, then ``approve=``, then a
Strands interrupt to the human; with none of them the write is refused. ``BYPASS_TOOL_CONSENT`` is ignored.
Topics a strands Robot streams are refused outright: command that robot through its own tool.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from . import profile as _profile
from .bus import RosBus
from .gate import MOTION_SURFACES, Approver, CommandGate
from .verbs import VERBS, dispatch

DOC = f"""Inspect and use a ROS 2 graph: list/echo/sample topics, call services, send action goals,
read/set parameters, look up TF, record a bag. Writes that can move a robot need an operator's yes.

Args:
    verb: one of {", ".join(VERBS)}.
    args: JSON object for the verb, e.g. {{"topic": "/joint_states", "count": 1}};
        publish {{"topic","type","fields","count","rate"}}; call {{"service","type","request"}};
        goal {{"action","type","goal","timeout"}}; params {{"node","names"}}; set_param {{"node","name","value"}};
        tf {{"target","source"}}; record {{"topics": [...], "seconds", "path"}}; stop {{"profile"}}.
"""


def _ok(data: Any) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": json.dumps(data, default=str)[:20000]}]}


def _err(e: Exception) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": f"{type(e).__name__}: {e}"}]}


def use_ros(bus: RosBus | None = None, *, approve: Approver | None = None, extra_surfaces: tuple[str, ...] = ()) -> Any:
    """Build the tool. ``bus`` defaults to the process's shared RosBus (made on first use)."""
    from strands import tool
    from strands.interrupt import InterruptException

    def interrupt_approver(ctx: Any) -> Approver | None:
        if approve is not None:
            return approve
        if ctx is None or not hasattr(ctx, "interrupt"):
            return None

        def ask(verb: str, target: str, preview: Mapping[str, Any]) -> bool:
            answer = ctx.interrupt(
                "use_ros-command-approval",
                reason={
                    "action": verb,
                    "target": target,
                    "preview": preview,
                    "warning": "this can move a real robot",
                    "how_to_answer": "reply yes to allow it once",
                },
            )
            return str(answer).strip().lower() in ("y", "yes", "approve", "approved")

        return ask

    @tool(name="use_ros", description=DOC, context=True)
    def _use_ros(verb: str, args: dict[str, Any] | None = None, tool_context: Any = None) -> dict[str, Any]:
        try:
            b = bus or RosBus.shared()
            gate = CommandGate(
                surfaces=MOTION_SURFACES + _profile.motion_surfaces() + tuple(extra_surfaces),
                approve=interrupt_approver(tool_context),
                owners=b.owners,
            )
            return _ok(dispatch(b, gate, verb, dict(args or {})))
        except InterruptException:  # the human is being asked: the agent must see it
            raise
        except Exception as e:  # noqa: BLE001 - the model reads the refusal and corrects its next call
            return _err(e)

    return _use_ros
