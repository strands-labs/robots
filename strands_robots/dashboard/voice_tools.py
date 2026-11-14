"""The voice agent's tools: the chat agent's (``sim.tool``, every ``robot.tool``) plus the page's run controls.

``dashboard_run`` starts a policy in the background as the run button does (the robot tool's ``run`` waits for the
run to end, which would hold a spoken turn), ``dashboard_stop`` ends it, ``dashboard_estop`` latches the e-stop. Each
is an agent's call: ``confirmed`` is always False, so a real robot refuses to run from a sentence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import chat

if TYPE_CHECKING:
    from .protocols import DashboardLike

SYSTEM_PROMPT = (
    "You are the voice of a strands-robots dashboard: you operate robots that share one simulated world (and maybe "
    "real ones). Always speak English, in one or two short sentences. Act at once on a clear request; observe first "
    "only when you need a joint value or a pose. Use the robot_<id> tools to observe, act (joint targets; seconds=1 "
    "reaches a pose smoothly) and describe, and the sim_world tool for the world. Every robot_* and sim_* call names "
    'its rpc and nests the arguments under it: {"rpc": "act", "act": {"action": {"gripper": 1.5}, "seconds": 1}}. '
    "To start a policy that keeps going (walking, a skill, a VLA) call dashboard_run: it returns at once while the "
    "robot moves; dashboard_stop ends it, dashboard_estop latches the emergency stop. A gripper opens toward the top "
    "of its range and closes toward the bottom. Say what you did after a tool returns; if a tool refuses, say why "
    "in a few words."
)


def _result(tid: str, fn: Callable[[], Any]) -> dict[str, Any]:
    from ..tools.wire import json_safe

    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001 — the model reads the refusal and says it
        return {"toolUseId": tid, "status": "error", "content": [{"text": f"{type(exc).__name__}: {exc}"}]}
    return {"toolUseId": tid, "status": "success", "content": [{"json": json_safe(out)}]}


def _robot_arg(dash: DashboardLike) -> dict[str, Any]:
    ids = sorted({*dash.sim.robots, *dash.real})
    return {"type": "string", "description": f"robot id, one of {ids}"}


def _presets() -> str:
    from ..policy.onnx.presets import _table

    return "; ".join(f"{n} ({', '.join(f['robots'])}): {f.get('summary', '')}" for n, f in _table().items())


def run_tools(dash: DashboardLike) -> list[Any]:
    """``dashboard_run`` / ``dashboard_stop`` / ``dashboard_estop``: what the page's run row and e-stop do."""
    from strands.tools.tools import PythonAgentTool

    from ..robot.run.source import REMOTE_SPEC, from_agent
    from . import runs

    robot = _robot_arg(dash)

    def run(tool_use: Any, **_state: Any) -> dict[str, Any]:
        args = dict(tool_use.get("input") or {})

        def go() -> Any:
            spec = args.get("policy")
            if (
                isinstance(spec, str) and REMOTE_SPEC in spec
            ):  # same rule as robot.tool: no network address from a model
                raise PermissionError(
                    f"a policy at a network address is the developer's to pass, not an agent's: {spec}"
                )
            # an agent's run: the robot's chain sees every step; a real body refuses without X-Confirm
            with from_agent():
                return runs.start(
                    dash._robot(str(args.get("robot"))),
                    spec,
                    instruction=args.get("instruction"),
                    seconds=args.get("seconds"),
                    hz=None,
                    confirmed=False,  # the operator's yes is a click on the page, never a sentence
                )

        return _result(tool_use["toolUseId"], go)

    def stop(tool_use: Any, **_state: Any) -> dict[str, Any]:
        rid = str((tool_use.get("input") or {}).get("robot"))
        return _result(tool_use["toolUseId"], lambda: runs.stop(dash._robot(rid)))

    def estop(tool_use: Any, **_state: Any) -> dict[str, Any]:
        rid = str((tool_use.get("input") or {}).get("robot"))
        return _result(tool_use["toolUseId"], lambda: dash.estop(rid))

    def tool(name: str, about: str, props: dict[str, Any], required: list[str], fn: Callable[..., Any]) -> Any:
        schema = {"type": "object", "properties": props, "required": required}
        spec = {"name": name, "description": about, "inputSchema": {"json": schema}}
        return PythonAgentTool(name, spec, fn)  # type: ignore[arg-type]

    run_props = {
        "robot": robot,
        "policy": {
            "type": "string",
            "description": "policy spec: a preset with its command, e.g. 'holosoma?vx=0.4' walks a g1 forward at "
            f"0.4 m/s (vx<0 back, wz turns); or a LeRobot Hub repo. Presets: {_presets()}",
        },
        "seconds": {"type": "number", "description": "how long; leave out to run until dashboard_stop"},
        "instruction": {"type": "string", "description": "the task text for a language-conditioned (VLA) policy"},
    }
    return [
        tool(
            "dashboard_run",
            "Start a policy on a robot in the background, as the page's run button does; returns at once.",
            run_props,
            ["robot", "policy"],
            run,
        ),
        tool("dashboard_stop", "Stop the robot's background run.", {"robot": robot}, ["robot"], stop),
        tool(
            "dashboard_estop",
            "Emergency stop: latch the robot's e-stop (only the operator can release it).",
            {"robot": robot},
            ["robot"],
            estop,
        ),
    ]


def build_tools(dash: DashboardLike) -> list[Any]:
    """The chat agent's tools plus the run controls."""
    return [*chat.build_tools(dash.sim, dash.real.values()), *run_tools(dash)]


def build_agent(dash: DashboardLike, model: Any) -> Any:
    from strands.bidi import BidiAgent

    return BidiAgent(model=model, tools=build_tools(dash), system_prompt=SYSTEM_PROMPT)
