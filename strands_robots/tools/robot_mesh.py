#!/usr/bin/env python3
"""
Robot Mesh Tool — agent-facing tool for robot mesh coordination.

Since every Robot() is already on the mesh, this tool just provides
the agent interface for discovery, messaging, and coordination.
"""

import json
import logging
from typing import Any, Dict

from strands import tool

logger = logging.getLogger(__name__)


@tool
def robot_mesh(
    action: str,
    target: str = "",
    instruction: str = "",
    command: str = "",
    policy_provider: str = "mock",
    policy_port: int = 0,
    duration: float = 30.0,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Robot mesh — discover and coordinate all robots on the Zenoh network.

    Every Robot() is automatically a mesh peer. This tool lets you see them,
    talk to them, and coordinate them. Also sees Reachy Mini.

    Args:
        action: What to do:
            - "peers": List all discovered robots and agents
            - "tell": Tell a robot to execute an instruction
            - "send": Send a raw command to a peer
            - "broadcast": Send command to ALL peers
            - "stop": Stop a specific peer's task
            - "emergency_stop": E-STOP all robots
            - "status": Mesh overview
            - "subscribe": Subscribe to any Zenoh topic (target=topic pattern)
            - "watch": Watch a robot's VLA execution stream (target=peer_id)
            - "inbox": Read buffered messages from subscriptions
        target: Peer ID for tell/send/stop, or topic for subscribe
        instruction: Natural language instruction for tell
        command: JSON command string for send/broadcast
        policy_provider: Policy for tell (groot, mock, lerobot_local, ...)
        policy_port: Policy port for tell
        duration: Task duration for tell
        timeout: Response timeout seconds

    Returns:
        Status dict with content

    Examples:
        robot_mesh(action="peers")
        robot_mesh(action="tell", target="so100_sim-a1b2", instruction="pick up the cube")
        robot_mesh(action="send", target="Mac-fc0610", command='{"action": "status"}')
        robot_mesh(action="emergency_stop")
    """
    try:
        from strands_robots.zenoh_mesh import _LOCAL_ROBOTS, get_peers

        if action == "peers":
            peers = get_peers()
            local = list(_LOCAL_ROBOTS.keys())

            text = f"🔗 Mesh: {len(local)} local, {len(peers)} remote\n\n"
            if local:
                text += "**Local (this process):**\n"
                for rid in local:
                    m = _LOCAL_ROBOTS[rid]
                    text += f"  ✅ {rid} ({m.peer_type})\n"
                text += "\n"
            if peers:
                text += "**Discovered peers:**\n"
                for p in peers:
                    icon = {"robot": "🤖", "sim": "🎮", "agent": "🧠"}.get(p.get("type", ""), "🔧")
                    text += (
                        f"  {icon} {p['peer_id']} ({p.get('type', '?')}) "
                        f"— {p.get('hostname', '?')}, {p.get('age', 0)}s ago\n"
                    )
                    if p.get("task_status"):
                        text += f"      Task: {p['task_status']} — {p.get('instruction', '')}\n"
            elif not local:
                text += "No peers. Create a Robot() — it auto-joins the mesh.\n"

            return {"status": "success", "content": [{"text": text}]}

        elif action == "tell":
            if not target or not instruction:
                return {"status": "error", "content": [{"text": "target and instruction required"}]}
            mesh = _any_mesh()
            if not mesh:
                return {"status": "error", "content": [{"text": "No local robots on mesh"}]}
            cmd = {
                "action": "execute",
                "instruction": instruction,
                "policy_provider": policy_provider,
                "duration": duration,
            }
            if policy_port:
                cmd["policy_port"] = policy_port
            r = mesh.send(target, cmd, timeout=timeout)
            return {
                "status": "success",
                "content": [{"text": f"📨 → {target}: {instruction}\n\n{json.dumps(r, indent=2, default=str)[:2000]}"}],
            }

        elif action == "send":
            if not target:
                return {"status": "error", "content": [{"text": "target required"}]}
            mesh = _any_mesh()
            if not mesh:
                return {"status": "error", "content": [{"text": "No local robots on mesh"}]}
            cmd = json.loads(command) if command else {"action": "status"}
            r = mesh.send(target, cmd, timeout=timeout)
            return {
                "status": "success",
                "content": [{"text": f"📨 {target}:\n{json.dumps(r, indent=2, default=str)[:2000]}"}],
            }

        elif action == "broadcast":
            mesh = _any_mesh()
            if not mesh:
                return {"status": "error", "content": [{"text": "No local robots on mesh"}]}
            cmd = json.loads(command) if command else {"action": "status"}
            rs = mesh.broadcast(cmd, timeout=timeout)
            return {
                "status": "success",
                "content": [{"text": f"📢 {len(rs)} responses:\n{json.dumps(rs, indent=2, default=str)[:3000]}"}],
            }

        elif action == "stop":
            if not target:
                return {"status": "error", "content": [{"text": "target required"}]}
            mesh = _any_mesh()
            if not mesh:
                return {"status": "error", "content": [{"text": "No local robots on mesh"}]}
            r = mesh.send(target, {"action": "stop"}, timeout=5.0)
            return {"status": "success", "content": [{"text": f"🛑 {target}: {json.dumps(r, default=str)}"}]}

        elif action == "emergency_stop":
            mesh = _any_mesh()
            if not mesh:
                return {"status": "error", "content": [{"text": "No local robots on mesh"}]}
            rs = mesh.emergency_stop()
            return {"status": "success", "content": [{"text": f"🚨 E-STOP → {len(rs)} responses"}]}

        elif action == "status":
            local = list(_LOCAL_ROBOTS.keys())
            peers = get_peers()
            text = f"Mesh: {len(local)} local robots, {len(peers)} peers\n"
            for rid in local:
                m = _LOCAL_ROBOTS[rid]
                text += f"  • {rid} ({m.peer_type}) alive={m.alive}\n"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "subscribe":
            if not target:
                return {
                    "status": "error",
                    "content": [
                        {"text": "target (topic pattern) required. E.g. 'reachy_mini/*' or '*/joint_positions'"}
                    ],
                }
            mesh = _any_mesh()
            if not mesh:
                return {"status": "error", "content": [{"text": "No local robots on mesh"}]}
            name = mesh.subscribe(target)
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"📡 Subscribed to: {target}\n"
                            f"Messages buffered in inbox['{name}']\n"
                            f"Use action='inbox' to read."
                        )
                    }
                ],
            }

        elif action == "watch":
            if not target:
                return {"status": "error", "content": [{"text": "target (peer_id) required"}]}
            mesh = _any_mesh()
            if not mesh:
                return {"status": "error", "content": [{"text": "No local robots on mesh"}]}
            name = mesh.on_stream(target)
            return {
                "status": "success",
                "content": [{"text": f"👁️ Watching VLA stream from: {target}\nMessages in inbox['{name}']"}],
            }

        elif action == "inbox":
            mesh = _any_mesh()
            if not mesh or not hasattr(mesh, "inbox"):
                return {"status": "success", "content": [{"text": "No subscriptions active"}]}
            text = f"📬 Inbox ({len(mesh.inbox)} subscriptions):\n"
            for name, msgs in mesh.inbox.items():
                text += f"  • {name}: {len(msgs)} messages\n"
                if msgs:
                    last = msgs[-1]
                    text += f"    Latest: {json.dumps(last[1], default=str)[:200]}\n"
            return {"status": "success", "content": [{"text": text}]}

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Unknown action: {action}. Try: peers, tell, send, broadcast, "
                            f"stop, emergency_stop, status, subscribe, watch, inbox"
                        )
                    }
                ],
            }

    except ImportError as e:
        return {"status": "error", "content": [{"text": f"Mesh unavailable: {e}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}


def _any_mesh():
    """Get any local mesh instance."""
    from strands_robots.zenoh_mesh import _LOCAL_ROBOTS

    return next(iter(_LOCAL_ROBOTS.values()), None) if _LOCAL_ROBOTS else None
