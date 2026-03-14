#!/usr/bin/env python3
"""
Robot Mesh Tool — agent-facing tool for robot mesh coordination.

Uses Device Connect's discover_devices and invoke_device for registry-based
discovery and structured RPC invocation.  Falls back to the plain Zenoh
mesh if Device Connect is not available.
"""

import json
import logging
import os
from typing import Any, Dict

from strands import tool

logger = logging.getLogger(__name__)

_dc_connected = False


class _ToolResult(dict):
    """Dict that prints its text content cleanly."""

    def __str__(self):
        content = self.get("content", [])
        if content and isinstance(content[0], dict):
            return content[0].get("text", super().__str__())
        return super().__str__()


def _ensure_connected():
    """Ensure Device Connect agent-side connection is established."""
    global _dc_connected
    if _dc_connected:
        return
    # Set P2P defaults before import (MessagingConfig reads env at import time)
    os.environ.setdefault("MESSAGING_BACKEND", "zenoh")
    os.environ.setdefault("DEVICE_CONNECT_ALLOW_INSECURE", "true")
    from device_connect_agent_tools.connection import connect, get_connection
    try:
        get_connection()
    except Exception:
        connect()
    _dc_connected = True


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
    """Robot mesh — discover and coordinate all robots on the network.

    Every Robot() is automatically a network peer. This tool lets you see them,
    talk to them, and coordinate them. Also sees Reachy Mini.

    Args:
        action: What to do:
            - "peers": List all discovered robots and agents
            - "tell": Tell a robot to execute an instruction
            - "send": Send a raw command to a peer
            - "broadcast": Send command to ALL peers
            - "stop": Stop a specific peer's task
            - "emergency_stop": E-STOP all robots
            - "status": Network overview
            - "subscribe": Subscribe to any Zenoh topic (target=topic pattern)
            - "watch": Watch a robot's VLA execution stream (target=peer_id)
            - "inbox": Read buffered messages from subscriptions
        target: Device ID for tell/send/stop, or topic for subscribe
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
        robot_mesh(action="tell", target="so100-lab-1", instruction="pick up the cube")
        robot_mesh(action="emergency_stop")
    """
    try:
        _ensure_connected()
        return _device_connect_dispatch(action, target, instruction, command,
                                policy_provider, policy_port, duration, timeout)
    except Exception as e:
        logger.debug(f"Device Connect dispatch failed, falling back to Zenoh mesh: {e}")
        return _mesh_dispatch(action, target, instruction, command,
                              policy_provider, policy_port, duration, timeout)


def _device_connect_dispatch(action, target, instruction, command,
                             policy_provider, policy_port, duration, timeout):
    """Dispatch using Device Connect's discover_devices + invoke_device."""
    try:
        from device_connect_agent_tools.connection import get_connection
        conn = get_connection()

        if action == "peers":
            devices = conn.list_devices()
            text = f"Discovered {len(devices)} device(s):\n"
            for d in devices:
                dtype = d.get("device_type", "?")
                icon = {"strands_robot": "robot", "strands_sim": "sim",
                        "reachy_mini": "reachy"}.get(dtype, dtype)
                status = d.get("status", {})
                avail = status.get("availability", "?") if isinstance(status, dict) else "?"
                text += f"  [{icon}] {d['device_id']} — {avail}\n"
                funcs = d.get("functions", [])
                if funcs:
                    names = [f["name"] if isinstance(f, dict) else f for f in funcs]
                    text += f"    Functions: {', '.join(names)}\n"
            return _ToolResult({"status": "success", "content": [{"text": text}]})

        elif action == "tell":
            if not target or not instruction:
                return _ToolResult({"status": "error", "content": [{"text": "target and instruction required"}]})
            params = {
                "instruction": instruction,
                "policy_provider": policy_provider,
                "duration": duration,
            }
            if policy_port:
                params["policy_port"] = policy_port
            result = conn.invoke(target, "execute", params, timeout=timeout)
            r = result.get("result", result)
            return _ToolResult({"status": "success", "content": [{"text": f"-> {target}: {instruction}\n  {json.dumps(r, default=str)}"}]})

        elif action == "send":
            if not target:
                return _ToolResult({"status": "error", "content": [{"text": "target required"}]})
            func = "getStatus"
            params = {}
            if command:
                cmd = json.loads(command)
                func = cmd.pop("action", cmd.pop("function", "getStatus"))
                params = cmd
            result = conn.invoke(target, func, params, timeout=timeout)
            r = result.get("result", result)
            return _ToolResult({"status": "success", "content": [{"text": f"{target}:\n{json.dumps(r, indent=2, default=str)[:2000]}"}]})

        elif action == "stop":
            if not target:
                return _ToolResult({"status": "error", "content": [{"text": "target required"}]})
            result = conn.invoke(target, "stop", timeout=5.0)
            r = result.get("result", result)
            return _ToolResult({"status": "success", "content": [{"text": f"Stop {target}: {json.dumps(r, default=str)}"}]})

        elif action == "emergency_stop":
            devices = conn.list_devices()
            stopped = 0
            for d in devices:
                try:
                    conn.invoke(d["device_id"], "stop", timeout=3.0)
                    stopped += 1
                except Exception:
                    pass
            return _ToolResult({"status": "success", "content": [{"text": f"E-STOP: {stopped}/{len(devices)} devices stopped"}]})

        elif action == "broadcast":
            func = "getStatus"
            params = {}
            if command:
                cmd = json.loads(command)
                func = cmd.pop("action", cmd.pop("function", "getStatus"))
                params = cmd
            results = conn.broadcast(func, params, timeout=timeout)
            text = f"Broadcast '{func}' -> {len(results)} response(s):\n"
            for r in results:
                status_str = "ok" if "result" in r else f"error: {r.get('error', '?')}"
                text += f"  {r['device_id']}: {status_str}\n"
            return _ToolResult({"status": "success", "content": [{"text": text}]})

        elif action == "subscribe":
            if not target:
                return _ToolResult({"status": "error", "content": [{"text": "target (subject pattern) required. E.g. 'device-connect.default.*.event.>'"}]})
            name = conn.subscribe_buffered(target)
            return _ToolResult({"status": "success", "content": [{"text": f"Subscribed to: {target}\nMessages buffered in inbox['{name}']\nUse action='inbox' to read."}]})

        elif action == "watch":
            if not target:
                return _ToolResult({"status": "error", "content": [{"text": "target (device_id) required"}]})
            subject = f"device-connect.{conn.zone}.{target}.event.>"
            name = conn.subscribe_buffered(subject, name=f"stream:{target}")
            return _ToolResult({"status": "success", "content": [{"text": f"Watching events from: {target}\nMessages in inbox['{name}']"}]})

        elif action == "inbox":
            inbox = conn.get_inbox()
            if not inbox:
                return _ToolResult({"status": "success", "content": [{"text": "No subscriptions active"}]})
            text = f"Inbox ({len(inbox)} subscription(s)):\n"
            for name, msgs in inbox.items():
                text += f"  {name}: {len(msgs)} messages\n"
                if msgs:
                    last = msgs[-1]
                    # Messages are (subject, data) tuples — matching Zenoh mesh format
                    text += f"    Latest: {json.dumps(last[1], default=str)[:200]}\n"
            return _ToolResult({"status": "success", "content": [{"text": text}]})

        elif action == "status":
            devices = conn.list_devices()
            text = f"Network: {len(devices)} device(s)\n"
            for d in devices:
                status = d.get("status", {})
                avail = status.get("availability", "?") if isinstance(status, dict) else "?"
                text += f"  {d['device_id']} ({d.get('device_type', '?')}) — {avail}\n"
            return _ToolResult({"status": "success", "content": [{"text": text}]})

        else:
            return _ToolResult({"status": "error", "content": [{"text": f"Unknown action: {action}. Try: peers, tell, send, broadcast, stop, emergency_stop, status, subscribe, watch, inbox"}]})

    except Exception as e:
        return _ToolResult({"status": "error", "content": [{"text": f"Error: {e}"}]})


def _mesh_dispatch(action, target, instruction, command,
                   policy_provider, policy_port, duration, timeout):
    """Fallback: dispatch using plain Zenoh mesh."""
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
                    text += f"  {icon} {p['peer_id']} ({p.get('type','?')}) — {p.get('hostname','?')}, {p.get('age',0)}s ago\n"
                    if p.get("task_status"):
                        text += f"      Task: {p['task_status']} — {p.get('instruction', '')}\n"
            elif not local:
                text += "No peers. Create a Robot() — it auto-joins the mesh.\n"

            return _ToolResult({"status": "success", "content": [{"text": text}]})

        elif action == "tell":
            if not target or not instruction:
                return _ToolResult({"status": "error", "content": [{"text": "target and instruction required"}]})
            mesh = _any_mesh()
            if not mesh:
                return _ToolResult({"status": "error", "content": [{"text": "No local robots on mesh"}]})
            cmd = {"action": "execute", "instruction": instruction,
                   "policy_provider": policy_provider, "duration": duration}
            if policy_port:
                cmd["policy_port"] = policy_port
            r = mesh.send(target, cmd, timeout=timeout)
            return _ToolResult({"status": "success", "content": [{"text": f"📨 → {target}: {instruction}\n\n{json.dumps(r, indent=2, default=str)[:2000]}"}]})

        elif action == "send":
            if not target:
                return _ToolResult({"status": "error", "content": [{"text": "target required"}]})
            mesh = _any_mesh()
            if not mesh:
                return _ToolResult({"status": "error", "content": [{"text": "No local robots on mesh"}]})
            cmd = json.loads(command) if command else {"action": "status"}
            r = mesh.send(target, cmd, timeout=timeout)
            return _ToolResult({"status": "success", "content": [{"text": f"📨 {target}:\n{json.dumps(r, indent=2, default=str)[:2000]}"}]})

        elif action == "broadcast":
            mesh = _any_mesh()
            if not mesh:
                return _ToolResult({"status": "error", "content": [{"text": "No local robots on mesh"}]})
            cmd = json.loads(command) if command else {"action": "status"}
            rs = mesh.broadcast(cmd, timeout=timeout)
            return _ToolResult({"status": "success", "content": [{"text": f"📢 {len(rs)} responses:\n{json.dumps(rs, indent=2, default=str)[:3000]}"}]})

        elif action == "stop":
            if not target:
                return _ToolResult({"status": "error", "content": [{"text": "target required"}]})
            mesh = _any_mesh()
            if not mesh:
                return _ToolResult({"status": "error", "content": [{"text": "No local robots on mesh"}]})
            r = mesh.send(target, {"action": "stop"}, timeout=5.0)
            return _ToolResult({"status": "success", "content": [{"text": f"🛑 {target}: {json.dumps(r, default=str)}"}]})

        elif action == "emergency_stop":
            mesh = _any_mesh()
            if not mesh:
                return _ToolResult({"status": "error", "content": [{"text": "No local robots on mesh"}]})
            rs = mesh.emergency_stop()
            return _ToolResult({"status": "success", "content": [{"text": f"🚨 E-STOP → {len(rs)} responses"}]})

        elif action == "status":
            local = list(_LOCAL_ROBOTS.keys())
            peers = get_peers()
            text = f"Mesh: {len(local)} local robots, {len(peers)} peers\n"
            for rid in local:
                m = _LOCAL_ROBOTS[rid]
                text += f"  • {rid} ({m.peer_type}) alive={m.alive}\n"
            return _ToolResult({"status": "success", "content": [{"text": text}]})

        elif action == "subscribe":
            if not target:
                return _ToolResult({"status": "error", "content": [{"text": "target (topic pattern) required. E.g. 'reachy_mini/*' or '*/joint_positions'"}]})
            mesh = _any_mesh()
            if not mesh:
                return _ToolResult({"status": "error", "content": [{"text": "No local robots on mesh"}]})
            name = mesh.subscribe(target)
            return _ToolResult({"status": "success", "content": [{"text": f"📡 Subscribed to: {target}\nMessages buffered in inbox['{name}']\nUse action='inbox' to read."}]})

        elif action == "watch":
            if not target:
                return _ToolResult({"status": "error", "content": [{"text": "target (peer_id) required"}]})
            mesh = _any_mesh()
            if not mesh:
                return _ToolResult({"status": "error", "content": [{"text": "No local robots on mesh"}]})
            name = mesh.on_stream(target)
            return _ToolResult({"status": "success", "content": [{"text": f"👁️ Watching VLA stream from: {target}\nMessages in inbox['{name}']"}]})

        elif action == "inbox":
            mesh = _any_mesh()
            if not mesh or not hasattr(mesh, 'inbox'):
                return _ToolResult({"status": "success", "content": [{"text": "No subscriptions active"}]})
            text = f"📬 Inbox ({len(mesh.inbox)} subscriptions):\n"
            for name, msgs in mesh.inbox.items():
                text += f"  • {name}: {len(msgs)} messages\n"
                if msgs:
                    last = msgs[-1]
                    text += f"    Latest: {json.dumps(last[1], default=str)[:200]}\n"
            return _ToolResult({"status": "success", "content": [{"text": text}]})

        else:
            return _ToolResult({"status": "error", "content": [{"text": f"Unknown action: {action}. Try: peers, tell, send, broadcast, stop, emergency_stop, status, subscribe, watch, inbox"}]})

    except ImportError as e:
        return _ToolResult({"status": "error", "content": [{"text": f"Mesh unavailable: {e}"}]})
    except Exception as e:
        return _ToolResult({"status": "error", "content": [{"text": f"Error: {e}"}]})


def _any_mesh():
    """Get any local mesh instance."""
    from strands_robots.zenoh_mesh import _LOCAL_ROBOTS
    return next(iter(_LOCAL_ROBOTS.values()), None) if _LOCAL_ROBOTS else None
