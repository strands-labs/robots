"""Connect: other peers' robots over the mesh, and policies served elsewhere, as cards and specs on this page.

A remote robot is ``mesh.connect(id, images=True)``: a ``Robot`` whose body is the device on its peer. It is a
hardware robot to this page (``is_real``), so every act, run and resume from here needs the operator's
``X-Confirm: yes``, and the device's own safety chain runs on its side too; an e-stop is never gated. Attached, it
lives in ``dash.real`` beside the hardware robots passed with ``robots=``, so the run, stop, e-stop and chat routes
treat both alike. A connected policy is a spec string ``Policy(spec)`` builds, probed once and listed with the rest.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from .connect_policy import KINDS, policy_spec, probe

if TYPE_CHECKING:
    from .dashboard import Dashboard

FRAME_S = 0.12  # one observe serves every camera <img> of a robot for this long


def mesh_url(address: str) -> str:
    """``lan`` → zero-config scouting; ``host:port`` / ``tcp/host:port`` → that peer; a ``zenoh://`` url as is."""
    a = address.strip()
    if a in ("lan", "zenoh://"):
        return "zenoh://"
    if a.startswith(("zenoh://", "mem://")):
        return a
    link, _, hostport = a.rpartition("/")
    host, _, port = hostport.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"address {address!r}: expected 'lan', 'host:port', '<link>/host:port' or a zenoh:// url")
    return f"zenoh://?connect={link or 'tcp'}/{hostport}"


class Connections:
    """The page's meshes, attached remote robots and connected policies."""

    def __init__(self, dash: Dashboard) -> None:
        self.dash = dash
        self.meshes: dict[str, Any] = {}  # url -> Mesh
        self._own: set[str] = set()  # the urls this page opened (closed with it)
        if dash.mesh is not None:
            self.meshes[next((t.url for t in getattr(dash.mesh, "transports", ())), "mesh")] = dash.mesh
        self.remotes: dict[str, tuple[str, Any]] = {}  # robot id -> (mesh url, Robot)
        self.policies: dict[str, dict[str, Any]] = {}
        self._frames: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def join(self, address: str, tenant: str | None = None) -> dict[str, Any]:
        """Join the mesh at ``address`` (idempotent); its devices then show in ``peers()``."""
        from ..mesh import Mesh

        url = mesh_url(address)
        with self._lock:
            if url not in self.meshes:
                self.meshes[url] = Mesh(url, tenant=tenant or self._tenant())
                self._own.add(url)
            m = self.meshes[url]
        return {"url": url, "tenant": m.tenant}

    def _tenant(self) -> str | None:
        return next((m.tenant for m in self.meshes.values()), None)

    def peers(self) -> dict[str, Any]:
        local = set(self.dash.sim.robots) | (set(self.dash.real) - set(self.remotes))
        devices = []
        for url, m in list(self.meshes.items()):
            for d in m.discover("device(*)"):
                man = getattr(d, "manifest", None) or {}
                devices.append(
                    {
                        "id": d.id,
                        "type": d.device_type,
                        "name": (man.get("description") or {}).get("name") or man.get("name"),
                        "mesh": url,
                        "labels": getattr(d, "labels", {}),
                        "action_keys": list(man.get("action_keys") or ()),
                        "attached": d.id in self.remotes,
                        "local": d.id in local,
                    }
                )
        meshes = [{"url": u, "tenant": m.tenant} for u, m in self.meshes.items()]
        return {"meshes": meshes, "devices": devices}

    def attach(self, peer: str | None, address: str | None, timeout: float = 3.0) -> dict[str, Any]:
        """Attach the device ``peer`` (on the mesh at ``address``, joined first, if given) as a card."""
        url = self.join(address)["url"] if address else None
        meshes = [self.meshes[url]] if url else list(self.meshes.values())
        if not meshes:
            raise ValueError('no mesh joined yet: POST /api/connect/robot {"address": "host:port"} (or "lan")')
        if peer is None:
            found = [d for d in meshes[0].discover(min_count=1, timeout=timeout) if d.id not in self.remotes]
            if len(found) != 1:
                raise ValueError(f"name the device to attach (peer=); unattached devices: {[d.id for d in found]}")
            peer = found[0].id
        if peer in self.remotes:
            return self.card(peer)
        if peer in self.dash.sim.robots or peer in self.dash.real:
            raise ValueError(f"{peer!r} is already a robot on this page")
        for m in meshes:
            if m.discover(f"device(id:{peer})", min_count=1, timeout=timeout):
                robot = m.connect(peer, images=True)
                with self._lock:
                    self.remotes[peer] = (next(u for u, x in self.meshes.items() if x is m), robot)
                    self.dash.real[peer] = robot
                return self.card(peer)
        raise KeyError(f"no device {peer!r} on {[u for u, x in self.meshes.items() if x in meshes]}")

    def detach(self, rid: str) -> dict[str, Any]:
        with self._lock:
            if rid not in self.remotes:
                raise KeyError(f"{rid!r} is not an attached remote robot")
            _url, robot = self.remotes.pop(rid)
            self.dash.real.pop(rid, None)
            self._frames.pop(rid, None)
        robot.close()
        return {"id": rid, "attached": False}

    def card(self, rid: str) -> dict[str, Any]:
        """A non-sim robot's card: its joints, latch, task and camera names (remote or hardware)."""
        robot = self.dash.real[rid]
        obs = robot.observe(images=False)
        task = robot._task
        return {
            "id": rid,
            "name": robot.name,
            "kind": "remote" if rid in self.remotes else "hardware",
            "mesh": self.remotes[rid][0] if rid in self.remotes else None,
            "twin": None if robot.twin is None else robot.twin.id,
            "joints": dict(zip(obs.joints.names, obs.joints.positions.tolist(), strict=True)),
            "estop": bool(robot.estopped) or self._device_latched(rid),
            "task": None if task is None else task.status,
            "cameras": [{"name": c, "url": f"/api/robots/{rid}/cameras/{c}.png"} for c in self._cameras(rid)],
        }

    def _device_latched(self, rid: str) -> bool:
        """A remote device's own latch (its operator, or another peer, may have stopped it)."""
        if rid not in self.remotes:
            return False
        url, _robot = self.remotes[rid]
        return bool((self.meshes[url].invoke(rid, "lockout") or {}).get("engaged"))

    def robots(self) -> list[dict[str, Any]]:
        out = []
        for rid in list(self.dash.real):
            try:
                out.append(self.card(rid))
            except Exception as exc:  # noqa: BLE001 — a peer gone mid-read is a card that says so, not a 500
                out.append({"id": rid, "kind": "remote" if rid in self.remotes else "hardware", "error": str(exc)})
        return out

    def _images(self, rid: str) -> dict[str, Any]:
        """The robot's camera frames now: one observe per ``FRAME_S`` serves every camera of it."""
        robot = self.dash.real[rid]
        hit = self._frames.get(rid)
        if hit is not None and time.monotonic() - hit[0] < FRAME_S:
            return hit[1]
        images = dict(robot.observe(images=True).images)
        self._frames[rid] = (time.monotonic(), images)
        return images

    def _cameras(self, rid: str) -> list[str]:
        hit = self._frames.get(rid)
        return sorted(hit[1]) if hit is not None else sorted(self._images(rid))

    def camera(self, rid: str, cam: str) -> Any:
        """The frame camera ``cam`` of robot ``rid`` sees: a real camera on hardware, the device's own on a peer."""
        if rid not in self.dash.real:
            raise KeyError(f"no hardware or remote robot {rid!r}; sim cameras are /api/cameras/<name>.png")
        images = self._images(rid)
        if cam not in images:
            raise KeyError(f"{rid} has no camera {cam!r}; cameras: {sorted(images)}")
        return images[cam]

    def act(self, rid: str, targets: Any, seconds: Any, *, confirmed: bool) -> dict[str, Any]:
        """Reach ``targets`` over ``seconds`` (≤ 5) on a non-sim robot: asked once, X-Confirm is the operator's yes."""
        robot = self.dash.real[rid]
        if not isinstance(targets, dict) or not targets or set(targets) - set(robot.action_keys):
            raise ValueError(f'body must be {{"targets": {{joint: value}}, "seconds": s}}; joints: {robot.action_keys}')
        s = float(seconds) if isinstance(seconds, (int, float)) and 0 < seconds <= 5 else 1.0
        if not confirmed:
            raise PermissionError(f"{rid} is a real robot to this page: move it with header X-Confirm: yes")
        obs = robot.confirm_act({k: float(v) for k, v in targets.items()}, seconds=s, confirmed=True)
        return {"t": obs.t, "joints": dict(zip(obs.joints.names, obs.joints.positions.tolist(), strict=True))}

    def resume(self, rid: str, approved_by: str | None) -> str:
        """A remote robot's latch lives on its device: ask it there (its authority decides, by default this page's
        operator, who stopped it); returns the name it accepted, for the latch here."""
        if rid not in self.remotes:
            raise KeyError(f"{rid!r} is not an attached remote robot")
        mesh = self.meshes[self.remotes[rid][0]]
        by = approved_by or mesh.operator
        mesh.invoke(rid, "resume", approved_by=by)
        return by

    def add_policy(self, endpoint: Any, kind: Any, deployment: Any = None, name: Any = None) -> dict[str, Any]:
        """Probe and remember a policy endpoint; it then shows in ``/api/policies`` and runs with ``POST /run``."""
        if not isinstance(endpoint, str) or not isinstance(kind, str):
            raise ValueError('body must be {"endpoint": str, "kind": one of ' + f"{list(KINDS)}" + ', "deployment"?}')
        spec = policy_spec(endpoint, kind, deployment if isinstance(deployment, str) and deployment else None)
        info = probe(spec, kind)
        entry: dict[str, Any] = {
            "name": spec,
            "label": name or spec,
            "kind": kind,
            "ready": True,
            "robots": [],
            "info": info,
        }
        entry["hz"] = info.get("hz", 30)
        entry["description"] = info.get("summary", kind)
        self.policies[spec] = entry
        return entry

    def close(self) -> None:
        for rid in list(self.remotes):
            self.detach(rid)
        for url in list(self._own):
            self.meshes.pop(url).close()
            self._own.discard(url)
