#!/usr/bin/env python3
"""Read-only fleet dashboard: a subscribe-only mesh peer rendered in Rerun.

Goal: One dashboard that serves every example in the fleet suite unchanged,
because it attaches to the MESH, not to a simulator or a backend (epic
#2179, decision D9). It joins as its own peer (``peer_type="dashboard"``),
subscribes to presence, health and safety topics, tails the signed audit
log, and renders two things: a fleet table (peer id, type, presence age,
current task, last safety event) and an event timeline (dispatch / estop /
resume / HITL decisions, as recorded in the audit trail).

READ-ONLY is enforced, not narrated: ``restrict_to_subscribe_only`` replaces
every command-capable method on the peer (``send`` / ``tell`` / ``broadcast``
/ ``emergency_stop`` / ``publish_step``) with a refusal, and confines raw
``publish`` to the peer's own ``strands/{peer_id}/...`` namespace - so a
command, an estop or a resume cannot be published from this peer by accident.
Two write paths are deliberately kept, because the peer's own mesh loops need
them: ``publish`` under that namespace (presence and health, so the dashboard
is visible as a peer) and ``publish_safety_event``, which the mesh's own
safety handlers call to record this peer's lockout transitions. So the
dashboard does append to the signed audit trail it renders - for its own
``peer_id`` only. This is a guard against misuse, not a security boundary:
the refusals are instance attributes, so code that deliberately reaches for
the class attribute is not contained by them (``restrict_to_subscribe_only``
itself does exactly that, via ``type(mesh).publish``, to keep the confined
path working). HITL approvals stay in the operator terminal; a write-capable
UI is an explicit epic non-goal.

Dependencies: pip install "strands-robots[mesh]"
              (rendering upgrades from terminal to Rerun when rerun-sdk is
              installed: pip install rerun-sdk)
Expected output: a live-updating fleet table plus an event timeline, in the
                 Rerun viewer when available and on the terminal otherwise.
                 On a headless or remote host, ``--serve-web`` serves the
                 Rerun web viewer + live log stream from this process
                 instead of spawning a native viewer window; it binds
                 127.0.0.1 by default and prints the ready-to-open URL plus
                 the SSH tunnel recipe.
Runtime: until Ctrl+C, or --duration seconds.

Note: Run any fleet example in another terminal (e.g.
      ``python examples/fleet/02_cross_zone_transport.py``) and this peer
      shows its zones, dispatches and handoffs live. Two processes need a
      discovery channel: multicast scouting is OFF by default, so set
      ``STRANDS_MESH_MULTICAST=true`` on BOTH processes (trusted networks
      only) or configure connect endpoints. Point STRANDS_MESH_AUDIT_DIR at
      the same directory the examples use so the audit tail reads the trail
      they write.

Part of the fleet suite (epic #2179, dashboard v1 per #2181).
"""

from __future__ import annotations

import os

os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")

import argparse
import ipaddress
import subprocess
import sys
import tempfile
import time
import urllib.parse
from collections import deque
from collections.abc import Callable
from typing import Any

from strands_robots.mesh import init_mesh
from strands_robots.mesh.audit import read_audit_log
from strands_robots.utils import require_optional, tcp_port_error

DASHBOARD_ID = "fleet-dashboard"

# --serve-web defaults. Loopback is the repo's network-exposure convention
# ("bind to 127.0.0.1 by default, not 0.0.0.0"); the ports match the rerun
# CLI's own defaults so the tunnel recipe below is the one users already know.
DEFAULT_BIND = "127.0.0.1"
DEFAULT_WEB_PORT = 9090
DEFAULT_GRPC_PORT = 9876

#: How long the web server child gets to bind its ports before --serve-web
#: gives up and raises (a duration, so it is measured on time.monotonic()).
SERVE_READY_TIMEOUT_S = 15.0

#: The line the rerun server prints once its gRPC listener is bound (0.26.x).
#: Readiness is read from the child's own log rather than probed with a TCP
#: connect: when the port is already taken, a connect probe reaches the
#: squatter and reports the server ready exactly when it is not - measured,
#: that false pass left the dashboard streaming into a socket that speaks no
#: gRPC instead of refusing with the child's own bind error.
_SERVE_READY_LINE = b"Listening for gRPC connections on"

#: And the line it prints instead when that listener cannot bind (0.26.x
#: keeps serving the viewer HTML and only logs this ERROR, so the process
#: staying alive proves nothing). Recognising it turns a 15s timeout into an
#: immediate refusal; if the wording ever changes, the timeout still catches
#: the failure and quotes the log.
_SERVE_FAILED_LINE = b"message proxy server crashed"

# The subscribe surface: presence and health for the fleet table, the two
# fleet-wide safety topics plus per-peer safety events for the timeline.
SUBSCRIBE_TOPICS: tuple[str, ...] = (
    "strands/*/presence",
    "strands/*/health",
    "strands/*/safety/event",
    "strands/safety/estop",
    "strands/safety/resume",
)

# Command-capable Mesh methods a read-only peer must not hold. Every one of
# these is reachable only from caller code: the mesh's own loops never call
# them (``tell`` delegates to ``send`` and ``emergency_stop`` to
# ``broadcast``, and both callers are themselves refused), so removing them
# cannot break the peer.
#
# Two write-capable methods are deliberately absent, because the peer's own
# mesh loops DO call them and a refusal would break the peer rather than
# guard it:
#   ``publish``              - confined to the peer's own namespace instead
#                              (the presence/health loops publish there).
#   ``publish_safety_event`` - the mesh's own ``_on_safety_estop`` /
#                              ``_on_safety_resume`` handlers call it to
#                              record this peer's lockout transitions, two of
#                              them outside a try, and the guarded ones catch
#                              only (TypeError, ValueError, OSError). Adding
#                              it here raises inside a Zenoh subscription
#                              callback on the safety path.
_COMMAND_METHODS: tuple[str, ...] = ("send", "tell", "broadcast", "emergency_stop", "publish_step")


def restrict_to_subscribe_only(mesh: Any) -> Any:
    """Turn a started Mesh peer into a subscribe-only surface, in place.

    Every method in :data:`_COMMAND_METHODS` is replaced with a refusal that
    raises ``RuntimeError`` naming the dashboard's contract, and raw
    ``publish`` is confined to the peer's own ``strands/{peer_id}/...``
    namespace - which keeps the presence/health liveness loops honest (the
    dashboard is visible as a peer) while making every command topic
    (another peer's ``cmd``, ``broadcast``, ``safety/estop``,
    ``safety/resume``) unreachable. Returns the same mesh object.

    ``publish_safety_event`` is exempt and stays callable: the mesh's own
    safety handlers use it to record this peer's lockout transitions, so
    refusing it would raise on the safety path rather than guard anything
    (see the note on :data:`_COMMAND_METHODS`). It writes only this peer's
    own ``strands/{peer_id}/safety/event`` topic and audit records carrying
    this peer's own ``peer_id`` - it cannot name another peer - but that does
    mean a caller holding the restricted mesh can append to the signed audit
    trail this dashboard renders. Scoping that write is a mesh-side contract
    question, not something this example decides.
    """

    def _refuse_command(name: str) -> Callable[..., Any]:
        def refused(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(f"read-only dashboard: {name} is disabled on this peer (subscribe-only surface)")

        return refused

    for name in _COMMAND_METHODS:
        setattr(mesh, name, _refuse_command(name))

    own_prefix = f"strands/{mesh.peer_id}/"
    inner_publish = type(mesh).publish

    def confined_publish(key: str, payload: dict[str, Any]) -> None:
        if not key.startswith(own_prefix):
            raise RuntimeError(
                f"read-only dashboard: publish to {key!r} refused (only {own_prefix}* is permitted on this peer)"
            )
        inner_publish(mesh, key, payload)

    mesh.publish = confined_publish
    return mesh


def build_snapshot(
    peers: list[dict[str, Any]],
    health: dict[str, dict[str, Any]],
    safety: dict[str, str],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pure snapshot builder: mesh state in, render-ready rows out.

    ``peers`` is the presence registry (``mesh.peers``), ``health`` the last
    health payload per peer, ``safety`` the last safety event type per peer
    (with the ``"fleet"`` key carrying the fleet-wide estop/resume state),
    and ``events`` the timeline. Rows are sorted by peer id so successive
    renders are stable.
    """
    rows = []
    for peer in sorted(peers, key=lambda p: str(p.get("peer_id"))):
        peer_id = str(peer.get("peer_id"))
        battery = health.get(peer_id, {}).get("battery")
        rows.append(
            {
                "peer": peer_id,
                "type": str(peer.get("type", "?")),
                "age_s": peer.get("age"),
                "task": str(peer.get("task_status") or peer.get("instruction") or "-"),
                "battery": battery if battery is not None else "-",
                "safety": safety.get(peer_id, "-"),
            }
        )
    return {"fleet_safety": safety.get("fleet", "ok"), "peers": rows, "events": list(events)}


class FleetDashboard:
    """Collects mesh + audit state and hands snapshots to a renderer.

    The mesh peer is injected already started (and already restricted -
    :func:`main` composes ``init_mesh`` + :func:`restrict_to_subscribe_only`
    + ``attach``), so everything here is observation: subscription callbacks
    fold wire events into local state, ``poll_audit`` tails the signed audit
    log, and ``tick`` renders one snapshot.
    """

    def __init__(self, mesh: Any, renderer: Any, *, tail_audit: bool = True, max_events: int = 200) -> None:
        self._mesh = mesh
        self._renderer = renderer
        self._tail_audit = tail_audit
        self._health: dict[str, dict[str, Any]] = {}
        self._safety: dict[str, str] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._audit_since = time.time()

    def attach(self) -> None:
        """Declare every read-side subscription; raise if any is refused."""
        for topic in SUBSCRIBE_TOPICS:
            if self._mesh.subscribe(topic, callback=self._on_sample, name=f"dashboard:{topic}") is None:
                raise RuntimeError(f"dashboard could not subscribe to {topic!r}; is the mesh up?")

    def _on_sample(self, key: str, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        if key.endswith("/health"):
            peer_id = str(data.get("peer_id", key.split("/")[1]))
            self._health[peer_id] = data
            return
        if key == "strands/safety/estop":
            issuer = str(data.get("peer_id", "?"))
            self._safety["fleet"] = f"LOCKOUT (estop by {issuer})"
            self._events.append({"ts": data.get("t", time.time()), "source": "wire", "event": "estop", "peer": issuer})
            return
        if key == "strands/safety/resume":
            self._safety["fleet"] = "ok (resumed)"
            self._events.append({"ts": data.get("t", time.time()), "source": "wire", "event": "resume", "peer": "-"})
            return
        if key.endswith("/safety/event"):
            peer_id = str(data.get("peer_id", key.split("/")[1]))
            self._safety[peer_id] = str(data.get("type", "?"))
            self._events.append(
                {"ts": data.get("t", time.time()), "source": "wire", "event": str(data.get("type")), "peer": peer_id}
            )
        # presence samples need no folding here: the peer registry the mesh
        # already maintains is the fleet-table source of truth.

    def poll_audit(self) -> None:
        """Fold new signed audit records into the event timeline."""
        if not self._tail_audit:
            return
        records = read_audit_log(since=self._audit_since)
        for record in records:
            ts = record.get("ts")
            if isinstance(ts, (int, float)):
                self._audit_since = max(self._audit_since, float(ts) + 1e-6)
            self._events.append(
                {
                    "ts": ts,
                    "source": "audit",
                    "event": str(record.get("event")),
                    "peer": str(record.get("peer_id")),
                }
            )

    def snapshot(self) -> dict[str, Any]:
        return build_snapshot(self._mesh.peers, self._health, self._safety, list(self._events))

    def tick(self) -> dict[str, Any]:
        self.poll_audit()
        snapshot = self.snapshot()
        self._renderer.render(snapshot)
        return snapshot


class TerminalRenderer:
    """Fallback renderer: the same snapshot, printed as plain text."""

    def __init__(self, *, max_event_lines: int = 8) -> None:
        self._max_event_lines = max_event_lines
        self._seen_events = 0

    def close(self) -> None:
        """No resources to release; mirrors :meth:`RerunRenderer.close`."""

    def render(self, snapshot: dict[str, Any]) -> None:
        print(f"\n== fleet ({time.strftime('%H:%M:%S')})  safety: {snapshot['fleet_safety']} ==")
        header = f"{'peer':<20} {'type':<10} {'age_s':>6} {'battery':>8} {'safety':<24} task"
        print(header)
        for row in snapshot["peers"]:
            age = f"{row['age_s']:.1f}" if isinstance(row["age_s"], (int, float)) else "?"
            print(
                f"{row['peer']:<20} {row['type']:<10} {age:>6} {row['battery']!s:>8} {row['safety']:<24} {row['task']}"
            )
        new_events = snapshot["events"][self._seen_events :]
        self._seen_events = len(snapshot["events"])
        for event in new_events[-self._max_event_lines :]:
            print(f"  [{event['source']}] {event['event']} ({event['peer']})")


def serve_connect_host(bind: str) -> str:
    """The address a local client (the SDK stream, the readiness probe, the
    printed URL) uses to reach a server bound on ``bind``.

    ``0.0.0.0`` accepts on every interface but is not itself a destination,
    so loopback stands in for it; any other bind address is reachable only at
    that address.
    """
    return "127.0.0.1" if bind == "0.0.0.0" else bind


def is_loopback_bind(bind: str) -> bool:
    """Whether a server bound to ``bind`` is reachable only from this host.

    Membership in the loopback class, not equality with one spelling of it:
    ``127.0.0.1`` is the default, but the whole ``127.0.0.0/8`` block is
    loopback too, and the Rerun CLI accepts it - measured on rerun-sdk 0.26.2,
    ``--bind 127.0.0.2`` binds both listeners and serves. An operator who
    isolates the viewer on its own loopback address is still on loopback, so
    the startup message owes them the tunnel recipe rather than a network-
    exposure warning.

    A hostname is deliberately not classified. ``--bind`` takes an IP literal
    only (measured: ``--bind localhost`` is refused with "invalid IP address
    syntax"), so parsing the address here means this classification and the
    server's own accepted domain cannot disagree - a name the server refuses
    never reaches the message at all.
    """
    try:
        return ipaddress.ip_address(bind).is_loopback
    except ValueError:
        # Not an IP literal, so not an address the server would bind either.
        return False


def url_host(host: str) -> str:
    """``host`` as a URL authority: an IPv6 literal is bracketed (RFC 3986).

    Unbracketed, an IPv6 address runs into the port separator and the result
    parses as neither (``http://::1:9090``), so the URL the message prints
    would not open and the ``ssh -L`` forward would not parse. The bracket
    keeps whatever :func:`is_loopback_bind` classifies spellable; no IPv6
    bind is reachable on rerun-sdk 0.26.2 (measured: the gRPC listener binds
    ``[::1]`` and the process then exits ``Could not parse address``), so this
    is what keeps the two halves consistent rather than a reachable fix.
    """
    try:
        if ipaddress.ip_address(host).version == 6:
            return f"[{host}]"
    except ValueError:
        # Not an IP literal, so not a form that collides with the port separator.
        pass
    return host


def rerun_binary() -> str:
    """Resolve the native rerun-cli binary that ships inside the rerun-sdk wheel.

    The child must BE the binary for :meth:`RerunRenderer.close` to actually
    stop the server: ``{sys.executable} -m rerun`` wraps the binary in
    ``subprocess.call`` (see ``rerun_cli.__main__``), so terminating that
    wrapper orphans the grandchild - measured on 0.26.2, the dashboard exited
    and both ports stayed bound by a leaked ``rerun_cli/rerun`` process,
    which is exactly the state that makes the next ``--serve-web`` fail.
    ``RERUN_CLI_PATH`` is honoured because it is the wrapper's own documented
    override for the same resolution.
    """
    override = os.environ.get("RERUN_CLI_PATH")
    if override:
        if not os.path.exists(override):
            raise RuntimeError(f"RERUN_CLI_PATH={override!r} does not exist")
        return override
    cli = require_optional("rerun_cli", pip_install="rerun-sdk", purpose="the fleet dashboard's --serve-web server")
    suffix = ".exe" if sys.platform.startswith("win") else ""
    path = os.path.join(os.path.dirname(cli.__file__), "rerun" + suffix)
    if not os.path.exists(path):
        raise RuntimeError(f"rerun-sdk is installed but its CLI binary is missing at {path!r}")
    return path


def serve_web_command(*, binary: str, bind: str, web_port: int, grpc_port: int) -> list[str]:
    """Pure argv builder for the Rerun web server child process.

    rerun-sdk 0.26's in-process serve API (``rr.serve_grpc`` +
    ``rr.serve_web_viewer``) exposes no bind address and binds ``0.0.0.0``
    (measured on 0.26.2: both listeners show ``0.0.0.0`` in ``ss -ltn``
    while the returned URI says ``127.0.0.1``), so the only surface that can
    honour the loopback-by-default convention is the CLI's ``--bind`` - and
    the CLI's own default is ``0.0.0.0``, so the address is always set here
    deliberately rather than inherited. ``binary`` comes from
    :func:`rerun_binary`, which keeps the server in the same environment as
    the SDK that streams to it.
    """
    return [
        binary,
        "--serve-web",
        "--bind",
        bind,
        "--port",
        str(grpc_port),
        "--web-viewer-port",
        str(web_port),
    ]


def web_viewer_lines(*, bind: str, web_port: int, grpc_port: int) -> list[str]:
    """The startup message for --serve-web: URL, tunnel recipe, exposure note.

    The URL carries the gRPC stream address in its ``?url=`` query (the
    ``rerun%2Bhttp...`` form) so the browser connects to the live stream
    without any manual viewer configuration. Loopback binding gets the SSH
    tunnel recipe - both ports must be forwarded, because the browser fetches
    the viewer from the web port and then dials the gRPC port itself. Any
    wider bind says so explicitly instead: opting into network exposure is
    the caller's deliberate act and the output should read like one.

    Which posture a bind has is :func:`is_loopback_bind`'s class membership,
    not equality with the default spelling, and the recipe forwards to the
    address the server is actually on: ``--bind 127.0.0.2`` serves on
    loopback, so a warning about who can reach it would be false and a
    forward to ``127.0.0.1`` would reach nothing.
    """
    host = url_host(serve_connect_host(bind))
    grpc_uri = f"rerun+http://{host}:{grpc_port}/proxy"
    url = f"http://{host}:{web_port}/?url={urllib.parse.quote(grpc_uri, safe='')}"
    lines = [f"Rerun web viewer: {url}"]
    if is_loopback_bind(bind):
        lines.append(f"bound to {bind} (loopback only). From a remote machine, tunnel both ports first:")
        lines.append(f"  ssh -N -L {web_port}:{host}:{web_port} -L {grpc_port}:{host}:{grpc_port} user@this-host")
        lines.append("then open the URL above in your local browser.")
    else:
        lines.append(
            f"WARNING: bound to {bind} (network exposure beyond loopback); "
            f"anyone who can reach ports {web_port} and {grpc_port} can watch this dashboard."
        )
    return lines


class RerunRenderer:
    """Rerun renderer: fleet table as a text document, events as a text log.

    Two transports, one render path. The default spawns the native viewer
    (``spawn=True``); ``serve_web=True`` instead starts a Rerun web server
    child bound to ``bind`` (loopback unless the caller opted wider) and
    streams the same log data into it over gRPC, so a browser - locally or
    through an SSH tunnel - watches the dashboard live on a headless host.
    The web viewer is a view only: nothing here widens the mesh peer's
    subscribe-only surface, which is enforced upstream by
    :func:`restrict_to_subscribe_only`.
    """

    def __init__(
        self,
        *,
        spawn: bool = True,
        serve_web: bool = False,
        bind: str = DEFAULT_BIND,
        web_port: int = DEFAULT_WEB_PORT,
        grpc_port: int = DEFAULT_GRPC_PORT,
    ) -> None:
        self._rr = require_optional(
            "rerun",
            pip_install="rerun-sdk",
            purpose="the fleet dashboard's Rerun rendering",
        )
        self._server: subprocess.Popen[bytes] | None = None
        if serve_web:
            self._server = self._start_web_server(bind=bind, web_port=web_port, grpc_port=grpc_port)
            self._rr.init("strands-fleet-dashboard", spawn=False)
            self._rr.connect_grpc(f"rerun+http://{serve_connect_host(bind)}:{grpc_port}/proxy")
            for line in web_viewer_lines(bind=bind, web_port=web_port, grpc_port=grpc_port):
                print(line)
        else:
            self._rr.init("strands-fleet-dashboard", spawn=spawn)
        self._seen_events = 0

    @staticmethod
    def _start_web_server(*, bind: str, web_port: int, grpc_port: int) -> subprocess.Popen[bytes]:
        """Start the web server child and wait until it serves (or raise).

        A port already in use makes the child exit; without this readiness
        gate the dashboard would keep running with its log stream pointed at
        nothing (or at whatever process holds the port), which is exactly the
        silent fall-through --serve-web exists to refuse. The child's output
        goes to a temp file (not a pipe: nothing drains a pipe here, and a
        full pipe buffer would block the server), and readiness is that
        file's own "listening" line - see :data:`_SERVE_READY_LINE` for why a
        TCP probe would pass on the one failure this gate is for.
        """
        cmd = serve_web_command(binary=rerun_binary(), bind=bind, web_port=web_port, grpc_port=grpc_port)
        log = tempfile.NamedTemporaryFile(
            mode="w+b", prefix="strands-dashboard-rerun-serve-", suffix=".log", delete=False
        )
        with log:
            # Popen dups the descriptor, so the parent's handle can close
            # here; the child keeps writing to the (kept) file.
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

        def _log_tail() -> str:
            with open(log.name, "rb") as handle:
                return handle.read()[-2000:].decode("utf-8", errors="replace")

        deadline = time.monotonic() + SERVE_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            # Exit status first, then the log: a ready line found after the
            # child died must not return a dead server.
            exited = proc.poll() is not None
            with open(log.name, "rb") as handle:
                output = handle.read()
            if _SERVE_FAILED_LINE in output:
                proc.terminate()
                raise RuntimeError(
                    f"the Rerun web server could not bind its gRPC listener on port {grpc_port} "
                    f"(already in use?). Its output:\n{_log_tail()}"
                )
            if exited:
                raise RuntimeError(
                    f"the Rerun web server exited (code {proc.returncode}) before serving "
                    f"(ports {web_port}/{grpc_port} already in use?). Its output:\n{_log_tail()}"
                )
            if _SERVE_READY_LINE in output:
                return proc
            time.sleep(0.1)
        proc.terminate()
        raise RuntimeError(
            f"the Rerun web server did not report its gRPC listener on port {grpc_port} "
            f"within {SERVE_READY_TIMEOUT_S:.0f}s. Its output:\n{_log_tail()}"
        )

    def close(self) -> None:
        """Flush and detach the log stream, then stop the web server child.

        The stream detaches first (``rr.disconnect`` flushes and closes the
        gRPC connection): killing the server while the sink is attached makes
        the SDK's shutdown flush fail against a vanished peer, and the last
        rendered snapshot goes with it. The child then gets a terminate: the
        server's whole job is to outlive renders, not the dashboard, and an
        orphaned child would keep both ports bound, so the next run's
        --serve-web fails on addresses this one leaked.
        """
        if self._server is not None:
            self._rr.disconnect()
            if self._server.poll() is None:
                self._server.terminate()
                try:
                    self._server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._server.kill()
                    self._server.wait(timeout=5)
        self._server = None

    def render(self, snapshot: dict[str, Any]) -> None:
        lines = [f"fleet safety: {snapshot['fleet_safety']}", ""]
        lines.append("| peer | type | age_s | battery | safety | task |")
        lines.append("|---|---|---|---|---|---|")
        for row in snapshot["peers"]:
            age = f"{row['age_s']:.1f}" if isinstance(row["age_s"], (int, float)) else "?"
            lines.append(
                f"| {row['peer']} | {row['type']} | {age} | {row['battery']} | {row['safety']} | {row['task']} |"
            )
        self._rr.log(
            "fleet/table",
            self._rr.TextDocument("\n".join(lines), media_type=self._rr.MediaType.MARKDOWN),
        )
        new_events = snapshot["events"][self._seen_events :]
        self._seen_events = len(snapshot["events"])
        for event in new_events:
            self._rr.log("fleet/events", self._rr.TextLog(f"[{event['source']}] {event['event']} ({event['peer']})"))


def make_renderer(
    *,
    prefer_rerun: bool = True,
    spawn: bool = True,
    serve_web: bool = False,
    bind: str = DEFAULT_BIND,
    web_port: int = DEFAULT_WEB_PORT,
    grpc_port: int = DEFAULT_GRPC_PORT,
) -> Any:
    """Rerun when available, terminal otherwise - degradation is loud.

    ``serve_web`` is an explicit ask for the web viewer, so it does NOT
    degrade: with rerun-sdk absent the ``ImportError`` (carrying the
    ``pip install rerun-sdk`` hint) propagates instead of silently rendering
    tables to a terminal nobody is watching on a headless host.
    """
    if serve_web:
        return RerunRenderer(serve_web=True, bind=bind, web_port=web_port, grpc_port=grpc_port)
    if prefer_rerun:
        try:
            return RerunRenderer(spawn=spawn)
        except ImportError as exc:
            print(f"rerun unavailable ({exc}); rendering to the terminal instead")
    return TerminalRenderer()


class _DashboardOwner:
    """Minimal mesh owner for the dashboard peer (it holds no robot)."""

    tool_name_str = DASHBOARD_ID


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to run (0 = until Ctrl+C)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between renders")
    parser.add_argument("--no-rerun", action="store_true", help="force the terminal renderer")
    parser.add_argument("--no-spawn", action="store_true", help="do not spawn the Rerun viewer process")
    parser.add_argument("--no-audit", action="store_true", help="do not tail the signed audit log")
    parser.add_argument(
        "--serve-web",
        action="store_true",
        help="serve the Rerun web viewer + live log stream from this process "
        "instead of spawning a native viewer (for headless/remote hosts)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=None,
        help=f"HTTP port for the web viewer (default {DEFAULT_WEB_PORT}; requires --serve-web)",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=None,
        help=f"gRPC port for the live log stream (default {DEFAULT_GRPC_PORT}; requires --serve-web)",
    )
    parser.add_argument(
        "--bind",
        default=None,
        help=f"bind address for --serve-web (default {DEFAULT_BIND}, loopback only; "
        "pass 0.0.0.0 to deliberately expose the viewer on every interface)",
    )
    args = parser.parse_args(argv)
    if args.interval <= 0:
        raise SystemExit("--interval must be a positive number of seconds")
    if args.serve_web and args.no_rerun:
        raise SystemExit("--serve-web and --no-rerun are mutually exclusive: the web viewer is Rerun rendering")
    if not args.serve_web and (args.bind is not None or args.web_port is not None or args.grpc_port is not None):
        # Silently ignoring these would be a dropped ask: they configure the
        # web server, and without --serve-web no web server exists.
        raise SystemExit("--bind/--web-port/--grpc-port configure the web viewer; they require --serve-web")
    bind = args.bind if args.bind is not None else DEFAULT_BIND
    web_port = args.web_port if args.web_port is not None else DEFAULT_WEB_PORT
    grpc_port = args.grpc_port if args.grpc_port is not None else DEFAULT_GRPC_PORT
    for value, name in ((web_port, "--web-port"), (grpc_port, "--grpc-port")):
        if (port_error := tcp_port_error(value, name, parser.prog)) is not None:
            raise SystemExit(port_error)
    if web_port == grpc_port:
        raise SystemExit(
            f"--web-port and --grpc-port must differ (both are {web_port}); "
            "the HTTP and gRPC servers each bind their own port"
        )
    if args.serve_web and not bind.strip():
        raise SystemExit("--bind must be a non-empty address")

    # The renderer comes first: if --serve-web cannot start (rerun-sdk absent,
    # port in use), the refusal lands before a mesh peer ever announces itself.
    renderer = make_renderer(
        prefer_rerun=not args.no_rerun,
        spawn=not args.no_spawn,
        serve_web=args.serve_web,
        bind=bind,
        web_port=web_port,
        grpc_port=grpc_port,
    )
    try:
        mesh = init_mesh(_DashboardOwner(), peer_id=DASHBOARD_ID, peer_type="dashboard")
        if mesh is None:
            raise SystemExit("mesh is disabled (STRANDS_MESH=0); the dashboard has nothing to attach to")
        if not mesh.alive:
            raise SystemExit('mesh did not start (is eclipse-zenoh installed? pip install "strands-robots[mesh]")')
        restrict_to_subscribe_only(mesh)

        dashboard = FleetDashboard(mesh, renderer, tail_audit=not args.no_audit)
        dashboard.attach()
        print(f"dashboard peer {DASHBOARD_ID} attached (subscribe-only); Ctrl+C to exit")

        deadline = time.monotonic() + args.duration if args.duration > 0 else None
        try:
            while deadline is None or time.monotonic() < deadline:
                dashboard.tick()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            # Ctrl+C is the documented way to stop the dashboard; announce the
            # shutdown so the interrupt is visibly handled, then fall through to
            # the mesh.stop() cleanup below.
            print("dashboard interrupted; detaching from the mesh")
        finally:
            mesh.stop()
    finally:
        renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
