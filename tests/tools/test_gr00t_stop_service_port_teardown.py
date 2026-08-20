"""Pin: gr00t_inference port teardown signals every pid and reports the truth.

``_stop_service`` is the teardown half of the service lifecycle: the ``stop``
action calls it directly and ``restart`` calls it to free the port before
rebinding. Two properties have to hold, and they pull in opposite directions:

  1. **Per-pid teardown is best effort.** The pid list comes from a scan, so a
     pid can exit in the window between being listed and being signalled. That
     makes ``kill`` exit non-zero ("No such process") -- which is the outcome
     teardown wanted. It must not stop the remaining pids from being signalled,
     and it must not skip the SIGKILL escalation.

  2. **The report must be true.** ``"status": "success"`` tells the caller the
     port is free. When something still holds the port after SIGTERM *and*
     SIGKILL -- a pid owned by another user, a failing ``docker exec`` -- saying
     success sends ``restart`` on to a bind that cannot succeed.

  3. **``restart`` has to consult that verdict.** The rebind cannot detect the
     collision on its own: the launch is a detached ``docker exec -d`` and the
     readiness wait is a bare TCP connect to the port, which the surviving
     server answers. So a discarded error does not merely make the follow-on
     failure name the wrong thing -- there is no follow-on failure at all. The
     restart reports ``success`` naming the new checkpoint while the old one
     keeps serving.

Both branches (``docker exec`` into a GR00T container, and the host ``lsof``
fallback) implement the same escalation, so every property is pinned on both.
"""

from __future__ import annotations

import importlib
import subprocess
from typing import Any
from unittest.mock import patch

import pytest

# ``from strands_robots.tools import gr00t_inference`` resolves via the package's
# lazy __getattr__ to the *tool function*, not the module -- import_module
# returns the module object so ``gi.subprocess`` is patchable. This mirrors the
# idiom in test_gr00t_docker_lifecycle_failsoft.py.
gi = importlib.import_module("strands_robots.tools.gr00t_inference")

PORT = 5555
CONTAINER = "gr00t-jetson"


class _Result:
    """Stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode: int, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


class FakeHost:
    """A process table that answers ``pgrep``/``lsof``/``kill`` consistently.

    Args:
        in_container: When True the service pids live inside a GR00T container
            (so ``docker ps`` lists one and ``pgrep`` answers); when False they
            live on the host and only ``lsof`` answers.
        live: Pids that are running and respond to SIGKILL.
        stale: Pids the *first* scan lists and that exit immediately after --
            the race this module exists to survive.
        unkillable: Pids that stay alive no matter what signal arrives, i.e.
            ``kill`` keeps failing (another user's process, a broken
            ``docker exec``).
    """

    def __init__(
        self,
        *,
        in_container: bool,
        live: tuple[str, ...] = (),
        stale: tuple[str, ...] = (),
        unkillable: tuple[str, ...] = (),
    ) -> None:
        self.in_container = in_container
        self.alive = set(live) | set(stale) | set(unkillable)
        self._stale = set(stale)
        self._unkillable = set(unkillable)
        self._scans = 0
        self.signals: list[str] = []
        # Every ``docker exec -d`` argv the tool ran, i.e. each inference server
        # it launched. A restart that was refused must not have launched one.
        self.launched: list[list[str]] = []

    def run(self, cmd: Any, *_args: Any, **kwargs: Any) -> _Result:
        argv = [str(part) for part in cmd]
        if argv[:3] == ["docker", "exec", "-d"]:
            # A detached exec: the daemon accepts it and returns 0 whether or not
            # the server inside goes on to bind the port.
            self.launched.append(argv)
            return _Result(0, "")
        if argv[:2] == ["docker", "ps"]:
            if not self.in_container:
                return _Result(0, "")
            return _Result(0, f"{CONTAINER}\tnvcr.io/nvidia/isaac-gr00t:latest\tUp 3 hours\t{PORT}->{PORT}")
        if "pgrep" in argv or argv[0] == "lsof":
            asks_container = "pgrep" in argv
            if asks_container is not self.in_container:
                return _Result(1, "")
            self._scans += 1
            listed = sorted(self.alive, key=int)
            if self._scans == 1:
                # The scan sees the stale pids; they exit before anyone signals.
                self.alive -= self._stale
            return _Result(0, "\n".join(listed)) if listed else _Result(1, "")
        if "kill" in argv:
            signal, pid = argv[-2], argv[-1]
            self.signals.append(f"{signal} {pid}")
            if pid not in self.alive or pid in self._unkillable:
                if kwargs.get("check"):
                    raise subprocess.CalledProcessError(1, argv, stderr=f"kill: ({pid}): No such process")
                return _Result(1)
            if signal == "-KILL":
                self.alive.discard(pid)
            return _Result(0)
        return _Result(0, "")


def _stop(host: FakeHost) -> dict[str, Any]:
    """Drive ``_stop_service`` against ``host`` with the teardown pause elided."""
    with (
        patch.object(gi.subprocess, "run", side_effect=host.run),
        patch.object(gi.time, "sleep"),
    ):
        return gi._stop_service(PORT)


def _signalled(host: FakeHost, pid: str) -> bool:
    return any(entry.endswith(f" {pid}") for entry in host.signals)


# --- a pid that exited between the scan and the signal is not a failure ------


@pytest.mark.parametrize("in_container", [True, False], ids=["container", "host"])
def test_a_stale_pid_does_not_stop_the_remaining_pids_from_being_signalled(in_container: bool) -> None:
    """Pid 111 exits before it is signalled; pid 222 must still get SIGTERM."""
    host = FakeHost(in_container=in_container, live=("222",), stale=("111",))
    _stop(host)
    assert _signalled(host, "222"), (
        f"pid 222 was never signalled: {host.signals}. Pid 111 had already exited, so 'kill' exited "
        "non-zero and aborted the sweep before reaching the pid that is actually holding the port."
    )


@pytest.mark.parametrize("in_container", [True, False], ids=["container", "host"])
def test_the_sigkill_escalation_still_runs_after_a_stale_pid(in_container: bool) -> None:
    """A pid that ignores SIGTERM is still escalated to SIGKILL."""
    host = FakeHost(in_container=in_container, live=("222",), stale=("111",))
    _stop(host)
    assert "-KILL 222" in host.signals, (
        f"the SIGKILL escalation never ran: {host.signals}. A pid that survives SIGTERM is exactly "
        "what the escalation exists for."
    )


@pytest.mark.parametrize("in_container", [True, False], ids=["container", "host"])
def test_a_stale_pid_alone_still_reports_the_port_stopped(in_container: bool) -> None:
    """Every pid gone (one exited on its own) is a successful teardown."""
    host = FakeHost(in_container=in_container, live=("222",), stale=("111",))
    result = _stop(host)
    assert result["status"] == "success", result
    assert host.alive == set(), f"pid(s) {sorted(host.alive)} survived a teardown reported as success"


# --- a port still held after the escalation is not a success -----------------


@pytest.mark.parametrize("in_container", [True, False], ids=["container", "host"])
def test_a_port_still_held_after_sigkill_is_not_reported_as_stopped(in_container: bool) -> None:
    """An unkillable owner must not be reported as a stopped service.

    ``success`` here tells ``restart`` the port is free, and the bind that
    follows then fails while naming the bind rather than the surviving process.
    """
    host = FakeHost(in_container=in_container, unkillable=("222",))
    result = _stop(host)
    assert result["status"] == "error", (
        f"reported {result} while pid 222 still holds port {PORT}. A caller cannot distinguish this "
        "from a port that was actually freed."
    )


@pytest.mark.parametrize("in_container", [True, False], ids=["container", "host"])
def test_a_port_still_held_names_the_port_and_the_surviving_pid(in_container: bool) -> None:
    """The refusal has to identify what to go look at."""
    host = FakeHost(in_container=in_container, unkillable=("222",))
    message = _stop(host).get("message", "")
    assert str(PORT) in message, f"message does not name the port: {message!r}"
    assert "222" in message, f"message does not name the surviving pid: {message!r}"


# --- controls: the paths that already worked must keep working ---------------


@pytest.mark.parametrize("in_container", [True, False], ids=["container", "host"])
def test_a_single_live_pid_is_stopped_and_reported_success(in_container: bool) -> None:
    """The ordinary teardown path is unchanged."""
    host = FakeHost(in_container=in_container, live=("222",))
    result = _stop(host)
    assert result["status"] == "success", result
    assert result["port"] == PORT
    assert host.alive == set()


def test_a_free_port_reports_success_without_signalling_anything() -> None:
    """Nothing to stop is a success, and must not signal a pid."""
    host = FakeHost(in_container=False)
    result = _stop(host)
    assert result["status"] == "success", result
    assert f"No service running on port {PORT}" in result["message"]
    assert host.signals == []


def test_an_unexpected_error_is_still_converted_to_an_error_tool_result() -> None:
    """The fail-soft dispatch contract still holds (see the failsoft module)."""
    with patch.object(gi, "_find_gr00t_containers", side_effect=RuntimeError("daemon gone")):
        result = gi._stop_service(PORT)
    assert result["status"] == "error"
    assert "Failed to stop service" in result["message"]


# --- restart consults the teardown verdict before it rebinds the port --------
#
# ``restart`` is the only caller that performs the bind ``_stop_service``'s
# refusal is written for ("The port is still held, so a restart will fail to
# bind it"). The ``stop`` action returns that verdict; ``restart`` used to
# discard it.

NEW_CHECKPOINT = "/data/checkpoints/policy-v2"


def _restart(host: FakeHost, *, checkpoint_path: str = NEW_CHECKPOINT) -> dict[str, Any]:
    """Drive the real ``restart`` dispatch against ``host``.

    ``_is_service_running`` is forced True to model the readiness probe the tool
    actually performs: a TCP connect to the port, which a server that survived
    the teardown answers just as readily as one the restart launched. That is
    exactly why the teardown verdict has to be consulted instead of re-derived
    from the port.
    """
    with (
        patch.object(gi.subprocess, "run", side_effect=host.run),
        patch.object(gi.time, "sleep"),
        patch.object(gi, "_is_service_running", return_value=True),
    ):
        return gi.gr00t_inference(action="restart", checkpoint_path=checkpoint_path, port=PORT)


def _launched_model_paths(host: FakeHost) -> list[str]:
    """``--model-path`` of every inference server the tool launched."""
    return [argv[argv.index("--model-path") + 1] for argv in host.launched if "--model-path" in argv]


def test_a_restart_that_cannot_free_the_port_is_refused() -> None:
    """An unkillable owner must not be reported as a completed checkpoint swap."""
    host = FakeHost(in_container=True, unkillable=("999",))
    result = _restart(host)
    assert result["status"] == "error", (
        f"reported {result.get('status')!r} ({result.get('message')!r}) while pid 999 still holds "
        f"port {PORT}, so the new checkpoint cannot have bound it. The response is indistinguishable "
        "from a restart that really did swap the checkpoint."
    )


def test_the_refusal_names_the_port_and_the_checkpoint_that_was_not_started() -> None:
    """The caller has to learn which request was refused, not just that one was."""
    host = FakeHost(in_container=True, unkillable=("999",))
    message = _restart(host).get("message", "")
    assert str(PORT) in message, f"message does not name the port: {message!r}"
    assert NEW_CHECKPOINT in message, f"message does not name the checkpoint that was not started: {message!r}"


def test_the_refusal_carries_the_teardown_reason() -> None:
    """The teardown already says what to go look at; the refusal must not drop it."""
    host = FakeHost(in_container=True, unkillable=("999",))
    message = _restart(host).get("message", "")
    assert "999" in message, f"message does not name the surviving pid the teardown found: {message!r}"
    assert "SIGKILL" in message, f"message does not carry the teardown's own reason: {message!r}"


def test_no_inference_server_is_launched_when_the_port_could_not_be_freed() -> None:
    """A refused restart must not leave a second server racing for the port."""
    host = FakeHost(in_container=True, unkillable=("999",))
    _restart(host)
    assert _launched_model_paths(host) == [], (
        f"launched {_launched_model_paths(host)} into a port still held by pid 999. The detached exec "
        "returns 0 regardless, so the failed bind is never reported."
    )


def test_a_host_side_owner_that_survives_the_escalation_is_also_refused() -> None:
    """The host ``lsof`` fallback reaches the same refusal as the container branch.

    Reached with an explicit ``container_name`` so the dispatch does not stop
    earlier at "no running GR00T containers found" - the port owner here is a
    process on the host, which is the case the teardown's second refusal names.
    """
    host = FakeHost(in_container=False, unkillable=("4242",))
    with (
        patch.object(gi.subprocess, "run", side_effect=host.run),
        patch.object(gi.time, "sleep"),
        patch.object(gi, "_is_service_running", return_value=True),
    ):
        result = gi.gr00t_inference(
            action="restart", checkpoint_path=NEW_CHECKPOINT, port=PORT, container_name=CONTAINER
        )
    assert result["status"] == "error", f"reported {result} while pid 4242 still holds port {PORT}"
    assert "4242" in result.get("message", ""), result


def test_every_stop_service_call_site_consults_the_verdict() -> None:
    """No caller may discard ``_stop_service``'s result.

    The root-cause pin. ``_stop_service`` reports an error precisely when the
    port is still held, and every caller of it is about to act on that fact -
    so a bare-expression call is the defect itself rather than a style point.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(gi))
    discarded = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "_stop_service"
    ]
    assert discarded == [], (
        f"_stop_service result discarded at line(s) {discarded}. Its error means the port is still "
        "held; a caller that drops it acts on a precondition it never read."
    )


# --- controls: the paths that already worked must keep working ---------------


def test_a_restart_that_frees_the_port_starts_the_new_checkpoint() -> None:
    """The ordinary restart path is unchanged, and reports the new checkpoint."""
    host = FakeHost(in_container=True, live=("222",))
    result = _restart(host)
    assert result["status"] == "success", result
    assert result["checkpoint_path"] == NEW_CHECKPOINT, result
    assert _launched_model_paths(host) == [NEW_CHECKPOINT], (
        f"launched {_launched_model_paths(host)}; the freed port must be rebound for the new checkpoint"
    )


def test_a_restart_onto_a_free_port_still_starts_the_new_checkpoint() -> None:
    """Nothing to stop is not a failed teardown, so the restart proceeds.

    ``container_name`` is explicit because with no port owner there is also no
    running GR00T container for the launch to discover - that refusal is a
    different one and would mask the property under test.
    """
    host = FakeHost(in_container=False)
    with (
        patch.object(gi.subprocess, "run", side_effect=host.run),
        patch.object(gi.time, "sleep"),
        patch.object(gi, "_is_service_running", return_value=True),
    ):
        result = gi.gr00t_inference(
            action="restart", checkpoint_path=NEW_CHECKPOINT, port=PORT, container_name=CONTAINER
        )
    assert result["status"] == "success", result
    assert _launched_model_paths(host) == [NEW_CHECKPOINT], result


def test_a_restart_without_a_checkpoint_is_refused_before_any_teardown() -> None:
    """The pre-existing argument guard still runs first: nothing is torn down."""
    host = FakeHost(in_container=True, live=("222",))
    with (
        patch.object(gi.subprocess, "run", side_effect=host.run),
        patch.object(gi.time, "sleep"),
    ):
        result = gi.gr00t_inference(action="restart", port=PORT)
    assert result["status"] == "error"
    assert "Checkpoint path required" in result["message"], result
    assert host.signals == [], f"tore down the running service for a request it then refused: {host.signals}"


def test_the_stop_action_still_returns_the_teardown_verdict_unchanged() -> None:
    """``stop`` already returned the verdict; its response must not have moved."""
    host = FakeHost(in_container=True, unkillable=("999",))
    with (
        patch.object(gi.subprocess, "run", side_effect=host.run),
        patch.object(gi.time, "sleep"),
    ):
        via_action = gi.gr00t_inference(action="stop", port=PORT)
    direct = _stop(FakeHost(in_container=True, unkillable=("999",)))
    assert via_action == direct, f"the stop action reshaped the teardown verdict: {via_action} != {direct}"
