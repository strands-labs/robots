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
     success sends ``restart`` on to a bind that cannot succeed, and the
     resulting error names the wrong thing.

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

    def run(self, cmd: Any, *_args: Any, **kwargs: Any) -> _Result:
        argv = [str(part) for part in cmd]
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
