"""Lifecycle tests for the VERA server runners (offline — no server, GPU, docker).

Covers the supervision logic that command-construction tests leave untested:
the loopback probe in :func:`_port_open`, the subprocess runner's start /
ready-wait / stop state machine (reuse, early-exit, timeout, SIGTERM->SIGKILL
escalation, stdout pump), and the Docker runner's container detection, launch
failure surfacing, readiness wait, log tail and conditional stop.

Everything is mocked at the ``subprocess`` / ``socket`` boundary so the tests
exercise the real orchestration code paths without launching anything.
"""

from __future__ import annotations

import subprocess

import pytest

from strands_robots.policies.vera import VeraConfig
from strands_robots.policies.vera import server_runner as sr


# --------------------------------------------------------------------------- #
# _port_open — the TCP readiness probe
# --------------------------------------------------------------------------- #
class TestPortOpen:
    def test_returns_true_when_connection_succeeds(self, monkeypatch):
        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(sr.socket, "create_connection", lambda *a, **k: _Conn())
        assert sr._port_open("127.0.0.1", 8820) is True

    def test_returns_false_when_connection_refused(self, monkeypatch):
        def _refuse(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(sr.socket, "create_connection", _refuse)
        assert sr._port_open("127.0.0.1", 8820) is False

    def test_probes_loopback_for_bind_all_address(self, monkeypatch):
        seen: dict[str, object] = {}

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _capture(address, timeout=None):
            seen["address"] = address
            return _Conn()

        monkeypatch.setattr(sr.socket, "create_connection", _capture)
        # 0.0.0.0 is a bind address, not connectable — must probe loopback.
        assert sr._port_open("0.0.0.0", 8820) is True
        assert seen["address"] == ("127.0.0.1", 8820)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeProc:
    """Scriptable stand-in for ``subprocess.Popen``."""

    def __init__(self, poll_values=None, stdout=None, wait_raises=0):
        # poll() yields each value in turn, then sticks on the last.
        self._poll_values = list(poll_values if poll_values is not None else [None])
        self.stdout = stdout
        self.returncode = self._poll_values[-1]
        self.terminated = False
        self.killed = False
        self._wait_raises = wait_raises  # how many leading wait() calls raise TimeoutExpired

    def poll(self):
        val = self._poll_values[0]
        if len(self._poll_values) > 1:
            self._poll_values.pop(0)
        return val

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self._wait_raises > 0:
            self._wait_raises -= 1
            raise subprocess.TimeoutExpired(cmd="vera", timeout=timeout or 0)
        return 0


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# --------------------------------------------------------------------------- #
# _require_vera_installed — git-only install gate (PR #950 regression)
# --------------------------------------------------------------------------- #
class TestRequireVeraInstalled:
    """VERA is git-only (not a PyPI extra); the runner must gate on it clearly.

    Pins the fix for the v0.4.1 PyPI-upload failure: the `vera` extra carried a
    `git+https` direct reference that PyPI rejects. With the extra removed, a
    missing `vera` must raise an actionable ImportError *before* we spawn the
    server (previously it surfaced as an opaque "exited early (code 1)").
    """

    def test_raises_importerror_with_git_hint_when_vera_absent(self, monkeypatch):
        # Probe subprocess reports non-zero -> `import vera` failed.
        monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: _FakeCompleted(returncode=1))
        with pytest.raises(ImportError, match="git\\+https://github.com/sizhe-li/VERA"):
            sr._require_vera_installed("python3")

    def test_passes_silently_when_vera_importable(self, monkeypatch):
        # Probe subprocess reports success -> `import vera` worked.
        monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: _FakeCompleted(returncode=0))
        sr._require_vera_installed("python3")  # must not raise

    def test_probes_the_target_interpreter(self, monkeypatch):
        seen: dict[str, object] = {}

        def _capture(cmd, **kwargs):
            seen["cmd"] = cmd
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(sr.subprocess, "run", _capture)
        sr._require_vera_installed("/custom/python")
        assert seen["cmd"] == ["/custom/python", "-c", "import vera"]


# --------------------------------------------------------------------------- #
# VeraServerRunner lifecycle
# --------------------------------------------------------------------------- #
class TestVeraServerRunnerStart:
    def test_reuses_already_listening_server(self, monkeypatch):
        monkeypatch.setattr(sr, "_port_open", lambda *a, **k: True)

        def _no_launch(*a, **k):
            raise AssertionError("Popen must not be called when a server is already up")

        monkeypatch.setattr(sr.subprocess, "Popen", _no_launch)
        runner = sr.VeraServerRunner(VeraConfig(embodiment="pusht"))
        runner.start()
        # Reuse path: nothing launched, so nothing for us to supervise.
        assert runner.is_running() is False

    def test_launches_and_returns_once_ready(self, monkeypatch):
        # Bypass the git-only vera install gate; this test covers supervision.
        monkeypatch.setattr(sr, "_require_vera_installed", lambda *a, **k: None)
        # First probe (reuse check) -> down; second probe (ready poll) -> up.
        probes = iter([False, True])
        monkeypatch.setattr(sr, "_port_open", lambda *a, **k: next(probes))
        captured: dict[str, object] = {}

        def _fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return _FakeProc(poll_values=[None], stdout=None)

        monkeypatch.setattr(sr.subprocess, "Popen", _fake_popen)
        runner = sr.VeraServerRunner(VeraConfig(embodiment="pusht"))
        runner.start()
        assert runner.is_running() is True
        assert isinstance(captured["cmd"], list)
        assert "vera.server.start_vera_server" in captured["cmd"]

    def test_raises_when_server_exits_before_ready(self, monkeypatch):
        monkeypatch.setattr(sr, "_require_vera_installed", lambda *a, **k: None)
        monkeypatch.setattr(sr, "_port_open", lambda *a, **k: False)
        # poll() reports an exit code -> process died during startup.
        monkeypatch.setattr(sr.subprocess, "Popen", lambda *a, **k: _FakeProc(poll_values=[1], stdout=None))
        runner = sr.VeraServerRunner(VeraConfig(embodiment="pusht"))
        with pytest.raises(RuntimeError, match="exited early"):
            runner.start()

    def test_times_out_when_never_ready(self, monkeypatch):
        monkeypatch.setattr(sr, "_require_vera_installed", lambda *a, **k: None)
        monkeypatch.setattr(sr, "_port_open", lambda *a, **k: False)
        proc = _FakeProc(poll_values=[None], stdout=None)
        monkeypatch.setattr(sr.subprocess, "Popen", lambda *a, **k: proc)
        # Zero readiness budget -> the wait loop never iterates and times out.
        cfg = VeraConfig(embodiment="pusht", server_ready_timeout=0.0)
        runner = sr.VeraServerRunner(cfg)
        with pytest.raises(TimeoutError, match="did not become ready"):
            runner.start()
        # Timeout must tear the process down.
        assert proc.terminated is True


class TestVeraServerRunnerStop:
    def test_stop_is_noop_without_process(self):
        runner = sr.VeraServerRunner(VeraConfig(embodiment="pusht"))
        runner.stop()  # no proc -> must not raise
        assert runner.is_running() is False

    def test_stop_terminates_running_process(self):
        runner = sr.VeraServerRunner(VeraConfig(embodiment="pusht"))
        proc = _FakeProc(poll_values=[None], stdout=None)
        runner._proc = proc
        runner.stop()
        assert proc.terminated is True
        assert runner.is_running() is False

    def test_stop_escalates_to_kill_on_sigterm_timeout(self):
        runner = sr.VeraServerRunner(VeraConfig(embodiment="pusht"))
        # First wait() (after terminate) times out -> escalate to kill().
        proc = _FakeProc(poll_values=[None], stdout=None, wait_raises=1)
        runner._proc = proc
        runner.stop()
        assert proc.terminated is True
        assert proc.killed is True

    def test_log_pump_streams_stdout_to_logger(self, caplog):
        runner = sr.VeraServerRunner(VeraConfig(embodiment="pusht"))
        runner._proc = _FakeProc(poll_values=[None], stdout=["loading model\n", "ready\n"])
        with caplog.at_level("INFO", logger=sr.logger.name):
            runner._start_log_pump()
            runner._log_thread.join(timeout=5)
        text = "\n".join(caplog.messages)
        assert "loading model" in text
        assert "ready" in text


# --------------------------------------------------------------------------- #
# DockerServerRunner lifecycle
# --------------------------------------------------------------------------- #
class TestDockerServerRunner:
    def _runner(self, **overrides):
        cfg = VeraConfig(embodiment="pusht", server_mode="docker", **overrides)
        return sr.DockerServerRunner(cfg)

    def test_docker_missing_raises_actionable_error(self, monkeypatch):
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(RuntimeError, match="docker not found"):
            self._runner()._docker()

    def test_container_running_reflects_docker_ps(self, monkeypatch):
        runner = self._runner()
        monkeypatch.setattr(runner, "_docker", lambda: "/usr/bin/docker")
        name = runner._container_name()
        monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: _FakeCompleted(stdout=f"{name}\n"))
        assert runner.is_running() is True
        monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: _FakeCompleted(stdout="other\n"))
        assert runner.is_running() is False

    def test_start_reuses_listening_server(self, monkeypatch):
        monkeypatch.setattr(sr, "_port_open", lambda *a, **k: True)

        def _fail(*a, **k):
            raise AssertionError("must not run docker when a server is already up")

        monkeypatch.setattr(sr.subprocess, "run", _fail)
        self._runner().start()  # returns via reuse path, no docker call

    def test_start_surfaces_launch_failure(self, monkeypatch):
        runner = self._runner(ckpt_root="/data/ckpts", docker_image="vera:latest")
        monkeypatch.setattr(sr, "_port_open", lambda *a, **k: False)
        monkeypatch.setattr(runner, "_docker", lambda: "/usr/bin/docker")
        monkeypatch.setattr(runner, "_container_running", lambda: False)
        monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: _FakeCompleted(returncode=1, stderr="no such image"))
        with pytest.raises(RuntimeError, match="failed to start VERA container"):
            runner.start()

    def test_wait_raises_when_container_exits(self, monkeypatch):
        runner = self._runner()
        runner._started_container = True
        monkeypatch.setattr(sr, "_port_open", lambda *a, **k: False)
        monkeypatch.setattr(runner, "_container_running", lambda: False)
        monkeypatch.setattr(runner, "_tail_logs", lambda lines=40: "boom: CUDA OOM")
        with pytest.raises(RuntimeError, match="exited before becoming ready"):
            runner._wait_until_ready()

    def test_tail_logs_returns_combined_output(self, monkeypatch):
        runner = self._runner()
        monkeypatch.setattr(runner, "_docker", lambda: "/usr/bin/docker")
        monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: _FakeCompleted(stdout="out\n", stderr="err\n"))
        logs = runner._tail_logs()
        assert "out" in logs and "err" in logs

    def test_stop_is_noop_for_preexisting_container(self, monkeypatch):
        runner = self._runner()
        # We never started it -> stop() must leave it alone (no docker call).
        monkeypatch.setattr(
            sr.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not stop"))
        )
        runner.stop()

    def test_stop_stops_container_we_started(self, monkeypatch):
        runner = self._runner()
        runner._started_container = True
        monkeypatch.setattr(runner, "_docker", lambda: "/usr/bin/docker")
        calls: list[list[str]] = []

        def _run(cmd, *a, **k):
            calls.append(cmd)
            return _FakeCompleted()

        monkeypatch.setattr(sr.subprocess, "run", _run)
        runner.stop()
        assert runner._started_container is False
        assert any("stop" in c for c in calls)

    def test_build_run_command_includes_all_optional_flags(self):
        runner = self._runner(
            ckpt_root="/data/ckpts",
            wan_ckpt_root="/data/wan",
            docker_image="vera:dev",
            dynamics_run_id="run42",
            text_prompt="stack the blocks",
            sample_steps=12,
            teacache=False,
            docker_extra_args=["--shm-size", "16g"],
        )
        # Pin the docker exe so the command builds without docker installed.
        runner._docker = lambda: "/usr/bin/docker"  # type: ignore[method-assign]
        cmd = runner._build_run_command()
        assert "VERA_DYNAMICS_RUN_ID=run42" in cmd
        assert "VERA_TEXT_PROMPT=stack the blocks" in cmd
        assert "VERA_SAMPLE_STEPS=12" in cmd
        assert "VERA_NO_TEACACHE=1" in cmd
        assert "--shm-size" in cmd and "16g" in cmd
        # WAN base bind-mounts the dedicated root read-only at /wan.
        assert "/data/wan:/wan:ro" in cmd

    def test_start_launches_container_and_returns_once_ready(self, monkeypatch):
        runner = self._runner(ckpt_root="/data/ckpts", docker_image="vera:latest")
        # reuse probe -> down; readiness probe -> up.
        probes = iter([False, True])
        monkeypatch.setattr(sr, "_port_open", lambda *a, **k: next(probes))
        monkeypatch.setattr(runner, "_docker", lambda: "/usr/bin/docker")
        # Down at the launch decision, up while we wait for readiness.
        states = iter([False, True, True])
        monkeypatch.setattr(runner, "_container_running", lambda: next(states))
        calls: list[list[str]] = []

        def _run(cmd, *a, **k):
            calls.append(cmd)
            return _FakeCompleted(stdout="abc123def456")

        monkeypatch.setattr(sr.subprocess, "run", _run)
        runner.start()
        assert runner._started_container is True
        assert any("run" in c for c in calls)

    def test_tail_logs_failsoft_on_error(self, monkeypatch):
        runner = self._runner()
        monkeypatch.setattr(runner, "_docker", lambda: "/usr/bin/docker")

        def _boom(*a, **k):
            raise OSError("docker daemon down")

        monkeypatch.setattr(sr.subprocess, "run", _boom)
        # _tail_logs must never raise — it is called on the error path.
        logs = runner._tail_logs()
        assert "could not read container logs" in logs

    def test_stop_failsoft_when_docker_stop_errors(self, monkeypatch):
        runner = self._runner()
        runner._started_container = True
        monkeypatch.setattr(runner, "_docker", lambda: "/usr/bin/docker")
        monkeypatch.setattr(sr.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("daemon down")))
        runner.stop()  # swallows the error, still clears the flag
        assert runner._started_container is False
