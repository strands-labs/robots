"""Managed VERA policy-server subprocess.

Launches ``python -m vera.server.start_vera_server`` with **list args** (never a
shell string - see PR #621 feedback), health-checks the websocket before
returning, streams the server's stdout/stderr to the logger, and shuts the
process down cleanly on :meth:`stop`.

The server holds the GPU and the two-stage model; this provider talks to it over
the websocket (see :mod:`client`). Auto-launch is optional - point the provider
at an already-running server by setting ``auto_launch_server=False``.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING


def _require_vera_installed(python_executable: str) -> None:
    """Verify the git-only ``vera`` package is importable, else raise clearly.

    VERA (https://github.com/sizhe-li/VERA) is distributed only as a git
    repository and cannot ship as a ``strands-robots`` extra, because PyPI
    rejects any distribution whose metadata carries a direct VCS/URL reference.
    The server is launched as ``python -m vera.server...`` in *python_executable*,
    so that is the interpreter we probe.

    Raises:
        ImportError: with a git install instruction if ``vera`` is absent.
    """
    import subprocess

    probe = subprocess.run(  # noqa: S603 - list args, no shell
        [python_executable, "-c", "import vera"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise ImportError(
            "The 'vera' package is required to run the VERA policy server but is "
            f"not importable in {python_executable}.\n"
            "VERA is git-only (not on PyPI, and cannot be a strands-robots extra "
            "because PyPI rejects direct VCS references). Install it directly:\n"
            "  pip install 'vera @ git+https://github.com/sizhe-li/VERA.git'\n"
            "  # plus the client deps: pip install websockets msgpack 'numpy>=1.24'\n"
            "See strands_robots/policies/vera/__init__.py for the full quickstart."
        )


if TYPE_CHECKING:
    from .config import VeraConfig

logger = logging.getLogger(__name__)


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if a TCP connection to ``host:port`` succeeds (server is listening)."""
    # 0.0.0.0 is a bind address, not connectable - probe loopback instead.
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with socket.create_connection((probe_host, port), timeout=timeout):
            return True
    except OSError:
        return False


class VeraServerRunner:
    """Launch and supervise a ``vera.server.start_vera_server`` subprocess.

    Args:
        config: The :class:`~strands_robots.policies.vera.config.VeraConfig`
            driving embodiment, ports, checkpoints and planner knobs.
    """

    def __init__(self, config: VeraConfig) -> None:
        self.config = config
        self._proc: subprocess.Popen[str] | None = None
        self._log_thread: threading.Thread | None = None

    # -- command construction ----------------------------------------------

    def _build_command(self) -> list[str]:
        """Assemble the server launch argv as a list (no shell string)."""
        cfg = self.config
        python = cfg.python_executable or sys.executable
        cmd: list[str] = [
            python,
            "-m",
            "vera.server.start_vera_server",
            "--embodiment",
            str(cfg.embodiment),
            "--host",
            str(cfg.host),
            "--port",
            str(cfg.server_port),
        ]
        if cfg.vis_port:
            cmd += ["--vis-port", str(cfg.vis_port)]
        if cfg.algo_config is not None:
            cmd += ["--algo-config", str(cfg.algo_config)]
        if cfg.dynamics_run_id:
            cmd += ["--dynamics-run-id", str(cfg.dynamics_run_id)]
        if cfg.text_prompt:
            cmd += ["--text", str(cfg.text_prompt)]
        if cfg.sample_steps is not None:
            cmd += ["--sample-steps", str(cfg.sample_steps)]
        if not cfg.teacache:
            cmd += ["--no-teacache"]
        else:
            cmd += ["--teacache-thresh", str(cfg.teacache_thresh)]
        return cmd

    # -- lifecycle ----------------------------------------------------------

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Launch the server (idempotent) and block until its websocket is up."""
        cfg = self.config

        # Already serving (ours or someone else's) - reuse it.
        if _port_open(cfg.host, int(cfg.server_port or 0)):
            logger.info("VERA server already listening on %s:%s; reusing", cfg.host, cfg.server_port)
            return

        # Pre-flight: the git-only `vera` package must be importable in the
        # target interpreter before we spawn `python -m vera.server...`. Without
        # this check a missing install surfaces only as an opaque "server exited
        # early (code 1)" RuntimeError several seconds later. VERA is NOT on PyPI
        # (and cannot be a strands-robots extra: PyPI rejects direct VCS refs), so
        # it must be installed directly from git.
        _require_vera_installed(cfg.python_executable or sys.executable)

        cmd = self._build_command()
        env = {**os.environ, **cfg.server_env()}
        logger.info("launching VERA server: %s", " ".join(cmd))

        self._proc = subprocess.Popen(  # noqa: S603 - list args, no shell
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._start_log_pump()
        self._wait_until_ready()

    def _start_log_pump(self) -> None:
        """Stream server stdout/stderr to the logger on a daemon thread."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return

        def _pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                logger.info("[vera.server] %s", line.rstrip())

        self._log_thread = threading.Thread(target=_pump, name="vera-server-log", daemon=True)
        self._log_thread.start()

    def _wait_until_ready(self) -> None:
        """Poll the websocket port until ready, or raise on timeout / early exit."""
        cfg = self.config
        deadline = time.monotonic() + cfg.server_ready_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                code = self._proc.returncode
                raise RuntimeError(
                    f"VERA server exited early (code {code}) before becoming ready. "
                    f"Check the [vera.server] log lines above; common causes are "
                    f"missing checkpoints (set VERA_CKPT_ROOT / ckpt_root) or CUDA OOM."
                )
            if _port_open(cfg.host, int(cfg.server_port or 0)):
                logger.info("VERA server ready on %s:%s", cfg.host, cfg.server_port)
                return
            time.sleep(1.0)
        self.stop()
        raise TimeoutError(
            f"VERA server did not become ready on {cfg.host}:{cfg.server_port} "
            f"within {cfg.server_ready_timeout:.0f}s (WAN model load can be slow - "
            f"raise server_ready_timeout / VERA_SERVER_READY_TIMEOUT if needed)."
        )

    def stop(self) -> None:
        """Terminate the server subprocess cleanly (SIGTERM, then SIGKILL)."""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("VERA server did not stop on SIGTERM; killing")
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("VERA server unresponsive to SIGKILL")
        self._proc = None
        logger.info("VERA server stopped")


class DockerServerRunner:
    """Launch and supervise the VERA policy server as a **Docker container**.

    This eliminates the need to install VERA's heavy / conflicting GPU stack
    (torch 2.6 / CUDA 12.4 + VGGT) into the host robots venv: the server runs
    inside ``strands-vera-server`` (see ``docker/Dockerfile``) and the provider
    connects over the same websocket protocol. Checkpoints are bind-mounted
    read-only from the host ``ckpt_root`` to ``/ckpts`` in the container.

    All ``docker`` invocations use **list args** (no shell strings).

    Args:
        config: The :class:`~strands_robots.policies.vera.config.VeraConfig`
            (``docker_image``, ``docker_container_name``, ``docker_gpus``,
            ``ckpt_root``, ports, embodiment).
    """

    def __init__(self, config: VeraConfig) -> None:
        self.config = config
        self._started_container = False

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _docker() -> str:
        from shutil import which

        exe = which("docker")
        if exe is None:
            raise RuntimeError(
                "docker not found on PATH. Install Docker + the NVIDIA Container "
                "Toolkit, or use server_mode='subprocess' with a local VERA install."
            )
        return exe

    def _container_name(self) -> str:
        return self.config.docker_container_name or f"vera-server-{self.config.embodiment}"

    def _container_running(self) -> bool:
        import subprocess

        name = self._container_name()
        out = subprocess.run(  # noqa: S603 - list args, no shell
            [self._docker(), "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        return name in out.stdout.split()

    def _build_run_command(self) -> list[str]:
        """Assemble the ``docker run`` argv as a list (no shell string)."""
        cfg = self.config
        name = self._container_name()
        cmd: list[str] = [
            self._docker(),
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "--gpus",
            str(cfg.docker_gpus),
            "--ipc=host",  # PyTorch needs ample shared memory for dataloader/CUDA
            "-p",
            f"{cfg.server_port}:{cfg.server_port}",
        ]
        if cfg.vis_port:
            cmd += ["-p", f"{cfg.vis_port}:{cfg.vis_port}"]
        if cfg.ckpt_root is not None:
            cmd += ["-v", f"{cfg.ckpt_root}:/ckpts:ro"]
        # WAN base (frozen Wan2.1-T2V-1.3B) - REQUIRED for mimicgen/omni, unused for
        # pusht. Mounted read-only at /wan; the container's algo_config reads it via
        # ${oc.env:VERA_WAN_CKPT_ROOT}. Falls back to the ckpt root (harmless no-op
        # bind) so pusht works without the separate download.
        wan_root = cfg.wan_ckpt_root or cfg.ckpt_root
        if wan_root is not None:
            cmd += ["-v", f"{wan_root}:/wan:ro"]
        # Env overlay consumed by the container entrypoint.
        cmd += ["-e", f"VERA_EMBODIMENT={cfg.embodiment}"]
        cmd += ["-e", f"VERA_PORT={cfg.server_port}"]
        cmd += ["-e", f"VERA_VIS_PORT={cfg.vis_port or 0}"]
        cmd += ["-e", "VERA_HOST=0.0.0.0"]
        cmd += ["-e", "VERA_WAN_CKPT_ROOT=/wan"]
        cmd += ["-e", "VERA_MIMICGEN_CKPT_DIR=/ckpts/mimicgen-wan-1.3b"]
        # IDM checkpoint (wandb run id). When unset, the entrypoint defaults
        # mimicgen to the PROVEN run 37oa162u; pass an explicit override through.
        if cfg.dynamics_run_id:
            cmd += ["-e", f"VERA_DYNAMICS_RUN_ID={cfg.dynamics_run_id}"]
        if cfg.text_prompt:
            cmd += ["-e", f"VERA_TEXT_PROMPT={cfg.text_prompt}"]
        if cfg.sample_steps is not None:
            cmd += ["-e", f"VERA_SAMPLE_STEPS={cfg.sample_steps}"]
        if not cfg.teacache:
            cmd += ["-e", "VERA_NO_TEACACHE=1"]
        if cfg.docker_extra_args:
            cmd += list(cfg.docker_extra_args)
        cmd += [cfg.docker_image]
        return cmd

    # -- lifecycle ----------------------------------------------------------

    def is_running(self) -> bool:
        return self._container_running()

    def start(self) -> None:
        """Start the container (idempotent) and block until its websocket is up."""
        import subprocess

        cfg = self.config

        if _port_open(cfg.host, int(cfg.server_port or 0)):
            logger.info("VERA server already listening on %s:%s; reusing", cfg.host, cfg.server_port)
            return
        if self._container_running():
            logger.info("VERA container %s already running; waiting for readiness", self._container_name())
        else:
            cmd = self._build_run_command()
            logger.info("starting VERA container: %s", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603 - list args
            if res.returncode != 0:
                raise RuntimeError(f"failed to start VERA container (exit {res.returncode}):\n{res.stderr.strip()}")
            self._started_container = True
            logger.info("VERA container started: %s", res.stdout.strip()[:12])

        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        """Poll the websocket port until ready, or raise on timeout / container exit."""
        cfg = self.config
        deadline = time.monotonic() + cfg.server_ready_timeout
        while time.monotonic() < deadline:
            if self._started_container and not self._container_running():
                logs = self._tail_logs()
                raise RuntimeError(
                    f"VERA container {self._container_name()} exited before becoming ready. Last logs:\n{logs}"
                )
            if _port_open(cfg.host, int(cfg.server_port or 0)):
                logger.info("VERA server ready on %s:%s", cfg.host, cfg.server_port)
                return
            time.sleep(2.0)
        self.stop()
        raise TimeoutError(
            f"VERA container did not become ready on {cfg.host}:{cfg.server_port} "
            f"within {cfg.server_ready_timeout:.0f}s (WAN model load can be slow - "
            f"raise server_ready_timeout / VERA_SERVER_READY_TIMEOUT)."
        )

    def _tail_logs(self, lines: int = 40) -> str:
        import subprocess

        try:
            out = subprocess.run(  # noqa: S603 - list args
                [self._docker(), "logs", "--tail", str(lines), self._container_name()],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return (out.stdout + out.stderr).strip()
        except Exception as e:  # noqa: BLE001
            return f"(could not read container logs: {e})"

    def stop(self) -> None:
        """Stop the container we started (leaves a pre-existing one running)."""
        import subprocess

        if not self._started_container:
            return
        name = self._container_name()
        try:
            subprocess.run(  # noqa: S603 - list args
                [self._docker(), "stop", name], capture_output=True, text=True, timeout=30
            )
            logger.info("VERA container %s stopped", name)
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to stop VERA container %s: %s", name, e)
        self._started_container = False


def make_server_runner(config: VeraConfig):
    """Return the right server runner for ``config.server_mode``.

    ``"subprocess"`` (default) launches ``python -m vera.server...`` locally;
    ``"docker"`` runs the ``strands-vera-server`` container. Both expose the
    same ``start()`` / ``stop()`` / ``is_running()`` interface.
    """
    mode = (config.server_mode or "subprocess").lower()
    if mode == "docker":
        return DockerServerRunner(config)
    if mode == "subprocess":
        return VeraServerRunner(config)
    raise ValueError(f"unknown server_mode {config.server_mode!r}; use 'subprocess' or 'docker'")
