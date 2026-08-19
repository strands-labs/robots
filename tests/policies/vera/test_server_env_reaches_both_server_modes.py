"""Every server-environment value the config declares reaches either server mode.

``VeraConfig.server_env()`` is the one place that says what the policy server
must be configured with - checkpoint roots, the IDM run id, the point-tracker
backend. Two runner classes consume that intent, and only one of them calls the
method: ``VeraServerRunner`` hands the overlay to ``Popen(env=...)``, while
``DockerServerRunner`` re-enumerates the same vocabulary by hand as ``docker
run -e`` flags. A value present in one enumeration and absent from the other is
silently dropped - ``docker run`` does not reject an environment variable nobody
passed it, so the server simply starts with a default the caller overrode.

``make_server_runner`` promises both runners drive the same server, and
``docs/policies/vera.md`` tabulates each kwarg against the server input it maps
to without qualifying either mode, so the vocabularies have to agree.

The headline check derives what must be carried from ``server_env()`` itself
rather than from a copied list, so a fifth value added to the overlay is graded
here without touching this file.

A value reaches the container by one of three routes, all of which put it in
front of the server process:

* directly, as ``-e VAR=value`` (a scalar, e.g. the tracker backend);
* translated, as ``-e VAR=<container path>`` beside ``-v <host path>:...`` (a
  host path that must be renamed to where it was mounted);
* by mount alone, where the entrypoint defaults the variable to the mount point
  (``VERA_CKPT_ROOT`` -> ``/ckpts``).

So the rule below counts a value as carried when the container command either
names the variable in an ``-e`` flag or bind-mounts the value it holds.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from strands_robots.policies.vera.config import VeraConfig
from strands_robots.policies.vera.server_runner import (
    DockerServerRunner,
    VeraServerRunner,
    make_server_runner,
)

# A config that exercises every branch of the overlay at once: two host paths,
# one wandb run id and one backend name.
FULLY_CONFIGURED: dict[str, Any] = {
    "embodiment": "mimicgen",
    "server_port": 8800,
    "ckpt_root": "/data/vera-ckpts",
    "wan_ckpt_root": "/data/wan",
    "dynamics_run_id": "37oa162u",
    "tracker_backend": "cotracker",
    "text_prompt": "stack the cube",
    "sample_steps": 10,
}

# Non-vacuity floor for the headline sweep: the overlay must really be
# populated, or "every value is carried" would be a statement about nothing.
MINIMUM_OVERLAY_VALUES = 4


def _config(mode: str, **overrides: Any) -> VeraConfig:
    return VeraConfig(server_mode=mode, **{**FULLY_CONFIGURED, **overrides})


def _docker_argv(cfg: VeraConfig) -> list[str]:
    """The real ``docker run`` argv, with only the ``docker`` lookup stubbed.

    Nothing is started. ``_docker`` shells out to ``which`` purely to find the
    binary, which a test host need not have; the command under inspection is
    composed by shipped code.
    """
    runner = make_server_runner(cfg)
    assert isinstance(runner, DockerServerRunner), f"server_mode={cfg.server_mode!r} chose {type(runner).__name__}"
    runner._docker = lambda: "/usr/bin/docker"  # type: ignore[method-assign]
    return list(runner._build_run_command())


def _env_flags(argv: list[str]) -> dict[str, str]:
    """The ``-e NAME=value`` pairs of a ``docker run`` argv, keyed by name."""
    flags: dict[str, str] = {}
    for flag, payload in zip(argv, argv[1:], strict=False):
        if flag == "-e" and "=" in payload:
            name, _, value = payload.partition("=")
            flags[name] = value
    return flags


def _mounted_host_paths(argv: list[str]) -> set[str]:
    """The host side of every ``-v <host>:<container>[:opts]`` bind mount."""
    return {payload.split(":", 1)[0] for flag, payload in zip(argv, argv[1:], strict=False) if flag == "-v"}


def _unreachable_in_docker(cfg: VeraConfig) -> dict[str, str]:
    """Overlay values the container command carries by no route at all."""
    argv = _docker_argv(cfg)
    named = _env_flags(argv)
    mounted = _mounted_host_paths(argv)
    return {name: value for name, value in cfg.server_env().items() if name not in named and value not in mounted}


class TestTheOverlayIsCarriedByBothModes:
    def test_the_overlay_is_populated(self):
        """Premise: a clean result below has to be about a real vocabulary."""
        overlay = _config("subprocess").server_env()
        assert len(overlay) >= MINIMUM_OVERLAY_VALUES, overlay

    def test_the_subprocess_mode_carries_the_overlay_wholesale(self):
        """The mode that calls ``server_env()`` is the reference to grade against.

        It carries the overlay by merging it into the child environment rather
        than by naming values one at a time, which is why it cannot omit one.
        """
        launch = inspect.getsource(VeraServerRunner.start)
        assert "cfg.server_env()" in launch, "the subprocess mode no longer reads the overlay"
        assert "env=env" in launch, "the overlay no longer reaches the child process"

    def test_the_container_command_carries_every_overlay_value(self):
        """The headline: no overlay value is dropped on the container path.

        Pre-fix this reports ``VERA_TRACKER_BACKEND``, the one value the hand-
        written ``-e`` list omitted.
        """
        missing = _unreachable_in_docker(_config("docker"))
        assert not missing, (
            "server_env() declares these for the server, and the docker run command "
            f"neither names them in an -e flag nor mounts their value: {missing}. "
            "A dropped value means the server starts with a default the caller overrode."
        )

    @pytest.mark.parametrize("backend", ["cotracker", "alltracker", "vggt"])
    def test_the_tracker_backend_reaches_the_container(self, backend):
        """A backend name is forwarded verbatim - no path translation applies."""
        cfg = _config("docker", tracker_backend=backend)
        assert cfg.server_env()["VERA_TRACKER_BACKEND"] == backend, "premise: the overlay carries it"
        assert _env_flags(_docker_argv(cfg))["VERA_TRACKER_BACKEND"] == backend

    def test_both_modes_agree_on_the_tracker_backend(self):
        """The same config configures the same tracker whichever mode runs it."""
        sub = _config("subprocess")
        dock = _config("docker")
        assert _env_flags(_docker_argv(dock))["VERA_TRACKER_BACKEND"] == sub.server_env()["VERA_TRACKER_BACKEND"]


class TestNothingElseMoved:
    """Controls: these hold before and after, and fail for the shortcut fixes."""

    def test_an_unset_tracker_backend_is_not_forwarded(self):
        """No empty override is injected when the caller expressed no preference.

        The entrypoint treats an empty value as unset, but emitting one states a
        choice the config never made.
        """
        cfg = _config("docker", tracker_backend=None)
        assert cfg.tracker_backend is None, "premise: nothing to forward"
        assert "VERA_TRACKER_BACKEND" not in " ".join(_docker_argv(cfg))

    def test_a_host_path_is_forwarded_as_its_container_path(self):
        """``wan_ckpt_root`` is mounted, so the container sees ``/wan``.

        Forwarding the host spelling verbatim - the shortcut a scalar fix invites
        - would name a directory that does not exist inside the container.
        """
        cfg = _config("docker")
        argv = _docker_argv(cfg)
        assert _env_flags(argv)["VERA_WAN_CKPT_ROOT"] == "/wan"
        assert "/data/wan:/wan:ro" in argv

    def test_the_checkpoint_root_is_carried_by_its_mount(self):
        """``ckpt_root`` needs no ``-e``: the entrypoint defaults it to the mount."""
        argv = _docker_argv(_config("docker"))
        assert "/data/vera-ckpts:/ckpts:ro" in argv
        assert "VERA_CKPT_ROOT" not in _env_flags(argv)

    def test_the_run_id_is_still_forwarded_directly(self):
        """The one scalar that was already carried keeps its route."""
        assert _env_flags(_docker_argv(_config("docker")))["VERA_DYNAMICS_RUN_ID"] == "37oa162u"

    def test_the_subprocess_launch_argv_is_unchanged(self):
        """The mode that already worked composes exactly the same command.

        ``tracker_backend`` has never been a server flag - it travels by
        environment - so it must not appear in the subprocess argv either.
        """
        cfg = _config("subprocess")
        runner = make_server_runner(cfg)
        assert isinstance(runner, VeraServerRunner)
        argv = runner._build_command()
        assert "--tracker-backend" not in argv
        assert "cotracker" not in argv

    def test_the_command_stays_list_args(self):
        """Every token is a plain string: no shell string is ever assembled."""
        argv = _docker_argv(_config("docker"))
        assert all(isinstance(token, str) for token in argv)
        assert argv[-1] == VeraConfig(server_mode="docker", embodiment="mimicgen").docker_image
