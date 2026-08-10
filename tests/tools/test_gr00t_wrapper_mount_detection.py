"""The wrapper-mount probe behind the ``deterministic=True`` idempotent skip.

:func:`~strands_robots.tools.gr00t_inference._container_has_wrapper_mount` is the
predicate that decides whether an already-running container may be handed back
for a deterministic ``start``. A container started without the wrapper mount
cannot serve one: the exec'd ``python /srv_wrap.py`` would die inside the
container long after ``start_container`` returned success, so the probe must
report the mount absent and let the caller fail fast with the actionable
"recreate with force=True" error.

The two lifecycle tests that reach that branch replace the probe with a stub
returning ``True``/``False``, so they pin what the *caller* does with each
answer and nothing about how the answer is produced. This module pins the probe
itself: how it reads ``docker inspect`` output, and that both lifecycle outcomes
still hold when the real probe - not a stub - supplies the answer.

Every claim its docstring makes is pinned here: a mount at the wrapper path is
detected, anything else is reported absent, and any docker invocation failure
(non-zero exit, missing binary, OS error) is reported absent rather than
passing silently.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

gi = importlib.import_module("strands_robots.tools.gr00t_inference")

from tests.tools.test_gr00t_deterministic import _container_kwargs  # noqa: E402

WRAPPER_PATH = gi._DETERMINISTIC_WRAPPER_CONTAINER_PATH

# Shape captured from a real ``docker inspect --format
# '{{range .Mounts}}{{.Destination}}<LF>{{end}}'`` run (Docker 29.0.0): one
# destination per line, plus a trailing empty line from the final range
# iteration. The ``<LF>`` is a real newline - the format string is a plain
# Python literal, so the escape is resolved before docker ever sees it.
MOUNTED = f"{WRAPPER_PATH}\n/data\n\n"
NOT_MOUNTED = "/data\n/checkpoints\n\n"
NO_MOUNTS = "\n"


def _inspect_returning(stdout: str, returncode: int = 0) -> Any:
    """A ``subprocess.run`` stand-in for the probe's ``docker inspect`` call."""

    def fake_run(cmd: list[str], *a: Any, **kw: Any) -> Any:
        return MagicMock(stdout=stdout, stderr="", returncode=returncode)

    return fake_run


class TestTheProbeReadsDockerInspectOutput:
    """The probe's own contract: what it makes of what docker prints."""

    def test_a_mount_at_the_wrapper_path_is_detected(self) -> None:
        with patch.object(gi.subprocess, "run", side_effect=_inspect_returning(MOUNTED)):
            assert gi._container_has_wrapper_mount("gr00t") is True

    def test_other_mounts_only_are_reported_absent(self) -> None:
        with patch.object(gi.subprocess, "run", side_effect=_inspect_returning(NOT_MOUNTED)):
            assert gi._container_has_wrapper_mount("gr00t") is False

    def test_a_container_with_no_mounts_is_reported_absent(self) -> None:
        with patch.object(gi.subprocess, "run", side_effect=_inspect_returning(NO_MOUNTS)):
            assert gi._container_has_wrapper_mount("gr00t") is False

    @pytest.mark.parametrize(
        "destination",
        [f"{WRAPPER_PATH}.bak", f"{WRAPPER_PATH}2", f"/opt{WRAPPER_PATH}"],
        ids=["suffixed", "digit-suffixed", "nested-under-another-prefix"],
    )
    def test_a_path_merely_containing_the_wrapper_path_is_not_a_match(self, destination: str) -> None:
        """The probe matches a whole destination, not a substring of one.

        A container bind-mounting ``/srv_wrap.py.bak`` has no wrapper at
        ``/srv_wrap.py``, so treating it as mounted would let a deterministic
        start skip onto a container that cannot serve it - the exact outcome the
        probe exists to prevent.
        """
        with patch.object(gi.subprocess, "run", side_effect=_inspect_returning(f"{destination}\n\n")):
            assert gi._container_has_wrapper_mount("gr00t") is False

    def test_a_non_zero_docker_exit_is_reported_absent(self) -> None:
        """``docker inspect`` failing (a vanished container) must not read as mounted."""
        with patch.object(gi.subprocess, "run", side_effect=_inspect_returning(MOUNTED, returncode=1)):
            assert gi._container_has_wrapper_mount("gr00t") is False

    @pytest.mark.parametrize(
        "failure",
        [FileNotFoundError("docker"), OSError("docker daemon unreachable")],
        ids=["docker-binary-missing", "os-error"],
    )
    def test_a_failed_docker_invocation_is_reported_absent(self, failure: Exception) -> None:
        with patch.object(gi.subprocess, "run", side_effect=failure):
            assert gi._container_has_wrapper_mount("gr00t") is False

    def test_the_probe_asks_docker_for_the_named_container_s_mount_destinations(self) -> None:
        """Pins the premise the fixtures above rest on: one destination per line."""
        seen: list[list[str]] = []

        def fake_run(cmd: list[str], *a: Any, **kw: Any) -> Any:
            seen.append(list(cmd))
            return MagicMock(stdout=MOUNTED, stderr="", returncode=0)

        with patch.object(gi.subprocess, "run", side_effect=fake_run):
            gi._container_has_wrapper_mount("some-container")

        assert len(seen) == 1
        cmd = seen[0]
        assert cmd[:3] == ["docker", "inspect", "--format"]
        assert cmd[-1] == "some-container"
        fmt = cmd[3]
        assert ".Mounts" in fmt and ".Destination" in fmt
        # A real newline, not a literal backslash-n: docker prints one
        # destination per line, which is what makes the token split exact.
        assert "\n" in fmt


class TestTheDeterministicSkipThroughTheRealProbe:
    """Both lifecycle outcomes, with the real probe supplying the answer."""

    @staticmethod
    def _start_with_inspect(stdout: str) -> dict[str, Any]:
        def fake_run(cmd: list[str], *a: Any, **kw: Any) -> Any:
            if cmd[:2] == ["docker", "inspect"]:
                return MagicMock(stdout=stdout, stderr="", returncode=0)
            raise AssertionError(f"docker must not be invoked for {cmd[:2]}")

        with (
            patch.object(gi, "_container_state", return_value="running"),
            patch.object(gi.subprocess, "run", side_effect=fake_run),
        ):
            return gi._start_container(**_container_kwargs(deterministic=True, force=False))

    def test_a_running_container_without_the_mount_fails_fast(self) -> None:
        result = self._start_with_inspect(NOT_MOUNTED)
        assert result["status"] == "error"
        assert "force=True" in result["message"]
        assert WRAPPER_PATH in result["message"]

    def test_a_running_container_with_the_mount_is_skipped(self) -> None:
        result = self._start_with_inspect(MOUNTED)
        assert result["status"] == "success"
        assert result["skipped"] is True
