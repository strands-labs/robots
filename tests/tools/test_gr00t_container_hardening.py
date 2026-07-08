"""Pin: gr00t_inference does not let an agent mount host paths or pick images.

The agent-facing tool must not expose ``volumes``, ``image_name``, or
``container_command`` -- those let a prompt-injected agent mount the host
filesystem, the docker socket, or run an arbitrary container command
(host RCE). Container topology is operator-config-driven.

Defence in depth: the private ``_start_container`` entry point (reachable
by operators/tests) also rejects dangerous mounts and off-allowlist images.

These tests fail on pre-fix code, where ``_start_container`` appended any
caller-supplied ``-v host:container`` mount straight into the docker argv.
"""

from __future__ import annotations

import importlib
import inspect
from unittest.mock import MagicMock, patch

import pytest

# ``from strands_robots.tools import gr00t_inference`` resolves via the package's
# lazy __getattr__ (tools/__init__.py) to the *tool function* (a
# DecoratedFunctionTool), not the module -- so it exposes no ``subprocess`` or
# ``_is_allowed_image`` attribute. These tests need the module object itself for
# the private helpers and ``patch.object(gi.subprocess, ...)``. import_module
# returns the canonical sys.modules entry without a plain ``import a.b.c``
# statement, which would otherwise trip CodeQL py/import-and-import-from against
# the from-import of this module elsewhere in the package. import_module is the
# package's own module-loading idiom (see both __init__.py files).
gi = importlib.import_module("strands_robots.tools.gr00t_inference")

# --- agent surface: dangerous params are gone --------------------------


def _tool_params() -> set[str]:
    fn = getattr(gi.gr00t_inference, "__wrapped__", None) or gi.gr00t_inference
    return set(inspect.signature(fn).parameters)


def test_tool_signature_drops_volumes():
    assert "volumes" not in _tool_params()


def test_tool_signature_drops_image_name():
    assert "image_name" not in _tool_params()


def test_tool_signature_drops_container_command():
    assert "container_command" not in _tool_params()


def test_tool_rejects_volumes_kwarg():
    # Passing the removed kwarg must raise (TypeError) rather than silently
    # mount anything.
    with pytest.raises(TypeError):
        gi.gr00t_inference(action="start_container", volumes={"/": "/host"})


# --- image allowlist ---------------------------------------------------


def test_is_allowed_image_accepts_canonical():
    assert gi._is_allowed_image("gr00t:latest") is True
    assert gi._is_allowed_image("gr00t:n1.7") is True
    assert gi._is_allowed_image("nvcr.io/nvidia/isaac-gr00t:n1.7") is True


def test_is_allowed_image_rejects_arbitrary():
    assert gi._is_allowed_image("alpine:latest") is False
    assert gi._is_allowed_image("evil/image:tag") is False
    assert gi._is_allowed_image("") is False


def test_image_allowlist_env_extends(monkeypatch):
    monkeypatch.setenv("STRANDS_GR00T_IMAGE_ALLOW", "myreg/gr00t:*")
    assert gi._is_allowed_image("myreg/gr00t:v1") is True


def test_image_allowlist_literal_pattern_pins_exact_ref(monkeypatch):
    # An operator pattern with NO trailing "*" is matched literally, not as a
    # prefix. Pinning "myreg/gr00t:v1.0" must admit that exact ref only -- a
    # sibling tag ("myreg/gr00t:v1.0-evil") or a longer image sharing the
    # prefix must be rejected, so a literal pin cannot be silently widened into
    # a wildcard. The wildcard branch is exercised above; this locks in the
    # complementary literal-match branch that guards the exact-pin contract.
    monkeypatch.setenv("STRANDS_GR00T_IMAGE_ALLOW", "myreg/gr00t:v1.0")
    assert gi._is_allowed_image("myreg/gr00t:v1.0") is True
    assert gi._is_allowed_image("myreg/gr00t:v1.0-evil") is False
    assert gi._is_allowed_image("myreg/gr00t:v2.0") is False


# --- _start_container guards (defence in depth) ------------------------


def _no_run_patches():
    """Patch subprocess.run + _container_state so a failing guard is the
    only reason docker run would be skipped."""
    return (
        patch.object(gi, "_container_state", return_value="absent"),
        patch.object(gi.subprocess, "run", side_effect=AssertionError("docker run must NOT be called")),
    )


def test_start_container_rejects_host_root_mount():
    state_p, run_p = _no_run_patches()
    with state_p, run_p:
        result = gi._start_container(
            image_name="alpine:latest",
            container_name="x",
            port=5555,
            volumes={"/": "/host"},
            hf_token=None,
            container_command="sh -c 'id'",
            hf_local_dir=None,
            force=True,
        )
    assert result["status"] == "error"


def test_start_container_rejects_docker_socket_mount():
    state_p, run_p = _no_run_patches()
    with state_p, run_p:
        result = gi._start_container(
            image_name="gr00t:latest",
            container_name="x",
            port=5555,
            volumes={"/var/run/docker.sock": "/var/run/docker.sock"},
            hf_token=None,
            container_command="docker ps",
            hf_local_dir=None,
            force=True,
        )
    assert result["status"] == "error"
    assert "docker" in result["message"].lower() or "socket" in result["message"].lower()


def test_start_container_rejects_etc_mount():
    state_p, run_p = _no_run_patches()
    with state_p, run_p:
        result = gi._start_container(
            image_name="gr00t:latest",
            container_name="x",
            port=5555,
            volumes={"/etc": "/host_etc"},
            hf_token=None,
            container_command="tail -f /dev/null",
            hf_local_dir=None,
            force=True,
        )
    assert result["status"] == "error"


# --- _check_volume_safety: child-of-protected-dir prefix coverage ------
# Regression for the exact-match gap (PR #372 review): mounting a *child*
# of a protected dir (e.g. /etc/shadow, /root/.ssh) must be rejected, not
# just the bare dir. These fail on the pre-fix `norm in blocked_dirs` code.


@pytest.mark.parametrize(
    "host_path",
    [
        "/etc/shadow",
        "/root/.ssh",
        "/root/.ssh/id_rsa",
        "/home/ubuntu/.aws/credentials",
        "/proc/1/environ",
        "/sys/kernel",
        "/var/run/docker.sock.bak",
        "/etc",
        "/",
    ],
)
def test_check_volume_safety_rejects_protected_paths(host_path):
    assert gi._check_volume_safety({host_path: "/x"}) is not None


@pytest.mark.parametrize(
    "host_path",
    ["/mnt/models", "/data/checkpoints", "/opt/gr00t", "/srv/data"],
)
def test_check_volume_safety_allows_legit_mounts(host_path):
    # A prefix check must not over-block: legitimate non-protected mounts
    # (and especially anything that merely starts with "/") still pass.
    assert gi._check_volume_safety({host_path: "/x"}) is None


def test_check_volume_safety_expands_user_home():
    # ~ expansion lands under /home or the real HOME -> must be rejected when
    # it resolves under a protected dir.
    import os

    reason = gi._check_volume_safety({"~/.ssh/id_rsa": "/x"})
    home = os.path.expanduser("~")
    if home.startswith(("/home", "/root", "/Users")):
        # /Users is not in the Linux blocklist; only assert when protected.
        if home.startswith(("/home", "/root")):
            assert reason is not None


def test_start_container_rejects_off_allowlist_image():
    state_p, run_p = _no_run_patches()
    with state_p, run_p:
        result = gi._start_container(
            image_name="alpine:latest",
            container_name="x",
            port=5555,
            volumes=None,
            hf_token=None,
            container_command="tail -f /dev/null",
            hf_local_dir=None,
            force=True,
        )
    assert result["status"] == "error"
    assert "allowlist" in result["message"]


def test_start_container_allows_safe_defaults():
    """The legitimate path (allowlisted image, default checkpoint volumes)
    still reaches docker run."""
    runs: list[list[str]] = []

    def fake_run(cmd, *a, **kw):
        runs.append(list(cmd))
        return MagicMock(stdout="", stderr="", returncode=0)

    with (
        patch.object(gi, "_container_state", return_value="absent"),
        patch.object(gi.subprocess, "run", side_effect=fake_run),
    ):
        result = gi._start_container(
            image_name="gr00t:latest",
            container_name="gr00t",
            port=5555,
            volumes=None,
            hf_token=None,
            container_command="tail -f /dev/null",
            hf_local_dir="/data/cp",
            force=True,
        )
    assert result["status"] == "success"
    argv = next(c for c in runs if c[:2] == ["docker", "run"])
    joined = " ".join(argv)
    # No host root / etc / socket mount in the emitted argv.
    assert "-v /:/host" not in joined
    assert "/var/run/docker.sock" not in joined
    assert "/data/cp:/data/checkpoints" in joined


# --- Pin: hf_local_dir flows through _check_volume_safety (R3 regression) ---


class TestHfLocalDirVolumeSafety:
    """Pin: agent-supplied hf_local_dir is validated against volume blocklist.

    Before this fix, _start_container checked _check_volume_safety(volumes)
    where volumes=None from the agent path, then built effective_volumes using
    hf_local_dir WITHOUT re-checking. A prompt-injected agent could call
    gr00t_inference(action="start_container", hf_local_dir="/etc") and mount
    a protected host directory into the container.

    The fix re-runs _check_volume_safety(effective_volumes) after the
    effective_volumes dict is fully assembled, catching protected paths
    introduced by hf_local_dir.
    """

    @pytest.mark.parametrize(
        "bad_path",
        [
            "/etc",
            "/etc/shadow",
            "/root",
            "/root/.ssh",
            "/root/.ssh/id_rsa",
            "/home",
            "/home/ubuntu/.aws/credentials",
            "/proc",
            "/proc/1/environ",
            "/sys",
            "/sys/kernel",
            "/var",
            "/var/run/docker.sock",
        ],
    )
    def test_start_container_rejects_hf_local_dir_under_protected_path(self, bad_path):
        """hf_local_dir pointing at a protected host path must be rejected."""
        gi = importlib.import_module("strands_robots.tools.gr00t_inference")

        with patch.object(gi.subprocess, "run", side_effect=AssertionError("docker must not be called")):
            with patch.object(gi, "_container_state", return_value="absent"):
                result = gi._start_container(
                    image_name="gr00t:latest",
                    container_name="test",
                    port=5555,
                    hf_token=None,
                    volumes=None,
                    container_command="tail -f /dev/null",
                    hf_local_dir=bad_path,
                    force=False,
                )
        assert result["status"] == "error", f"Expected error for hf_local_dir={bad_path!r}, got: {result}"
        # Confirm the error message references the protected path
        assert "protected" in result["message"].lower() or "refusing" in result["message"].lower(), (
            f"Error message should mention protection/refusal: {result['message']}"
        )

    @pytest.mark.parametrize(
        "safe_path",
        [
            "/mnt/models/gr00t-checkpoints",
            "/data/checkpoints",
            "/opt/gr00t/weights",
            "/srv/ml/gr00t",
        ],
    )
    def test_start_container_allows_hf_local_dir_safe_paths(self, safe_path):
        """Legitimate hf_local_dir paths must NOT be rejected."""
        gi = importlib.import_module("strands_robots.tools.gr00t_inference")

        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(gi.subprocess, "run", mock_run):
            with patch.object(gi, "_container_state", return_value="absent"):
                result = gi._start_container(
                    image_name="gr00t:latest",
                    container_name="test",
                    port=5555,
                    hf_token=None,
                    volumes=None,
                    container_command="tail -f /dev/null",
                    hf_local_dir=safe_path,
                    force=False,
                )
        assert result["status"] == "success", f"Expected success for hf_local_dir={safe_path!r}, got: {result}"
        # Verify the path was mounted
        call_args = mock_run.call_args[0][0]
        assert "-v" in call_args
        vol_str = f"{safe_path}:/data/checkpoints"
        assert vol_str in " ".join(call_args), f"Expected volume mount {vol_str} in docker cmd"


class TestHfLocalDirDownloadSafety:
    """Pin: agent-supplied hf_local_dir is validated for the host-fs WRITE sinks.

    hf_local_dir reaches a second host-fs surface beyond the bind mount:
    _download_checkpoint writes the snapshot to it directly on the host (no
    docker mediation), and lifecycle="full" downloads BEFORE it starts the
    container -- so guarding only _start_container left an arbitrary host-fs
    write primitive for action="download_checkpoint" / action="lifecycle".

    The fix validates hf_local_dir once at the agent dispatch boundary (and
    defence-in-depth at the top of _download_checkpoint), so every action
    rejects a prompt-injected path before any mkdir / snapshot_download runs.
    """

    _BAD_PATHS = [
        "/etc",
        "/etc/cron.d",
        "/root/.ssh",
        "/home/ubuntu/.aws/credentials",
        "/proc/1/environ",
        "/sys/kernel",
        "/var/run",
    ]

    @pytest.mark.parametrize("bad_path", _BAD_PATHS)
    def test_download_checkpoint_rejects_protected_hf_local_dir(self, bad_path):
        """action='download_checkpoint' with a protected hf_local_dir must
        error before any host-fs write or network call."""
        gi = importlib.import_module("strands_robots.tools.gr00t_inference")

        with patch("pathlib.Path.mkdir", side_effect=AssertionError("mkdir must not be called")):
            with patch.object(gi, "require_optional", side_effect=AssertionError("hub import must not be reached")):
                result = gi.gr00t_inference(
                    action="download_checkpoint",
                    hf_repo="attacker/payload",
                    hf_local_dir=bad_path,
                )
        assert result["status"] == "error", f"Expected error for hf_local_dir={bad_path!r}, got: {result}"
        assert "protected" in result["message"].lower() or "refusing" in result["message"].lower(), (
            f"Error message should mention protection/refusal: {result['message']}"
        )

    @pytest.mark.parametrize("bad_path", _BAD_PATHS)
    def test_lifecycle_rejects_protected_hf_local_dir(self, bad_path):
        """action='lifecycle' downloads before starting the container; a
        protected hf_local_dir must be rejected at dispatch before download."""
        gi = importlib.import_module("strands_robots.tools.gr00t_inference")

        with patch("pathlib.Path.mkdir", side_effect=AssertionError("mkdir must not be called")):
            with patch.object(gi.subprocess, "run", side_effect=AssertionError("docker must not be called")):
                result = gi.gr00t_inference(
                    action="lifecycle",
                    lifecycle="full",
                    hf_repo="attacker/payload",
                    hf_local_dir=bad_path,
                )
        assert result["status"] == "error", f"Expected error for hf_local_dir={bad_path!r}, got: {result}"
        assert "protected" in result["message"].lower() or "refusing" in result["message"].lower(), (
            f"Error message should mention protection/refusal: {result['message']}"
        )

    def test_download_checkpoint_internal_entry_guarded(self):
        """Defence in depth: _download_checkpoint itself rejects a protected
        path even when reached directly (not via the tool dispatch)."""
        gi = importlib.import_module("strands_robots.tools.gr00t_inference")

        with patch("pathlib.Path.mkdir", side_effect=AssertionError("mkdir must not be called")):
            with patch.object(gi, "require_optional", side_effect=AssertionError("hub import must not be reached")):
                result = gi._download_checkpoint(
                    hf_repo="attacker/payload",
                    hf_subfolder=None,
                    hf_local_dir="/etc/cron.d",
                    hf_token=None,
                    force=False,
                )
        assert result["status"] == "error"
        assert "protected" in result["message"].lower() or "refusing" in result["message"].lower()

    def test_check_hf_local_dir_safety_allows_safe_and_none(self):
        """The shared helper passes legitimate paths and None through."""
        gi = importlib.import_module("strands_robots.tools.gr00t_inference")

        assert gi._check_hf_local_dir_safety(None) is None
        assert gi._check_hf_local_dir_safety("") is None
        assert gi._check_hf_local_dir_safety("/mnt/models/gr00t") is None
        assert gi._check_hf_local_dir_safety("/data/checkpoints") is None
        assert gi._check_hf_local_dir_safety("/etc") is not None
        assert gi._check_hf_local_dir_safety("/root/.ssh") is not None


# --- Pin: build source repo URL/tag/dir are NOT agent-controllable ----------
#
# ``build_image`` / ``lifecycle`` clone a git repo and ``bash docker/build.sh``
# it -- host RCE if the source repo is agent-supplied. ``repo_url`` /
# ``repo_tag`` / ``source_dir`` are therefore removed from the agent signature
# and resolved from operator env (``STRANDS_GR00T_REPO_URL`` /
# ``STRANDS_GR00T_REPO_TAG``), with the URL exact-matched against an allowlist
# and the tag shape-validated, before any ``git`` / ``bash`` subprocess.


class TestBuildSourceNotAgentControllable:
    """The clone URL/tag/dir cannot be steered by the agent or a bad env."""

    def test_tool_signature_drops_repo_params(self):
        params = _tool_params()
        assert "repo_url" not in params
        assert "repo_tag" not in params
        assert "source_dir" not in params

    @pytest.mark.parametrize("kwarg", ["repo_url", "repo_tag", "source_dir"])
    def test_tool_rejects_removed_repo_kwarg(self, kwarg):
        # Passing a removed param must raise rather than silently clone.
        with pytest.raises(TypeError):
            gi.gr00t_inference(action="build_image", **{kwarg: "/x"})

    @pytest.mark.parametrize(
        "bad_url",
        [
            "https://github.com/attacker/evil-repo",
            "https://github.com/NVIDIA/Isaac-GR00T-evil",
            "https://github.com/NVIDIA/Isaac-GR00T.evil.com",
            "--upload-pack=/bin/sh",
            "file:///etc",
        ],
    )
    def test_build_image_rejects_off_allowlist_url(self, monkeypatch, bad_url):
        """A misconfigured operator URL fails closed before git runs."""
        monkeypatch.setenv("STRANDS_GR00T_REPO_URL", bad_url)
        with patch.object(gi.subprocess, "run", side_effect=AssertionError("git/bash must not run")):
            result = gi.gr00t_inference(action="build_image")
        assert result["status"] == "error"
        assert "allowlist" in result["message"].lower()

    def test_lifecycle_rejects_off_allowlist_url(self, monkeypatch):
        monkeypatch.setenv("STRANDS_GR00T_REPO_URL", "https://github.com/attacker/evil")
        with patch.object(gi.subprocess, "run", side_effect=AssertionError("git/bash must not run")):
            result = gi.gr00t_inference(action="lifecycle", lifecycle="full", hf_repo="x/y")
        assert result["status"] == "error"
        assert "allowlist" in result["message"].lower()

    @pytest.mark.parametrize("bad_tag", ["--upload-pack=x", "a; rm -rf /", "a b", "$(id)", "-x"])
    def test_build_image_rejects_unsafe_tag(self, monkeypatch, bad_tag):
        monkeypatch.setenv("STRANDS_GR00T_REPO_TAG", bad_tag)
        with patch.object(gi.subprocess, "run", side_effect=AssertionError("git/bash must not run")):
            result = gi.gr00t_inference(action="build_image")
        assert result["status"] == "error"
        assert "git ref" in result["message"].lower()

    def test_build_image_internal_entry_guarded(self):
        """Defence in depth: _build_image rejects an off-allowlist URL even when
        called directly (operator/test entry), before any git/bash subprocess."""
        with patch.object(gi.subprocess, "run", side_effect=AssertionError("git/bash must not run")):
            result = gi._build_image(
                repo_url="https://github.com/attacker/evil",
                repo_tag="main",
                image_name="gr00t:latest",
                force=True,
            )
        assert result["status"] == "error"
        assert "allowlist" in result["message"].lower()

    def test_build_image_allows_default_clones_to_fixed_dir(self, monkeypatch, tmp_path):
        """The allowlisted default URL clones to the operator-fixed dir
        (_isaac_gr00t_dir), never a caller path, and uses the resolved URL."""
        runs: list[list[str]] = []

        def fake_run(cmd, *a, **kw):
            runs.append(list(cmd))
            return MagicMock(stdout="", stderr="", returncode=0)

        fixed = tmp_path / "Isaac-GR00T"
        with (
            patch.object(gi, "_isaac_gr00t_dir", return_value=fixed),
            patch.object(gi, "_image_exists", return_value=False),
            patch("pathlib.Path.is_dir", return_value=False),
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch.object(gi.subprocess, "run", side_effect=fake_run),
        ):
            result = gi.gr00t_inference(action="build_image", force=True)
        assert result["status"] == "success", result
        clone = next(c for c in runs if c[:2] == ["git", "clone"])
        # Resolved default URL is used; destination is the fixed dir.
        assert gi._DEFAULT_REPO_URL in clone
        assert str(fixed) in clone

    def test_operator_extends_url_allowlist(self, monkeypatch):
        mirror = "https://git.internal/mirror/Isaac-GR00T"
        monkeypatch.setenv("STRANDS_GR00T_REPO_URL_ALLOW", mirror)
        monkeypatch.setenv("STRANDS_GR00T_REPO_URL", mirror)
        assert gi._is_allowed_repo_url(mirror) is True


# --- Class invariant: NO agent-controlled string reaches a privileged sink ---
#
# This pins the *class* of vulnerability closed, not just one instance. The
# gr00t_inference tool shells out to git/docker/bash and writes the host fs;
# every parameter that could steer a host path, image, command, volume, or
# clone source has been removed from the agent surface and moved to
# operator-config + allowlist. If a future change re-introduces any of these
# as an agent parameter, this test fails -- terminating the whack-a-mole.


def test_no_agent_controlled_host_exec_or_path_params():
    forbidden = {
        "repo_url",
        "repo_tag",
        "source_dir",
        "image_name",
        "volumes",
        "container_command",
    }
    leaked = forbidden & _tool_params()
    assert not leaked, (
        f"agent-facing params reintroduce a host-exec/mount surface: {sorted(leaked)}. "
        "These must be operator-config + allowlist driven, not agent parameters."
    )


# --------------------------------------------------------------------------- #
# #384 item 2 - symlink (realpath) resolution in volume safety                 #
# --------------------------------------------------------------------------- #
def test_check_volume_safety_resolves_symlink_into_protected_dir(tmp_path):
    """A host symlink whose target is a protected dir must be rejected.

    Pre-fix (#384): ``_normalize_host_path`` ran ``normpath`` only, so a
    pre-existing symlink ``/data/ckpt -> /etc`` passed the blocklist while
    docker would mount the resolved ``/etc``. The realpath check closes this.
    """
    link = tmp_path / "sneaky_ckpt"
    link.symlink_to("/etc")  # target is a protected dir
    reason = gi._check_volume_safety({str(link): "/data/checkpoints"})
    assert reason is not None, "symlink resolving to /etc must be rejected; the realpath guard is missing"
    assert "/etc" in reason or "protected" in reason


def test_check_volume_safety_allows_symlink_into_safe_dir(tmp_path):
    """A symlink resolving to a non-protected dir must still be allowed."""
    safe_target = tmp_path / "real_models"
    safe_target.mkdir()
    link = tmp_path / "models_link"
    link.symlink_to(safe_target)
    assert gi._check_volume_safety({str(link): "/data/checkpoints"}) is None


def test_resolve_host_path_does_not_raise_on_missing_path():
    """``_resolve_host_path`` must resolve best-effort on a not-yet-created
    mount source (realpath resolves as far as it can, never raises)."""
    out = gi._resolve_host_path("/does/not/exist/yet/ckpt")
    assert isinstance(out, str) and out.startswith("/")
