"""Security tests for LLM-supplied simulation output-path guards.

Exercises the pure helpers in :mod:`strands_robots.simulation.safe_output` that
back ``run_policy(video={"path": ...})`` and
``start_cameras_recording(output_dir=..., name=...)``: metacharacter / backslash
/ symlink / traversal rejection, opt-in sandbox confinement, atomic writes, and
filename-component sanitization. GL-free so they run in CI without an OpenGL
context and on both Linux and macOS.
"""

import os
import sys

import pytest

from strands_robots.simulation.safe_output import (
    atomic_write_bytes,
    env_flag,
    sanitize_name_component,
    validate_output_path,
    video_sandbox_args,
)

# --- guards that apply unconditionally (guards-only mode: sandbox_root=None) ---


@pytest.mark.parametrize("meta", [";", "|", "$", "`", ">", "<", "\n", "\r", "\x00"])
def test_rejects_shell_metacharacters(meta):
    """A path containing any shell metacharacter is rejected."""
    with pytest.raises(ValueError, match="metacharacters"):
        validate_output_path(f"clip{meta}.mp4", sandbox_root=None, allow_abs=True)


def test_rejects_backslash_separator():
    """Backslash separators (Windows-style traversal) are rejected outright."""
    with pytest.raises(ValueError, match="backslash"):
        validate_output_path("..\\..\\etc\\passwd", sandbox_root=None, allow_abs=True)


def test_rejects_empty():
    """An empty / whitespace-only path is rejected."""
    with pytest.raises(ValueError, match="empty"):
        validate_output_path("   ", sandbox_root=None, allow_abs=True)


def test_rejects_symlink_target(tmp_path):
    """A symlink planted at the target is refused (arbitrary-write vector)."""
    victim = tmp_path / "victim.mp4"
    victim.write_bytes(b"important")
    link = tmp_path / "clip.mp4"
    link.symlink_to(victim)
    with pytest.raises(ValueError, match="symlink"):
        validate_output_path(str(link), sandbox_root=None, allow_abs=True)


@pytest.mark.parametrize("bad", ["../escape.mp4", "../../etc/x.mp4", "sub/../../etc/x.mp4"])
def test_rejects_traversal_even_without_sandbox(bad):
    """`..` segments are rejected outright, even in guards-only mode."""
    with pytest.raises(ValueError, match="traversal"):
        validate_output_path(bad, sandbox_root=None, allow_abs=True)


def test_allows_plain_absolute_path_without_sandbox(tmp_path):
    """A plain absolute path (no `..`) is permitted in guards-only mode."""
    target = tmp_path / "sub" / "clip.mp4"
    resolved = validate_output_path(str(target), sandbox_root=None, allow_abs=True)
    assert resolved == target.resolve()


# --- opt-in sandbox confinement ------------------------------------------------


def test_sandbox_rejects_path_outside_root(tmp_path):
    """With a sandbox root and allow_abs=False, a path outside it is rejected."""
    root = tmp_path / "videos"
    root.mkdir()
    with pytest.raises(ValueError, match="outside the sandbox"):
        validate_output_path("/etc/cron.d/evil", sandbox_root=root.resolve(), allow_abs=False)


def test_sandbox_rejects_traversal_escape(tmp_path):
    """`..` that would escape the sandbox root is rejected as traversal."""
    root = tmp_path / "videos"
    root.mkdir()
    escape = root / ".." / "outside.mp4"
    with pytest.raises(ValueError, match="traversal"):
        validate_output_path(str(escape), sandbox_root=root.resolve(), allow_abs=False)


def test_sandbox_accepts_path_inside_root(tmp_path):
    """A path inside the sandbox root resolves cleanly."""
    root = tmp_path / "videos"
    root.mkdir()
    target = root / "sub" / "clip.mp4"
    resolved = validate_output_path(str(target), sandbox_root=root.resolve(), allow_abs=False)
    assert resolved == target.resolve()


def test_allow_abs_bypasses_sandbox(tmp_path):
    """allow_abs=True permits an absolute path outside the sandbox root."""
    root = tmp_path / "videos"
    root.mkdir()
    resolved = validate_output_path(str(tmp_path / "elsewhere.mp4"), sandbox_root=root.resolve(), allow_abs=True)
    assert resolved == (tmp_path / "elsewhere.mp4").resolve()


def test_sandbox_rejects_intermediate_symlink_escape(tmp_path):
    """A dir symlink planted *inside* the sandbox that points outside cannot
    smuggle a write past confinement.

    The final path component is a plain filename (so the ``is_symlink()`` guard,
    which only inspects the leaf, does not fire); the escape lives in an
    intermediate directory component. Confinement holds because ``resolve()``
    expands the intermediate symlink before the ``relative_to(sandbox_root)``
    check, so the true on-disk destination is seen to be outside the root.
    Regression guard: a future refactor that resolved leniently (or short-
    circuited the confinement check) would silently reopen an arbitrary-write
    hole here.
    """
    root = tmp_path / "videos"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "escape"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the sandbox"):
        validate_output_path(str(link / "clip.mp4"), sandbox_root=root.resolve(), allow_abs=False)


def test_sandbox_accepts_intermediate_symlink_staying_inside(tmp_path):
    """A dir symlink whose target is still under the sandbox root is accepted.

    Confinement is defined by the *resolved* destination, not a blanket ban on
    symlinked path components, so a symlink that stays inside the root must not
    be rejected (that would break legitimate layouts that route through a link).
    """
    root = tmp_path / "videos"
    root.mkdir()
    real = root / "real"
    real.mkdir()
    link = root / "alias"
    link.symlink_to(real, target_is_directory=True)
    resolved = validate_output_path(str(link / "clip.mp4"), sandbox_root=root.resolve(), allow_abs=False)
    assert resolved == (real / "clip.mp4").resolve()


# --- video_sandbox_args env policy --------------------------------------------


def test_video_sandbox_args_default_allows_abs(monkeypatch):
    """Without STRANDS_ROBOTS_VIDEO_ROOT, video paths are unconfined (historic contract)."""
    monkeypatch.delenv("STRANDS_ROBOTS_VIDEO_ROOT", raising=False)
    root, allow_abs = video_sandbox_args()
    assert root is None
    assert allow_abs is True


def test_video_sandbox_args_confines_when_root_set(monkeypatch, tmp_path):
    """Setting STRANDS_ROBOTS_VIDEO_ROOT switches to sandbox-confined mode."""
    monkeypatch.setenv("STRANDS_ROBOTS_VIDEO_ROOT", str(tmp_path / "vids"))
    monkeypatch.delenv("STRANDS_ROBOTS_VIDEO_ALLOW_ABS", raising=False)
    root, allow_abs = video_sandbox_args()
    assert root == (tmp_path / "vids").resolve()
    assert allow_abs is False


def test_video_sandbox_args_allow_abs_override(monkeypatch, tmp_path):
    """STRANDS_ROBOTS_VIDEO_ALLOW_ABS re-permits absolute paths inside sandbox mode."""
    monkeypatch.setenv("STRANDS_ROBOTS_VIDEO_ROOT", str(tmp_path / "vids"))
    monkeypatch.setenv("STRANDS_ROBOTS_VIDEO_ALLOW_ABS", "1")
    _, allow_abs = video_sandbox_args()
    assert allow_abs is True


@pytest.mark.parametrize("val,expected", [("1", True), ("true", True), ("YES", True), ("0", False), ("", False)])
def test_env_flag(monkeypatch, val, expected):
    """env_flag treats 1/true/yes (case-insensitive) as opt-in."""
    monkeypatch.setenv("SOME_FLAG", val)
    assert env_flag("SOME_FLAG") is expected


# --- name-component sanitization (interpolated into per-camera filenames) ------


@pytest.mark.parametrize("bad", ["../escape", "a/b", "a\\b", "..", ".", "rec;rm", "..leading"])
def test_sanitize_name_rejects_unsafe(bad):
    """A recording `name` carrying separators / traversal / metachars is rejected."""
    with pytest.raises(ValueError):
        sanitize_name_component(bad)


@pytest.mark.parametrize("ok", ["rec_01", "grasp-cube", "episode.0", "T1"])
def test_sanitize_name_accepts_safe(ok):
    """A plain filename tag passes through unchanged."""
    assert sanitize_name_component(ok) == ok


# --- atomic write --------------------------------------------------------------


def test_atomic_write_creates_file_with_owner_perms(tmp_path):
    """atomic_write_bytes writes the payload and sets 0o600 / 0o700 perms."""
    target = tmp_path / "deep" / "clip.bin"
    atomic_write_bytes(target, b"payload")
    assert target.read_bytes() == b"payload"
    if sys.platform != "win32":
        assert oct(target.stat().st_mode & 0o777) == "0o600"
        assert oct((tmp_path / "deep").stat().st_mode & 0o777) == "0o700"


def test_atomic_write_preserves_existing_on_failure(tmp_path, monkeypatch):
    """If os.replace fails mid-write, the existing target and dir are untouched."""
    target = tmp_path / "keep.bin"
    target.write_bytes(b"ORIGINAL")

    def boom(_src, _dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_bytes(target, b"NEW-DATA-THAT-SHOULD-NOT-LAND")

    assert target.read_bytes() == b"ORIGINAL"
    leftovers = [p for p in target.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
