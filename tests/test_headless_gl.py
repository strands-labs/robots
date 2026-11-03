"""Headless Linux renders with no setup: ``import strands_robots`` picks MUJOCO_GL before anything imports mujoco."""

import os
import subprocess
import sys

import pytest

from strands_robots import _gl

LINUX_EGL = sys.platform.startswith("linux") and _gl._loads("libEGL.so.1")


def _run(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    clean = {k: v for k, v in os.environ.items() if k not in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "DISPLAY")}
    cmd = [sys.executable, "-c", code]
    return subprocess.run(cmd, env={**clean, **env}, capture_output=True, text=True, timeout=120)


def _fake_host(monkeypatch, system, env):
    monkeypatch.setenv("MUJOCO_GL", "")  # recorded, so the value ensure_gl() writes itself is undone after the test
    for key in ("MUJOCO_GL", "DISPLAY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(_gl.platform, "system", lambda: system)
    monkeypatch.setattr(_gl, "_loads", lambda lib: lib == "libEGL.so.1")


@pytest.mark.parametrize(
    ("system", "env", "expect"),
    [
        ("Linux", {}, "egl"),  # headless: picked
        ("Linux", {"MUJOCO_GL": "osmesa"}, "osmesa"),  # the user's choice is never clobbered
        ("Linux", {"DISPLAY": ":0"}, ""),  # a display: MuJoCo's own default
        ("Darwin", {}, ""),  # macOS / Windows: the native context
    ],
)
def test_ensure_gl_picks_only_when_headless_and_unset(monkeypatch, system, env, expect):
    _fake_host(monkeypatch, system, env)
    assert _gl.ensure_gl() == expect
    assert os.environ.get("MUJOCO_GL", "") == expect


def test_picking_for_a_faked_headless_host_leaves_an_unset_mujoco_gl_unset():
    # Unset as on a Mac, where a leaked egl fails every later `import mujoco`; CI's own MUJOCO_GL=egl would hide it.
    with pytest.MonkeyPatch.context() as outer:
        outer.setenv("MUJOCO_GL", "")  # recorded, so this test too puts back what the process had
        outer.delenv("MUJOCO_GL")
        with pytest.MonkeyPatch.context() as mp:
            _fake_host(mp, "Linux", {})
            assert _gl.ensure_gl() == "egl"
        left = os.environ.get("MUJOCO_GL")  # compared alone: a failing assert on os.environ would print it
    assert left is None


@pytest.mark.skipif(not LINUX_EGL, reason="needs Linux with libEGL")
def test_the_package_import_sets_mujoco_gl_before_mujoco_is_imported():
    out = _run("import os, sys, strands_robots; print(os.environ.get('MUJOCO_GL'), 'mujoco' in sys.modules)")
    assert out.stdout.split() == ["egl", "False"], out.stderr


@pytest.mark.skipif(not LINUX_EGL, reason="needs Linux with libEGL")
def test_a_user_import_of_mujoco_after_the_package_still_renders():
    code = (
        "import strands_robots, mujoco\n"
        "img = strands_robots.Robot('so101').sim.render()\n"
        "print(img.shape, int(img.reshape(-1, 3).std(0).sum() > 0))"
    )
    out = _run(code)
    assert out.returncode == 0 and out.stdout.strip().endswith("1"), out.stderr[-2000:]


@pytest.mark.skipif(not LINUX_EGL, reason="needs Linux with libEGL")
def test_mujoco_imported_first_names_the_fix():
    code = (
        "import mujoco, strands_robots\n"
        "try:\n    strands_robots.Robot('so101').sim.render()\n"
        "except RuntimeError as e:\n    print(e)"
    )
    out = _run(code)
    assert "set MUJOCO_GL=egl before the first import of mujoco" in out.stdout, out.stdout + out.stderr[-2000:]
