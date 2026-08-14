"""Contract tests for the shared MuJoCo GL-availability probe.

These pin the behaviour that the render-test gating relies on: the probe reports
a boolean, honours the ``ROBOT_TEST_MUJOCO=0`` force-skip escape hatch, exposes
a reusable ``requires_gl`` skip marker, and - the safety half - constructs the
probe renderer at most once per process however often the cache on the public
entry point is cleared. On a host whose first attempt failed, a second
construction aborts the interpreter uncatchably and takes the rest of the
session with it, so a cleared cache must not be able to reach one.

They run without a GL context: the force-skip, marker-shape and
build-at-most-once assertions never construct a renderer, and the one that
exercises a *failing* probe supplies its own failure rather than needing a
headless host.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib
import subprocess
import sys
import textwrap
from collections.abc import Iterator

import pytest

from tests.simulation.mujoco import _gl_probe
from tests.simulation.mujoco._gl_probe import gl_available, requires_gl


def _refuse_to_construct(*args: object, **kwargs: object) -> None:
    """Stand in for ``mujoco.Renderer`` and fail if anything tries to build one."""
    raise AssertionError("the probe renderer was constructed a second time")


@pytest.fixture(autouse=True)
def _unforced_probe_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give every case the unforced environment its assertions assume.

    ``ROBOT_TEST_MUJOCO=0`` is a supported host setting - this module documents
    it for "a known-bad runner" - and it short-circuits :func:`gl_available`
    ahead of the probe. Left ambient it means the latch is never primed, so the
    cases below would assert about a probe that never ran. The cases that
    exercise the force-skip set the variable themselves.

    An unset latch is *primed* rather than probed. That setting exists to keep a
    known-bad runner from attempting GL at all, so calling the probe here would
    attempt it on exactly the host that opted out - measured as one offscreen
    renderer constructed under ``ROBOT_TEST_MUJOCO=0``, against none when the
    value is primed. ``monkeypatch`` also restores it, so a case that latches a
    swallowed failure cannot leave that as the process-wide answer for every
    later caller.
    """
    monkeypatch.delenv("ROBOT_TEST_MUJOCO", raising=False)
    if _gl_probe._HARDWARE_PROBE_RESULT is None:
        monkeypatch.setattr(_gl_probe, "_HARDWARE_PROBE_RESULT", False)
    _gl_probe.gl_available.cache_clear()
    yield
    _gl_probe.gl_available.cache_clear()


def test_gl_available_returns_bool() -> None:
    """The probe result is a plain bool the skipif condition can consume."""
    assert isinstance(gl_available(), bool)


def test_robot_test_mujoco_zero_forces_no_gl(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROBOT_TEST_MUJOCO=0 forces a negative result without probing hardware."""
    monkeypatch.setenv("ROBOT_TEST_MUJOCO", "0")
    _gl_probe.gl_available.cache_clear()
    try:
        assert gl_available() is False
    finally:
        # Do not leak the forced-negative result into other tests.
        _gl_probe.gl_available.cache_clear()


def test_requires_gl_is_a_skip_marker() -> None:
    """requires_gl is a usable skipif MarkDecorator (applies cleanly to tests)."""
    assert isinstance(requires_gl, pytest.MarkDecorator)
    assert requires_gl.name == "skipif"


def test_a_hardware_answer_is_latched_before_any_case_runs() -> None:
    """An answer is on the latch by the time a case runs, so it is never retried.

    Non-vacuity for the tests below: "no second renderer was constructed" only
    means something if the process already holds an answer. On a host that
    exports the force-skip the import-time probe never ran, and the fixture
    primes the latch rather than probing - which is why this asserts that an
    answer is present rather than where it came from.
    """
    assert _gl_probe._HARDWARE_PROBE_RESULT is not None


def test_gl_available_reports_the_latched_hardware_answer() -> None:
    """The cached entry point answers from the latch rather than re-probing."""
    assert gl_available() is _gl_probe._HARDWARE_PROBE_RESULT


def test_a_cleared_cache_cannot_reprobe_the_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """cache_clear() re-reads the environment but never re-runs the construction.

    On a host whose first probe failed, a second renderer construction aborts
    the interpreter uncatchably, so the cleared cache must not be able to reach
    one. A renderer that refuses to be built proves nothing tries: this holds on
    a GL host and on a headless one, so the pin lives where CI can see it.
    """
    mj = pytest.importorskip("mujoco")
    monkeypatch.setattr(mj, "Renderer", _refuse_to_construct)
    latched = _gl_probe._HARDWARE_PROBE_RESULT
    _gl_probe.gl_available.cache_clear()
    try:
        assert gl_available() is latched
    finally:
        _gl_probe.gl_available.cache_clear()


def test_a_first_probe_failure_is_latched_and_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A graceful first failure is remembered; the retry that would abort never runs.

    This is the headless host's own path, driven on a host that does have GL by
    resetting the latch and making the construction fail.
    """
    mj = pytest.importorskip("mujoco")
    attempts: list[str] = []

    def _failing(*args: object, **kwargs: object) -> None:
        attempts.append("constructed")
        raise RuntimeError("X11: The DISPLAY environment variable is missing")

    monkeypatch.setattr(mj, "Renderer", _failing)
    monkeypatch.setattr(_gl_probe, "_HARDWARE_PROBE_RESULT", None)

    assert _gl_probe._probe_gl_once() is False
    assert _gl_probe._probe_gl_once() is False
    assert attempts == ["constructed"], "the failed probe was retried"


def test_the_force_skip_leaves_the_hardware_latch_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """ROBOT_TEST_MUJOCO=0 forces the skip without consuming or poisoning the latch.

    The force-skip short-circuits ahead of the probe, so the real hardware
    answer survives it and comes back unprobed once the variable is gone.
    """
    mj = pytest.importorskip("mujoco")
    monkeypatch.setattr(mj, "Renderer", _refuse_to_construct)
    latched = _gl_probe._HARDWARE_PROBE_RESULT

    monkeypatch.setenv("ROBOT_TEST_MUJOCO", "0")
    _gl_probe.gl_available.cache_clear()
    try:
        assert gl_available() is False
        assert _gl_probe._HARDWARE_PROBE_RESULT is latched
    finally:
        _gl_probe.gl_available.cache_clear()

    monkeypatch.delenv("ROBOT_TEST_MUJOCO", raising=False)
    _gl_probe.gl_available.cache_clear()
    try:
        assert gl_available() is latched
    finally:
        _gl_probe.gl_available.cache_clear()


def test_the_force_skip_avoids_the_probe_construction_entirely() -> None:
    """``ROBOT_TEST_MUJOCO=0`` answers without building a renderer at all.

    The latch makes this unobservable inside a process that has already probed:
    by the time any test runs, the import-time probe has answered and a second
    call constructs nothing either way. So it runs in a child interpreter where
    the force-skip is set before the module is first imported, which is the only
    place the ordering between the force-skip and the probe is visible.
    """
    pytest.importorskip("mujoco")
    root = pathlib.Path(inspect.getfile(_gl_probe)).resolve().parents[3]
    assert (root / "pyproject.toml").is_file(), root
    code = textwrap.dedent(
        """
        import mujoco

        built = []
        real = mujoco.Renderer

        def counting(*args, **kwargs):
            built.append(1)
            return real(*args, **kwargs)

        mujoco.Renderer = counting

        from tests.simulation.mujoco._gl_probe import gl_available

        print(f"answer={gl_available()} built={len(built)}")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "ROBOT_TEST_MUJOCO": "0", "PYTHONPATH": str(root)},
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "answer=False built=0" in proc.stdout, proc.stdout


def test_every_probe_exit_returns_the_latch() -> None:
    """``_probe_gl_once`` reports the latched value rather than a literal.

    The value reported and the value remembered have to be the same one: a
    literal return can be changed without the latch beside it, leaving the
    process answering one thing and remembering another for every later caller.
    Pinned structurally because the two agree today, so no input distinguishes
    them - the divergence this forbids is a future edit, not current behaviour.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(_gl_probe._probe_gl_once)))
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    assert len(returns) == 4, f"expected four exits, found {len(returns)}"
    for node in returns:
        assert isinstance(node.value, ast.Name) and node.value.id == "_HARDWARE_PROBE_RESULT", (
            f"exit on line {node.lineno} returns "
            f"{ast.unparse(node.value) if node.value else 'None'} instead of the latch"
        )


def test_this_module_is_green_under_the_force_skip() -> None:
    """Every case here holds with ``ROBOT_TEST_MUJOCO=0`` in the environment.

    That is the configuration the skip reason points an operator at on a
    known-bad runner - the host class this gating exists for - so these cases
    must not depend on the ambient value. With the force-skip set before the
    module is imported the import-time probe never runs and the latch stays
    unset, and an assertion written against a latched answer fails there with
    text (``assert None is not None``) that reads as the safety latch being
    broken rather than as an environment sensitivity. A child pytest over this
    file pins it, deselecting this case so the recursion is one level deep.
    """
    here = pathlib.Path(__file__).resolve()
    root = here.parents[3]
    assert (root / "pyproject.toml").is_file(), root
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(here),
            "-q",
            "--no-cov",
            "-p",
            "no:randomly",
            "-k",
            "not is_green_under_the_force_skip",
        ],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "ROBOT_TEST_MUJOCO": "0"},
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
