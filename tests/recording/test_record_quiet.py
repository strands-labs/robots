"""``record=`` writes a dataset without flooding the terminal with LeRobot's encoder banner and progress bars."""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import subprocess
import sys
import textwrap

import pytest

from strands_robots.recording.lerobot_v3 import quiet

pytest.importorskip("lerobot.datasets.lerobot_dataset")
import av.logging  # noqa: E402  (LeRobot's own dependency)

_SCRIPT = textwrap.dedent(
    """
    import sys
    import av.logging
    from datasets.utils.logging import is_progress_bar_enabled
    from strands_robots import Robot
    from strands_robots.policy import SequencePolicy
    arm = Robot("so101")
    arm.sim.camera("wrist", size=(64, 48), attach=arm)
    for _ in range(2):
        arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=5, record=sys.argv[1])
    av.logging.log(av.logging.WARNING, "swscaler", "a notice after the runs")  # as each frame decoded on macOS does
    print("BARS", is_progress_bar_enabled())
    """
)


def _record(tmp_path, **env: str) -> subprocess.CompletedProcess[str]:
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path / "ds")],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, **env},
    )
    assert out.returncode == 0, out.stderr[-1500:]
    return out


@pytest.mark.parametrize(
    ("env", "banner"),
    [({}, False), ({"SVT_LOG": "3"}, True)],
    ids=["quiet-by-default", "a-callers-SVT_LOG-wins"],
)
def test_two_recorded_episodes_print_no_encoder_banner_or_progress_bar(tmp_path, env, banner):
    out = _record(tmp_path, **env)
    assert ("Svt[info]" in out.stderr) is banner, out.stderr[-1500:]
    assert "a notice after the runs" not in out.stderr, out.stderr[-1500:]  # libav's own printer is not left on
    assert "Map:" not in out.stderr and " @ 0x" not in out.stderr, out.stderr[-1500:]  # datasets bar, libav notice
    assert out.stdout.split() == ["BARS", "True"]  # the caller's progress-bar setting is put back


_said = itertools.count()


def _heard(capfd, caplog) -> dict[str, str]:
    """Where libav sends a WARNING and an ERROR: its own printer (stderr), PyAV's (Python logging) or nowhere."""
    capfd.readouterr()
    caplog.clear()
    said = {name: f"{name} message {next(_said)}" for name in ("WARNING", "ERROR")}  # PyAV drops a repeated one
    for name, text in said.items():
        av.logging.log(getattr(av.logging, name), "swscaler", text + "\n")
    err = capfd.readouterr().err
    return {n: "stderr" if t in err else "logging" if t in caplog.messages else "nowhere" for n, t in said.items()}


def _write(raises: bool) -> None:
    with quiet():
        av.logging.restore_default_callback()  # what LeRobot's encoder does after each video
        if raises:
            raise RuntimeError("the encoder failed")


@pytest.mark.parametrize("raises", [False, True], ids=["write-returns", "write-raises"])
@pytest.mark.parametrize("level", [None, av.logging.ERROR, av.logging.INFO], ids=["off", "ERROR", "INFO"])
def test_after_a_write_libav_logs_where_the_caller_had_it(capfd, caplog, level, raises):
    """LeRobot's encoder leaves libav's own stderr printer in place of PyAV's callback; a write puts PyAV's back, at the
    caller's level (``None``, PyAV's default, drops libav's log), and does so when it raises too."""
    caplog.set_level(logging.WARNING, logger="libav")  # PyAV's own default for its logger
    av.logging.set_level(level)
    try:
        before = _heard(capfd, caplog)
        with pytest.raises(RuntimeError) if raises else contextlib.nullcontext():
            _write(raises)
        after = av.logging.get_level(), _heard(capfd, caplog)
    finally:
        av.logging.set_level(None)
    assert after == (level, before)
