"""``record=`` writes a dataset without flooding the terminal with LeRobot's encoder banner and progress bars."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("lerobot.datasets.lerobot_dataset")

_SCRIPT = textwrap.dedent(
    """
    import sys
    from datasets.utils.logging import is_progress_bar_enabled
    from strands_robots import Robot
    from strands_robots.policy import SequencePolicy
    arm = Robot("so101")
    arm.sim.camera("wrist", size=(64, 48), attach=arm)
    for _ in range(2):
        arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=5, record=sys.argv[1])
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
    assert "Map:" not in out.stderr and " @ 0x" not in out.stderr, out.stderr[-1500:]  # datasets bar, libav notice
    assert out.stdout.split() == ["BARS", "True"]  # the caller's progress-bar setting is put back
