"""Acceptance: a real lerobot checkpoint drives the sim SO-101 through ``Policy``, RTC off and on.

``Robot("so101", mode="sim")`` gets the three cameras ``lerobot/smolvla_base``
declares and runs it through ``run_policy(policy_provider="lerobot_local")``
for three seconds at 30 Hz. The run is graded on what the arm did, read from
the observer stream: every step applied, the requested RTC posture reported
back by both the policy and the runner, and a joint that moved. Real MuJoCo,
real lerobot weights on a real GPU, no doubles.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="needs lerobot"),
    pytest.mark.timeout(900),
]

CHECKPOINT = "lerobot/smolvla_base"
FPS = 30
SECONDS = 3.0
CAMERAS = {
    "camera1": ([0.6, 0.0, 0.4], [0.0, 0.0, 0.1]),
    "camera2": ([0.0, 0.6, 0.4], [0.0, 0.0, 0.1]),
    "camera3": ([0.3, 0.3, 0.8], [0.0, 0.0, 0.0]),
}


@pytest.mark.parametrize("rtc", [False, True], ids=["rtc_off", "rtc_on"])
def test_a_lerobot_checkpoint_drives_the_sim_arm(rtc: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lerobot.policies.smolvla.modeling_smolvla", reason="needs the [smolvla] extra")
    monkeypatch.setenv("STRANDS_TRUST_REMOTE_CODE", "1")
    from strands_robots import Robot

    robot = Robot("so101", mode="sim")
    try:
        for name, (position, target) in CAMERAS.items():
            added = robot.add_camera(name, position=position, target=target, width=256, height=256)
            assert added["status"] == "success", added
        joints = robot.robot_action_keys("so101")
        trace: list[list[float]] = []

        def observe(event: Any) -> None:
            if type(event).__name__ == "RunPolicyStep":
                trace.append([float(event.observation[k]) for k in joints])

        ran = robot.run_policy(
            robot_name="so101",
            policy_provider="lerobot_local",
            policy_config={"pretrained_name_or_path": CHECKPOINT, "device": "cuda", "rtc_enabled": rtc},
            instruction="pick up the cube",
            duration=SECONDS,
            control_frequency=FPS,
            async_rtc=rtc,
            observer=observe,
        )
    finally:
        robot.destroy()

    assert ran["status"] == "success", ran
    report = next(c["json"] for c in ran["content"] if "json" in c)
    steps = int(FPS * SECONDS)
    assert (report["actions_applied"], report["action_errors"], len(trace)) == (steps, 0, steps)
    assert (report["policy_rtc_enabled"], report["rtc_async_enabled"]) == (rtc, rtc)
    moved = max(abs(q - q0) for row in trace for q, q0 in zip(row, trace[0], strict=True))
    assert moved > 0.01, f"no joint moved more than 0.01 rad over {steps} steps ({moved:.4f})"
