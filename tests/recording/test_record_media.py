"""Frames read back out of a recorded dataset, in a fresh process, are the pictures the camera took.

``RECORDING_MEDIA=<dir>`` keeps the PNG strip and GIF that this test renders (otherwise they go to tmp).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from strands_robots import Robot

pytest.importorskip("lerobot.datasets.lerobot_dataset")
pytest.importorskip("PIL")

_RENDER = """
import json, sys
import numpy as np
from PIL import Image, ImageDraw
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root, out = sys.argv[1], sys.argv[2]
ds = LeRobotDataset("local/x", root=root)
cam = next(k for k, v in ds.meta.features.items() if v["dtype"] == "video")
frames, labels = [], []
for i in range(ds.num_frames):
    item = ds[i]
    rgb = (item[cam].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    frames.append(rgb)
    ep, t, pan = int(item['episode_index']), int(item['frame_index']), float(item['action'][0])
    labels.append(f"ep {ep} · t {t} · pan {pan:+.2f}")
pick = np.linspace(0, len(frames) - 1, 6).round().astype(int)
h, w = frames[0].shape[:2]
strip = Image.new("RGB", (w * len(pick), h + 16), "white")
for j, i in enumerate(pick):
    strip.paste(Image.fromarray(frames[i]), (j * w, 16))
    ImageDraw.Draw(strip).text((j * w + 3, 2), labels[i], fill="black")
strip.save(f"{out}/recording-strip.png")
gif = [Image.fromarray(f) for f in frames]
gif[0].save(f"{out}/recording.gif", save_all=True, append_images=gif[1:], duration=int(1000 / ds.fps), loop=0)
diff = [float(np.abs(frames[i].astype(int) - frames[0].astype(int)).mean()) for i in pick]
print(json.dumps({"frames": len(frames), "episodes": ds.num_episodes, "size": [w, h], "std": float(frames[0].std()),
                  "diff_vs_first": [round(d, 2) for d in diff], "camera": cam}))
"""


def test_frames_reread_in_a_fresh_process_show_the_arm_moving(tmp_path):
    arm = Robot("so101")
    arm.sim.camera("front", position=[0.45, -0.45, 0.35], target=[0, 0, 0.12], size=(160, 120), attach=arm)
    target = tmp_path / "wave"
    for ep, sign in enumerate((1.0, -1.0, 1.0)):
        arm.run(
            lambda o, s=sign: {"shoulder_pan": s * 0.9 * math.sin(o.t * 3.0), "elbow_flex": 0.6 * math.sin(o.t * 2.0)},
            hz=20,
            steps=30,
            record=target,
            instruction=f"wave {ep}",
        )
    out = Path(os.environ.get("RECORDING_MEDIA", tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(
        [sys.executable, "-c", _RENDER, str(target), str(out)], capture_output=True, text=True, timeout=300
    )
    assert run.returncode == 0, run.stderr[-2000:]
    got = json.loads(run.stdout.strip().splitlines()[-1])
    assert got["frames"] == 90 and got["episodes"] == 3 and got["size"] == [160, 120]
    assert got["std"] > 5  # a picture, not a blank frame
    assert max(got["diff_vs_first"]) > 2  # the arm moved between the first and later frames
    assert (out / "recording-strip.png").is_file() and (out / "recording.gif").is_file()
    print(f"[measured] media: {json.dumps(got)} -> {out}")
