"""One run = one episode: opened before step one, written by ``finish``. The run loop is the only caller."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..description import Description
from ..types import Observation
from ..units import Units
from .features import ACTION, IMAGES, STATE
from .target import is_repo_id, locate


class Episode:
    """``open(observe)`` before step one; ``add(obs, action)`` per control step, in the units the policy saw and did;
    ``finish()`` -> the dataset path.

    A run that raises or ends on a policy fault records nothing (``RunResult.recorded`` is None); a run stopped by
    ``stop()`` keeps its frames. Only an integer ``hz`` can record (a LeRobot dataset has an integer fps).
    Nothing is uploaded unless ``push=True``, which needs an ``hf://owner/name`` target and runs LeRobot's
    ``push_to_hub`` after the episode. ``wants_images``: every frame needs the cameras, whatever the policy looks at.
    """

    wants_images = True

    def __init__(
        self,
        target: str | Path,
        *,
        description: Description,
        hz: float,
        instruction: str | None,
        units: Units | None,
        push: bool = False,
    ) -> None:
        if push and not is_repo_id(target):
            raise ValueError(f"push=True uploads to the Hub and needs record='hf://owner/name', got {target!r}")
        fps = int(round(hz))
        if abs(fps - hz) > 1e-6:
            raise ValueError(f"a LeRobot dataset needs an integer fps; run at hz={fps} to record (got {hz})")
        from .lerobot_v3 import check_target, require_lerobot

        require_lerobot()  # before step one: a run that cannot be recorded never moves the robot
        self.repo_id, self.root = locate(target)
        self.target = target
        self.fps = fps
        self.description = description
        self.task = instruction or description.name
        self.units = units
        self.push = push
        self.frames = 0
        self._writer: Any = None
        self._created = False
        check_target(self)

    def _in_units(self, rad: Any) -> np.ndarray:
        q = np.asarray(rad, dtype=float)
        return np.asarray(q if self.units is None else self.units.from_rad(q), dtype=np.float32)

    def open(self, observe: Callable[[], Observation]) -> None:
        """Create or resume the dataset from one observation (its joints and cameras are the features), before
        step one: its setup is then no tick's lateness, and a dataset this run cannot append to is refused before
        step one writes."""
        if self._writer is None:
            from .lerobot_v3 import open_dataset, quiet

            first = observe()
            with quiet():
                self._writer, self._created = open_dataset(self, first)

    def add(self, obs: Observation, action: Any) -> None:
        self.open(lambda: obs)
        frame: dict[str, Any] = {
            STATE: self._in_units(obs.joints.positions),
            ACTION: self._in_units(action),
            "task": self.task,
        }
        for cam, img in obs.images.items():
            frame[IMAGES + cam] = np.ascontiguousarray(img)
        self._writer.add_frame(frame)
        self.frames += 1

    def finish(self) -> str | None:
        """Save the episode and finalize; returns the dataset directory (None when no frame was added)."""
        if not self.frames:  # opened before step one, ended before a frame: no episode, and no new empty dataset
            self.abort()
            return None
        from .lerobot_v3 import quiet

        with quiet():
            self._writer.save_episode(parallel_encoding=False)  # no process pool: a script must not re-import
            self._writer.finalize()
        if self.push:
            self._writer.push_to_hub()
        return str(self._writer.root)

    def close(self, stopped_by: str) -> str | None:
        """End of a run: a fault drops the episode (``None``); any other ending saves it and returns the path."""
        if stopped_by == "fault":
            self.abort()
            return None
        return self.finish()

    def abort(self) -> None:
        """A run that raised or faulted: drop its frames, close the writer, and remove a dataset this run created."""
        if self._writer is None:
            return
        self._writer.clear_episode_buffer()
        self._writer.finalize()
        if self._created and int(self._writer.meta.total_episodes) == 0:
            shutil.rmtree(self._writer.root, ignore_errors=True)
        self._writer = None


class NotRecording:
    """What a run without ``record=`` holds: every call is a no-op, so the loop has no recording branches."""

    wants_images = False

    def open(self, observe: Callable[[], Observation]) -> None:
        return None

    def add(self, obs: Observation, action: Any) -> None:
        return None

    def close(self, stopped_by: str) -> str | None:
        return None

    def abort(self) -> None:
        return None


def episode(
    record: str | Path | None,
    *,
    description: Description,
    hz: float,
    instruction: str | None,
    units: Units | None,
    push: bool = False,
) -> Episode | NotRecording:
    """The run loop's one entry: an ``Episode`` for ``record=``, else ``NotRecording``."""
    if record is None:
        if push:
            raise ValueError("push=True uploads a recording; pass record='hf://owner/name' too")
        return NotRecording()
    return Episode(record, description=description, hz=hz, instruction=instruction, units=units, push=push)
