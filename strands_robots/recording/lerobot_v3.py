"""LeRobot v3 through lerobot's own ``LeRobotDataset.create / resume``; imported only when a run records."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._extras import LEROBOT
from ..types import Observation
from .features import features_from, units_meta

UNITS_FILE = "units.json"

if TYPE_CHECKING:
    from .episode import Episode


def require_lerobot() -> Any:
    """LeRobot's dataset class, or an ImportError naming the install line (raised before the robot moves)."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise ImportError(
            f"record= writes a LeRobot dataset and needs LeRobot with its dataset extra: {LEROBOT} ({e})"
        ) from e
    return LeRobotDataset


@contextmanager
def quiet() -> Iterator[None]:
    """LeRobot's writer without the SVT-AV1 banner, libav notices or ``datasets`` bars; a caller's ``SVT_LOG`` wins."""
    import av.logging
    from datasets.utils import logging as hf

    logging.getLogger("lerobot").setLevel(logging.ERROR)
    svt, bars, libav = os.environ.get("SVT_LOG"), hf.is_progress_bar_enabled(), logging.getLogger("libav")
    level, pyav = libav.level, av.logging.get_level()  # PyAV's own level: None, its default, drops libav's log
    os.environ["SVT_LOG"] = svt or "1"  # read by the encoder when it opens: errors only
    libav.setLevel(logging.ERROR)
    hf.disable_progress_bar()
    av.logging.set_libav_level(av.logging.ERROR)  # lerobot hands libav back its own printer after each video
    try:
        yield
    finally:
        if svt is None:
            del os.environ["SVT_LOG"]
        libav.setLevel(level)
        av.logging.set_libav_level(av.logging.INFO)  # libav's own default: PyAV has no getter for it
        av.logging.set_level(pyav)  # and PyAV's callback back in place of that printer, at the caller's level
        (hf.enable_progress_bar if bars else hf.disable_progress_bar)()


def check_target(ep: Episode) -> None:
    """Refuse an existing dataset with another fps, robot or units before step one (no observation needed)."""
    root = ep.root if ep.root is not None else _hub_cache(ep.repo_id)
    info = root / "meta" / "info.json"
    if not info.is_file():
        return
    have = json.loads(info.read_text())
    if int(have.get("fps", ep.fps)) != ep.fps:
        raise ValueError(f"{ep.target}: the dataset there is {have['fps']} fps; this run records at {ep.fps}")
    robot = _robot_type(ep)
    if have.get("robot_type") not in (None, robot):
        raise ValueError(f"{ep.target}: the dataset there is robot {have['robot_type']!r}; this run is {robot!r}")
    _same_units(ep, root, units_meta(ep.units, ep.description))


def _robot_type(ep: Episode) -> str:
    return ep.description.hardware.lerobot_type or ep.description.name


def open_dataset(ep: Episode, first: Observation) -> tuple[Any, bool]:
    """``(dataset, created)``: resume the one already at the target (same features, fps, units) or create it."""
    LeRobotDataset = require_lerobot()
    features = features_from(first, ep.description.action_keys)
    said = units_meta(ep.units, ep.description)
    root = ep.root if ep.root is not None else _hub_cache(ep.repo_id)
    if (root / "meta" / "info.json").is_file():
        _same_units(ep, root, said)
        ds = LeRobotDataset.resume(ep.repo_id, root=root)
        _same(ep, ds, features)
        return ds, False
    ds = LeRobotDataset.create(
        ep.repo_id, fps=ep.fps, features=features, root=root, robot_type=_robot_type(ep), use_videos=True
    )
    (Path(ds.root) / "meta" / UNITS_FILE).write_text(json.dumps(said, indent=2) + "\n")
    return ds, True


def _same_units(ep: Episode, root: Path, said: dict[str, Any]) -> None:
    path = root / "meta" / UNITS_FILE
    have = json.loads(path.read_text()) if path.is_file() else None
    if have is not None and have != said:
        raise ValueError(
            f"{ep.target}: the dataset there is stored in units {have['units']!r}; this run records {said['units']!r}"
            + (_differences(have, said) if have["units"] == said["units"] else "")  # one scheme: say what differs
        )


def _differences(have: dict[str, Any], said: dict[str, Any]) -> str:
    """Each entry the dataset's ``meta/units.json`` states otherwise than this run's, with both values; ``none``
    where one has no such entry (every ``mode``, in a dataset recorded before ``record=`` named them)."""
    there, here = _entries(have), _entries(said)
    keys = [k for k in dict.fromkeys([*here, *there]) if (k in there, there.get(k)) != (k in here, here.get(k))]
    told = [f"{k.removeprefix('joints ')} {_shown(there, k)} (this run: {_shown(here, k)})" for k in keys]
    return f", but meta/units.json there has {', '.join(told)}" if told else ""


def _entries(value: Any, at: str = "") -> dict[str, Any]:
    """A JSON value as one entry per leaf, keyed by its path: ``{"joints gripper.pos mode": "range_0_100", ...}``."""
    if not isinstance(value, dict) or not value:
        return {at: value}
    flat: dict[str, Any] = {}
    for key, sub in value.items():
        flat.update(_entries(sub, f"{at} {key}".lstrip()))
    return flat


def _shown(entries: dict[str, Any], key: str) -> str:
    return repr(entries[key]) if key in entries else "none"


def _hub_cache(repo_id: str) -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / repo_id


def _same(ep: Episode, ds: Any, features: dict[str, dict[str, Any]]) -> None:
    have = {k: (tuple(v["shape"]), list(v.get("names") or [])) for k, v in ds.meta.features.items() if k in features}
    want = {k: (tuple(v["shape"]), list(v["names"])) for k, v in features.items()}
    if have != want:
        raise ValueError(f"{ep.target}: the dataset there has features {have}; this run records {want}")
    if int(ds.fps) != ep.fps:
        raise ValueError(f"{ep.target}: the dataset there is {ds.fps} fps; this run records at {ep.fps}")
