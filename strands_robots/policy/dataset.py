"""``Policy("<dataset dir>[?episode=n]")``: replay one recorded episode's actions, in the units it was stored in."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._extras import LEROBOT
from ..units import Units, lerobot_units
from ..units.lerobot import norm_hint, unnamed
from .sequence import SequencePolicy

if TYPE_CHECKING:
    from ..description import Description


class ReplayPolicy(SequencePolicy):
    """A recorded episode's ``action`` column played back at the dataset's fps, then held on its last row.

    ``units`` is what ``meta/units.json`` says the dataset stores (``record=`` writes it), and ``units_for``
    rebuilds those units for the robot it replays on, with the LeRobot norm mode the file names for each joint.
    A dataset without that file (one written by another tool) needs ``units=`` from the caller, because guessing
    radians for degrees moves a real arm by a factor of 57. One whose file names no modes (written before
    ``record=`` named them, or with ``units="lerobot"`` alone) takes them from ``norm=``; without it a real body
    refuses the replay, because the registry's modes are a guess there, and a simulated body takes them.
    """

    def __init__(
        self, root: Path, episode: int = 0, *, units: str | None = None, norm: Mapping[str, str] | None = None
    ) -> None:
        info = json.loads((root / "meta" / "info.json").read_text())
        stored = root / "meta" / "units.json"
        said: dict[str, Any] = json.loads(stored.read_text()) if stored.is_file() else {}
        if "units" in said and units is not None and units != said["units"]:  # the dataset's word is the record
            raise ValueError(
                f"{root}: meta/units.json says the actions are stored in {said['units']!r}, but units={units!r} was "
                "passed; drop units= (the dataset knows) or replay a dataset recorded in those units"
            )
        given = dict(norm) if norm else None  # an empty norm names no mode
        self.names = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]]
        if given is not None and (stray := sorted(set(given) - {"*", *self.names})):
            raise ValueError(  # a misspelt joint would take the "*" mode on a real arm
                f"{root}: norm= names {', '.join(stray)}, which the dataset does not record "
                f"(it recorded {', '.join(self.names)})"
            )
        units = said.get("units", units or ("lerobot" if given else None))
        if units is None:
            raise ValueError(
                f"{root}: no meta/units.json says what the actions are in; pass units='rad', or norm= with the "
                "LeRobot norm mode each joint was recorded in"
            )
        modes = {k.removesuffix(".pos"): j["mode"] for k, j in said.get("joints", {}).items() if "mode" in j}
        if given is not None and units != "lerobot":
            raise ValueError(f"{root}: norm= names LeRobot norm modes, but the actions are stored in {units!r}")
        if given is not None and modes and {k: given.get(k, given.get("*")) for k in modes} != modes:
            raise ValueError(
                f"{root}: meta/units.json says the joints are in {modes}, but norm={given} was passed; drop norm= "
                "(the dataset knows)"
            )
        self.norm: dict[str, str] | None = modes or given  # joint (or "*") -> LeRobot norm mode; None: unknown
        self.units, self.hz, self.root, self.episode = units, float(info["fps"]), root, episode
        self.recorded = _episode_actions(root, episode)
        super().__init__(list(self.recorded))

    def units_for(self, description: Description) -> Units | None:
        """The recorded units for this robot: LeRobot's in the modes the dataset or ``norm=`` names (else the
        registry's, which ``attach`` refuses on a real body); any other scheme by its name. A ``norm=`` whose ``"*"``
        would give a joint the registry names (the gripper) another mode is refused: the joint must be named."""
        if self.units == "lerobot":
            if self.norm is not None and (left := unnamed(description, self.norm)):
                gives = ", ".join(f"{k} {m!r}" for k, m in left.items())
                raise ValueError(  # LeRobot's SO gripper is range_0_100 in both modes: its 50 % is not 50 degrees
                    f"{self!r}: norm= leaves {', '.join(left)} to its '*' mode {self.norm['*']!r}, but "
                    f"{description.name}'s LeRobot convention gives {gives}: name each, norm={ {**self.norm, **left} }"
                )
            return lerobot_units(description, self.norm)
        return Units.for_robot(description, str(self.units))  # never None: __init__ refuses a dataset without units

    def attach(self, robot: Any) -> None:
        """Order the recorded columns by this robot's action keys; a key the dataset lacks is refused by name."""
        if robot.is_real and self.units == "lerobot" and self.norm is None:
            raise ValueError(  # the registry's modes would be a guess, and a wrong one moves a real arm off target
                f"{self!r} does not say which LeRobot norm mode each joint was recorded in, and {robot.id} is a real "
                f"body: pass {norm_hint(robot.description)} with the modes it was recorded in (record= names them "
                "in meta/units.json)"
            )
        missing = [k for k in robot.action_keys if k not in self.names]
        if missing:
            raise ValueError(f"{self!r} has no action for {', '.join(missing)} (it recorded {', '.join(self.names)})")
        order = [self.names.index(k) for k in robot.action_keys]
        self.actions = list(self.recorded[:, order])

    def __repr__(self) -> str:
        return f"Policy('{self.root}?episode={self.episode}')"


def _episode_actions(root: Path, episode: int) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(f"replaying a dataset reads its parquet files and needs LeRobot: {LEROBOT} ({e})") from e
    files = sorted((root / "data").glob("**/*.parquet"))
    rows = [pq.read_table(f, columns=["episode_index", "frame_index", "action"]).to_pydict() for f in files]
    found = sorted({e for r in rows for e in r["episode_index"]})
    frames = sorted((f, a) for r in rows for e, f, a in zip(*r.values(), strict=True) if e == episode)
    if not frames:
        raise ValueError(f"{root}: no episode {episode}; it has episodes {found}")
    return np.asarray([a for _, a in frames], dtype=float)
