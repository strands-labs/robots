"""What a policy needs from the body that feeds it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..types import Observation


@dataclass(frozen=True)
class Requirements:
    """What a policy needs from the body that feeds it: camera names (and sizes) and the state length.

    ``Robot.run`` checks these BEFORE the first inference so a missing camera is a one-line refusal naming it,
    not a shape error inside a model. ``images`` maps camera name -> (width, height) or None when any size is fine;
    ``state_dim`` None means "whatever the robot has".
    """

    images: Mapping[str, tuple[int, int] | None] = field(default_factory=dict)
    state_dim: int | None = None

    def check(self, obs: Observation, *, who: str = "policy") -> None:
        have = {k: (v.shape[1], v.shape[0]) for k, v in obs.images.items()}
        missing = [c for c in self.images if c not in have]
        if missing:
            raise ValueError(
                f"{who} needs camera(s) {missing} but the observation has {sorted(have) or 'none'}: "
                f"add them with sim.camera(name, size=(w, h), attach=robot) or cameras={{name: index}} on hardware"
            )
        wrong = {c: (have[c], want) for c, want in self.images.items() if want is not None and have[c] != tuple(want)}
        if wrong:
            desc = ", ".join(f"{c}: has {h[0]}x{h[1]}, needs {w[0]}x{w[1]}" for c, (h, w) in wrong.items())
            raise ValueError(f"{who} camera size mismatch — {desc}")
        if self.state_dim is not None and len(obs.joints.positions) != self.state_dim:
            raise ValueError(
                f"{who} expects a {self.state_dim}-dim state, the robot reports {len(obs.joints.positions)} joints "
                f"{list(obs.joints.names)}"
            )
