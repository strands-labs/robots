"""The edge between a strands robot and a LeRobot deployment: names, units, cameras, one frame."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._extras import hint
from ...description import Description
from ...types import Observation
from ...units import Units, lerobot_units, semantics_for, units_for_semantics
from ...units.semantics import table
from ..requirements import Requirements
from .quiet import quiet_load

if TYPE_CHECKING:
    from lerobot.remote_inference.configs import ServerConfig

STATE, ACTION, IMAGES = "observation.state", "action", "observation.images."


def lerobot() -> Any:
    """Import lerobot (with its remote extra) or fail with the one install line."""
    try:
        import lerobot as lr
        import lerobot.remote_inference  # noqa: F401  (zenoh + msgpack)
    except ImportError as e:
        raise ImportError(f"LeRobot policies need the lerobot extra: {hint('lerobot')} ({e})") from e
    return lr


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass(frozen=True)
class Contract:
    """A deployment's features read against one robot: the index maps, the cameras and the units."""

    description: Description
    features: tuple[Any, ...]  # lerobot FeatureSpec, server order
    action_feature: Any
    semantics: str

    @property
    def state_names(self) -> tuple[str, ...]:
        return next((f.names for f in self.features if f.name == STATE), ())

    @property
    def cameras(self) -> dict[str, tuple[int, int]]:
        """Camera name -> (width, height), from every ``observation.images.<cam>`` (HWC)."""
        return {f.name[len(IMAGES) :]: (f.shape[1], f.shape[0]) for f in self.features if f.name.startswith(IMAGES)}

    def _index(self, names: Sequence[str], what: str) -> np.ndarray:
        keys = self.description.action_keys
        idx = []
        for n in names:
            key = n[: -len(".pos")] if n.endswith(".pos") else n
            if key not in keys:
                raise ValueError(
                    f"deployment {what} component {n!r} is not a joint of {self.description.name} "
                    f"(action keys: {list(keys)}); a LeRobot name is '<action key>.pos'"
                )
            idx.append(keys.index(key))
        return np.array(idx, dtype=int)

    def check(self) -> None:
        """Every state/action component names one of this robot's joints, and the action covers them all."""
        self._index(self.state_names, "state")
        a = self._index(self.action_feature.names, "action")
        if sorted(a.tolist()) != list(range(len(self.description.action_keys))):
            raise ValueError(
                f"deployment actions {list(self.action_feature.names)} do not drive every joint of "
                f"{self.description.name} {list(self.description.action_keys)}"
            )

    def units(self) -> Units | None:
        return units_for_semantics(self.semantics, self.description)

    def requirements(self) -> Requirements:
        return Requirements(images=self.cameras, state_dim=None)

    def frame(self, obs: Observation) -> dict[str, np.ndarray]:
        """One LeRobot frame from an observation already in the policy's units (state in server order)."""
        out: dict[str, np.ndarray] = {}
        if self.state_names:
            out[STATE] = np.asarray(obs.joints.positions, dtype=np.float32)[self._index(self.state_names, "state")]
        for cam in self.cameras:
            out[IMAGES + cam] = np.ascontiguousarray(obs.images[cam], dtype=np.uint8)
        return out

    def to_keys(self, action: Any) -> np.ndarray:
        """A canonical LeRobot action (server order) -> this robot's action-key order."""
        a = np.asarray(action, dtype=float).reshape(-1)
        out = np.empty(len(self.description.action_keys))
        out[self._index(self.action_feature.names, "action")] = a
        return out

    def to_keys_chunk(self, chunk: Any) -> np.ndarray:
        c = np.asarray(chunk, dtype=float).reshape(-1, len(self.action_feature.names))
        out = np.empty((len(c), len(self.description.action_keys)))
        out[:, self._index(self.action_feature.names, "action")] = c
        return out


# What LeRobot's ``_normalize`` can emit per mode: range modes are clamped (exact), degrees is one motor turn.
_CAN_EMIT = {"range_m100_100": (-100.0, 100.0), "range_0_100": (0.0, 100.0), "degrees": (-180.0, 180.0)}


def _misfit(units: Units, stats: Mapping[str, np.ndarray], graded: set[str]) -> str | None:
    """The first saved min/max of a ``graded`` joint that LeRobot's normalizer cannot emit in its mode, or None."""
    for feat in (STATE, ACTION):
        mn, mx = stats.get(f"{feat}.min"), stats.get(f"{feat}.max")
        if mn is None or mx is None or len(mn) != len(units.keys):
            continue
        for i, (k, mode) in enumerate(zip(units.keys, units.modes, strict=False)):
            if k not in graded:
                continue
            lo, hi = _CAN_EMIT.get(mode, (-np.inf, np.inf))
            if mn[i] < lo - 1e-3 or mx[i] > hi + 1e-3:
                return f"{feat} {k} spans {mn[i]:.1f}..{mx[i]:.1f}, impossible in {mode} ({lo:.0f}..{hi:.0f})"
    return None


def check_units(description: Description, artifact: Path, checkpoint: str, semantics: str) -> None:
    """Refuse a checkpoint whose own normalizer stats cannot be in ``semantics``: units are checked, not assumed.

    A LeRobot checkpoint saves its dataset's per-joint min/max (``policy_preprocessor_*normalizer*.safetensors``).
    A degrees dataset read as ``range_m100_100`` drives a joint toward its limit; this names the joint and the
    labels its numbers do fit. Only a joint some row of this robot reads in another mode is graded: a misfit no
    label can change (the gripper, a pre-clamp or sim-recorded excursion) is no evidence. No min/max: not checked.
    """
    files = sorted(artifact.glob("policy_preprocessor*normalizer*.safetensors"))
    units = units_for_semantics(semantics, description)
    if not files or units is None:
        return
    from safetensors.numpy import load_file

    stats = load_file(str(files[0]))
    mine = {k: v["lerobot_norm"] for k, v in table().items() if "lerobot_norm" in v and k != semantics}
    rows = [n for k, n in mine.items() if k.startswith(f"{description.name}-")]
    modes = [dict(zip(u.keys, u.modes, strict=False)) for u in [units, *(lerobot_units(description, n) for n in rows)]]
    graded = {k for k in units.keys if len({m.get(k) for m in modes}) > 1}
    why = _misfit(units, stats, graded)
    if why is None:
        return
    fits = [f"norm={n}" for n in rows if _misfit(lerobot_units(description, n), stats, graded) is None]
    raise ValueError(
        f"{checkpoint} was not trained in {semantics}: its saved stats say {why}. "
        f"Pass the units its data was recorded in ({' or '.join(fits) or 'semantics=<label>'})"
    )


def server_config(
    description: Description,
    checkpoint: str,
    *,
    deployment: str = "strands",
    endpoint: str = "tcp/127.0.0.1:7447",
    device: str | None = None,
    revision: str | None = None,
    semantics: str | None = None,
    norm: Mapping[str, str] | None = None,
    fps: float = 30.0,
    modes: Sequence[str] = ("chunk",),
    n_action_steps: int | None = None,
) -> ServerConfig:
    """A LeRobot ``ServerConfig`` for this robot and checkpoint: nobody hand-writes features again.

    State/action names are ``<action key>.pos`` in the robot's order, cameras are the checkpoint's own
    ``observation.images.*`` at its trained size (HWC), the semantics label comes from ``norm`` (default:
    the registry's ``hardware.lerobot_norm``) through ``units/semantics.json``, and checked against the
    checkpoint's own saved min/max (``check_units``): a checkpoint whose numbers cannot be in those units is refused.
    """
    lerobot()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.inference import FeatureSpec
    from lerobot.remote_inference.configs import ExecutionConfig, ModelConfig, ServerConfig
    from lerobot.remote_inference.deployment import resolve_artifact
    from lerobot.transport import ZenohConfig

    artifact = resolve_artifact(checkpoint, revision)
    with quiet_load():  # only its features are read here; ModelConfig below carries the device
        cfg = PreTrainedConfig.from_pretrained(artifact)
    sem = semantics or semantics_for(description, dict(norm) if norm is not None else None)
    check_units(description, artifact, checkpoint, sem)
    names = tuple(f"{k}.pos" for k in description.action_keys)
    feats = []
    for key, ft in cfg.input_features.items():
        shape = tuple(ft.shape)
        if key == STATE:
            if shape != (len(names),):
                raise ValueError(f"{checkpoint} state is {shape}; {description.name} has {len(names)} joints {names}")
            feats.append(FeatureSpec(STATE, shape, "float32", names=names, semantics=sem))
        elif key.startswith(IMAGES):
            c, h, w = shape
            feats.append(FeatureSpec(key, (h, w, c), "uint8", kind="rgb", semantics="rgb-v1"))
        else:
            raise ValueError(f"{checkpoint} needs input {key!r}, which a strands robot does not produce yet")
    if tuple(cfg.output_features[ACTION].shape) != (len(names),):
        raise ValueError(
            f"{checkpoint} acts on {tuple(cfg.output_features[ACTION].shape)}; {description.name}: {names}"
        )
    return ServerConfig(
        deployment=deployment,
        model=ModelConfig(repo_or_path=checkpoint, revision=revision, device=device or default_device()),
        execution=ExecutionConfig(supported_modes=list(modes), action_fps=fps, n_action_steps=n_action_steps),
        zenoh=ZenohConfig(listen_endpoints=[endpoint]),
        semantics=sem,
        robot_type=description.hardware.lerobot_type or description.name,
        features=feats,
        action_feature=FeatureSpec(ACTION, (len(names),), "float32", names=names, semantics=sem),
    )


def write_server_yaml(cfg: ServerConfig, path: str | Path) -> Path:
    """``lerobot-policy-server --config_path=<path>`` serves exactly this config (plain YAML, no tags)."""
    import dataclasses
    import enum

    import yaml

    def plain(v: Any) -> Any:
        if isinstance(v, enum.Enum):
            return v.value
        if isinstance(v, Mapping):
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v

    data = plain(dataclasses.asdict(cfg))
    if data["execution"]["supported_modes"] == ["chunk"]:
        data["execution"].pop("rtc", None)  # LeRobot's defaults; only an RTC deployment sets them
    p = Path(path)
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p
