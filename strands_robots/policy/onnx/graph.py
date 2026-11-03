"""``OnnxPolicy``: one ONNX graph on ``onnxruntime`` (CPU), plus the base-frame math the locomotion presets share.

The graph's ``custom_metadata_map`` is exported next to it as ``<name>.json`` so a reader without a metadata API
(onnxruntime-web) sees the same layout. ``onnxruntime`` and ``huggingface_hub`` are imported only when a graph is
loaded, never at ``import strands_robots``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..._extras import hint
from ...types import Action, Observation
from ..base import Policy

_GRAVITY = np.array([0.0, 0.0, -1.0])
CPU = ("CPUExecutionProvider",)


def quat_rotate_inverse(quat_wxyz: Any, vec: Any) -> np.ndarray:
    """Express a world-frame vector in the body frame of a (w, x, y, z) quaternion (normalised first)."""
    q = np.asarray(quat_wxyz, dtype=float).reshape(-1)
    v = np.asarray(vec, dtype=float).reshape(-1)
    if q.shape != (4,) or v.shape != (3,):
        raise ValueError(f"quat_rotate_inverse: quat must be (4,) and vec (3,), got {q.shape} and {v.shape}")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("quat_rotate_inverse: quaternion has ~zero norm and describes no rotation")
    unit = q / norm
    w, xyz = unit[0], unit[1:]
    t = np.cross(xyz, v) * 2.0
    return np.asarray(v - w * t + np.cross(xyz, t))


def projected_gravity(quat_wxyz: Any) -> np.ndarray:
    """World -Z in the body frame: ``[0, 0, -1]`` when upright."""
    return quat_rotate_inverse(quat_wxyz, _GRAVITY)


def base_twist(obs: Observation, who: str) -> tuple[np.ndarray, np.ndarray]:
    """(angular_velocity, orientation) of the base, or a refusal that names the body's gap."""
    if obs.pose is None or obs.pose.angular_velocity is None:
        raise ValueError(
            f"{who} needs the base twist: obs.pose.angular_velocity is None (a floating body in a world reports it; "
            "a fixed base or a servo bus does not)"
        )
    return np.asarray(obs.pose.angular_velocity, dtype=float), np.asarray(obs.pose.orientation, dtype=float)


def resolve_weights(source: str | Path, *, repo: str | None = None, sha256: str | None = None) -> Path:
    """A local ``.onnx`` path as is; a bare file name with a ``repo`` always comes from the Hub. ``sha256`` is checked
    on whichever file is returned."""
    p = Path(source).expanduser()
    if repo is None or "/" in str(source):
        if not p.is_file():
            raise FileNotFoundError(f"ONNX file not found: {source} (pass repo='owner/name' to fetch a bare file name)")
        path = p
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(f"huggingface_hub is not installed: {hint('onnx')}") from exc
        path = Path(hf_hub_download(repo, str(source)))
    if sha256 is not None:
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha256:
            raise OSError(f"{path} hashes to {got}, expected {sha256}: refusing to run unknown weights")
    return path


def _session(path: str | Path, providers: Sequence[str]) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(f"onnxruntime is not installed: {hint('onnx')}") from exc
    return ort.InferenceSession(str(path), providers=list(providers))


def read_metadata(path: str | Path) -> dict[str, str]:
    """The graph's ``custom_metadata_map`` (what the presets are configured from)."""
    return dict(_session(path, CPU).get_modelmeta().custom_metadata_map)


def write_sidecar(path: Path, session: Any) -> Path:
    """``<file>.onnx.json`` next to the graph: inputs, outputs and metadata, for readers without a metadata API.

    The name cannot collide with a ``<file>.json`` the user exported, and a file that is not ours is never written."""
    side = path.with_name(path.name + ".json")
    doc = {
        "file": path.name,
        "inputs": [{"name": i.name, "shape": list(i.shape)} for i in session.get_inputs()],
        "outputs": [{"name": o.name, "shape": list(o.shape)} for o in session.get_outputs()],
        "metadata": dict(session.get_modelmeta().custom_metadata_map),
    }
    text = json.dumps(doc, indent=1)
    if not side.exists() or (not side.is_symlink() and side.read_text() != text):
        try:
            side.write_text(text)
        except OSError:  # a read-only cache is fine; the sidecar is a convenience
            pass
    return side


class OnnxPolicy(Policy):
    """One ONNX graph, one input, one output. ``act`` feeds ``joints.positions`` unless a subclass lays out more."""

    hz: float = 50.0
    units = "rad"

    def __init__(
        self,
        source: str | Path,
        *,
        repo: str | None = None,
        sha256: str | None = None,
        providers: Sequence[str] = CPU,
    ) -> None:
        if isinstance(providers, str) or not isinstance(providers, Sequence):
            raise TypeError("providers must be a sequence of onnxruntime provider names, e.g. ['CPUExecutionProvider']")
        self.path = resolve_weights(source, repo=repo, sha256=sha256)
        self.session = _session(self.path, providers)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"{self.path.name}: expected one input and one output, got {len(inputs)}/{len(outputs)}")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        shape = inputs[0].shape
        self.input_width = int(shape[-1]) if shape and isinstance(shape[-1], int) else -1
        self.metadata: dict[str, str] = dict(self.session.get_modelmeta().custom_metadata_map)
        self.sidecar = write_sidecar(self.path, self.session)

    def infer(self, vec: np.ndarray) -> np.ndarray:
        """Run the graph on one observation row; returns the output row as float32."""
        x = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        if self.input_width > 0 and x.shape[1] != self.input_width:
            raise ValueError(f"{self.path.name} takes {self.input_width} inputs, observation has {x.shape[1]}")
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def observation(self, obs: Observation) -> np.ndarray:
        return obs.joints.positions.astype(np.float32)

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        return self.infer(self.observation(obs)).astype(float)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path.name!r})"
