"""Is the thing training just called a checkpoint actually a policy on disk?"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = [
    "CONFIG_NAMES",
    "WEIGHT_SUFFIXES",
    "WEIGHT_NAMES",
    "MIN_WEIGHT_BYTES",
    "artifact_verdict",
]

#: What a policy directory calls its config. HF-native backends write one of these; the
#: trainer's own discovery keys off exactly this, which is why it can be alone.
CONFIG_NAMES = ("config.json", "train_config.json")

#: Weight file extensions across the backends this dashboard can train with: safetensors
#: (LeRobot, GR00T, converted Cosmos), torch pickles (RL algos), and DCP shards.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".distcp")

#: Names that are weights even without a telling suffix.
WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors")

#: Under this, a weight file is not weights: it is the first block of a write that never
#: finished. A real policy head is megabytes; 4KB is one filesystem block.
MIN_WEIGHT_BYTES = 4096


def _is_file(p: Path) -> bool:
    """``Path.is_file`` that cannot raise."""
    try:
        return p.is_file()
    except OSError:
        return False


def _is_weight(p: Path) -> bool:
    return p.name in WEIGHT_NAMES or p.suffix.lower() in WEIGHT_SUFFIXES


def _weights(directory: Path) -> list[Path]:
    """Weight files in this directory, plus one level down."""
    found: list[Path] = []
    try:
        for entry in sorted(directory.iterdir()):
            if _is_file(entry) and _is_weight(entry):
                found.append(entry)
            elif entry.is_dir():
                try:
                    found.extend(c for c in sorted(entry.iterdir()) if _is_file(c) and _is_weight(c))
                except OSError:
                    continue
    except OSError:
        return []
    return found


def artifact_verdict(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    """What can be said about an exported artifact WITHOUT loading it. Returns ``{"ok": bool, "path":
    str, ...}``.
    """
    if not path or not str(path).strip():
        return {
            "ok": False,
            "path": "",
            "reason": "empty_path",
            "message": (
                "the trainer reported success but named no artifact path, so there is "
                "nothing to deploy. Check the training log for the export step."
            ),
        }

    p = Path(str(path)).expanduser()
    out: dict[str, Any] = {"ok": False, "path": str(p)}

    if not p.exists():
        out["reason"] = "missing"
        out["message"] = (
            f"nothing exists at {p}. An output directory on a removable or network volume "
            "that is no longer mounted looks exactly like this, as does a run whose output "
            "was cleaned up. Re-export from the job, or check the volume is mounted."
        )
        return out

    if _is_file(p):
        # A single-file artifact (a converted safetensors) is legitimate.
        size = p.stat().st_size
        if _is_weight(p) and size >= MIN_WEIGHT_BYTES:
            return {
                "ok": True,
                "path": str(p),
                "kind": "file",
                "weight_bytes": size,
                "note": "checked on disk only - nothing here loads the model",
            }
        out["reason"] = "truncated" if _is_weight(p) else "not_a_policy"
        out["message"] = (
            f"{p.name} is {size} bytes, which is a write that never finished, not a policy."
            if _is_weight(p)
            else f"{p} is a file, but not a weights file the policy loader accepts."
        )
        return out

    configs = [c for c in CONFIG_NAMES if _is_file(p / c)]
    weights = _weights(p)
    big = [w for w in weights if w.stat().st_size >= MIN_WEIGHT_BYTES]

    if not configs and not weights:
        out["reason"] = "empty"
        out["message"] = (
            f"{p} exists but holds no policy config and no weights. That is what an output "
            "directory looks like before the first checkpoint is saved: the run may have "
            "died in its first epoch. The training log will say."
        )
        return out

    if configs and not weights:
        out["reason"] = "config_without_weights"
        out["configs"] = configs
        out["message"] = (
            f"{p} has {configs[0]} but no weights file. A checkpoint directory is discovered "
            "BY ITS CONFIG, so a run killed between writing the config and writing "
            "model.safetensors - a crash, an OOM, a full disk, a closed lid - exports and "
            "deploys as if it were finished. This one would fail when the policy is loaded "
            "on the robot. Re-export, or re-run from the last complete checkpoint."
        )
        return out

    if weights and not big:
        out["reason"] = "truncated"
        out["weights"] = [w.name for w in weights]
        out["weight_bytes"] = max((w.stat().st_size for w in weights), default=0)
        out["message"] = (
            f"the weights in {p} are {out['weight_bytes']} bytes - a policy head is megabytes, "
            "so this is the first block of a write that never finished (a full disk or a "
            "killed process). Re-export; if it repeats, check free space on that volume."
        )
        return out

    verdict: dict[str, Any] = {
        "ok": True,
        "path": str(p),
        "kind": "dir",
        "weights": [w.name for w in big],
        "weight_bytes": sum(w.stat().st_size for w in big),
        "note": "checked on disk only - nothing here loads the model",
    }
    if not configs:
        # Weights with no config: some loaders infer the architecture, so this is not a refusal - but
        # it is worth saying, because it is also what a half-written checkpoint looks like from the
        # other side.
        verdict["warning"] = (
            f"{p} has weights but no {CONFIG_NAMES[0]}/{CONFIG_NAMES[1]} - the policy family "
            "may have to be chosen by hand when this is run."
        )
    return verdict
