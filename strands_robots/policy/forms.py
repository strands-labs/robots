"""What a policy spec string looks like: the form tests ``Policy(spec)`` and the Hub allowlist both apply."""

from __future__ import annotations

import re
from pathlib import Path

REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
PRESET = re.compile(r"^[a-z][a-z0-9_]*(\?[^\s/]*)?$")


def is_preset(spec: str) -> bool:
    if not PRESET.match(spec):
        return False
    from .onnx import preset_names

    return spec.partition("?")[0] in preset_names()


def is_dataset(local: Path) -> bool:
    return (local / "meta" / "info.json").is_file()


def is_checkpoint(local: Path) -> bool:
    return (local / "config.json").is_file()
