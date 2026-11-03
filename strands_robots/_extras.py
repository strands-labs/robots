"""One place that says how to install an optional extra — so every hint agrees and can be corrected once.

The PyPI name ``strands-robots`` currently points at the previous library (0.5.x), so until this package is published
under its own name the honest line is the editable install from this repository.
"""

from __future__ import annotations

PACKAGE = "strands-robots"
REPO_HINT = 'uv pip install -e ".[{extra}]"'
# LeRobot's remote_inference is not on PyPI yet, and a wheel cannot depend on a git commit, so it is its own line.
# One line for every LeRobot feature: policies (remote), the SO-101 bus (feetech) and record= (dataset).
LEROBOT = 'pip install "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@53feaef37b67e8f4c23b6953e880ff535e39d6fe"'


def hint(extra: str) -> str:
    """``pip install 'strands-robots[lerobot]'  (from a checkout: uv pip install -e ".[lerobot]")``."""
    line = f"pip install '{PACKAGE}[{extra}]'  (from a checkout: {REPO_HINT.format(extra=extra)})"
    return f"{line}, then {LEROBOT}" if extra == "lerobot" else line
