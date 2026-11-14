"""One place that says how to install an optional extra — so every hint agrees and can be corrected once.

The PyPI name ``strands-robots`` currently points at the previous library (0.5.x), so until this package is published
under its own name the honest line is the editable install from this repository.
"""

from __future__ import annotations

PACKAGE = "strands-robots"
REPO_HINT = 'uv pip install -e ".[{extra}]"'


def hint(extra: str) -> str:
    """``pip install 'strands-robots[lerobot]'  (from a checkout: uv pip install -e ".[lerobot]")``."""
    return f"pip install '{PACKAGE}[{extra}]'  (from a checkout: {REPO_HINT.format(extra=extra)})"
