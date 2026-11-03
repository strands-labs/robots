"""A LeRobot without its remote extra is refused before the first step with the pinned install line, not with PyPI's.

LeRobot imports msgpack lazily (``require_package`` at the first encode), so importing ``lerobot.remote_inference``
alone succeeds without it, and LeRobot's own hint (``pip install 'lerobot[remote]'``) would replace a git LeRobot
with the PyPI release that has no remote inference.
"""

from __future__ import annotations

import re
import sys

import pytest

from strands_robots._extras import LEROBOT
from strands_robots.policy.lerobot.contract import lerobot


@pytest.mark.parametrize("missing", ["msgpack", "zenoh"])
def test_a_missing_remote_dependency_names_the_one_install_line(missing, monkeypatch):
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ImportError, match=re.escape(LEROBOT)):
        lerobot()
