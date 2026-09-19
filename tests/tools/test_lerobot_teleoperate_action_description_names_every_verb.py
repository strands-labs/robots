"""The ``action`` parameter's description names every verb the dispatcher answers.

The Args entry for ``action`` is what strands publishes as that parameter's
description in ``tool_spec``, so it is the only list of verbs an agent ever
sees. ``dagger`` was dispatched (``action == "dagger"``) and documented in the
Actions block, but the Args entry still read ``(start, stop, list, status,
replay)``, so an agent reading the spec could never learn the verb existed.
"""

from __future__ import annotations

import importlib
import inspect
import re

from strands_robots.tools.lerobot_teleoperate import lerobot_teleoperate

# The package's lazy ``__getattr__`` hands back the tool for this name, so the
# module itself has to be imported by path.
_module = importlib.import_module("strands_robots.tools.lerobot_teleoperate")


def _dispatched_actions() -> set[str]:
    source = inspect.getsource(_module)
    return set(re.findall(r'action == "([a-z_]+)"', source))


def test_action_description_names_every_dispatched_verb() -> None:
    dispatched = _dispatched_actions()
    assert "dagger" in dispatched, "the dispatcher no longer answers dagger; update this test's premise"
    description = lerobot_teleoperate.tool_spec["inputSchema"]["json"]["properties"]["action"]["description"]
    missing = sorted(verb for verb in dispatched if not re.search(rf"\b{verb}\b", description))
    assert not missing, f"action description {description!r} omits dispatched verb(s) {missing}"
