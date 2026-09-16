"""Registry-declared tool frame for a robot whose model ships no site.

``move_to`` drives the frame :func:`~strands_robots.simulation.ik.discover_ee_frame`
finds: a tool-point site first, else a hand/wrist body. The SO-100 in the
Menagerie (``trs_so_arm100``) has **no sites**, so discovery settled on the
``Wrist_Pitch_Roll`` body - about 16 cm short of the jaw tips - and every low
target the README's first robot was sent to was "reached" by the wrist while
the fingers hung in the air (0/3 in the v0.5.2 devx replay). The SO-101 model
ships a ``gripper`` site at the jaw and the same prompt lands 4/4.

The model files are vendored from upstream and are not edited; the registry
is ours. A robot entry may therefore declare the tool point the model lacks:

.. code-block:: json

    "tool_frame": {"body": "Fixed_Jaw", "pos": [0.0, -0.0995, 0.001], "site": "tcp"}

``body`` is the model's own (un-namespaced) body name, ``pos`` is the tool
point in that body's frame in meters, ``site`` is the name of the site to
create (default ``"tcp"``, which :data:`~strands_robots.simulation.ik._SITE_SEARCH_HINTS`
ranks above every body hint). The backend adds the site to the robot's spec
before attaching it, so it is namespaced with the rest of the robot and
discovery, ``get_robot_state``'s ``end_effector`` line and ``move_to`` all
follow it. A model that already has a tool site needs no declaration.

Shape-checked here, once, the way the sibling ``gripper`` block is: a
malformed block (possible via the user overlay ``user_robots.json``) is a
loud refusal of ``add_robot``, never a silent fall-back to the wrist.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from strands_robots.registry.robots import get_robot
from strands_robots.utils import refusal_repr

DEFAULT_TOOL_SITE = "tcp"
"""Site name used when a ``tool_frame`` block names none."""

__all__ = ["DEFAULT_TOOL_SITE", "ToolFrame", "ToolFrameRefused", "registry_tool_frame", "tool_frame_from_block"]


class ToolFrameRefused(ValueError):
    """A declared tool frame the loaded model cannot carry (unknown body, taken site name)."""


@dataclass(frozen=True)
class ToolFrame:
    """A tool point to add to a robot model: ``site`` at ``pos`` in ``body``."""

    body: str
    pos: tuple[float, float, float]
    site: str = DEFAULT_TOOL_SITE


def tool_frame_from_block(data_config: str, block: Any) -> tuple[ToolFrame | None, str | None]:
    """Shape-check a registry ``tool_frame`` block.

    Returns ``(frame, None)`` for a well-formed block and ``(None, reason)``
    otherwise. The reason names the entry and spells the expected shape so the
    fix is one edit away.
    """
    expected = (
        "Expected {'body': <model body name>, 'pos': [x, y, z] meters in that body's frame, "
        f"'site': <name, default '{DEFAULT_TOOL_SITE}'>}}."
    )
    # ``refusal_repr``, not ``{block!r}``: this text IS the answer to an
    # unusable block, so building it must not be able to raise - the package
    # rule for refusal text, since a value a guard has already refused may
    # raise from its own ``__repr__``.
    prefix = f"registry tool_frame for data_config '{data_config}' is malformed: {refusal_repr(block)}. "
    if not isinstance(block, dict):
        return None, prefix + expected
    unknown = sorted(set(block) - {"body", "pos", "site"})
    if unknown:
        return None, prefix + f"Unknown key(s) {unknown}. " + expected
    body = block.get("body")
    if not isinstance(body, str) or not body:
        return None, prefix + "'body' must be a non-empty body name. " + expected
    pos = block.get("pos")
    if (
        not isinstance(pos, (list, tuple))
        or len(pos) != 3
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pos)
    ):
        return None, prefix + "'pos' must be three finite numbers. " + expected
    site = block.get("site", DEFAULT_TOOL_SITE)
    if not isinstance(site, str) or not site or "/" in site:
        return None, prefix + "'site' must be a non-empty name without '/'. " + expected
    return ToolFrame(body=body, pos=(float(pos[0]), float(pos[1]), float(pos[2])), site=site), None


def registry_tool_frame(
    data_config: str | None,
    lookup: Callable[[str], dict[str, Any] | None] = get_robot,
) -> tuple[ToolFrame | None, str | None]:
    """The ``tool_frame`` a registry entry declares for *data_config*, if any.

    ``(None, None)`` when there is no entry or the entry declares no tool
    frame - the model's own sites and bodies decide, as before.
    ``(None, reason)`` when a block exists but is malformed.
    """
    if not data_config:
        return None, None
    info = lookup(data_config)
    if not info or "tool_frame" not in info:
        return None, None
    return tool_frame_from_block(data_config, info["tool_frame"])
