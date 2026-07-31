"""Shared numeric-option guard for the tools that drive a ROS graph.

:mod:`~strands_robots.tools.use_ros`, :mod:`~strands_robots.tools.use_rtps` and
:mod:`~strands_robots.tools.use_rosbridge` reach a ROS graph over three
different transports - in-process rclpy, raw RTPS, and a rosbridge WebSocket -
but each exposes an agent the same three numeric options and consumes them the
same way: ``count`` is a ``range()`` bound, ``rate`` becomes an inter-message
period ``1 / rate``, and ``timeout`` is a wait budget. Only a positive finite
value can be honored by any of them, so this module is the single owner of that
rule: the accepted domain cannot differ between two transports onto the same
graph, where the same publish would otherwise be refused by one and silently
mis-paced by the other.

Each tool keeps its own per-action table of which options an action reads,
because that is a property of the transport rather than of the domain: every
rosbridge action is a timeout-bounded WebSocket round trip, while rclpy graph
introspection reads no caller budget at all. Passing the table in keeps the
scoping decision - and the "never refuse a value this action ignores" contract
it encodes - beside the dispatch that owns it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from strands_robots.utils import positive_count_error, positive_finite_number_error

# The domain each option is checked against, in the order errors are reported.
# ``count`` is a discrete ``range()`` bound, so only a true ``int`` can be
# honored; ``timeout`` and ``rate`` are a span of seconds and a frequency in Hz,
# where a fractional value is perfectly usable.
_OPTION_DOMAINS: tuple[tuple[str, Callable[[Any, str, str], str | None]], ...] = (
    ("timeout", positive_finite_number_error),
    ("count", positive_count_error),
    ("rate", positive_finite_number_error),
)


def numeric_option_error(
    action: str,
    options_by_action: dict[str, tuple[str, ...]],
    *,
    timeout: Any,
    count: Any,
    rate: Any,
) -> str | None:
    """Error text for the first numeric option ``action`` consumes but cannot honor.

    ``timeout``, ``count`` and ``rate`` are agent-supplied, so each is checked
    against the shared domain for its kind before any transport entity is
    created: ``count`` against
    :func:`~strands_robots.utils.positive_count_error` and ``timeout`` / ``rate``
    against :func:`~strands_robots.utils.positive_finite_number_error`.

    Only the options ``action`` actually reads are checked. A caller must never
    be refused for a value the requested action never looks at, so the scoping
    is driven by ``options_by_action`` rather than by validating the whole
    signature unconditionally.

    Args:
        action: The requested action; decides which options are effective.
        options_by_action: The calling tool's map of action name -> the option
            names that action consumes. An action absent from the map reads
            none of them and is never refused here.
        timeout: Seconds to wait, as supplied.
        count: Message/sample count, as supplied.
        rate: Publish rate in Hz, as supplied.

    Returns:
        An error message naming the action and the option, or ``None`` when
        every option this action reads is usable.
    """
    consumed = options_by_action.get(action, ())
    supplied: dict[str, Any] = {"timeout": timeout, "count": count, "rate": rate}
    for param, check in _OPTION_DOMAINS:
        if param in consumed:
            error = check(supplied[param], param, action)
            if error:
                return error
    return None
