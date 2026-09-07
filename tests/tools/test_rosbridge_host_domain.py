"""The host half of a rosbridge address is graded before anything spends it.

Both rosbridge surfaces interpolate a caller-supplied host and port into one
websocket address, ``ws://<host>:<port>``. The port half is graded twice: by
:func:`~strands_robots.utils.tcp_port_error`, the domain every port in the
package shares, and then by the transport's own narrower ceiling. The host half
beside it went straight to the transport's allowlist pattern, and a pattern can
only be offered a string - so every non-string host raised ``TypeError`` out of
``re``, out of a tool whose every other refusal is a result dict and out of a
constructor documented to report a malformed host as ``ValueError``.

These pins hold the two halves against each other: an unusable value is
*reported* by whichever half it was given to, the shared domain refuses no host
the transport admits, and the transport keeps the narrower allowlist that is
its own posture rather than the domain's.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import itertools
import string
from typing import Any

import pytest

from strands_robots.mesh.rosbridge_robot import RosbridgeRobot
from strands_robots.tools.use_rosbridge import use_rosbridge
from strands_robots.utils import dial_host_error

# Values a tool call carries verbatim from JSON, an env read or an agent, none of
# which a websocket URI can name a host with.
NON_STRING_HOSTS: list[Any] = [None, 9090, True, 1.5, ["localhost"], b"localhost", {"host": "x"}]

# Hosts the shared domain admits as a string a URI can carry, and that this
# transport's allowlist narrows anyway. The narrowing is the transport's own
# posture - the same reason its port ceiling sits beside it and not in the shared
# domain - so it is pinned as a deliberate asymmetry, not repaired.
ADMITTED_BY_DOMAIN_NARROWED_BY_TRANSPORT = ["[::1]", "host!", "*", "h\u00e9llo", "host%2f"]

# Values neither half of the address can carry. ``9090`` is deliberately absent:
# it is a legal port and not a host, which is the asymmetry itself - one value,
# two answers, decided by which half of ``ws://<host>:<port>`` it was given to.
UNUSABLE_FOR_EITHER_HALF: list[Any] = [None, True, 1.5, ["localhost"], b"localhost", {"host": "x"}]


def _text(result: dict[str, Any]) -> str:
    return " ".join(part.get("text", "") for part in result["content"])


def _tool_host_verdict(host: Any) -> str | None:
    """Return the tool's refusal text for ``host``, or ``None`` when admitted.

    Read through the public tool with an action it does not offer: the host is
    graded ahead of the action, so an admitted host reports the unknown action
    and no host is ever dialled - the verdict is observed without a backend and
    without reaching into the module's private pattern.
    """
    result = use_rosbridge(action="__no_such_action__", host=host)
    assert result["status"] == "error", result
    text = _text(result)
    return None if "unknown action" in text else text


def _constructor_host_verdict(host: Any) -> str | None:
    """Return ``RosbridgeRobot``'s refusal text for ``host``, or ``None``."""
    try:
        RosbridgeRobot("node", "/cmd_vel", "/odom", host=host)
    except ValueError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize("host", NON_STRING_HOSTS)
def test_a_non_string_host_is_reported_as_a_result_not_raised(host: Any) -> None:
    """The tool reports it, rather than raising ``re``'s TypeError past dispatch."""
    verdict = _tool_host_verdict(host)
    assert verdict is not None
    assert "host" in verdict


@pytest.mark.parametrize("host", NON_STRING_HOSTS)
def test_a_non_string_host_is_the_valueerror_the_constructor_documents(host: Any) -> None:
    """``RosbridgeRobot`` documents a malformed host as ``ValueError``; a
    non-string one used to leave as ``TypeError`` instead."""
    verdict = _constructor_host_verdict(host)
    assert verdict is not None
    assert "host" in verdict


@pytest.mark.parametrize("value", UNUSABLE_FOR_EITHER_HALF)
def test_both_halves_of_one_address_report_the_same_unusable_value(value: Any) -> None:
    """Host and port are interpolated into one address, so a value neither can
    carry is reported the same way whichever half it is given to."""
    host_half = use_rosbridge(action="__no_such_action__", host=value)
    port_half = use_rosbridge(action="__no_such_action__", port=value)
    assert host_half["status"] == "error"
    assert port_half["status"] == "error"
    assert "host" in _text(host_half)
    assert "port" in _text(port_half)


def test_the_shared_domain_refuses_no_host_this_transport_admits() -> None:
    """Consolidating onto the shared domain narrows nothing: every hostname the
    transport allowlist admits is a hostname the domain admits too."""
    alphabet = string.ascii_letters + string.digits + "._-"
    probes = list(alphabet)
    probes += ["".join(pair) for pair in itertools.product(alphabet, repeat=2)]
    probes += ["localhost", "0.0.0.0", "127.0.0.1", "my-robot.local", "a" * 63]
    admitted = [probe for probe in probes if _tool_host_verdict(probe) is None]
    assert len(admitted) > 4000, "the allowlist probe set collapsed; the oracle is not reading it"
    refused_by_domain = [probe for probe in admitted if dial_host_error(probe, "host", "x") is not None]
    assert refused_by_domain == []


@pytest.mark.parametrize("host", ADMITTED_BY_DOMAIN_NARROWED_BY_TRANSPORT)
def test_the_transport_keeps_its_own_narrower_host_allowlist(host: Any) -> None:
    """The allowlist is this transport's posture, not the shared domain's: it
    still refuses hostnames a websocket URI could carry."""
    assert dial_host_error(host, "host", "x") is None, "probe no longer isolates the transport's narrowing"
    assert _tool_host_verdict(host) is not None
    assert _constructor_host_verdict(host) is not None


def test_an_empty_host_is_still_refused_by_the_domain_that_replaced_the_falsiness_check() -> None:
    """A truthiness test used to catch ``""`` ahead of the pattern; the shared
    domain owns it now, and names the address it cannot build."""
    for verdict in (_tool_host_verdict(""), _constructor_host_verdict("")):
        assert verdict is not None
        assert "host" in verdict


@pytest.mark.parametrize(
    ("module_name", "owner"),
    [
        ("strands_robots.tools.use_rosbridge", "use_rosbridge"),
        ("strands_robots.mesh.rosbridge_robot", "__init__"),
    ],
)
def test_both_surfaces_read_the_shared_host_domain(module_name: str, owner: str) -> None:
    """Graded on the call graph rather than the source text: a domain named but
    not called - in a table, say - reads as consolidated while grading nothing."""
    module = importlib.import_module(module_name)
    tree = ast.parse(inspect.getsource(module))
    owners = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == owner]
    assert owners, f"{owner} not found in {module_name}"
    called = {
        node.func.id for node in ast.walk(owners[0]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "dial_host_error" in called, f"{module_name}:{owner} does not read the shared host domain"
