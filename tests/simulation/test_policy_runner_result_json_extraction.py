"""Contract for the send_action result-JSON parser used by fail-fast accounting.

When a policy emits action keys the robot cannot resolve, MuJoCo's
``send_action`` reports the per-key breakdown as an ``{"json": {...}}`` content
block carrying ``unresolved_keys`` / ``applied``. ``PolicyRunner`` reads that
block through :func:`strands_robots.simulation.policy_runner._extract_result_json`
to decide, on the error path, whether a step drove zero actuators (which feeds
the 100%-unresolved fail-fast probe).

The parser is deliberately defensive: a backend that is not MuJoCo, or a coarse
error without a per-key breakdown, returns a result that carries no structured
``json`` block -- and a hypothetical backend could even return a non-dict. In
every such case the parser must return ``None`` so the runner falls back to
treating the whole step as unresolved rather than crashing while trying to read
a breakdown that isn't there. The public ``run_policy`` surface can't force a
non-dict result (``send_action`` always returns a dict), so this pins the
parser's input->output contract directly, including that defensive branch.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.simulation.policy_runner import _extract_result_json


@pytest.mark.parametrize(
    "result",
    [
        pytest.param("boom", id="non-dict-str"),
        pytest.param(None, id="non-dict-none"),
        pytest.param(42, id="non-dict-int"),
        pytest.param([{"json": {"unresolved_keys": ["a"]}}], id="non-dict-list-even-with-json"),
    ],
)
def test_non_dict_result_yields_none(result: Any) -> None:
    # A backend that returns something other than a status dict must not crash
    # the parser: it degrades to None so the caller treats the step coarsely.
    assert _extract_result_json(result) is None


@pytest.mark.parametrize(
    "result",
    [
        pytest.param({"status": "error"}, id="no-content-key"),
        pytest.param({"content": None}, id="content-none"),
        pytest.param({"content": []}, id="content-empty"),
        pytest.param({"content": [{"text": "coarse error, no per-key breakdown"}]}, id="text-block-only"),
        pytest.param({"content": ["not-a-dict", 7]}, id="non-dict-blocks"),
        pytest.param({"content": [{"json": "not-a-dict"}]}, id="json-value-not-dict"),
    ],
)
def test_dict_without_json_block_yields_none(result: dict[str, Any]) -> None:
    # A coarse error / non-MuJoCo backend carries no structured json block, so
    # the per-key breakdown is unavailable and the parser returns None.
    assert _extract_result_json(result) is None


def test_returns_json_payload_when_present() -> None:
    payload = {"unresolved_keys": ["wrist"], "applied": ["shoulder"]}
    result = {"status": "error", "content": [{"text": "unresolved"}, {"json": payload}]}
    assert _extract_result_json(result) == payload


def test_first_json_block_wins() -> None:
    # Documented "first structured block" contract: later json blocks are ignored.
    result = {"content": [{"json": {"k": 1}}, {"json": {"k": 2}}]}
    assert _extract_result_json(result) == {"k": 1}
