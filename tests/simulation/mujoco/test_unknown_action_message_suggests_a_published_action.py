"""An unknown ``action`` is answered with the action the caller probably meant.

``action`` is the one parameter every call to this tool must supply and the only
one with no usable default, so it is where a typo is most likely to land. It was
also the only parameter whose refusal named neither a candidate nor a way to find
one: a misspelled *robot* got ``" Did you mean: ...? Available robots: [...]. Use
action='list_robots' to see all."`` from :func:`close_match_hint`, while a
misspelled *action* one frame away got a bare ``"Unknown action: renderr"``.

Four unknown-entity messages already route through that helper
(``_unknown_model_msg``, ``_unknown_object_msg``, ``_unknown_camera_msg``,
``_unknown_robot_msg``), each with a docstring rejecting the dead-end form for
forcing "an agent driving the API blind into a discovery round-trip on every
typo". This pins the fifth caller, so the parameter most likely to be misspelled
is not the one left without the treatment.

The suggestion corpus is the published enum, never every dispatchable method.
Those differ: the router resolves by ``getattr`` with no allowlist while the two
agent-facing entry points refuse a non-enum name (#2093), so suggesting a
dispatchable-but-unpublished action would answer one refusal with a second. The
control below is what holds that line.
"""

from __future__ import annotations

import asyncio
import re

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.base import close_match_hint  # noqa: E402
from strands_robots.simulation.mujoco.simulation import (  # noqa: E402
    _PUBLISHED_ACTIONS,
    Simulation,
)
from tests.simulation.mujoco.test_tool_spec import _PYTHON_ONLY_ACTIONS  # noqa: E402

#: One-edit misspellings of published actions, each a plausible slip: a dropped
#: or added character, a singular/plural confusion, a transposition. Every
#: ``want`` is in the enum, so the refusal has a correct answer available.
_TYPOS = [
    ("set_joint_position", "set_joint_positions"),
    ("get_stat", "get_state"),
    ("list_robot", "list_robots"),
    ("renderr", "render"),
    ("add_objec", "add_object"),
    ("remove_objct", "remove_object"),
]

_NOT_FOR_AGENTS = "is not available to an agent"


@pytest.fixture
def sim():
    s = Simulation(tool_name="unknown_action_message_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _text(result):
    return result["content"][0]["text"]


def _suggestions(text):
    """The names offered by the ``" Did you mean: a, b, c?"`` fragment."""
    match = re.search(r"Did you mean: ([^?]+)\?", text)
    if match is None:
        return []
    return [name.strip() for name in match.group(1).split(",")]


def _stream_once(sim, action):
    """Drive one call through ``stream``, the other agent-facing entry point."""

    async def _run():
        tool_use = {"toolUseId": "tu-1", "input": {"action": action}}
        return [event async for event in sim.stream(tool_use, {})]

    events = asyncio.run(_run())
    assert len(events) == 1
    return events[0].tool_result


@pytest.mark.parametrize(("typo", "want"), _TYPOS)
def test_a_one_edit_typo_names_the_action_it_probably_meant(sim, typo, want):
    """The refusal for a near-miss names the published action a character away.

    Without this the caller is told only that the name it sent is wrong, so
    recovering from a dropped ``s`` costs a round-trip through the schema.
    """
    result = sim(action=typo)
    assert result["status"] == "error"
    text = _text(result)
    assert want in _suggestions(text), f"{typo!r} -> {text!r}"


def test_the_refusal_says_how_many_actions_exist_and_where_they_are_listed(sim):
    """A name with no close match still learns the vocabulary is closed.

    ``close_match_hint`` returns ``""`` when nothing scores well enough, so the
    pointer has to travel independently of the suggestion or the worst case -
    a name resembling nothing - stays the dead end this fixes.
    """
    text = _text(sim(action="zzzzzzzzzz"))
    assert _suggestions(text) == []
    assert str(len(_PUBLISHED_ACTIONS)) in text
    assert "tool_spec" in text


def test_the_action_refusal_suggests_under_the_same_rule_as_the_entity_refusals(sim):
    """The fifth caller of the shared helper, byte for byte.

    Asserting the helper's own output is in the message pins the *rule* rather
    than one wording, so the action refusal cannot drift away from the four
    entity refusals that already use it.
    """
    typo = "set_joint_position"
    expected = close_match_hint(typo, sorted(_PUBLISHED_ACTIONS))
    assert expected, "premise: the helper has a suggestion for this typo"
    assert expected in _text(sim(action=typo))


def test_the_stream_entry_point_gets_the_same_suggestion(sim):
    """Both agent-facing entry points share the refusal, so both recover.

    ``stream`` is the form a Strands agent actually invokes; a fix applied only
    to ``__call__`` would leave the agent path unchanged.
    """
    text = _text(_stream_once(sim, "renderr"))
    assert "render" in _suggestions(text)


@pytest.mark.parametrize("typo", [name[:-1] for name in sorted(_PYTHON_ONLY_ACTIONS)])
def test_every_suggested_name_is_one_the_agent_can_actually_call(sim, typo):
    """Control: suggestions come from the enum, not from every dispatchable name.

    Each probe is a truncated Python-only action, so the nearest dispatchable
    name is one this boundary refuses. Offering it would answer a refusal with a
    second refusal; drawing the corpus from the enum is what prevents that.
    Passes before the fix too - there were no suggestions to be wrong.
    """
    offered = set(_suggestions(_text(sim(action=typo))))
    assert offered <= _PUBLISHED_ACTIONS
    assert not offered & _PYTHON_ONLY_ACTIONS


def test_a_python_only_action_keeps_its_own_verdict(sim):
    """Control: a curated omission is still reported as one, not as a typo.

    The two verdicts answer different questions, and turning the held-back
    capability into "unknown" would send a reader hunting for a misspelling
    that is not there.
    """
    text = _text(sim(action="send_action"))
    assert _NOT_FOR_AGENTS in text
    assert "Did you mean" not in text


def test_the_unknown_action_verdict_still_reads_as_before(sim):
    """Control: the verdict is unchanged, only the recovery is added."""
    text = _text(sim(action="teleprot"))
    assert text.startswith("Unknown action: teleprot")
    assert _NOT_FOR_AGENTS not in text


def test_a_published_action_still_dispatches(sim):
    """Control: no advertised action was made unreachable."""
    assert sim(action="list_robots")["status"] == "success"
