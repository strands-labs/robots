"""A second ``create_world`` is refused with the remedies that fit the request.

A world cannot be rebuilt under a live scene, so a second ``create_world`` is
refused. ``Robot("so101")`` returns an engine whose world is already created and
populated, so that refusal is the *first* thing such a caller meets - and it
offered exactly two remedies, ``destroy`` and ``reset``, for every request:

* ``reset`` applies no ``create_world`` parameter. It calls ``mj_resetData`` and
  restores the home poses, at the values the world was built with. A caller who
  asked ``create_world(terrain="rough")``, followed the advice and called
  ``reset`` got ``status="success"`` and a world with zero heightfields; the same
  for ``timestep=0.001`` (still ``0.002``) and ``gravity`` (still ``-9.81``). The
  advertised remedy reported success and changed nothing about the request.
* ``destroy`` then ``create_world`` does apply the parameters, but discards every
  robot and object in the live world. The refusal named neither the contents nor
  that cost, so on a ``Robot(...)`` engine it read as routine bookkeeping while
  throwing away the robot the factory had just added.
* ``set_timestep`` and ``set_gravity`` - published actions that apply exactly
  those two parameters to the live world and keep its contents - were not
  mentioned at all.

The refusal now names what the live world holds and routes by the arguments
actually passed: ``timestep`` / ``gravity`` to the setters that can apply them in
place, ``ground_plane`` / ``terrain`` / ``difficulty`` to ``destroy`` with its
cost stated, and a bare call to the world that is already usable. ``reset`` is
never offered as a way to obtain a *different* world.

These pin that routing, that every action the refusal advertises is one the agent
schema publishes, and - deliberately - that the refusal itself is unchanged:
``create_world`` on a live world is still an error that leaves the world exactly
as it was, so a later change that makes it idempotent or auto-resets the scene
fails here instead of silently discarding a populated world.
"""

import inspect
import re
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

#: Requests that a live world CAN adopt, with the value to read back afterwards.
_IN_PLACE_REQUESTS: list[tuple[str, dict[str, Any], str]] = [
    ("timestep", {"timestep": 0.001}, "set_timestep"),
    ("gravity", {"gravity": [0.0, 0.0, -1.62]}, "set_gravity"),
]

#: Requests compiled in at creation, so only a new world can carry them.
_STRUCTURAL_REQUESTS: list[dict[str, Any]] = [
    {"terrain": "rough"},
    {"ground_plane": False},
    {"terrain": "stairs", "difficulty": 0.3},
]


@pytest.fixture
def live() -> Any:
    """A live, populated world - what ``Robot("so101")`` hands back."""
    engine = Simulation(tool_name="test_create_world_refusal", mesh=False)
    assert engine.create_world()["status"] == "success"
    assert engine.add_object(name="cube", shape="box", position=[0.3, 0.0, 0.05])["status"] == "success"
    yield engine
    engine.destroy()


def _text(result: dict[str, Any]) -> str:
    content = result.get("content") or []
    return str(content[0].get("text", "")) if content else ""


def _advertised_actions(text: str) -> set[str]:
    """Every published action name the refusal mentions.

    Matched against the schema's own action vocabulary so a word that merely
    reads like an action ("world") is not counted, and so an action added to the
    message is picked up without restating the list here.
    """
    from strands_robots.simulation.mujoco.simulation import _PUBLISHED_ACTIONS

    words = set(re.findall(r"[a-z_][a-z0-9_]*", text))
    return words & set(_PUBLISHED_ACTIONS)


def _read_back(sim: Any, param: str) -> Any:
    assert sim._world is not None
    opt = sim._world._model.opt
    return float(opt.timestep) if param == "timestep" else [float(v) for v in opt.gravity]


class TestResetIsNotOfferedForAParameterItCannotApply:
    """``reset`` reports success and applies no ``create_world`` parameter."""

    @pytest.mark.parametrize(("param", "request_kw", "_setter"), _IN_PLACE_REQUESTS, ids=lambda v: str(v)[:24])
    def test_a_physics_parameter_request_does_not_route_to_reset(
        self, live: Any, param: str, request_kw: dict[str, Any], _setter: str
    ) -> None:
        refusal = live.create_world(**request_kw)
        assert refusal["status"] == "error"
        if "reset" not in _advertised_actions(_text(refusal)):
            return
        assert live.reset()["status"] == "success"
        got = _read_back(live, param)
        pytest.fail(
            f"create_world({param}={request_kw[param]!r}) was refused with a message advertising "
            f"reset; reset then reported success and left {param}={got!r}, not "
            f"{request_kw[param]!r}. Message: {_text(refusal)!r}"
        )

    @pytest.mark.parametrize("request_kw", _STRUCTURAL_REQUESTS, ids=lambda v: ",".join(v))
    def test_a_structural_request_does_not_route_to_reset(self, live: Any, request_kw: dict[str, Any]) -> None:
        refusal = live.create_world(**request_kw)
        assert refusal["status"] == "error"
        if "reset" not in _advertised_actions(_text(refusal)):
            return
        assert live.reset()["status"] == "success"
        assert live._world is not None
        hfields = int(live._world._model.nhfield)
        pytest.fail(
            f"create_world({request_kw!r}) was refused with a message advertising reset; reset "
            f"then reported success with nhfield={hfields} and the ground unchanged. "
            f"Message: {_text(refusal)!r}"
        )


class TestTheSetterThatCanApplyTheRequestIsNamed:
    """The two parameters a live world can adopt have published setters."""

    @pytest.mark.parametrize(("param", "request_kw", "setter"), _IN_PLACE_REQUESTS, ids=lambda v: str(v)[:24])
    def test_the_refusal_names_it(self, live: Any, param: str, request_kw: dict[str, Any], setter: str) -> None:
        refusal = live.create_world(**request_kw)
        assert setter in _text(refusal), (
            f"create_world({param}=...) was refused without naming {setter}, the published action "
            f"that applies it to the live world: {_text(refusal)!r}"
        )

    @pytest.mark.parametrize(("param", "request_kw", "setter"), _IN_PLACE_REQUESTS, ids=lambda v: str(v)[:24])
    def test_following_it_applies_the_request_and_keeps_the_contents(
        self, live: Any, param: str, request_kw: dict[str, Any], setter: str
    ) -> None:
        """The remedy is only worth naming if it does what the caller asked."""
        live.create_world(**request_kw)
        assert getattr(live, setter)(request_kw[param])["status"] == "success"
        assert _read_back(live, param) == pytest.approx(request_kw[param])
        assert "cube" in live._world.objects


class TestTheCostOfDestroyIsStated:
    """``destroy`` drops the scene, so the refusal that offers it says so."""

    @pytest.mark.parametrize("request_kw", _STRUCTURAL_REQUESTS, ids=lambda v: ",".join(v))
    def test_the_live_contents_are_named(self, live: Any, request_kw: dict[str, Any]) -> None:
        text = _text(live.create_world(**request_kw))
        assert "cube" in text, f"the refusal did not name the object destroy would discard: {text!r}"

    def test_a_bare_call_names_them_too(self, live: Any) -> None:
        """The bare call is the one an agent makes first, before reading docs."""
        text = _text(live.create_world())
        assert "cube" in text, f"the refusal did not name the live world's contents: {text!r}"

    def test_destroy_is_still_the_route_for_a_structural_request(self, live: Any) -> None:
        """A control: the parameter with no setter must still point at destroy."""
        assert "destroy" in _text(live.create_world(terrain="rough"))


class TestTheRefusalOnlyAdvertisesSelectableActions:
    """A refusal naming an action the schema omits is a dead end for an agent."""

    @pytest.mark.parametrize(
        "request_kw",
        [{}, *(kw for _, kw, _ in _IN_PLACE_REQUESTS), *_STRUCTURAL_REQUESTS],
        ids=lambda v: ",".join(v) or "no-args",
    )
    def test_every_action_named_is_published(self, live: Any, request_kw: dict[str, Any]) -> None:
        from strands_robots.simulation.mujoco.simulation import _PUBLISHED_ACTIONS

        text = _text(live.create_world(**request_kw))
        assert _advertised_actions(text), f"the refusal named no action to take: {text!r}"
        for action in _advertised_actions(text):
            assert action in _PUBLISHED_ACTIONS
            assert callable(getattr(live, action, None)), f"{action} is advertised but not callable"


class TestTheRefusalItselfIsUnchanged:
    """Refusing is the behaviour; only the message it carries is new."""

    @pytest.mark.parametrize(
        "request_kw",
        [{}, *(kw for _, kw, _ in _IN_PLACE_REQUESTS), *_STRUCTURAL_REQUESTS],
        ids=lambda v: ",".join(v) or "no-args",
    )
    def test_a_second_create_world_is_still_an_error(self, live: Any, request_kw: dict[str, Any]) -> None:
        """Not idempotent and not an auto-reset: a populated world is not discarded."""
        assert live.create_world(**request_kw)["status"] == "error"

    @pytest.mark.parametrize(
        "request_kw",
        [{}, *(kw for _, kw, _ in _IN_PLACE_REQUESTS), *_STRUCTURAL_REQUESTS],
        ids=lambda v: ",".join(v) or "no-args",
    )
    def test_the_refused_call_leaves_the_world_exactly_as_it_was(self, live: Any, request_kw: dict[str, Any]) -> None:
        assert live._world is not None
        before = (
            float(live._world._model.opt.timestep),
            [float(v) for v in live._world._model.opt.gravity],
            int(live._world._model.nhfield),
            sorted(live._world.objects),
        )
        live.create_world(**request_kw)
        after = (
            float(live._world._model.opt.timestep),
            [float(v) for v in live._world._model.opt.gravity],
            int(live._world._model.nhfield),
            sorted(live._world.objects),
        )
        assert after == before

    def test_the_first_create_world_is_untouched(self) -> None:
        """A world has to be built before the refusal can fire at all."""
        engine = Simulation(tool_name="test_create_world_refusal_first", mesh=False)
        try:
            result = engine.create_world()
            assert result["status"] == "success"
            assert "Simulation world created" in _text(result)
        finally:
            engine.destroy()

    def test_an_invalid_argument_still_outranks_the_world_guard(self, live: Any) -> None:
        """A bad terrain is a bad terrain whether or not a world exists."""
        text = _text(live.create_world(terrain="not-a-terrain"))
        assert "not-a-terrain" in text
        assert "already exists" not in text


class TestTheFactoryPathTheRefusalActuallyMeets:
    """``Robot(name)`` is the caller that hits this refusal on its first call."""

    def test_the_preloaded_robot_is_named_so_the_agent_knows_it_is_there(self) -> None:
        """The bare call an agent makes first must report what it already has,
        so "build a scene" does not begin by discarding the robot it was given."""
        robots = pytest.importorskip("strands_robots")
        sim = robots.Robot("so101", mesh=False)
        try:
            assert "so101" in sim.list_robots()
            text = _text(sim.create_world())
            assert "so101" in text, (
                f"Robot('so101') pre-populates the world, but its create_world refusal never "
                f"named the robot a destroy would discard: {text!r}"
            )
        finally:
            sim.destroy()

    def test_a_physics_request_on_the_factory_engine_routes_to_the_setter(self) -> None:
        robots = pytest.importorskip("strands_robots")
        sim = robots.Robot("so101", mesh=False)
        try:
            text = _text(sim.create_world(timestep=0.001))
            assert "set_timestep" in text
            assert "reset" not in _advertised_actions(text)
            assert sim.set_timestep(0.001)["status"] == "success"
            assert _read_back(sim, "timestep") == pytest.approx(0.001)
            assert "so101" in sim.list_robots(), "the setter route must keep the preloaded robot"
        finally:
            sim.destroy()


class TestTheSetterTableMatchesTheSignature:
    """The routing table is only correct if it names real parameters and actions."""

    def test_every_entry_is_a_create_world_parameter_with_a_published_setter(self) -> None:
        from strands_robots.simulation.mujoco.simulation import (
            _PUBLISHED_ACTIONS,
            _WORLD_PARAM_SETTERS,
        )
        from strands_robots.simulation.mujoco.simulation import Simulation as Sim

        params = inspect.signature(Sim.create_world).parameters
        assert _WORLD_PARAM_SETTERS, "the table cannot be empty: the refusal reads it to route"
        for param, action in _WORLD_PARAM_SETTERS:
            assert param in params, f"{param} is not a create_world parameter"
            assert action in _PUBLISHED_ACTIONS, f"{action} is not published to the agent"
            assert callable(getattr(Sim, action, None)), f"{action} is not a method"
            assert param in inspect.signature(getattr(Sim, action)).parameters, (
                f"{action} does not take a {param} argument, so it cannot apply the request"
            )

    def test_the_parameters_with_no_setter_are_absent(self) -> None:
        """``ground_plane`` / ``terrain`` / ``difficulty`` route to destroy instead."""
        from strands_robots.simulation.mujoco.simulation import _WORLD_PARAM_SETTERS

        named = {param for param, _ in _WORLD_PARAM_SETTERS}
        assert named.isdisjoint({"ground_plane", "terrain", "difficulty"})
