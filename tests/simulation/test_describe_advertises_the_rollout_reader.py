"""``describe()`` advertises the rollout reader its own entries send a caller to.

``list_policies_running`` was promoted onto :class:`SimEngine` so it answers on
every backend rather than only on MuJoCo, where it started. The promotion
reached the implementation and not the advertisement: the discovery surface a
caller enumerates to learn what an engine can do listed the verb only in
MuJoCo's ``describe()`` override, so on Newton and Isaac the verb answered
``status="success"`` while ``describe()["methods"]`` did not name it. An agent
that discovers capability through ``describe()`` - the surface this project
extends rather than aliasing a second name onto - could not find a working verb.

Three statements in the tree promised otherwise, which is what makes this a
defect rather than a gap:

* the base ``start_policy`` entry tells the caller a backend "advertises
  ``list_policies_running`` beside it, so this entry is how to tell which one
  you hold" - a promise about the same mapping that omitted it;
* ``docs/simulation/overview.md`` states the verb "answers on every backend ...
  a MuJoCo, Newton or Isaac engine names the robots it is driving";
* :meth:`SimEngine.list_policies_running` documents itself as "promoted here
  from the MuJoCo engine so it answers on every backend".

Its documented pair partner ``stop_policy`` was advertised on the base all
along, so the surface listed one half of a pair the docs describe as inseparable
("the two never report opposite facts about the same robot at the same
instant").

The pins below grade the promise structurally (an entry that says the surface
advertises a verb must name it), grade the symptom behaviourally against real
MuJoCo and Newton engines, and pin the one-owner shape of the entry text -
because a second copy of that text is exactly what let the promotion reach the
implementation and not the advertisement.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import pathlib
import re
import textwrap
from typing import Any

import pytest

from strands_robots.simulation.base import SimEngine

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

#: Public ``SimEngine`` verbs, the names an entry can send a caller to.
_VERBS = frozenset(n for n in dir(SimEngine) if not n.startswith("_") and callable(getattr(SimEngine, n, None)))

#: An entry saying the surface *advertises* a verb is a promise about the
#: mapping itself, not a mention of a related verb: "advertises
#: list_policies_running beside it" tells the caller to look it up there. A bare
#: mention (``replay_episode`` naming ``robot_action_keys`` as the source of a
#: default) makes no claim about the mapping and is deliberately out of scope.
_PROMISE = re.compile(r"advertis\w*\s+(?:it\s+)?(?:the\s+)?([a-z_][a-z0-9_]*)")

#: ``describe`` is the one verb an entry may name without advertising it: a
#: caller reading the entry is already holding this method's output.
_SELF_REFERENCE = "describe"

#: The two halves of the rollout-reporting pair. ``docs/simulation/overview.md``
#: derives one verdict from the other's population, so a surface that offers one
#: and hides the other tells a caller half a contract.
_ROLLOUT_REPORTING_PAIR = ("stop_policy", "list_policies_running")


class _BaseSurfaceEngine(SimEngine):
    """The thinnest concrete engine, so ``describe()`` is the base mapping alone."""

    def create_world(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def destroy(self) -> dict[str, Any]:
        return {"status": "success"}

    def reset(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        return {"status": "success"}

    def get_state(self) -> dict[str, Any]:
        return {"status": "success"}

    def add_robot(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def remove_robot(self, name: str) -> dict[str, Any]:
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return ["so101"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["1"]

    def add_object(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def remove_object(self, name: str) -> dict[str, Any]:
        return {"status": "success"}

    def get_observation(self, robot_name: str | None = None, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def send_action(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def render(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}


class _DeclaredSurfaceEngine(_BaseSurfaceEngine):
    """The thinnest engine that also implements every OPTIONAL base verb.

    ``SimEngine.describe()`` advertises ``load_scene`` / ``randomize`` /
    ``set_obs_noise`` / ``get_contacts`` only on a subclass that actually
    overrides them, because an entry for a method whose body is ``raise
    NotImplementedError`` is a false advertisement. So
    :class:`_BaseSurfaceEngine`, which overrides none of the four, reports the
    base mapping *minus* those - which is the right reference for the promise
    pins (what a caller is really told) and the wrong one for measuring whether
    another backend NARROWS the surface: against it, a backend that omits a
    raising stub hides nothing, because the base never offered it either.

    This subclass gives the narrowing pin the mapping as *declared*, so it keeps
    grading Newton's own narrowing rather than the base's.
    """

    def load_scene(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def randomize(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def set_obs_noise(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}

    def get_contacts(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success"}


def _base_methods() -> dict[str, str]:
    return dict(_BaseSurfaceEngine().describe()["methods"])


def _declared_base_methods() -> dict[str, str]:
    """The base mapping including the optional verbs, for the narrowing pin."""
    return dict(_DeclaredSurfaceEngine().describe()["methods"])


def _broken_promises(methods: dict[str, str]) -> list[tuple[str, str]]:
    """Entries promising the surface advertises a verb the surface does not name."""
    broken = []
    for entry, text in sorted(methods.items()):
        if not isinstance(text, str):
            continue
        for match in _PROMISE.finditer(text):
            verb = match.group(1)
            if verb in _VERBS and verb != _SELF_REFERENCE and verb not in methods:
                broken.append((entry, verb))
    return broken


def _live_engine(backend: str) -> Any:
    from strands_robots.simulation import create_simulation

    sim = create_simulation(backend=backend)
    sim.create_world()
    sim.add_robot("so101")
    return sim


_LIVE_BACKENDS = [
    pytest.param("mujoco", id="mujoco"),
    pytest.param("newton", id="newton", marks=pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp absent")),
]


class TestTheSurfaceKeepsItsOwnPromises:
    """An entry that says the surface advertises a verb must name that verb."""

    def test_the_base_surface_keeps_every_advertise_promise_it_makes(self) -> None:
        methods = _base_methods()
        promises = [
            (entry, m.group(1))
            for entry, text in methods.items()
            if isinstance(text, str)
            for m in _PROMISE.finditer(text)
            if m.group(1) in _VERBS and m.group(1) != _SELF_REFERENCE
        ]
        assert promises, "this scan found no advertise-promise at all, so it is grading nothing"
        assert _broken_promises(methods) == [], (
            "describe() tells a caller the surface advertises a verb it does not name, "
            "so following the entry leads nowhere"
        )

    @pytest.mark.parametrize("backend", _LIVE_BACKENDS)
    def test_a_live_backend_keeps_them_too(self, backend: str) -> None:
        sim = _live_engine(backend)
        try:
            assert _broken_promises(dict(sim.describe()["methods"])) == []
        finally:
            sim.destroy()


class TestTheRolloutReaderIsDiscoverable:
    """A verb that answers is a verb the discovery surface names."""

    def test_the_base_surface_advertises_it(self) -> None:
        assert "list_policies_running" in _base_methods()

    @pytest.mark.parametrize("backend", _LIVE_BACKENDS)
    def test_a_backend_that_answers_it_advertises_it(self, backend: str) -> None:
        sim = _live_engine(backend)
        try:
            answered = sim.list_policies_running()["status"] == "success"
            assert answered, "premise: this backend reports its in-flight population"
            assert "list_policies_running" in sim.describe()["methods"], (
                f"{backend} answers list_policies_running but describe() does not name it, "
                "so a caller enumerating the surface cannot find a verb that works"
            )
        finally:
            sim.destroy()

    @pytest.mark.parametrize("backend", _LIVE_BACKENDS)
    def test_both_halves_of_the_reporting_pair_are_advertised(self, backend: str) -> None:
        sim = _live_engine(backend)
        try:
            methods = sim.describe()["methods"]
            missing = [v for v in _ROLLOUT_REPORTING_PAIR if v not in methods]
            assert missing == [], f"{backend} advertises half the rollout-reporting pair, hiding {missing}"
        finally:
            sim.destroy()

    def test_isaac_inherits_the_base_surface(self) -> None:
        """Isaac cannot be constructed without ``isaacsim``, so grade the seam.

        Its ``describe()`` builds on ``super().describe()`` and removes nothing
        from the inherited mapping, which is how the base entry reaches it.
        """
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        source = inspect.getsource(IsaacSimulation.describe)
        assert "super().describe()" in source
        tree = ast.parse(textwrap.dedent(source))
        deletes = [n for n in ast.walk(tree) if isinstance(n, ast.Delete)]
        assert deletes == [], "Isaac removes an inherited entry, so the base surface does not reach it"
        assert ".pop(" not in source, "Isaac pops an inherited entry, so the base surface does not reach it"


class TestTheEntryTextHasOneOwner:
    """The advertisement is assembled twice, so its text is owned once."""

    def test_every_advertising_site_references_the_shared_entry(self) -> None:
        root = pathlib.Path(inspect.getfile(SimEngine)).parent
        literals = []
        sites = 0
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                for key, value in zip(node.keys, node.values, strict=True):
                    if not (isinstance(key, ast.Constant) and key.value == "list_policies_running"):
                        continue
                    sites += 1
                    if not (isinstance(value, ast.Name) and value.id == "LIST_POLICIES_RUNNING_DESCRIBE_ENTRY"):
                        literals.append(f"{path.relative_to(root)}:{key.lineno}")
        assert sites >= 2, f"expected the surface to be assembled in at least two places, found {sites}"
        assert literals == [], (
            "an advertising site spells the entry text itself instead of reading the shared one; "
            f"a second copy is what let the verb's promotion miss the advertisement: {literals}"
        )

    def test_the_shared_entry_names_the_return_shape(self) -> None:
        # Imported here, not at module scope: on a tree without the shared
        # entry every cell in this file would be a collection error, which
        # grades nothing.
        from strands_robots.simulation.base import LIST_POLICIES_RUNNING_DESCRIBE_ENTRY

        assert "-> dict" in LIST_POLICIES_RUNNING_DESCRIBE_ENTRY
        assert "stop_policy" in LIST_POLICIES_RUNNING_DESCRIBE_ENTRY


class TestNarrowingStaysHonest:
    """A backend may hide a base verb only when that verb does not answer on it.

    The control for the pins above: they require the surface to name what works,
    not to name everything. Newton legitimately hides two base entries, and both
    refuse rather than answering.
    """

    @pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp absent")
    def test_newton_hides_only_verbs_it_does_not_implement(self) -> None:
        sim = _live_engine("newton")
        try:
            methods = sim.describe()["methods"]
            hidden = sorted(set(_declared_base_methods()) - set(methods))
            assert hidden == ["get_contacts", "load_scene"], f"the hidden set moved: {hidden}"
            with pytest.raises(NotImplementedError):
                sim.get_contacts()
            with pytest.raises(NotImplementedError):
                sim.load_scene("/nonexistent/scene.xml")
            # Non-vacuity: an empty or collapsed mapping would satisfy the rule
            # above by hiding everything, so name a verb that must be present.
            assert "list_policies_running" in methods
        finally:
            sim.destroy()
