"""A contact success criterion is refused, loudly, where no contact query exists.

``eval_policy(success_fn="contact")`` resolves to the predicate DSL's
``contact_any``, whose never-raise contract turns a backend's
``NotImplementedError`` into ``False`` - every tick, at DEBUG. On a backend
whose ``get_contacts`` is still the ``SimEngine`` raising stub (Isaac, Newton,
any minimal engine), the full evaluation therefore ran to completion - GPU-hours
on Isaac - and reported ``success_rate: 0.0`` with ``success_measured: True``:
a wrong answer shaped exactly like a policy that failed every episode, with
nothing anywhere saying success was never measurable. ``base.py``'s own
``evaluate`` docstring names ``success_fn="contact"`` as the way "to measure
real task success", which is what sent callers into it.

Three layers, three fixes, pinned here:

* **The resolver refuses up front.** ``_resolve_success_fn("contact")`` raises
  ``ValueError`` - which ``evaluate`` already returns as its structured error
  envelope - when the backend's ``get_contacts`` is the base stub, BEFORE any
  rollout is spent. The test is structural (did the subclass override?) rather
  than a probe call, because a real backend's ``get_contacts`` can fail for
  world-lifecycle reasons that say nothing about the capability.
* **The predicates say why they answer False.** The contact read now has one
  owner, ``_read_contacts``, which treats ``NotImplementedError`` as the
  permanent fact it is - a WARNING once per backend class, naming the remedy -
  instead of a per-tick DEBUG line indistinguishable from a transient failure.
  The DSL stays never-raise: it is reachable directly through benchmark specs,
  where refusing is not this layer's call.
* **describe() stops advertising the stubs.** The base advertisement of
  ``load_scene`` / ``randomize`` / ``set_obs_noise`` / ``get_contacts`` is now
  conditional on the subclass overriding the raising stub, so a backend cannot
  advertise a capability whose call raises. (The Newton backend only avoided
  this by building its ``describe()`` from scratch - a per-backend workaround
  for the base-class defect.) The conditional-advertisement rule itself is
  pinned in test_sim_engine_describe_discovery.py beside the original
  discoverability claim; here the two shipped backends are graded.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from strands_robots.simulation import predicates
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner


def _implement_abstracts(namespace: dict[str, Any]) -> dict[str, Any]:
    """Fill ``SimEngine``'s abstract surface with inert success envelopes.

    These tests are about the OPTIONAL stubs (``get_contacts`` and friends),
    which are ordinary methods; the abstract surface only needs to exist so the
    class instantiates. Derived from ``__abstractmethods__`` rather than listed,
    so an abstract method added to the ABC later does not break this fixture.
    """
    for name in SimEngine.__abstractmethods__:
        namespace.setdefault(name, lambda self, *a, **k: {"status": "success", "content": []})
    return namespace


_StubEngine = type(
    "_StubEngine",
    (SimEngine,),
    _implement_abstracts(
        {
            "__doc__": "A backend that inherits every optional stub - the Isaac/Newton shape.",
            "__init__": lambda self: None,
            "list_robots": lambda self: [],
        }
    ),
)


class _ContactEngine(_StubEngine):  # type: ignore[misc, valid-type]
    """A backend with a real contact query - the MuJoCo shape."""

    def __init__(self, contacts: list[dict[str, Any]] | None = None) -> None:
        self._contacts = contacts or []

    def get_contacts(self) -> dict[str, Any]:
        return {
            "status": "success",
            "content": [{"json": {"contacts": self._contacts, "n_contacts": len(self._contacts)}}],
        }


def _runner(sim: Any) -> PolicyRunner:
    return PolicyRunner(sim)


@pytest.fixture(autouse=True)
def _fresh_warning_registry(monkeypatch):
    """Each test observes the once-per-backend warning from a clean slate.

    ``raising=False`` so the fixture does not itself fail on pre-fix code,
    where the registry does not exist - the controls in this file must keep
    passing there to stay controls.
    """
    monkeypatch.setattr(predicates, "_WARNED_NO_CONTACT_QUERY", set(), raising=False)


class TestTheResolverRefusesUpFront:
    def test_contact_on_a_stub_backend_raises_before_any_rollout(self) -> None:
        with pytest.raises(ValueError) as exc:
            _runner(_StubEngine())._resolve_success_fn("contact")

        text = str(exc.value)
        assert "_StubEngine" in text, "the refusal names the backend"
        assert "get_contacts" in text
        # It explains the wrong answer it prevents and names both remedies.
        assert "0.0" in text
        assert "MuJoCo" in text

    def test_evaluate_returns_it_as_the_structured_envelope(self) -> None:
        """The ValueError is the resolver's documented channel into evaluate's
        error dict - the caller sees a refusal, not a traceback."""

        class _Policy:
            requires_images = False

        result = _runner(_StubEngine()).evaluate(
            "bot",
            _Policy(),  # type: ignore[arg-type]
            n_episodes=1,
            success_fn="contact",
        )
        assert result["status"] == "error", result
        assert "get_contacts" in result["content"][0]["text"]

    def test_a_backend_with_a_contact_query_resolves(self) -> None:
        """Control: the refusal keys on the stub, not on the string."""
        check = _runner(_ContactEngine())._resolve_success_fn("contact")
        assert callable(check)

    def test_a_callable_success_fn_is_untouched(self) -> None:
        """The refusal is scoped to the one string that depends on the missing
        capability; a caller-supplied callable makes no claim about contacts."""
        fn = lambda obs: True  # noqa: E731
        assert _runner(_StubEngine())._resolve_success_fn(fn) is fn


class TestThePredicatesSayWhyTheyAnswerFalse:
    def test_a_stub_backend_warns_once_and_answers_false(self, caplog) -> None:
        pred = predicates.make_predicate("contact_any")
        sim = _StubEngine()
        with caplog.at_level(logging.WARNING, logger=predicates.logger.name):
            results = [pred(sim) for _ in range(5)]

        assert results == [False] * 5
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "once per backend, not once per polled tick"
        text = warnings[0].getMessage()
        assert "_StubEngine" in text
        assert "MuJoCo" in text, "the warning names a remedy"

    def test_the_warning_is_per_backend_class_not_per_predicate(self, caplog) -> None:
        sim = _StubEngine()
        with caplog.at_level(logging.WARNING, logger=predicates.logger.name):
            predicates.make_predicate("contact_any")(sim)
            predicates.make_predicate("contact_between", geom_a="a", geom_b="b")(sim)

        assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1

    def test_a_transient_failure_stays_a_debug_line(self, caplog) -> None:
        """The pre-existing degraded mode is unchanged: a read that fails for
        world-lifecycle reasons is not a missing capability and must not train
        operators to ignore the capability warning."""

        class _FlakyEngine(_ContactEngine):
            def get_contacts(self) -> dict[str, Any]:
                raise RuntimeError("world mid-teardown")

        with caplog.at_level(logging.WARNING, logger=predicates.logger.name):
            assert predicates.make_predicate("contact_any")(_FlakyEngine()) is False
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_a_real_contact_still_reads_true(self) -> None:
        """Control: the shared reader did not change the answering path."""
        sim = _ContactEngine(contacts=[{"geom1": "a", "geom2": "b", "active": True, "distance": -0.001}])
        assert predicates.make_predicate("contact_any")(sim) is True
        assert predicates.make_predicate("contact_between", geom_a="a", geom_b="b")(sim) is True
        assert predicates.make_predicate("contact_between", geom_a="a", geom_b="c")(sim) is False


class TestTheShippedBackendsAdvertiseHonestly:
    def test_isaac_advertises_exactly_what_it_overrides(self) -> None:
        """Derived from the override state rather than a snapshot of Isaac's
        capability set: at the time this gate landed, Isaac inherited the
        randomize / set_obs_noise / get_contacts stubs and had to stop
        advertising them - and sibling branches then IMPLEMENTED those very
        methods, which is precisely when a hardcoded "Isaac does not advertise
        X" assertion turns a correct advertisement into a red test. What is
        actually owed is the equivalence: advertised iff overridden."""
        pytest.importorskip("strands_robots.simulation.isaac")
        import threading

        from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation

        engine = IsaacSimulation.__new__(IsaacSimulation)
        engine._lock = threading.RLock()
        engine._robots = {}
        engine._cameras = {}
        engine._config = IsaacConfig()
        engine._world_created = False
        methods = engine.describe()["methods"]
        for name in ("load_scene", "randomize", "set_obs_noise", "get_contacts"):
            overridden = getattr(IsaacSimulation, name) is not getattr(SimEngine, name)
            assert (name in methods) == overridden, (
                f"{name!r}: advertised={name in methods} but overridden={overridden} - "
                f"describe() must advertise exactly the capabilities whose calls do not raise"
            )
        # load_scene is genuinely implemented there, so the equivalence is not
        # vacuously satisfied by four absences.
        assert "load_scene" in methods

    @pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec("mujoco") is None,
        reason="mujoco not installed",
    )
    def test_mujoco_still_advertises_all_of_them(self) -> None:
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        sim = MuJoCoSimEngine()
        try:
            methods = sim.describe()["methods"]
            for name in ("load_scene", "randomize", "set_obs_noise", "get_contacts"):
                assert name in methods, f"the gate over-pruned {name!r} on a backend that implements it"
        finally:
            sim.destroy()
