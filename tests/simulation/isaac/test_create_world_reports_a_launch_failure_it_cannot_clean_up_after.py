"""A launch failure is reported, not replaced by a crash in its own cleanup.

``create_world`` opens one ``try`` that covers the whole build, and its handler
tears the ``World`` singleton down rather than dropping the reference - ``World``
registers itself as ``SimulationContext.instance()``, so ``self._world = None``
releases nothing and the next ``create_world`` would be handed the stale
instance.

But the name that cleanup calls is bound by a function-local import that lives
INSIDE the same ``try``, and it sits AFTER the first fallible statement in it:

    self._app = _get_or_create_simulation_app(...)   # first thing that can fail
    ...
    from isaacsim.core.api import World              # what the handler needs

``SimulationApp`` raising ``RuntimeError`` / ``OSError`` is not an exotic case -
it is the common real launch failure for this backend: no GPU, no driver, no
display, a Kit that will not start. That lands in the handler with ``World``
never bound, and ``World.clear_instance()`` then raised ``UnboundLocalError``.

Which is the part that made it a defect rather than an untidy traceback:
``UnboundLocalError`` is a ``NameError`` subclass, so it is in neither the
handler's own ``(RuntimeError, OSError, AttributeError)`` tuple nor the outer
one - and deliberately, since that outer comment states "programming bugs
(NameError, ...) propagate". So it escaped ``create_world`` entirely. A method
whose contract is to return ``{"status": "error", ...}`` crashed instead, and the
actionable launch error the operator needed was replaced by ``cannot access
local variable 'World' where it is not associated with a value``.

The fix binds a separate ``world_cls`` to ``None`` above the ``try`` and skips
the teardown when it is still ``None``. That is not merely defensive: if the
import never ran then no ``World`` was constructed, so no singleton was
registered and there is nothing to clear. A distinct name rather than
pre-binding ``World`` itself, because rebinding it by import is a redefinition
mypy refuses.

These run without Isaac Sim: the app factory is stood in, so what is graded is
the handler's behaviour on each failure class rather than a real Kit boot.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac import simulation as isaac_simulation  # noqa: E402
from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402

#: The failure classes that reach the cleanup handler with the World import
#: never having run. ``ImportError`` is deliberately absent: it is answered by an
#: earlier handler that reports the missing install, and never reaches this one.
_PRE_IMPORT_FAILURES = [
    pytest.param(RuntimeError, id="RuntimeError-carb-or-kit-init"),
    pytest.param(OSError, id="OSError-display-or-driver"),
]


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []))


def _sim() -> Any:
    return IsaacSimulation(num_envs=1, headless=True, render_mode="headless")


class TestALaunchFailureBeforeTheWorldImportIsReported:
    @pytest.mark.parametrize("exc_type", _PRE_IMPORT_FAILURES)
    def test_it_returns_an_error_envelope(self, exc_type: type[BaseException], monkeypatch) -> None:
        """The contract: this method returns its failures."""

        def _boom(**_kwargs: Any) -> Any:
            raise exc_type("the GPU is not there")

        monkeypatch.setattr(isaac_simulation, "_get_or_create_simulation_app", _boom)

        result = _sim().create_world()

        assert result["status"] == "error"

    @pytest.mark.parametrize("exc_type", _PRE_IMPORT_FAILURES)
    def test_it_does_not_raise(self, exc_type: type[BaseException], monkeypatch) -> None:
        """Stated separately from the envelope because the regression was an
        ESCAPE, not a wrong payload: pre-fix this raised ``UnboundLocalError``
        out of the method and no envelope existed to inspect."""

        def _boom(**_kwargs: Any) -> Any:
            raise exc_type("the GPU is not there")

        monkeypatch.setattr(isaac_simulation, "_get_or_create_simulation_app", _boom)

        sim = _sim()
        try:
            sim.create_world()
        except NameError as escaped:  # UnboundLocalError is a NameError
            pytest.fail(f"create_world raised instead of reporting: {escaped!r}")

    @pytest.mark.parametrize("exc_type", _PRE_IMPORT_FAILURES)
    def test_the_report_names_the_original_failure(self, exc_type: type[BaseException], monkeypatch) -> None:
        """The whole cost of the bug: the operator was told about a local
        variable instead of about their GPU."""

        def _boom(**_kwargs: Any) -> Any:
            raise exc_type("the GPU is not there")

        monkeypatch.setattr(isaac_simulation, "_get_or_create_simulation_app", _boom)

        text = _text(_sim().create_world())

        assert "the GPU is not there" in text

    @pytest.mark.parametrize("exc_type", _PRE_IMPORT_FAILURES)
    def test_the_report_does_not_name_a_local_variable(self, exc_type: type[BaseException], monkeypatch) -> None:
        def _boom(**_kwargs: Any) -> Any:
            raise exc_type("the GPU is not there")

        monkeypatch.setattr(isaac_simulation, "_get_or_create_simulation_app", _boom)

        text = _text(_sim().create_world())

        assert "local variable" not in text
        assert "World" not in text or "world" in text.lower()

    def test_the_world_is_left_unset(self, monkeypatch) -> None:
        """The teardown's other half still holds: a failed create leaves no
        half-built world for a later call to find."""

        def _boom(**_kwargs: Any) -> Any:
            raise RuntimeError("the GPU is not there")

        monkeypatch.setattr(isaac_simulation, "_get_or_create_simulation_app", _boom)

        sim = _sim()
        sim.create_world()

        assert sim._world is None
        assert sim._world_created is False

    def test_a_second_attempt_is_still_allowed(self, monkeypatch) -> None:
        """A failed create must not latch the "already created" gate, or the
        operator cannot retry after fixing the driver."""

        def _boom(**_kwargs: Any) -> Any:
            raise RuntimeError("the GPU is not there")

        monkeypatch.setattr(isaac_simulation, "_get_or_create_simulation_app", _boom)

        sim = _sim()
        sim.create_world()
        second = sim.create_world()

        assert second["status"] == "error"
        assert "already created" not in _text(second)


class TestTheImportFailureStillReportsTheInstall:
    """The control: ``ImportError`` has its own handler and must keep its own
    message, which names the install rather than the hardware."""

    def test_it_names_isaac_sim(self, monkeypatch) -> None:
        def _boom(**_kwargs: Any) -> Any:
            raise ImportError("no module named isaacsim")

        monkeypatch.setattr(isaac_simulation, "_get_or_create_simulation_app", _boom)

        text = _text(_sim().create_world())

        assert "Isaac Sim" in text


class TestTheCleanupHandleIsBoundBeforeTheTry:
    """Structural, because the behavioural cells above pass for two different
    reasons and only one of them is the fix.

    A future edit that moves the ``World`` import back above the app launch would
    also make them pass, while leaving the next statement added between the two
    exposed again. What the fix actually guarantees is that the cleanup reads a
    name bound OUTSIDE the try.
    """

    def test_the_teardown_guards_on_a_name_bound_outside_the_try(self) -> None:
        import ast
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(IsaacSimulation.create_world))
        tree = ast.parse(source)

        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef)

        # Anchored on the try whose HANDLER does the teardown, not on the first
        # or lowest-numbered try in the function - ``create_world`` has several,
        # and the earliest is an unrelated one. Compared by line because
        # ``ast.walk`` is breadth-first, and because the pre-binding sits inside
        # a ``with self._lock:`` block rather than at the function's top level.
        try_line = next(
            (
                node.lineno
                for node in sorted(
                    (n for n in ast.walk(fn) if isinstance(n, ast.Try)),
                    key=lambda n: n.lineno,
                )
                if "clear_instance" in ast.dump(node)
            ),
            None,
        )
        assert try_line is not None, "no try in create_world tears the World singleton down"

        pre_try = {
            node.target.id
            for node in ast.walk(fn)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
            and node.lineno < try_line
        }

        assert pre_try, "no cleanup handle is pre-bound above create_world's try"
        assert "world_cls" in pre_try, f"expected world_cls pre-bound, found {sorted(pre_try)}"

    def test_clear_instance_is_reached_through_that_name(self) -> None:
        import inspect

        source = inspect.getsource(IsaacSimulation.create_world)

        assert "world_cls.clear_instance()" in source
        assert "if world_cls is not None:" in source
