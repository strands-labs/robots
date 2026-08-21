"""Tests for ``strands_robots._async_utils._resolve_coroutine``.

The helper bridges async results into sync call sites and has three
branches:

    1. plain (non-coroutine) value -> returned unchanged
    2. coroutine, no running event loop -> ``asyncio.run``
    3. coroutine, inside a running event loop -> offloaded to the reused
       single-worker thread executor

These tests pin all three branches deterministically, without touching any
robot or simulation surface, and pin that the branch guard classifies only the
loop check -- so an exception the coroutine raises is never mistaken for the
"no running loop" verdict, whatever its class.
"""

from __future__ import annotations

import asyncio

import pytest

from strands_robots._async_utils import _resolve_coroutine


class TestResolveCoroutine:
    """_resolve_coroutine() - the three resolution branches."""

    def test_plain_value_returned_unchanged(self) -> None:
        """A non-coroutine value is passed straight through."""
        sentinel = object()
        assert _resolve_coroutine(sentinel) is sentinel

    def test_plain_falsy_value_passthrough(self) -> None:
        """Falsy non-coroutine values must not be mistaken for 'no result'."""
        assert _resolve_coroutine(0) == 0
        assert _resolve_coroutine(None) is None
        assert _resolve_coroutine("") == ""

    def test_coroutine_no_running_loop_uses_asyncio_run(self) -> None:
        """With no running loop, the coroutine is resolved via asyncio.run."""

        async def produce() -> str:
            return "resolved-sync"

        assert _resolve_coroutine(produce()) == "resolved-sync"

    def test_coroutine_inside_running_loop_offloads_to_thread(self) -> None:
        """Inside a running loop, resolution is offloaded to the executor thread.

        Calling asyncio.run() directly here would raise 'cannot be called
        from a running event loop'; the helper must instead run the
        coroutine on its worker thread and return the value synchronously.
        """

        async def inner() -> int:
            return 42

        async def driver() -> int:
            # We are inside a running loop here, so this exercises the
            # get_running_loop()-succeeds (offload) branch.
            return _resolve_coroutine(inner())

        assert asyncio.run(driver()) == 42

    def test_coroutine_inside_running_loop_propagates_exception(self) -> None:
        """Exceptions raised inside the offloaded coroutine surface to the caller."""

        async def boom() -> None:
            raise ValueError("kaboom")

        async def driver() -> None:
            return _resolve_coroutine(boom())

        try:
            asyncio.run(driver())
        except ValueError as exc:
            assert str(exc) == "kaboom"
        else:  # pragma: no cover - explicit failure if no exception
            raise AssertionError("expected ValueError to propagate")


class TestTheOffloadReportsWhatTheCoroutineRaised:
    """The branch guard classifies the loop check only, not the awaited work.

    ``asyncio.get_running_loop()`` reports "no running loop" by raising
    ``RuntimeError``, so that class is the branch verdict. While the offload sat
    inside the same ``try``, a ``RuntimeError`` raised by the coroutine itself
    was read as that verdict: the fallback then called ``asyncio.run`` a second
    time, from inside the running loop and on an already-consumed coroutine, and
    the caller received "asyncio.run() cannot be called from a running event
    loop" instead of what the coroutine said.

    ``ValueError`` -- the class the neighbouring propagation test raises -- was
    never affected, which is why the whole ``RuntimeError`` family could reach a
    caller renamed.
    """

    @staticmethod
    def _raise_inside_a_running_loop(exc: BaseException) -> BaseException:
        """Resolve a coroutine that raises ``exc`` from inside a running loop."""

        async def boom() -> None:
            raise exc

        async def driver() -> None:
            return _resolve_coroutine(boom())

        try:
            asyncio.run(driver())
        except Exception as caught:  # noqa: BLE001 - the value under test
            return caught
        except BaseException as unexpected:
            # Nothing under test raises outside Exception; surfacing it beats
            # returning it for the caller to assert on.
            raise AssertionError(f"unexpected {type(unexpected).__name__} from the offload") from unexpected
        raise AssertionError(f"expected {type(exc).__name__} to propagate")

    @pytest.mark.parametrize(
        "exc_class",
        [RuntimeError, NotImplementedError, RecursionError],
        ids=lambda c: c.__name__,
    )
    def test_the_runtime_error_family_is_not_read_as_the_branch_verdict(self, exc_class: type[BaseException]) -> None:
        """A RuntimeError from the coroutine is its own, not "no running loop"."""
        caught = self._raise_inside_a_running_loop(exc_class("policy said this"))
        detail = (
            f"{exc_class.__name__} raised inside the offloaded coroutine reached the caller as "
            f"{type(caught).__name__}: {caught} -- the branch guard classified the coroutine's "
            f"own failure as its 'no running loop' verdict"
        )
        assert isinstance(caught, exc_class), detail
        assert "policy said this" in str(caught), detail

    def test_an_unrelated_exception_class_is_unaffected(self) -> None:
        """The classes that already propagated still do (the guard did not widen)."""
        caught = self._raise_inside_a_running_loop(ValueError("kaboom"))
        assert isinstance(caught, ValueError)
        assert str(caught) == "kaboom"

    def test_the_no_loop_branch_is_still_selected_by_the_same_verdict(self) -> None:
        """With no running loop the coroutine is still resolved, not refused."""

        async def produce() -> str:
            return "resolved"

        assert _resolve_coroutine(produce()) == "resolved"

    def test_a_failure_with_no_running_loop_also_reaches_the_caller(self) -> None:
        """The unchanged branch keeps reporting the coroutine's own failure."""

        async def boom() -> None:
            raise RuntimeError("policy said this")

        with pytest.raises(RuntimeError, match="policy said this"):
            _resolve_coroutine(boom())
