"""Async-to-sync helper for resolving coroutines in sync contexts."""

import asyncio
import concurrent.futures

# Module-level executor reused across calls to avoid creating threads at high frequency.
# A single worker is sufficient - we only need to offload one asyncio.run() at a time.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="strands_async")


def _resolve_coroutine(coro_or_result):  # type: ignore[no-untyped-def]
    """Safely resolve a potentially-async result to a sync value.

    Handles three cases:
        1. Already a plain value → return as-is
        2. Coroutine, no running loop → asyncio.run()
        3. Coroutine, inside running loop → offload to reused thread

    In every case an exception raised by the coroutine reaches the caller
    unchanged: the branch guard classifies only whether a loop is running, so no
    exception class the coroutine may raise is mistaken for that verdict.

    Args:
        coro_or_result: Either a coroutine or an already-resolved value.

    Returns:
        The resolved (sync) value.
    """
    if not asyncio.iscoroutine(coro_or_result):
        return coro_or_result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop. This is the ONLY thing the guard classifies: with the
        # offload inside it, a RuntimeError raised by the awaited coroutine
        # itself landed here too, and the fallback then re-ran an
        # already-consumed coroutine from inside the loop -- so the caller was
        # handed "asyncio.run() cannot be called from a running event loop"
        # in place of what the coroutine actually said. RuntimeError's whole
        # family (NotImplementedError, RecursionError) reached the caller that
        # way, and NotImplementedError is exactly the MRO-contract violation a
        # caller must be able to see.
        return asyncio.run(coro_or_result)
    return _EXECUTOR.submit(asyncio.run, coro_or_result).result()
