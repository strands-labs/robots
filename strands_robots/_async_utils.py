"""Async-to-sync helpers shared across robot.py and simulation backends."""

import asyncio
import concurrent.futures


def _resolve_coroutine(coro_or_result):
    """Safely resolve a potentially-async result to a sync value.

    Handles three cases:
        1. Already a plain value → return as-is
        2. Coroutine, no running loop → asyncio.run()
        3. Coroutine, inside running loop → offload to thread

    Args:
        coro_or_result: Either a coroutine or an already-resolved value.

    Returns:
        The resolved (sync) value.
    """
    if not asyncio.iscoroutine(coro_or_result):
        return coro_or_result
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro_or_result).result()
    except RuntimeError:
        return asyncio.run(coro_or_result)


def _run_async(coro_fn):
    """Run an async function from sync context, handling nested loops.

    Args:
        coro_fn: Zero-argument async callable (e.g. ``lambda: some_coro()``).

    Returns:
        The result of the coroutine.
    """
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro_fn()).result()
    except RuntimeError:
        return asyncio.run(coro_fn())
