"""An abandoned fixture work item must not turn a red test into a hung job.

``Robot.start_task`` submits the rollout to ``Robot._executor``, and the
fixtures that build that executor for a ``Robot`` assembled via ``__new__``
sometimes wait for the submitted work with ``future.result(timeout=...)``. When
that wait gives up, the work item is still running. A ``ThreadPoolExecutor``
worker is **not** a daemon thread and the class registers an interpreter-exit
hook that joins every worker it started, so the process cannot exit while the
work item is wedged: pytest prints the verdict and the job then hangs.

Measured on that shape -- one worker, a wedged item, a wait that gives up after
2s, ``shutdown(wait=False)`` in teardown -- pytest reported ``1 failed in
2.03s`` and a 45s wall clock had to kill the process (exit 124). A failing test
was delivered as a hung job.

:mod:`tests._daemon_executor` keeps the semantics those fixtures asked for and
runs the worker as a daemon, which the interpreter does not join. This module
pins both halves: that the helper really does let the process exit, and that the
``ThreadPoolExecutor`` shape really does not -- so the helper cannot be reverted
to the shape it replaced without a test saying so.

The verdict itself is untouched. Every test here asserts the wait still fails;
only the hang goes away.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from tests._daemon_executor import DaemonThreadExecutor

_TESTS_DIR = pathlib.Path(inspect.getfile(DaemonThreadExecutor)).resolve().parent
_ROOT = _TESTS_DIR.parent

# Derived from the helper module rather than written as a path literal, so a
# move of the test tree cannot silently point the scan somewhere empty.
assert (_ROOT / "pyproject.toml").is_file(), f"repository root not found at {_ROOT}"

_CHILD = """
import sys, threading
sys.path.insert(0, {root!r})
{import_line}
ex = {ctor}
wedged = threading.Event()
fut = ex.submit(wedged.wait)
try:
    fut.result(timeout=0.3)
except TimeoutError:
    print("VERDICT: the wait gave up", flush=True)
ex.shutdown(wait=False)
"""


def _child(import_line: str, ctor: str, budget: float) -> tuple[bool, str]:
    """Run the abandon-a-work-item shape in a child interpreter.

    Args:
        import_line: Import statement for the executor under test.
        ctor: Constructor expression for the executor.
        budget: Seconds to allow the child before declaring it stuck.

    Returns:
        ``(exited, stdout)`` where ``exited`` is whether the interpreter
        terminated on its own inside ``budget``.
    """
    code = _CHILD.format(root=str(_ROOT), import_line=import_line, ctor=ctor)
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", code],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=budget,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        return False, (expired.stdout or b"").decode() if isinstance(expired.stdout, bytes) else (expired.stdout or "")
    return True, done.stdout


class TestTheWorkerCannotOutliveTheInterpreter:
    """The one property that makes exit possible: the worker is a daemon."""

    def test_the_helper_worker_is_a_daemon_thread(self) -> None:
        executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix="probe")
        try:
            started = threading.Event()
            executor.submit(started.set).result(timeout=5)
            worker = executor.worker
            assert worker is not None, "submit did not start a worker"
            assert worker.daemon is True, "the worker would be joined at interpreter exit"
        finally:
            executor.shutdown(wait=True)

    def test_a_thread_pool_worker_is_not_a_daemon(self) -> None:
        """The premise: the shape this helper replaces is joined at exit."""
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="premise")
        try:
            pool.submit(lambda: None).result(timeout=5)
            workers = [t for t in threading.enumerate() if t.name.startswith("premise")]
            assert workers, "no worker to inspect"
            assert all(not t.daemon for t in workers), (
                "ThreadPoolExecutor workers are non-daemon; if that changed the helper is redundant"
            )
        finally:
            pool.shutdown(wait=True)


class TestAnAbandonedWorkItemDoesNotBlockInterpreterExit:
    """End to end, in a child interpreter, for both executor shapes."""

    def test_the_helper_lets_the_interpreter_exit(self) -> None:
        exited, out = _child(
            "from tests._daemon_executor import DaemonThreadExecutor",
            "DaemonThreadExecutor(max_workers=1)",
            budget=20.0,
        )
        assert "VERDICT" in out, f"the wait did not give up: {out!r}"
        assert exited, "the interpreter did not exit while a work item was wedged"

    def test_the_thread_pool_shape_does_not(self) -> None:
        """Why the helper exists. Reverting to a pool reinstates the hang."""
        exited, out = _child(
            "from concurrent.futures import ThreadPoolExecutor",
            "ThreadPoolExecutor(max_workers=1)",
            budget=5.0,
        )
        assert "VERDICT" in out, f"the wait did not give up: {out!r}"
        assert not exited, (
            "ThreadPoolExecutor no longer joins a running worker at exit; if that is now true the helper can be retired"
        )


class TestPytestDeliversTheVerdictAndThenExits:
    """The incident shape: a failing test must cost a failure, not a hung job."""

    def test_a_failing_wait_is_reported_and_the_run_ends(self, tmp_path: pathlib.Path) -> None:
        generated = tmp_path / "test_generated_abandon.py"
        generated.write_text(
            "import threading\n"
            "from typing import Any\n"
            "import pytest\n"
            "from tests._daemon_executor import DaemonThreadExecutor\n"
            "\n"
            "@pytest.fixture\n"
            "def pool() -> Any:\n"
            "    ex = DaemonThreadExecutor(max_workers=1)\n"
            "    try:\n"
            "        yield ex\n"
            "    finally:\n"
            "        ex.shutdown(wait=False)\n"
            "\n"
            "def test_the_wait_gives_up(pool: Any) -> None:\n"
            "    wedged = threading.Event()\n"
            "    pool.submit(wedged.wait).result(timeout=0.3)\n"
        )
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-m", "pytest", str(generated), "-q", "--no-cov", "-p", "no:cacheprovider"],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:  # pragma: no cover - the defect this pins
            pytest.fail("the child pytest run never exited after the failing wait")
        assert done.returncode != 0, "the abandoned wait should still fail the test"
        assert "1 failed" in done.stdout, done.stdout[-2000:]


class TestTheDropInContractHolds:
    """The helper stands in for ``ThreadPoolExecutor(max_workers=1)``."""

    def test_a_result_is_returned(self) -> None:
        executor = DaemonThreadExecutor()
        try:
            assert executor.submit(lambda: "chunk").result(timeout=5) == "chunk"
        finally:
            executor.shutdown(wait=True)

    def test_an_exception_reaches_the_caller(self) -> None:
        executor = DaemonThreadExecutor()

        def _boom() -> None:
            raise ValueError("from the worker")

        try:
            with pytest.raises(ValueError, match="from the worker"):
                executor.submit(_boom).result(timeout=5)
        finally:
            executor.shutdown(wait=True)

    def test_work_runs_in_submit_order_on_one_worker(self) -> None:
        executor = DaemonThreadExecutor()
        seen: list[int] = []
        try:
            futures = [executor.submit(seen.append, i) for i in range(5)]
            for future in futures:
                future.result(timeout=5)
            assert seen == [0, 1, 2, 3, 4]
        finally:
            executor.shutdown(wait=True)

    def test_submit_after_shutdown_is_refused(self) -> None:
        executor = DaemonThreadExecutor()
        executor.shutdown(wait=True)
        with pytest.raises(RuntimeError, match="after shutdown"):
            executor.submit(lambda: None)

    def test_shutdown_waiting_joins_the_worker(self) -> None:
        executor = DaemonThreadExecutor()
        executor.submit(lambda: None).result(timeout=5)
        worker = executor.worker
        assert worker is not None
        executor.shutdown(wait=True)
        assert worker.is_alive() is False

    def test_cancel_futures_cancels_queued_work(self) -> None:
        executor = DaemonThreadExecutor()
        release = threading.Event()
        running = threading.Event()

        def _hold() -> None:
            running.set()
            release.wait()

        # Wait until the holder is actually running, so the second submission is
        # the one still queued when the drain happens.
        holder = executor.submit(_hold)
        assert running.wait(5), "the worker never picked up the first item"
        queued = executor.submit(lambda: None)
        executor.shutdown(wait=False, cancel_futures=True)
        assert queued.cancelled() is True
        release.set()
        holder.result(timeout=5)

    def test_more_than_one_worker_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_workers must be 1"):
            DaemonThreadExecutor(max_workers=2)

    def test_robot_cleanup_drains_the_executor_it_is_handed(self) -> None:
        """The production call the helper must honour (``shutdown(wait=True)``)."""
        source = inspect.getsource(HwRobot.cleanup)
        assert "_executor.shutdown(wait=True)" in source, (
            "Robot.cleanup no longer drains the executor; the helper's shutdown contract was written against that call"
        )


def _executor_constructors(tree: ast.AST) -> list[tuple[int, str]]:
    """Return ``(line, constructor)`` for each ``*._executor = <Call>``."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "_executor":
                func = node.value.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                found.append((node.lineno, name))
    return found


def _abandons_a_work_item(tree: ast.AST) -> bool:
    """Whether the module gives up on a future via ``result(timeout=...)``."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "result"
        and any(keyword.arg == "timeout" for keyword in node.keywords)
        for node in ast.walk(tree)
    )


def _unsafe_modules(files: list[pathlib.Path]) -> dict[str, list[int]]:
    """Modules that abandon a work item on a non-daemon fixture executor."""
    offenders: dict[str, list[int]] = {}
    for path in files:
        tree = ast.parse(path.read_text())
        constructors = _executor_constructors(tree)
        if not constructors or not _abandons_a_work_item(tree):
            continue
        bad = [line for line, name in constructors if name != "DaemonThreadExecutor"]
        if bad:
            offenders[path.name] = bad
    return offenders


class TestEveryAbandoningFixtureUsesTheDaemonExecutor:
    """A fixture whose work item a test may abandon must use the helper.

    Scoped to executors assigned to ``*._executor`` -- the attribute
    ``Robot.start_task`` submits to -- so a pool a test builds and drains inside
    its own ``with`` block is out of scope by construction rather than by an
    exemption. The scan does not verify *ordering*: a module that drains through
    ``Robot.cleanup`` before its wait is safe today and would still be safe with
    the helper, so demanding the helper costs nothing and removes the dependence
    on that ordering. Fixtures that never abandon a work item are a wider class
    and are deliberately left alone.
    """

    def test_no_module_abandons_work_on_a_non_daemon_executor(self) -> None:
        offenders = _unsafe_modules(sorted(_TESTS_DIR.rglob("test_*.py")))
        assert offenders == {}, (
            "these fixtures leave a running work item for interpreter exit; "
            f"use tests._daemon_executor.DaemonThreadExecutor: {offenders}"
        )

    def test_the_scan_covers_the_modules_that_abandon_work(self) -> None:
        """Non-vacuity: the known abandoning modules are the ones scanned."""
        abandoning = {
            path.name
            for path in sorted(_TESTS_DIR.rglob("test_*.py"))
            if _executor_constructors(tree := ast.parse(path.read_text())) and _abandons_a_work_item(tree)
        }
        assert abandoning == {
            "test_hardware_bus_is_shared_with_the_mesh.py",
            "test_hardware_cleanup_disconnects.py",
            "test_hardware_policy_port_domain.py",
            "test_hardware_robot_lifecycle.py",
        }, abandoning

    def test_the_scan_reports_a_planted_thread_pool_fixture(self, tmp_path: pathlib.Path) -> None:
        planted = tmp_path / "test_planted_pool.py"
        planted.write_text(
            "from concurrent.futures import ThreadPoolExecutor\n"
            "def _make(hw):\n"
            "    hw._executor = ThreadPoolExecutor(max_workers=1)\n"
            "def test_x(hw):\n"
            "    hw._task_state.task_future.result(timeout=5)\n"
        )
        assert _unsafe_modules([planted]) == {"test_planted_pool.py": [3]}

    def test_the_scan_passes_a_planted_daemon_fixture(self, tmp_path: pathlib.Path) -> None:
        planted = tmp_path / "test_planted_daemon.py"
        planted.write_text(
            "from tests._daemon_executor import DaemonThreadExecutor\n"
            "def _make(hw):\n"
            "    hw._executor = DaemonThreadExecutor(max_workers=1)\n"
            "def test_x(hw):\n"
            "    hw._task_state.task_future.result(timeout=5)\n"
        )
        assert _unsafe_modules([planted]) == {}
