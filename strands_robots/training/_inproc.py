"""In-process execution helpers shared by the training backends (no subprocess).

Every backend drives its upstream pipeline by **importing the package and
calling its own function** - LeRobot's ``train(cfg)``, GR00T's
``experiment.run(config)``, Cosmos's ``convert_model_to_dcp(args)`` /
``train.launch(config, args)`` / ``export_model(args)``. None of them shell out
to a second interpreter or a ``torchrun`` binary.

Two primitives:

* :func:`call_callable` - run a Python callable in THIS interpreter with its
  stdout/stderr + root-logger output tee'd to a per-run log file (so the
  trainers can still parse a "RUNNING != learning" verdict, exactly as they did
  from the old subprocess log). Both halves go through one file object, so
  neither can overwrite the other's bytes.

* :func:`elastic_launch_callable` - multi-GPU single-node launch via torch's
  programmatic :class:`torch.distributed.launcher.api.elastic_launch` (the same
  elastic agent ``torchrun`` uses, driven in-process). It spawns one worker per
  GPU; each worker calls a Python callable with arguments passed as Python
  objects - there is no command line to assemble or inject into. The worker
  reads ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` that the agent sets, which is
  exactly what HF ``TrainingArguments`` / lerobot ``accelerate`` /
  cosmos ``distributed.init()`` expect.

Argument-injection safety for the few remaining string values (upstream config
flags built partly from the agent-supplied ``TrainSpec.extra``) lives in
:mod:`strands_robots.training._validate`, called fail-closed from every
backend's ``validate()``.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)


class _Tee(io.TextIOBase):
    """Write-through tee: forwards writes to both a live stream and a log file."""

    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:  # type: ignore[override]
        for stream in (self._primary, self._secondary):
            try:
                stream.write(s)
            except Exception:  # noqa: BLE001 - never let logging break training
                pass
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        for stream in (self._primary, self._secondary):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass


class capture_to_file:
    """Context manager: tee stdout/stderr + root-logger output into ``log_path``.

    Exactly one file object ever writes ``log_path``: the stream the tee holds,
    which the root-logger handler is pointed at too. Both halves therefore share
    one write offset and land in the order they were produced. Two file objects
    over one path would not - a buffered tee write and an appending handler each
    track their own offset, so whichever flushes second overwrites the other's
    bytes in place, and the log a trainer reads back for its
    "RUNNING != learning" verdict is left holding neither stream whole.

    ``log_path=None`` is a no-op (used by non-rank-0 workers so only rank 0
    writes the shared log).
    """

    def __init__(self, log_path: str | None) -> None:
        self.log_path = log_path
        self._stream: io.TextIOBase | None = None
        self._fh: logging.StreamHandler | None = None
        self._r_out: Any = None
        self._r_err: Any = None

    def __enter__(self) -> capture_to_file:
        if not self.log_path:
            return self
        self._stream = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115
        # Records go through the *same* file object the tee writes to. Opening
        # the path a second time (a FileHandler) gives two file objects with
        # independent write offsets over one file: the tee's buffered writes
        # land at its own offset and overwrite whatever the handler appended
        # there, so the log ends up holding neither stream whole.
        self._fh = logging.StreamHandler(self._stream)
        self._fh.setLevel(logging.INFO)
        self._fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(self._fh)
        self._r_out = contextlib.redirect_stdout(_Tee(sys.stdout, self._stream))
        self._r_err = contextlib.redirect_stderr(_Tee(sys.stderr, self._stream))
        self._r_out.__enter__()
        self._r_err.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._r_err is not None:
                self._r_err.__exit__(*exc)
            if self._r_out is not None:
                self._r_out.__exit__(*exc)
        finally:
            if self._fh is not None:
                logging.getLogger().removeHandler(self._fh)
                self._fh.close()
            if self._stream is not None:
                self._stream.close()


@contextlib.contextmanager
def resume_argv(config_path: str | None) -> Iterator[None]:
    """Expose ``--config_path=<train_config.json>`` on ``sys.argv`` for a resume.

    lerobot's ``TrainPipelineConfig.validate()`` resolves a resumed run by reading
    ``--config_path`` back off ``sys.argv`` (``parser.parse_arg``): draccus has
    already consumed the flag by the time ``validate()`` runs, so it is recovered
    from the raw args rather than the config object. The in-process path builds the
    ``TrainPipelineConfig`` directly and never populates ``sys.argv``, so a
    ``resume=True`` config would raise ``ValueError("A config_path is expected when
    resuming a run...")`` before training starts. This context manager injects the
    flag for the duration of the ``train(cfg)`` call and restores the original
    ``sys.argv`` in a ``finally``. A no-op when ``config_path`` is falsy.
    """
    if not config_path:
        yield
        return
    saved_argv = sys.argv
    sys.argv = [*saved_argv, f"--config_path={config_path}"]
    try:
        yield
    finally:
        sys.argv = saved_argv


def call_callable(
    fn: Callable[..., Any],
    *args: Any,
    log_path: str | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``fn(*args, **kwargs)`` in-process, output captured to ``log_path``.

    The purest "import the package and use it" path: the caller has already
    built the upstream's own config object and just needs its function invoked
    here. No argv, no shell, no nested interpreter.
    """
    with capture_to_file(log_path):
        return fn(*args, **kwargs)


def elastic_launch_callable(
    fn: Callable[..., Any],
    *,
    nproc_per_node: int,
    nnodes: int = 1,
    rdzv_endpoint: str = "",
    rdzv_backend: str = "c10d",
    run_id: str = "",
    fn_args: tuple[Any, ...] = (),
) -> Any:
    """Multi-process launch via torch's programmatic elastic launcher (no torchrun).

    Spawns ``nproc_per_node`` workers with torch's elastic agent (Python
    multiprocessing) and calls ``fn(*fn_args)`` in each. Arguments are Python
    objects, so there is no command line to inject into - the shell-free
    replacement for ``torchrun --nproc_per_node=N``. For ``nnodes > 1`` a shared
    ``rdzv_endpoint`` (host:port reachable by every node) is required.
    """
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

    config = LaunchConfig(
        min_nodes=nnodes,
        max_nodes=nnodes,
        nproc_per_node=nproc_per_node,
        run_id=run_id or "strands-train",
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint or "localhost:0",
        max_restarts=0,
        start_method="spawn",
    )
    return elastic_launch(config, fn)(*fn_args)
