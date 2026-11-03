"""LeRobot loads a checkpoint quietly: one module's print and one logger, never ``sys.stdout``."""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator

_quiet = threading.Lock()
_quiet_depth, _quiet_level = 0, logging.NOTSET


@contextlib.contextmanager
def quiet_load() -> Iterator[None]:
    """A checkpoint loads without LeRobot's warning about its saved device (the asked-for device replaces it)
    or its "Loading weights" print. Only that module's print is muted, never ``sys.stdout``: loads run on
    a run's thread, and another thread's output (or an overlapping load) must stay untouched."""
    global _quiet_depth, _quiet_level
    from lerobot.policies import pretrained

    log = logging.getLogger("lerobot.configs.policies")
    with _quiet:
        if _quiet_depth == 0:
            _quiet_level = log.level
            log.setLevel(logging.ERROR)
            pretrained.print = lambda *a, **k: None  # type: ignore[attr-defined]
        _quiet_depth += 1
    try:
        yield
    finally:
        with _quiet:
            _quiet_depth -= 1
            if _quiet_depth == 0:
                log.setLevel(_quiet_level)
                del pretrained.print  # type: ignore[attr-defined]
